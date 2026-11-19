"""Graph tools for analysts: connection finder and entity comparison.

Both work on the corpus knowledge graph that ``VerificationService`` already
holds, resolve names through the same alias resolver as verification, and keep
the source sentence, publisher and date on every relation they return.

* ``find_connections`` - the shortest evidence chains between two entities.
  Edges through very generic hubs (e.g. "government") are weighted up so
  specific links are preferred, and each hop is re-checked with the NLI
  verifier so the analyst sees which links the text actually states.
* ``compare_entities`` - what 2-3 actors / malware families share (tools,
  targets, techniques, CVEs) and what is unique to each.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx

from src.core.errors import EntityNotFoundError, InvalidInputError

MAX_HOPS = 4
MAX_PATHS = 3


# ------------------------------------------------------------------ shared helpers
def _index(service: Any) -> Dict[str, Any]:
    return service._graph()


def resolve_node(service: Any, name: str) -> Dict[str, Any]:
    if not isinstance(name, str) or not name.strip() or len(name) > 120:
        raise InvalidInputError("Entity names must be 1-120 characters")
    node = service._find_node(service.aliases.resolve(name.strip()))
    if node is None:
        raise EntityNotFoundError(f"{name!r} is not in the corpus knowledge graph",
                                  hint="Try another spelling or alias; only entities linked in the graph can be connected or compared.")
    return node


def _rel(link: Dict[str, Any]) -> str:
    return link.get("relation_type") or link.get("key") or "RELATED_TO"


def _evidence(link: Dict[str, Any]) -> Dict[str, Any]:
    ev = (link.get("evidence") or [{}])[0]
    return {
        "text": (link.get("evidence_text") or ev.get("evidence_text") or "")[:500],
        "publisher": link.get("source_publisher") or ev.get("source") or (link.get("sources") or [None])[0],
        "date": link.get("publication_date") or ev.get("publication_date"),
        "report_id": link.get("report_id") or ev.get("report_id"),
        "evidence_count": int(link.get("evidence_count", 1) or 1),
        "publishers": sorted(set(link.get("sources") or ([link["source_publisher"]] if link.get("source_publisher") else []))),
    }


def _node_out(n: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": n["id"], "name": n.get("name") or n["id"].split("::")[-1], "category": n.get("category") or "Entity",
            "mentions": int(n.get("mention_count", 0) or 0)}


def list_entities(service: Any) -> List[Dict[str, Any]]:
    """Entities that have at least one relation, most mentioned first (for autocomplete)."""
    gi = _index(service)
    out = [dict(_node_out(n), degree=len(gi["links"].get(nid, []))) for nid, n in gi["nodes"].items() if gi["links"].get(nid)]
    return sorted(out, key=lambda r: (-r["mentions"], r["name"].lower()))


# ------------------------------------------------------------------ connection finder
def build_nx(service: Any) -> nx.Graph:
    """Undirected view of the positive relations; parallel relations are kept on the edge."""
    cached = getattr(service, "_nx_graph", None)
    if cached is not None:
        return cached
    gi = _index(service)
    g = nx.Graph()
    seen = set()
    for links in gi["links"].values():
        for l in links:
            key = (l["source"], l["target"], _rel(l), l.get("report_id"), l.get("sentence_id"))
            if key in seen or l.get("polarity", "positive") != "positive":
                continue
            seen.add(key)
            if l["source"] == l["target"]:
                continue
            if g.has_edge(l["source"], l["target"]):
                g[l["source"]][l["target"]]["links"].append(l)
            else:
                g.add_edge(l["source"], l["target"], links=[l])
    deg = dict(g.degree())
    for u, v, d in g.edges(data=True):
        # Prefer specific links: passing through a hub with many relations costs more.
        d["w"] = 1.0 + 0.35 * (math.log2(1 + deg[u]) + math.log2(1 + deg[v])) / 2
    service._nx_graph = g
    return g


def _best_link(links: List[Dict[str, Any]]) -> Dict[str, Any]:
    return sorted(links, key=lambda l: (-int(l.get("evidence_count", 1) or 1), _rel(l)))[0]


def find_connections(
    service: Any, a: str, b: str, max_hops: int = MAX_HOPS, max_paths: int = MAX_PATHS, verify: bool = True,
) -> Dict[str, Any]:
    max_hops = max(1, min(int(max_hops), 6))
    max_paths = max(1, min(int(max_paths), 5))
    na, nb = resolve_node(service, a), resolve_node(service, b)
    if na["id"] == nb["id"]:
        raise InvalidInputError("Pick two different entities")
    g = build_nx(service)
    gi = _index(service)
    paths: List[List[str]] = []
    if na["id"] in g and nb["id"] in g:
        try:
            for p in nx.shortest_simple_paths(g, na["id"], nb["id"], weight="w"):
                if len(p) - 1 > max_hops:
                    # weighted order is not strictly by length; keep scanning a little for short ones
                    if len(paths) >= max_paths or len(p) - 1 > max_hops + 2:
                        break
                    continue
                paths.append(p)
                if len(paths) >= max_paths:
                    break
        except nx.NetworkXNoPath:
            paths = []

    verdicts: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def verdict(subj: str, rel: str, obj: str) -> Optional[Dict[str, Any]]:
        if not verify:
            return None
        key = (subj.lower(), rel, obj.lower())
        if key not in verdicts:
            try:
                v = service.verify_claim(subj, rel, obj, use_llm_summary=False)
                verdicts[key] = {"status": v.status, "support_tier": v.support_tier, "hedged_only": v.hedged_only,
                                 "supporting": len(v.supporting), "contradicting": len(v.contradicting)}
            except Exception as exc:  # a hop that cannot be verified is reported, not fatal
                verdicts[key] = {"status": "ERROR", "error": str(exc)[:200]}
        return verdicts[key]

    out_paths = []
    for p in paths:
        hops = []
        for u, v in zip(p, p[1:]):
            link = _best_link(g[u][v]["links"])
            s_node, o_node = gi["nodes"][link["source"]], gi["nodes"][link["target"]]
            s_name, o_name = _node_out(s_node)["name"], _node_out(o_node)["name"]
            rels = sorted({_rel(l) for l in g[u][v]["links"]})
            hops.append({
                "from": u, "to": v, "subject": s_name, "relation": _rel(link), "object": o_name,
                "forward": link["source"] == u, "all_relations": rels, **_evidence(link),
                "verdict": verdict(s_name, _rel(link), o_name),
            })
        confirmed = sum(1 for h in hops if (h["verdict"] or {}).get("status") == "SUPPORTED")
        out_paths.append({
            "nodes": [_node_out(gi["nodes"][nid]) for nid in p],
            "hops": hops, "length": len(hops), "confirmed_hops": confirmed,
        })
    return {
        "source": _node_out(na), "target": _node_out(nb), "max_hops": max_hops,
        "paths": out_paths, "verified": verify,
        "note": "Relations are extracted automatically; each hop shows whether the NLI verifier finds the source text states it.",
    }


# ------------------------------------------------------------------ comparison
GROUPS = [
    ("uses", "Uses (malware & tools)", "USES", True),
    ("used_by", "Used by", "USES", False),
    ("targets", "Targets", "TARGETS", True),
    ("techniques", "ATT&CK techniques", "USES_TECHNIQUE", True),
    ("exploits", "Exploits", "EXPLOITS", True),
    ("targeted_by", "Targeted by", "TARGETS", False),
]


def _neighbors(service: Any, node_id: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """group key -> other entity name (lower) -> {name, category, evidence[]}."""
    gi = _index(service)
    out: Dict[str, Dict[str, Dict[str, Any]]] = {k: {} for k, *_ in GROUPS}
    for l in gi["links"].get(node_id, []):
        if l.get("polarity", "positive") != "positive":
            continue
        outbound = l["source"] == node_id
        other_id = l["target"] if outbound else l["source"]
        if other_id == node_id:
            continue
        for key, _label, rel, is_out in GROUPS:
            if _rel(l) == rel and outbound == is_out:
                other = _node_out(gi["nodes"].get(other_id, {"id": other_id}))
                row = out[key].setdefault(other["name"].lower(), {"name": other["name"], "category": other["category"], "evidence": []})
                row["evidence"].append(_evidence(l))
    return out


def compare_entities(service: Any, names: List[str]) -> Dict[str, Any]:
    if not isinstance(names, list) or not 2 <= len(names) <= 3:
        raise InvalidInputError("Compare 2 or 3 entities")
    nodes = [resolve_node(service, n) for n in names]
    if len({n["id"] for n in nodes}) != len(nodes):
        raise InvalidInputError("Pick different entities (two names resolved to the same one)")
    ents = []
    for n in nodes:
        info = _node_out(n)
        try:
            e = service.get_entity(info["name"])
            info.update(mitre_id=e.mitre_id, aliases=list(e.aliases)[:10], first_seen=e.first_seen, last_seen=e.last_seen)
        except Exception:
            info.update(mitre_id=None, aliases=[], first_seen=n.get("first_seen"), last_seen=n.get("last_seen"))
        ents.append(info)
    neigh = [_neighbors(service, n["id"]) for n in nodes]
    k = len(nodes)
    groups = []
    all_sets = [set() for _ in nodes]
    for key, label, rel, is_out in GROUPS:
        names_in = sorted({nm for nb in neigh for nm in nb[key]})
        if not names_in:
            continue
        items = []
        for nm in names_in:
            by = {ents[i]["name"]: neigh[i][key][nm]["evidence"] for i in range(k) if nm in neigh[i][key]}
            first = next(neigh[i][key][nm] for i in range(k) if nm in neigh[i][key])
            items.append({"name": first["name"], "category": first["category"], "shared_by": list(by), "evidence": by})
            for i in range(k):
                if nm in neigh[i][key]:
                    all_sets[i].add((key, nm))
        items.sort(key=lambda it: (-len(it["shared_by"]), it["name"].lower()))
        groups.append({
            "key": key, "label": label, "relation": rel, "outbound": is_out, "items": items,
            "shared_all": sum(1 for it in items if len(it["shared_by"]) == k),
        })
    union = set().union(*all_sets)
    inter = set.intersection(*all_sets) if all_sets else set()
    pair_overlap = []
    for i in range(k):
        for j in range(i + 1, k):
            u = all_sets[i] | all_sets[j]
            pair_overlap.append({"a": ents[i]["name"], "b": ents[j]["name"], "shared": len(all_sets[i] & all_sets[j]),
                                 "jaccard": round(len(all_sets[i] & all_sets[j]) / len(u), 3) if u else 0.0})
    return {
        "entities": ents, "groups": groups, "shared_all": len(inter), "total_distinct": len(union),
        "pairs": pair_overlap,
        "note": "Built from automatically extracted, unreviewed graph relations. Shared items are leads; open each to verify it.",
    }


def overlap_layer(cmp: Dict[str, Any], tactics: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """ATT&CK Navigator layer: score = how many of the compared entities use the technique."""
    from src.analyst.navigator import build_layer, technique_id

    names = [e["name"] for e in cmp["entities"]]
    rows = []
    for g in cmp["groups"]:
        if g["key"] != "techniques":
            continue
        for it in g["items"]:
            tid = technique_id(it["name"])
            if tid:
                reports = sorted({ev.get("report_id") for evs in it["evidence"].values() for ev in evs if ev.get("report_id")})
                rows.append({"technique_id": tid, "name": it["name"], "count": len(it["shared_by"]), "reports": reports,
                             "example": f"Used by: {', '.join(it['shared_by'])}"})
    return build_layer(" vs ".join(names), rows, tactics, source_label="CyberRAG comparison",
                       note="Score = number of compared entities linked to the technique. Links are automatic and unreviewed.")


def comparison_csv(cmp: Dict[str, Any], to_row: Optional[Callable] = None) -> str:
    import csv
    import io

    names = [e["name"] for e in cmp["entities"]]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["group", "item", "category"] + names)
    for g in cmp["groups"]:
        for it in g["items"]:
            w.writerow([g["label"], it["name"], it["category"]] + ["yes" if n in it["shared_by"] else "" for n in names])
    return buf.getvalue()
