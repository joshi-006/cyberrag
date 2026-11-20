"""Answer entity questions: "What does APT28 target?", "Who uses Cobalt Strike?",
"Which techniques does Leviathan use?", "Tell me about Winnti".

The planner already finds the entity in such questions; this module works out
the relation and its direction from the wording, collects the matching graph
relations, and verifies each one with the NLI pipeline so the answer is a list
of checked facts with sources, not a raw graph dump.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.analyst.graphtools import _evidence, _node_out, _rel, resolve_node
from src.core.errors import InvalidInputError

MAX_ANSWERS = 15

_VERBS: List[Tuple[str, str]] = [
    (r"\btechniques?\b|\bttps?\b|\btactics?\b", "USES_TECHNIQUE"),
    (r"\bexploit(s|ed|ing)?\b|\bcves?\b|\bvulnerabilit(y|ies)\b", "EXPLOITS"),
    (r"\btarget(s|ed|ing)?\b|\battack(s|ed|ing)?\b|\bvictims?\b", "TARGETS"),
    (r"\b(use|uses|used|using|deliver(s|ed)?|deploy(s|ed)?|drop(s|ped)?|load(s|ed)?|distribut(e|es|ed)|run(s)?|leverag(e|es|ed))\b", "USES"),
]
_CATEGORY_WORDS: List[Tuple[str, str]] = [
    (r"\bmalware\b|\bfamil(y|ies)\b|\bransomware\b|\bloaders?\b|\btrojans?\b", "Malware"),
    (r"\btools?\b|\butilit(y|ies)\b", "Tool"),
    (r"\btechniques?\b|\bttps?\b", "Technique"),
    (r"\bcves?\b|\bvulnerabilit(y|ies)\b", "CVE"),
    (r"\bcountr(y|ies)\b|\bregions?\b|\blocations?\b", "Location"),
    (r"\bsectors?\b|\bindustr(y|ies)\b|\borgani[sz]ations?\b|\bcompan(y|ies)\b", "Organization"),
    (r"\b(threat )?actors?\b|\bgroups?\b|\bapts?\b", "ThreatActor"),
]
_QUESTION = re.compile(r"^\s*(what|which|who|whom|list|show|tell me about|describe)\b", re.I)


def is_entity_question(q: str) -> bool:
    return bool(_QUESTION.match(q or ""))


def parse(question: str, entity: str) -> Dict[str, Any]:
    """Relation, direction and category filter for a question about ``entity``."""
    q = question.lower()
    ent_pos = q.find(entity.lower())
    relation, verb_pos = None, -1
    for pat, rel in _VERBS:
        m = re.search(pat, q)
        if m:
            relation, verb_pos = rel, m.start()
            break
    category = None
    for pat, cat in _CATEGORY_WORDS:
        # "techniques"/"CVEs" also pick the relation; skip category words that are the entity itself
        m = re.search(pat, q)
        if m and not (ent_pos >= 0 and ent_pos <= m.start() < ent_pos + len(entity)):
            category = cat
            break
    # "Who uses X" / "Which groups target X": the entity comes after the verb -> it is the object.
    inbound = relation is not None and ent_pos >= 0 and verb_pos >= 0 and ent_pos > verb_pos
    if relation in ("USES_TECHNIQUE", "EXPLOITS") and category in ("Technique", "CVE"):
        inbound = False if ent_pos < verb_pos or category else inbound
    if category == "Technique" and relation == "USES":
        relation = "USES_TECHNIQUE"
    if category == "CVE" and relation in (None, "USES"):
        relation = "EXPLOITS"
    return {"relation": relation, "inbound": inbound, "category": category}


def _candidates(service: Any, node_id: str, relation: Optional[str], inbound: bool, category: Optional[str]) -> List[Dict[str, Any]]:
    gi = service._graph()
    rows: Dict[Tuple[str, str, bool], Dict[str, Any]] = {}
    for link in gi["links"].get(node_id, []):
        if link.get("polarity", "positive") != "positive" or link["source"] == link["target"]:
            continue
        outbound = link["source"] == node_id
        rel = _rel(link)
        if relation is not None and (rel != relation or outbound == inbound):
            continue
        other_id = link["target"] if outbound else link["source"]
        other = _node_out(gi["nodes"].get(other_id, {"id": other_id}))
        if category and other["category"] != category:
            continue
        key = (other["name"].lower(), rel, outbound)
        row = rows.setdefault(key, {"entity": other, "relation": rel, "outbound": outbound, "links": []})
        row["links"].append(link)
    out = list(rows.values())
    out.sort(key=lambda r: (-sum(int(l.get("evidence_count", 1) or 1) for l in r["links"]), r["entity"]["name"].lower()))
    return out


_ORDER = {"SUPPORTED": 0, "CONFLICTING_EVIDENCE": 1, "CONTRADICTED": 2, "INSUFFICIENT_EVIDENCE": 3, "ERROR": 4}


def answer(service: Any, question: str, max_answers: int = MAX_ANSWERS) -> Dict[str, Any]:
    if not isinstance(question, str) or not 3 <= len(question.strip()) <= 300:
        raise InvalidInputError("Ask a question of 3-300 characters")
    plan, _ = service.planner.plan(question)
    names = [e for e in plan.entities if e.lower() not in ("tell", "me", "about", "list", "show", "describe")]
    if not names:
        raise InvalidInputError("No known threat actor, malware or target found in the question",
                                hint="Try e.g. 'What does APT28 target?' or 'Who uses Cobalt Strike?'")
    node = resolve_node(service, names[0])
    ent = _node_out(node)
    p = parse(question, names[0])
    if p["relation"] is None:
        cands = _candidates(service, node["id"], None, False, p["category"])
    else:
        cands = _candidates(service, node["id"], p["relation"], p["inbound"], p["category"])
    truncated = len(cands) > max_answers
    items = []
    for c in cands[:max_answers]:
        subj, obj = (ent["name"], c["entity"]["name"]) if c["outbound"] else (c["entity"]["name"], ent["name"])
        best = sorted(c["links"], key=lambda l: -int(l.get("evidence_count", 1) or 1))[0]
        row = {"entity": c["entity"], "subject": subj, "relation": c["relation"], "object": obj,
               "outbound": c["outbound"], "graph_evidence": _evidence(best)}
        try:
            v = service.verify_claim(subj, c["relation"], obj, use_llm_summary=False)
            top = (v.supporting or v.contradicting or [None])[0]
            row.update(status=v.status, support_tier=v.support_tier, hedged_only=v.hedged_only,
                       supporting=len(v.supporting), contradicting=len(v.contradicting),
                       publishers=sorted({e.publisher for e in v.supporting if e.publisher}),
                       quote=None if top is None else {"text": (top.untrusted_text or "")[:400], "publisher": top.publisher,
                                                       "date": top.pub_date, "synthetic": top.synthetic})
        except Exception as exc:
            row.update(status="ERROR", error=str(exc)[:200])
        items.append(row)
    items.sort(key=lambda r: (_ORDER.get(r["status"], 9), -(r.get("supporting") or 0), r["entity"]["name"].lower()))
    confirmed = sum(1 for r in items if r["status"] == "SUPPORTED")
    return {
        "question": question.strip(), "entity": ent, "relation": p["relation"], "inbound": p["inbound"],
        "category": p["category"], "answers": items, "confirmed": confirmed, "truncated": truncated,
        "total_candidates": len(cands),
        "note": "Candidates come from automatically extracted graph relations; each was re-verified against the source passages.",
    }
