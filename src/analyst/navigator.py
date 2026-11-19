"""MITRE ATT&CK Navigator layer export.

Builds a layer file (format 4.5) that opens in https://mitre-attack.github.io/attack-navigator/
via "Open Existing Layer -> Upload from local". The score of a technique is the
number of distinct evidence sentences that link the entity to it; every comment
names the reports, so the heat map stays traceable.

Technique links in the baseline corpus are automatically extracted and not
human reviewed (relation precision about one third, see README section 3). The
layer description says so; treat it as a lead list, not an attribution.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

_TID = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

LAYER_FORMAT = "4.5"
NAVIGATOR_VERSION = "5.1.0"
UNVERIFIED_NOTE = (
    "Technique links were extracted automatically from report text and are not human reviewed. "
    "Use them as leads; check the cited sentences before relying on them."
)


def technique_id(text: str) -> Optional[str]:
    m = _TID.search(text or "")
    return m.group(0) if m else None


def collect_techniques(
    assertions: Iterable[Dict[str, Any]], names: Iterable[str]
) -> List[Dict[str, Any]]:
    """Positive USES_TECHNIQUE assertions whose subject is one of ``names`` (case-insensitive).

    Returns one row per technique id with the evidence count, reports and one
    example sentence, sorted by count then id.
    """
    wanted = {n.strip().lower() for n in names if n and n.strip()}
    rows: Dict[str, Dict[str, Any]] = {}
    for a in assertions:
        if a.get("predicate") != "USES_TECHNIQUE" or a.get("polarity", "positive") != "positive":
            continue
        if str(a.get("subject", "")).strip().lower() not in wanted:
            continue
        tid = technique_id(str(a.get("object", "")))
        if not tid:
            continue
        row = rows.setdefault(tid, {
            "technique_id": tid,
            "name": str(a.get("object", "")).split(" - ", 1)[-1].strip(),
            "evidence": set(),
            "reports": set(),
            "example": a.get("evidence_sentence") or a.get("claim_text") or "",
        })
        row["evidence"].add(a.get("relation_id") or a.get("assertion_id") or str(len(row["evidence"])))
        if a.get("report_id"):
            row["reports"].add(a["report_id"])
    out = []
    for r in rows.values():
        out.append({
            "technique_id": r["technique_id"], "name": r["name"], "count": len(r["evidence"]),
            "reports": sorted(r["reports"]), "example": r["example"][:400],
        })
    return sorted(out, key=lambda r: (-r["count"], r["technique_id"]))


def build_layer(
    entity_name: str,
    techniques: List[Dict[str, Any]],
    tactics: Optional[Dict[str, str]] = None,
    source_label: str = "CyberRAG corpus",
    note: str = UNVERIFIED_NOTE,
) -> Dict[str, Any]:
    """Navigator layer JSON. ``tactics`` maps technique id -> tactic name (for the comment only)."""
    tactics = tactics or {}
    max_score = max((t.get("count", 1) for t in techniques), default=1)
    entries = []
    for t in techniques:
        reports = t.get("reports") or []
        comment = f"{t.get('count', 1)} evidence sentence(s)"
        if reports:
            comment += f" in {', '.join(reports[:5])}" + (" …" if len(reports) > 5 else "")
        if tactics.get(t["technique_id"]):
            comment += f". Tactic: {tactics[t['technique_id']]}"
        entries.append({
            "techniqueID": t["technique_id"],
            "score": int(t.get("count", 1)),
            "comment": comment,
            "enabled": True,
            "showSubtechniques": "." in t["technique_id"],
            "metadata": [{"name": "example", "value": (t.get("example") or "")[:300]}] if t.get("example") else [],
        })
    return {
        "name": f"{entity_name} – {source_label}"[:100],
        "versions": {"layer": LAYER_FORMAT, "navigator": NAVIGATOR_VERSION},
        "domain": "enterprise-attack",
        "description": f"Techniques linked to {entity_name} in {source_label}. {note}",
        "sorting": 3,
        "hideDisabled": False,
        "techniques": entries,
        "gradient": {"colors": ["#ffe766", "#ff6666"], "minValue": 0, "maxValue": max(1, max_score)},
        "legendItems": [],
        "metadata": [{"name": "generated_by", "value": "CyberRAG"}, {"name": "technique_count", "value": str(len(entries))}],
        "showTacticRowBackground": False,
        "selectTechniquesAcrossTactics": True,
        "selectSubtechniquesWithParent": False,
    }


def tactic_map(registry: Dict[str, Any]) -> Dict[str, str]:
    """Technique id -> tactic string from the shipped MITRE registry."""
    out = {}
    for tid, rec in (registry.get("techniques") or {}).items():
        tac = rec.get("tactic")
        if isinstance(tac, list):
            tac = ", ".join(tac)
        if tac:
            out[tid] = str(tac)
    return out
