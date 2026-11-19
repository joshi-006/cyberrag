"""Markdown analyst brief for a workbench investigation.

Takes the JSON returned by ``POST /api/investigate`` and renders a self-contained
brief an analyst can paste into a ticket or wiki: verdict, evidence with source
and date, conflict resolution, the decision trace, and the system's stated
limits. Report text is quoted verbatim but flattened to one line and stripped of
Markdown control characters so a malicious report cannot inject headings,
links or HTML into the brief.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List

LIMITS = (
    "NLI scores measure how well the text of a passage matches the claim; they are not calibrated "
    "probabilities that the claim is true. Graph assertions are extracted automatically and not human "
    "reviewed. Source diversity counts distinct publishers, so syndicated copies are not counted twice."
)

_MD_CONTROL = re.compile(r"[`*_\[\]<>#|\\]")


def _safe(text: Any, limit: int = 600) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    s = _MD_CONTROL.sub(lambda m: "\\" + m.group(0), s)
    return s[:limit] + ("…" if len(s) > limit else "")


def _evidence_block(items: List[Dict[str, Any]], score_key: str, label: str, kind: str) -> List[str]:
    if not items:
        return [f"_No {kind} evidence._", ""]
    out = []
    for i, e in enumerate(items, 1):
        out.append(
            f"{i}. **{_safe(e.get('source') or 'unknown publisher', 80)}** · {_safe(e.get('publication_date') or 'undated', 20)}"
            f" · {label} {float(e.get(score_key) or 0):.2f}"
        )
        out.append(f"   > {_safe(e.get('text'))}")
        ref = re.sub(r"[`\s]", "", str(e.get("segment_id") or e.get("report_id") or "n/a"))[:160]
        out.append(f"   Passage `{ref}`")
    out.append("")
    return out


def build_brief(inv: Dict[str, Any], generated_at: str = "") -> str:
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sup = inv.get("supporting_evidence") or []
    con = inv.get("contradicting_evidence") or []
    publishers = sorted({e.get("source") for e in sup + con if e.get("source")})
    lines = [
        f"# CTI verification brief: {_safe(inv.get('query'), 200)}",
        "",
        f"*Generated {generated_at} by CyberRAG. Evidence text is quoted from third-party reports.*",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Verdict | **{_safe(inv.get('status'), 60)}** |",
        f"| Consensus evidence score | {float(inv.get('consensus_evidence_score') or 0):.2f} |",
        f"| Independent publishers | {int(inv.get('source_diversity') or 0)} ({_safe(', '.join(publishers), 300) or 'none'}) |",
        f"| Supporting / contradicting passages | {len(sup)} / {len(con)} |",
        f"| Entities matched | {_safe(', '.join(m.get('name', '') for m in inv.get('matched_entities') or []), 300) or 'none'} |",
        "",
    ]
    cr = inv.get("conflict_resolution")
    if cr:
        lines += [
            "## Conflict resolution",
            "",
            f"**{_safe(cr.get('status'), 60)}**"
            + (f" · {cr['delta_days']} days between reports (threshold {cr.get('threshold', 'n/a')})" if cr.get("delta_days") is not None else ""),
            "",
            _safe(cr.get("reason"), 1000),
            "",
        ]
    lines += ["## Supporting evidence", ""] + _evidence_block(sup, "p_entail", "entailment", "supporting")
    lines += ["## Contradicting evidence", ""] + _evidence_block(con, "p_contra", "contradiction", "contradicting")
    trace = inv.get("execution_trace") or []
    if trace:
        lines += ["## Decision trace", ""]
        for s in trace:
            state = re.sub(r"[^A-Za-z0-9_ -]", "", str(s.get("state") or ""))[:40]
            lines.append(f"{s.get('step_index', '?')}. `{state}` {_safe(s.get('action'), 300)}")
            if s.get("retry_or_termination_reason"):
                lines.append(f"   *{_safe(s['retry_or_termination_reason'], 300)}*")
        lines.append("")
    lines += ["## Limits", "", LIMITS, ""]
    return "\n".join(lines)
