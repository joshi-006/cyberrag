"""Batch claim verification.

Input is one claim per line, in either form:

* structured: ``APT28 | USES | Zebrocy`` (also ``,`` / ``;`` / tab separated);
  prefix the relation with ``NOT`` for a negative claim (``Emotet | NOT USES | TrickBot``)
* free text: ``Does APT29 use Cobalt Strike?`` - sent through the question planner

Blank lines and lines starting with ``#`` are skipped. A header row
``subject,relation,object`` is skipped too, so a CSV exported from a sheet works.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Any, Dict, List, Optional

from src.core.errors import CTIError

MAX_LINES = 25
RELATIONS = {"USES", "TARGETS", "EXPLOITS", "USES_TECHNIQUE"}
_RELATION_ALIASES = {"USE": "USES", "TARGET": "TARGETS", "EXPLOIT": "EXPLOITS", "USES TECHNIQUE": "USES_TECHNIQUE",
                     "TECHNIQUE": "USES_TECHNIQUE"}
_SPLIT = re.compile(r"\s*(?:\||\t|;|,)\s*")


def _relation(token: str):
    t = re.sub(r"\s+", " ", token.strip().upper())
    neg = False
    if t.startswith("NOT "):
        neg, t = True, t[4:].strip()
    t = _RELATION_ALIASES.get(t, t).replace(" ", "_")
    return (t, "negative" if neg else "positive") if t in RELATIONS else (None, None)


def parse_line(line: str) -> Optional[Dict[str, Any]]:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    parts = [p.strip().strip('"') for p in _SPLIT.split(raw)]
    if len(parts) == 3:
        if [p.lower() for p in parts] == ["subject", "relation", "object"]:
            return None
        rel, pol = _relation(parts[1])
        if rel and parts[0] and parts[2]:
            return {"kind": "claim", "input": raw, "subject": parts[0], "relation": rel, "object": parts[2], "polarity": pol}
        if "|" in raw or "\t" in raw:  # clearly meant as structured: say what is wrong
            return {"kind": "invalid", "input": raw,
                    "error": f"Unknown relation {parts[1]!r}; use one of {', '.join(sorted(RELATIONS))} (optionally prefixed with NOT)."}
    return {"kind": "question", "input": raw}


def parse_lines(text: str) -> List[Dict[str, Any]]:
    items = [p for p in (parse_line(l) for l in (text or "").splitlines()) if p]
    return items


def _row_from_verification(item: Dict[str, Any], v: Dict[str, Any]) -> Dict[str, Any]:
    claim = v.get("claim", {})
    sup = v.get("supporting") or []
    publishers = sorted({e.get("publisher") for e in sup if e.get("publisher")})
    return {
        "input": item["input"], "kind": item["kind"],
        "claim": f"{claim.get('subject')} {'NOT ' if v.get('polarity') == 'negative' else ''}{claim.get('relation')} {claim.get('object')}",
        "status": v.get("status"), "support_tier": v.get("support_tier"),
        "supporting": len(sup), "contradicting": len(v.get("contradicting") or []),
        "publishers": publishers, "hedged_only": v.get("hedged_only", False),
        "conflict": (v.get("conflict") or {}).get("type"), "verification_id": v.get("verification_id"),
    }


def run_batch(service: Any, text: str) -> Dict[str, Any]:
    items = parse_lines(text)
    truncated = len(items) > MAX_LINES
    items = items[:MAX_LINES]
    rows: List[Dict[str, Any]] = []
    for item in items:
        if item["kind"] == "invalid":
            rows.append({"input": item["input"], "kind": "invalid", "claim": None, "status": "ERROR",
                         "error": item["error"], "support_tier": None, "supporting": 0, "contradicting": 0,
                         "publishers": [], "hedged_only": False, "conflict": None, "verification_id": None})
            continue
        try:
            if item["kind"] == "claim":
                v = service.verify_claim(item["subject"], item["relation"], item["object"],
                                         polarity=item["polarity"], use_llm_summary=False)
                rows.append(_row_from_verification(item, v.model_dump()))
            else:
                q = service.verify_question(item["input"])
                results = q.get("results") or []
                if not results:
                    rows.append({"input": item["input"], "kind": "question", "claim": "(no checkable claim found)",
                                 "status": q.get("overall_status"), "support_tier": None, "supporting": 0,
                                 "contradicting": 0, "publishers": [], "hedged_only": False, "conflict": None,
                                 "verification_id": None})
                for r in results:
                    rows.append(_row_from_verification(item, r))
        except CTIError as exc:
            rows.append({"input": item["input"], "kind": item["kind"], "claim": None, "status": "ERROR",
                         "error": exc.message, "support_tier": None, "supporting": 0, "contradicting": 0,
                         "publishers": [], "hedged_only": False, "conflict": None, "verification_id": None})
    summary: Dict[str, int] = {}
    for r in rows:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    return {"rows": rows, "summary": summary, "lines_read": len(items), "truncated": truncated, "max_lines": MAX_LINES}


CSV_COLUMNS = ["input", "claim", "status", "support_tier", "supporting", "contradicting", "publishers",
               "hedged_only", "conflict", "verification_id", "error"]


def to_csv(result: Dict[str, Any]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLUMNS)
    for r in result["rows"]:
        w.writerow([("; ".join(r.get(c) or []) if c == "publishers" else r.get(c, "")) for c in CSV_COLUMNS])
    return buf.getvalue()
