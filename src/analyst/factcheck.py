"""Fact-check a new report against the baseline corpus.

Steps:
1. Run the same deterministic extractor used for uploads (NER + clause-aware
   relation extractor) over the new report.
2. De-duplicate the extracted (subject, relation, object, polarity) claims.
3. Verify each claim against the *baseline corpus only* (no upload session), so
   the new report can never corroborate itself.
4. Bucket each claim:

   ==========================  ==============================================
   CORROBORATED                corpus SUPPORTED the same claim
   CONTRADICTED_BY_CORPUS      corpus CONTRADICTED it
   DISPUTED                    corpus evidence conflicts (both directions)
   NEW_INTEL                   corpus has no gate-passing evidence for it
   ERROR                       the claim could not be verified (input error)
   ==========================  ==============================================

The extractor's relation precision is about one third (README section 3), so
every row carries the report sentence it came from and the UI labels the list
"automatically extracted - review each row".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.errors import CTIError

MAX_CLAIMS = 20

BUCKETS = ("CORROBORATED", "CONTRADICTED_BY_CORPUS", "DISPUTED", "NEW_INTEL", "ERROR")
_STATUS_TO_BUCKET = {
    "SUPPORTED": "CORROBORATED",
    "CONTRADICTED": "CONTRADICTED_BY_CORPUS",
    "CONFLICTING_EVIDENCE": "DISPUTED",
    "INSUFFICIENT_EVIDENCE": "NEW_INTEL",
}
EXTRACTION_NOTE = (
    "Claims were extracted automatically (measured relation precision about one third). "
    "Read the report sentence on each row before acting on it."
)


def bucket_for(status: str) -> str:
    return _STATUS_TO_BUCKET.get(status, "ERROR")


def dedupe_claims(relations: List[Dict[str, Any]], limit: int = MAX_CLAIMS) -> List[Dict[str, Any]]:
    """Keep the highest-confidence instance of each (subject, relation, object, polarity)."""
    best: Dict[tuple, Dict[str, Any]] = {}
    for r in relations:
        key = (str(r.get("subject", "")).lower(), r.get("relation"), str(r.get("object", "")).lower(),
               r.get("polarity", "positive"))
        if not key[0] or not key[2] or key[0] == key[2]:
            continue
        cur = best.get(key)
        if cur is None:
            best[key] = dict(r, occurrences=1)
        else:
            cur["occurrences"] += 1
            if float(r.get("confidence", 0)) > float(cur.get("confidence", 0)):
                best[key] = dict(r, occurrences=cur["occurrences"])
    rows = sorted(best.values(), key=lambda r: (-r["occurrences"], -float(r.get("confidence", 0)), r["subject"], r["object"]))
    return rows[:limit]


def _top_evidence(items: List[Any], n: int = 2) -> List[Dict[str, Any]]:
    out = []
    for e in items[:n]:
        out.append({
            "publisher": e.publisher, "pub_date": e.pub_date, "report_id": e.report_id,
            "text": (e.untrusted_text or "")[:400], "evidence_id": e.evidence_id,
        })
    return out


def factcheck_report(
    service: Any, text: str, *, title: str = "Analyst report", publisher: str = "Analyst Submission",
    pub_date: str = "", max_claims: int = MAX_CLAIMS,
) -> Dict[str, Any]:
    from src.core.uploads import validate_report_text

    text = validate_report_text(text, service.settings.max_upload_bytes)
    extraction = service._extract().extract(
        text, report_id="FACTCHECK", publisher=publisher, pub_date=pub_date or "1970-01-01", title=title
    )
    all_relations = extraction.get("relations", [])
    claims = dedupe_claims(all_relations, limit=max(1, min(max_claims, MAX_CLAIMS)))

    rows: List[Dict[str, Any]] = []
    for c in claims:
        row = {
            "subject": c["subject"], "relation": c["relation"], "object": c["object"],
            "polarity": c.get("polarity", "positive"), "report_sentence": (c.get("evidence_text") or "")[:500],
            "extractor_confidence": round(float(c.get("confidence", 0)), 3), "occurrences": c["occurrences"],
        }
        try:
            v = service.verify_claim(c["subject"], c["relation"], c["object"], polarity=row["polarity"], use_llm_summary=False)
            row.update({
                "bucket": bucket_for(v.status), "corpus_status": v.status, "support_tier": v.support_tier,
                "verification_id": v.verification_id, "conflict": v.conflict.model_dump(),
                "supporting": _top_evidence(v.supporting), "contradicting": _top_evidence(v.contradicting),
                "hedged_only": v.hedged_only,
            })
        except CTIError as exc:
            row.update({"bucket": "ERROR", "corpus_status": None, "error": exc.message})
        rows.append(row)

    summary = {b: 0 for b in BUCKETS}
    for r in rows:
        summary[r["bucket"]] += 1

    techniques = sorted({
        r["object"] for r in all_relations
        if r.get("relation") == "USES_TECHNIQUE" and r.get("polarity", "positive") == "positive"
    })
    return {
        "title": title, "publisher": publisher, "pub_date": pub_date or None,
        "sentences": len(extraction.get("segments", [])),
        "entities": extraction.get("entities", [])[:50],
        "claims_extracted": len(all_relations), "claims_checked": len(rows),
        "truncated": len(dedupe_claims(all_relations, limit=10_000)) > len(rows),
        "summary": summary, "claims": rows, "techniques": techniques,
        "note": EXTRACTION_NOTE,
    }
