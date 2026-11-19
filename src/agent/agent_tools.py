"""
Typed Agent Tools for CyberRAG.
Encapsulates HybridGraphRAG, CTIEvidenceVerifier, ContextResolver, and QueryExpander.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.llm.query_planner import SubClaim
from src.llm.query_expander import QueryExpander
from src.agent.decision_engine import EntityAwareEvidenceGate
from src.verification.evidence_verifier import is_epistemically_hedged


class AgentToolRegistry:
    """Provides unified, typed interfaces to the retrieval and verification engines."""

    def __init__(
        self,
        retriever=None,
        verifier=None,
        context_resolver=None,
        expander: Optional[QueryExpander] = None,
        gate: Optional[EntityAwareEvidenceGate] = None,
    ):
        self.retriever = retriever
        self.verifier = verifier
        self.context_resolver = context_resolver
        self.expander = expander or QueryExpander()
        self.gate = gate or EntityAwareEvidenceGate(expander=self.expander)

    def execute_retrieval(
        self, query_str: str, temporal_scope: Optional[Tuple[str, str]] = None
    ) -> List[Dict[str, Any]]:
        results = []
        if self.retriever is not None:
            raw = self.retriever.retrieve(query_str, top_k=8)
            for k in ("dense", "hybrid"):
                for it in raw.get(k, []):
                    item = dict(it)
                    item["type"] = "text"
                    results.append(item)
            for it in raw.get("graph", []):
                item = dict(it)
                item["type"] = "graph"
                results.append(item)
        else:
            results.append({
                "type": "text",
                "report_id": "STUB-REPORT-01",
                "segment_id": "p1::s1",
                "text": query_str,
                "source": "Zscaler",
                "publication_date": "2021-11-16",
            })

        if temporal_scope:
            start_d, end_d = temporal_scope
            filtered = []
            for item in results:
                p_date = item.get("publication_date")
                if p_date and start_d <= p_date <= end_d:
                    filtered.append(item)
                elif not p_date:
                    filtered.append(item)
            results = filtered

        return results

    def deduplicate_evidence(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped = []
        for it in items:
            key = (it.get("report_id"), it.get("segment_id"), it.get("assertion_id"))
            if key not in seen:
                seen.add(key)
                deduped.append(it)
        return deduped

    def execute_verification(
        self, probative_items: List[Dict[str, Any]], subject: str, predicate: str, obj: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        supporting = []
        contradicting = []
        verified_items = []

        sub_claim = SubClaim(subject=subject, predicate=predicate, object=obj)
        hypothesis = sub_claim.to_natural_phrase()

        for it in probative_items:
            text = it.get("text") or it.get("evidence") or ""
            it_copy = dict(it)

            if self.verifier is not None:
                res = self.verifier.classify_pair(evidence=text, claim=hypothesis)
                p_entail = getattr(res, "p_entailment", getattr(res, "entailment", 0.0))
                p_contra = getattr(res, "p_contradiction", getattr(res, "contradiction", 0.0))
                p_neutral = getattr(res, "p_neutral", getattr(res, "neutral", 0.0))
            else:
                raise RuntimeError(
                    "CTIEvidenceVerifier is not configured on AgentToolRegistry. "
                    "An active NLI cross-encoder model is required for claim verification; "
                    "heuristic keyword fallback has been removed to uphold scientific integrity."
                )

            it_copy["p_entail"] = p_entail
            it_copy["p_contra"] = p_contra
            it_copy["p_neutral"] = p_neutral
            it_copy["is_hedged"] = getattr(res, "is_hedged", False) or is_epistemically_hedged(text)

            if p_entail > 0.50:
                it_copy["nli_label"] = "ENTAILMENT"
                supporting.append(it_copy)
            elif p_contra > 0.50:
                # Contradiction Gate: require strict exact entity match without partial name overlap
                text_l = text.lower()
                s_terms = self.expander.get_search_terms(subject)
                o_terms = self.expander.get_search_terms(obj)
                s_ok = self.gate._entity_present(text_l, s_terms, subject)
                o_ok = self.gate._entity_present(text_l, o_terms, obj)
                if s_ok and o_ok:
                    it_copy["nli_label"] = "CONTRADICTION"
                    contradicting.append(it_copy)
                else:
                    it_copy["nli_label"] = "NEUTRAL"
            else:
                it_copy["nli_label"] = "NEUTRAL"

            verified_items.append(it_copy)

        return verified_items, supporting, contradicting

    def execute_conflict_resolution(
        self, supporting: List[Dict[str, Any]], contradicting: List[Dict[str, Any]], delta_threshold: int
    ) -> Dict[str, Any]:
        supp_dates = [s.get("publication_date") for s in supporting if s.get("publication_date")]
        contra_dates = [c.get("publication_date") for c in contradicting if c.get("publication_date")]

        if supp_dates and contra_dates:
            min_supp = min(supp_dates)
            max_contra = max(contra_dates)
            try:
                d1 = datetime.strptime(min_supp, "%Y-%m-%d")
                d2 = datetime.strptime(max_contra, "%Y-%m-%d")
                delta_days = abs((d2 - d1).days)
            except Exception:
                delta_days = 0
        else:
            delta_days = 0

        if delta_days >= delta_threshold:
            status = "TEMPORAL_SUPERSEDENCE"
            reason = (
                f"Contradicting reports have a {delta_days}-day temporal separation, "
                f"exceeding the configurable heuristic threshold of {delta_threshold} days. "
                f"Older operational intelligence is classified as superseded."
            )
        else:
            supp_vendors = set(s.get("source") for s in supporting if s.get("source"))
            contra_vendors = set(c.get("source") for c in contradicting if c.get("source"))
            if supp_vendors != contra_vendors:
                status = "ATTRIBUTION_DISPUTE"
                reason = (
                    f"Contradictory findings published within {delta_days} days across independent vendors "
                    f"({list(supp_vendors)} vs {list(contra_vendors)}). Preserved as active vendor dispute."
                )
            else:
                status = "DIRECT_CONTRADICTION"
                reason = "Direct factual disagreement within identical temporal window."

        return {
            "status": status,
            "delta_days": delta_days,
            "threshold": delta_threshold,
            "reason": reason,
        }
