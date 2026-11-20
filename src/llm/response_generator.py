"""
Grounded Response Generator for CyberRAG.
Produces strictly evidence-backed narrative briefs with explicit sentence-level provenance citations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from src.llm.llm_client import LocalLLMClient


class GroundedResponseGenerator:
    """Generates an evidence-backed CTI summary with attribution and uncertainty explanations."""

    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        self.llm_client = llm_client or LocalLLMClient()

    def generate_narrative(
        self,
        query: str,
        subject: str,
        predicate: str,
        obj: str,
        status: str,
        confidence: float,
        source_diversity: int,
        supporting: List[Dict[str, Any]],
        contradicting: List[Dict[str, Any]],
        conflict_result: Optional[Dict[str, Any]] = None,
        polarity: str = "positive",
    ) -> str:
        claim_str = f"{subject} {'NOT ' if polarity == 'negative' else ''}{predicate} {obj}"
        lines = [
            f"### ThreatIntel Evidence Verification Brief: `{claim_str}`",
            f"**Overall Status**: `{status}` | **Confidence Score**: `{confidence:.2f}` | **Source Diversity**: `{source_diversity} independent vendor(s)`",
            "",
            "#### Supporting Evidence Citations:",
        ]

        if supporting:
            for s in supporting:
                r_id = s.get("report_id", "N/A")
                s_id = s.get("segment_id", "N/A")
                date = s.get("publication_date", "N/A")
                vendor = s.get("source", "Unknown")
                p_e = s.get("p_entail", 0.0)
                text = (s.get("text") or s.get("evidence") or "").strip()
                lines.append(f"- **[{vendor} | {date} | {r_id}::{s_id}]** (Entailment: `{p_e:.4f}`):")
                lines.append(f"  > \"{text}\"")
        else:
            lines.append("- *No verified supporting passages found in corpus.*")

        if contradicting:
            lines.extend(["", "#### Contradicting Evidence Citations:"])
            for c in contradicting:
                r_id = c.get("report_id", "N/A")
                s_id = c.get("segment_id", "N/A")
                date = c.get("publication_date", "N/A")
                vendor = c.get("source", "Unknown")
                p_c = c.get("p_contra", 0.0)
                text = (c.get("text") or c.get("evidence") or "").strip()
                lines.append(f"- **[{vendor} | {date} | {r_id}::{s_id}]** (Contradiction: `{p_c:.4f}`):")
                lines.append(f"  > \"{text}\"")

        if conflict_result:
            lines.extend([
                "",
                "#### Context Conflict Analysis:",
                f"- **Resolution Tier**: `{conflict_result['status']}`",
                f"- **Temporal Gap**: `{conflict_result['delta_days']} days` (Threshold $\\theta={conflict_result['threshold']}$d)",
                f"- **Analysis**: {conflict_result['reason']}",
            ])

        if status == "WEAK_SUPPORT":
            lines.extend([
                "",
                "#### Epistemic Hedging Warning:",
                "- **Caution**: All retrieved supporting passages rely on modal hedging keywords (e.g. 'might link', 'suspected', 'alleged', or 'unconfirmed').",
                "- **Analysis**: The hypothesis is supported as plausible/tentative publisher speculation, but lacks definitive factual confirmation.",
            ])

        if self.llm_client.is_available() and supporting:
            llm_summary = self.generate_llm_summary(query, supporting, conflict_result)
            if llm_summary:
                lines.extend(["", "#### Grounded LLM Synthesis:", llm_summary])

        return "\n".join(lines)

    def generate_llm_summary(
        self,
        query: str,
        supporting: List[Dict[str, Any]],
        conflict_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        system = (
            "You are a Senior Cyber Threat Intelligence Analyst. "
            "Write a concise 2-sentence executive summary strictly grounded in the provided evidence. "
            "Do NOT extrapolate or add external knowledge. Include vendor and date citations."
        )
        context_snippets = []
        for s in supporting[:3]:
            context_snippets.append(f"[{s.get('source')}, {s.get('publication_date')}] {s.get('text')}")
        prompt = f"Query: {query}\n\nEvidence:\n" + "\n".join(context_snippets)
        if conflict_result:
            prompt += f"\n\nTemporal Resolution: {conflict_result.get('reason')}"
        return self.llm_client.chat_complete(system, prompt, max_tokens=150)
