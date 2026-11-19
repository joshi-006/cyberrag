"""
CyberRAG Autonomous Retrieval & Verification Agent.
Implements the frozen 5-stage Bounded ReAct State Machine:
1. Stage 1: LLM Query Planning (intent, sub-claims, temporal scope)
2. Stage 2: Autonomous Retrieval Agent (Bounded ReAct loop, max 3 iterations, KG, RRF, alias expansion)
3. Stage 3: Evidence Fusion & Context Building (deduplication & provenance tracking)
4. Stage 4: Multi-Stage Evidence Verification (Entity gate, DeBERTa NLI, temporal/scope conflict resolution)
5. Stage 5: Grounded LLM Reasoning & Reporting (synthesizes verified brief with citations)
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.llm.llm_client import LocalLLMClient
from src.llm.query_planner import (
    LLMQueryPlanner,
    QueryPlan,
    SubClaim,
    QueryIntent,
    CANONICAL_PREDICATES,
    RELATION_MAP,
)
from src.llm.query_expander import QueryExpander, KNOWN_ALIASES
from src.llm.response_generator import GroundedResponseGenerator
from src.agent.agent_state import AgentState, AgentStepTrace
from src.agent.decision_engine import EntityAwareEvidenceGate
from src.agent.agent_tools import AgentToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class GroundedReport:
    query: str
    status: str  # "VERIFIED", "CONTRADICTED", "TEMPORAL_SUPERSEDENCE", "INSUFFICIENT_EVIDENCE"
    confidence_score: float  # Multi-publisher heuristic consensus score (not Bayesian probability)
    source_diversity: int  # Count of distinct independent vendors
    supporting_evidence: List[Dict[str, Any]]
    contradicting_evidence: List[Dict[str, Any]]
    conflict_resolution: Optional[Dict[str, Any]]
    state_history: List[AgentState]
    execution_trace: List[AgentStepTrace]
    narrative: str

    @property
    def evidence_score(self) -> float:
        """Alias for confidence_score, indicating multi-vendor consensus score."""
        return self.confidence_score

    @property
    def consensus_evidence_score(self) -> float:
        """Standardized consensus evidence score (multi-publisher consensus heuristic)."""
        return self.confidence_score


class CyberRAGAgent:
    """
    Autonomous CTI Retrieval & Verification Agent.
    Coordinates Query Planning, KG Traversal, Hybrid RRF Retrieval,
    Entity-Aware Gating, DeBERTa Claim Verification, and Temporal Conflict Resolution.
    """

    def __init__(
        self,
        retriever=None,
        verifier=None,
        context_resolver=None,
        max_iterations: int = 3,
        temporal_delta_days: int = 90,
        llm_client: Optional[LocalLLMClient] = None,
        enable_claim_conditioned_retrieval: Optional[bool] = None,
    ):
        self.max_iterations = max_iterations
        self.temporal_delta_days = temporal_delta_days
        self.llm_client = llm_client or LocalLLMClient()
        if enable_claim_conditioned_retrieval is not None:
            self.enable_claim_conditioned_retrieval = enable_claim_conditioned_retrieval
        else:
            self.enable_claim_conditioned_retrieval = (
                os.environ.get("CTI_ENABLE_CLAIM_CONDITIONED_RETRIEVAL", "").lower() in ("1", "true", "yes")
            )

        # Ensure active NLI verifier and retriever are present; auto-initialize if not provided
        if verifier is None:
            try:
                from src.verification.evidence_verifier import CTIEvidenceVerifier
                verifier = CTIEvidenceVerifier()
                logger.info("CyberRAGAgent: CTIEvidenceVerifier initialized.")
            except Exception as e:
                logger.warning(f"CyberRAGAgent: Could not auto-initialize CTIEvidenceVerifier: {e}")

        if retriever is None:
            try:
                from src.hybrid_graph_rag import HybridGraphRAG
                retriever = HybridGraphRAG()
                logger.info("CyberRAGAgent: HybridGraphRAG initialized.")
            except Exception as e:
                logger.warning(f"CyberRAGAgent: Could not auto-initialize HybridGraphRAG: {e}")

        self.expander = QueryExpander()
        self.planner = LLMQueryPlanner(expander=self.expander, llm_client=self.llm_client)
        self.gate = EntityAwareEvidenceGate(expander=self.expander)
        self.tools = AgentToolRegistry(
            retriever=retriever,
            verifier=verifier,
            context_resolver=context_resolver,
            expander=self.expander,
        )
        self.response_generator = GroundedResponseGenerator(llm_client=self.llm_client)

    def run(self, query: str) -> GroundedReport:
        state_history: List[AgentState] = []
        trace: List[AgentStepTrace] = []

        def record_step(
            state: AgentState,
            action: str,
            tool_name: str,
            tool_in: Any,
            tool_out: str,
            decision_outcome: str = "",
            retry_or_termination_reason: str = "",
        ):
            state_history.append(state)
            trace.append(AgentStepTrace(
                step_index=len(trace) + 1,
                state=state,
                thought=action,
                action=action,
                tool_name=tool_name,
                tool_input=tool_in if isinstance(tool_in, dict) else {"arg": tool_in},
                tool_output_summary=tool_out,
                decision_outcome=decision_outcome or tool_out,
                retry_or_termination_reason=retry_or_termination_reason,
            ))

        # ----------------------------------------------------------------------
        # Stage 1: QUERY_ANALYSIS (LLM Query Planner)
        # ----------------------------------------------------------------------
        current_state = AgentState.QUERY_ANALYSIS
        plan = self.planner.plan(query)
        record_step(
            state=current_state,
            action=f"Decomposed query intent '{plan.intent.value}' into canonical entity slots ({plan.entities}) and sub-claims.",
            tool_name="LLMQueryPlanner.plan",
            tool_in={"query": query},
            tool_out=f"Entities: {plan.entities}, Claims: {[c.to_natural_phrase() for c in plan.sub_claims]}, TempScope: {plan.temporal_scope}",
            decision_outcome="Extracted search targets and temporal constraints for multi-source retrieval.",
        )

        subject = plan.sub_claims[0].subject if plan.sub_claims else (plan.entities[0] if plan.entities else "Unknown")
        predicate = plan.sub_claims[0].predicate if plan.sub_claims else "USES"
        obj = plan.sub_claims[0].object if plan.sub_claims else (plan.entities[1] if len(plan.entities) > 1 else "Unknown")

        # ----------------------------------------------------------------------
        # Stage 2: Bounded Autonomous Retrieval Loop (Max K=3)
        # ----------------------------------------------------------------------
        all_raw_evidence: List[Dict[str, Any]] = []
        probative_evidence: List[Dict[str, Any]] = []
        rejected_evidence: List[Dict[str, Any]] = []
        iteration = 0

        current_queries = list(plan.search_queries)

        while iteration < self.max_iterations:
            iteration += 1
            current_state = AgentState.RETRIEVAL

            iter_query = current_queries[(iteration - 1) % len(current_queries)]
            retrieved_items = self.tools.execute_retrieval(
                iter_query,
                plan.temporal_scope,
                subject=subject,
                predicate=predicate,
                obj=obj,
                enable_claim_conditioned=self.enable_claim_conditioned_retrieval,
            )
            all_raw_evidence.extend(retrieved_items)
            all_raw_evidence = self.tools.deduplicate_evidence(all_raw_evidence)

            tool_call_name = (
                "ClaimConditionedRetriever.retrieve_for_claim"
                if self.enable_claim_conditioned_retrieval
                else "HybridGraphRAG.retrieve"
            )
            record_step(
                state=current_state,
                action=f"Iteration {iteration}/{self.max_iterations}: Executed hybrid dense-sparse-graph retrieval for '{iter_query}'.",
                tool_name=tool_call_name,
                tool_in={"query": iter_query, "iteration": iteration},
                tool_out=f"Retrieved {len(retrieved_items)} items. Total unique pool: {len(all_raw_evidence)} items.",
                decision_outcome=f"Collected {len(retrieved_items)} candidate segments for entity relevance inspection.",
            )

            # ------------------------------------------------------------------
            # Stage 4.1: EVIDENCE_INSPECTION (Entity-Aware Evidence Gate)
            # ------------------------------------------------------------------
            current_state = AgentState.EVIDENCE_INSPECTION
            probative_evidence, rejected_evidence = self.gate.filter_evidence_list(
                all_raw_evidence, subject, obj
            )

            record_step(
                state=current_state,
                action=f"Applied Entity-Aware Evidence Gate requiring co-occurrence of '{subject}' and '{obj}'.",
                tool_name="EntityAwareEvidenceGate.filter",
                tool_in={"subject": subject, "object": obj, "total_candidates": len(all_raw_evidence)},
                tool_out=f"Probative count: {len(probative_evidence)}. Screened out: {len(rejected_evidence)}.",
                decision_outcome=f"{len(probative_evidence)} probative passages passed co-occurrence inspection.",
            )

            # Sufficient probative evidence collected
            if len(probative_evidence) >= 2 or iteration >= self.max_iterations:
                break

            # ------------------------------------------------------------------
            # Stage 2.2: RETRY_EXPANSION
            # ------------------------------------------------------------------
            if iteration < self.max_iterations:
                current_state = AgentState.RETRY_EXPANSION
                expanded_q = self.expander.expand_query(subject, obj, iteration)
                current_queries.append(expanded_q)
                record_step(
                    state=current_state,
                    action=f"Triggered query expansion due to insufficient probative evidence ({len(probative_evidence)} < 2).",
                    tool_name="QueryExpander.expand_query",
                    tool_in={"subject": subject, "object": obj, "iteration": iteration},
                    tool_out=f"Added expanded query: '{expanded_q}'.",
                    decision_outcome=f"Formulated iteration {iteration + 1} expanded query.",
                    retry_or_termination_reason=f"Insufficient evidence threshold not met ({len(probative_evidence)} probative items).",
                )

        # ----------------------------------------------------------------------
        # Termination on INSUFFICIENT_EVIDENCE
        # ----------------------------------------------------------------------
        if not probative_evidence:
            current_state = AgentState.INSUFFICIENT_EVIDENCE
            record_step(
                state=current_state,
                action=f"Terminated retrieval loop after {self.max_iterations} iterations without finding passages mentioning both '{subject}' and '{obj}'.",
                tool_name="AgentHalt.insufficient_evidence",
                tool_in={"subject": subject, "object": obj, "iterations": iteration},
                tool_out="Exiting to ANSWER_GENERATION with INSUFFICIENT_EVIDENCE.",
                decision_outcome="Bypassed NLI inference to prevent false contradiction hallucination on ungrounded entities.",
                retry_or_termination_reason="No co-occurring evidence found across 3 retrieval attempts.",
            )
            return self._synthesize_insufficient_report(
                query=query,
                subject=subject,
                predicate=predicate,
                obj=obj,
                state_history=state_history,
                trace=trace,
                plan=plan,
                rejected_count=len(rejected_evidence),
            )

        # ----------------------------------------------------------------------
        # Stage 4.2 & 4.3: VERIFICATION (DeBERTa NLI & Claim Aggregation)
        # ----------------------------------------------------------------------
        current_state = AgentState.VERIFICATION
        verified_evidence, supporting, contradicting = self.tools.execute_verification(
            probative_evidence, subject, predicate, obj
        )

        record_step(
            state=current_state,
            action=f"Evaluated {len(probative_evidence)} probative passages using DeBERTa-v3 NLI cross-encoder.",
            tool_name="CTIEvidenceVerifier.verify",
            tool_in={"hypothesis": f"{subject} {predicate} {obj}", "num_passages": len(probative_evidence)},
            tool_out=f"Results: {len(supporting)} Entailments, {len(contradicting)} Contradictions, {len(probative_evidence) - len(supporting) - len(contradicting)} Neutral.",
            decision_outcome=f"Classified evidence: {len(supporting)} support, {len(contradicting)} contradiction.",
        )

        # ----------------------------------------------------------------------
        # Stage 4.4: CONFLICT_RESOLUTION (Temporal & Scope Taxonomy)
        # ----------------------------------------------------------------------
        conflict_result = None
        if supporting and contradicting:
            current_state = AgentState.CONFLICT_RESOLUTION
            conflict_result = self.tools.execute_conflict_resolution(
                supporting, contradicting, self.temporal_delta_days
            )
            record_step(
                state=current_state,
                action=f"Analyzed conflicting evidence timestamps using Context Resolver with {self.temporal_delta_days}-day threshold.",
                tool_name="ContextResolver.resolve",
                tool_in={"delta_days_threshold": self.temporal_delta_days},
                tool_out=f"Resolution: {conflict_result.get('status')} (Delta: {conflict_result.get('delta_days')} days).",
                decision_outcome=f"Determined conflict classification: {conflict_result.get('status')} based on chronological ordering.",
            )

        # ----------------------------------------------------------------------
        # Stage 5: ANSWER_GENERATION (Grounded Report & Synthesis)
        # ----------------------------------------------------------------------
        current_state = AgentState.ANSWER_GENERATION
        report = self._synthesize_final_report(
            query=query,
            subject=subject,
            predicate=predicate,
            obj=obj,
            supporting=supporting,
            contradicting=contradicting,
            conflict_result=conflict_result,
            state_history=state_history,
            trace=trace,
        )

        record_step(
            state=current_state,
            action="Synthesized grounded final intelligence report with provenance citations and consensus evidence scoring.",
            tool_name="GroundedResponseGenerator.generate",
            tool_in={"status": report.status, "confidence": report.confidence_score},
            tool_out=f"Report created with {len(report.supporting_evidence)} supporting citations.",
            decision_outcome=f"Final intelligence verdict: {report.status} (Consensus Evidence Score: {report.confidence_score:.2f}).",
        )

        return report

    def _synthesize_final_report(
        self,
        query: str,
        subject: str,
        predicate: str,
        obj: str,
        supporting: List[Dict[str, Any]],
        contradicting: List[Dict[str, Any]],
        conflict_result: Optional[Dict[str, Any]],
        state_history: List[AgentState],
        trace: List[AgentStepTrace],
    ) -> GroundedReport:
        all_sources = set(it.get("source") for it in supporting if it.get("source"))
        source_diversity = len(all_sources)
        all_hedged = bool(supporting) and all(it.get("is_hedged", False) for it in supporting)

        if conflict_result and conflict_result.get("status") == "TEMPORAL_SUPERSEDENCE":
            if all_hedged:
                status = "WEAK_SUPPORT"
                conf = 0.50
                conflict_result = None  # Epistemic guard: hedged evidence cannot trigger supersedence
            else:
                status = "TEMPORAL_SUPERSEDENCE"
                conf = 0.85
        elif supporting and not contradicting:
            if all_hedged:
                status = "WEAK_SUPPORT"
                conf = 0.50
                conflict_result = None
            else:
                status = "VERIFIED"
                conf = min(0.99, 0.70 + (0.10 * source_diversity))
        elif contradicting and not supporting:
            status = "CONTRADICTED"
            conf = 0.90
        elif conflict_result:
            status = conflict_result.get("status", "CONFLICT_DETECTED")
            conf = 0.65
        else:
            status = "INSUFFICIENT_EVIDENCE"
            conf = 0.10

        narrative = self.response_generator.generate_narrative(
            query=query,
            subject=subject,
            predicate=predicate,
            obj=obj,
            status=status,
            confidence=conf,
            source_diversity=source_diversity,
            supporting=supporting,
            contradicting=contradicting,
            conflict_result=conflict_result,
        )

        return GroundedReport(
            query=query,
            status=status,
            confidence_score=conf,
            source_diversity=source_diversity,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            conflict_resolution=conflict_result,
            state_history=state_history,
            execution_trace=trace,
            narrative=narrative,
        )

    def _synthesize_insufficient_report(
        self,
        query: str,
        subject: str,
        predicate: str,
        obj: str,
        state_history: List[AgentState],
        trace: List[AgentStepTrace],
        plan: QueryPlan,
        rejected_count: int,
    ) -> GroundedReport:
        narrative = (
            f"### CyberRAG Evidence Verification Brief: `{subject} {predicate} {obj}`\n"
            f"**Overall Status**: `INSUFFICIENT_EVIDENCE` | **Confidence Score**: `0.00` | **Source Diversity**: `0`\n\n"
            f"**Audit Findings**:\n"
            f"- Autonomous retrieval agent performed {self.max_iterations} iterative search hops across dense, lexical, and graph indices.\n"
            f"- Checked threat actor / malware aliases: `{plan.aliases.get(subject, ['None'])}` and `{plan.aliases.get(obj, ['None'])}`.\n"
            f"- The **Entity-Aware Evidence Gate** screened {rejected_count} candidate passages and rejected them because neither passage co-occurred with both '{subject}' and '{obj}'.\n"
            f"- Cross-encoder NLI inference was **deliberately withheld** to prevent false contradiction hallucinations on irrelevant text.\n"
            f"- **Conclusion**: The indexed 104-report corpus contains no factual evidence supporting or refuting this relationship."
        )

        return GroundedReport(
            query=query,
            status="INSUFFICIENT_EVIDENCE",
            confidence_score=0.0,
            source_diversity=0,
            supporting_evidence=[],
            contradicting_evidence=[],
            conflict_resolution=None,
            state_history=state_history,
            execution_trace=trace,
            narrative=narrative,
        )
