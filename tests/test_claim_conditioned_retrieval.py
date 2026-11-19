"""
Unit and integration tests for 3-Stage Claim-Conditioned Retrieval.
Validates:
1. Stage 1: Entity-pair sentence co-occurrence pre-gating (100% gate pass rate).
2. Stage 2: Dual query retrieval (support + counter) retrieving takedown/cessation evidence.
3. Stage 3: Publisher capping and diversification (max_per_publisher constraint).
4. Safety toggle: enable_claim_conditioned_retrieval=False baseline preservation.
"""
from __future__ import annotations
import unittest
from typing import Any, Dict, List

from src.retrieval import ClaimConditionedRetriever
from src.agent.decision_engine import EntityAwareEvidenceGate
from src.agent.retrieval_agent import CyberRAGAgent


class MockHybridRetriever:
    """Mock hybrid retriever simulating dense + BM25 results across publishers."""

    def __init__(self):
        self.documents = [
            {
                "report_id": "RPT-01",
                "segment_id": "p1::s1",
                "publisher": "VendorA",
                "source": "VendorA",
                "publication_date": "2020-05-01",
                "text": "APT28 deployed Zebrocy malware in targeted spear-phishing campaigns against European foreign ministries.",
            },
            {
                "report_id": "RPT-02",
                "segment_id": "p2::s1",
                "publisher": "VendorA",
                "source": "VendorA",
                "publication_date": "2020-06-01",
                "text": "VendorA observed APT28 delivering Zebrocy payloads via macro-enabled documents.",
            },
            {
                "report_id": "RPT-03",
                "segment_id": "p3::s1",
                "publisher": "VendorA",
                "source": "VendorA",
                "publication_date": "2020-07-01",
                "text": "Additional Zebrocy telemetry confirmed APT28 command and control infrastructure.",
            },
            {
                "report_id": "RPT-04",
                "segment_id": "p1::s1",
                "publisher": "VendorB",
                "source": "VendorB",
                "publication_date": "2020-08-15",
                "text": "VendorB intelligence correlates APT28 activity with Zebrocy dropper variants.",
            },
            {
                "report_id": "RPT-05",
                "segment_id": "p1::s1",
                "publisher": "VendorC",
                "source": "VendorC",
                "publication_date": "2021-02-10",
                "text": "Irrelevant passage: Emotet botnet sends high-volume spam without any APT28 involvement.",
            },
            {
                "report_id": "RPT-TAKEDOWN",
                "segment_id": "p1::s1",
                "publisher": "Europol / CISA",
                "source": "Europol / CISA",
                "publication_date": "2021-05-20",
                "text": "Following global law enforcement disruption, Emotet no longer uses TrickBot payloads.",
            },
        ]

    def retrieve(self, query: str, top_k: int = 5, candidate_k: int = 50) -> List[Dict[str, Any]]:
        q_l = query.lower()
        results = []
        for doc in self.documents:
            text_l = doc["text"].lower()
            score = 0.0
            if "apt28" in q_l and "apt28" in text_l:
                score += 0.5
            if "zebrocy" in q_l and "zebrocy" in text_l:
                score += 0.5
            if "emotet" in q_l and "emotet" in text_l:
                score += 0.5
            if "trickbot" in q_l and "trickbot" in text_l:
                score += 0.5
            if any(term in q_l for term in ("stopped", "no longer", "disrupted", "takedown")) and any(
                term in text_l for term in ("no longer", "disruption", "disrupted")
            ):
                score += 0.8

            if score > 0:
                item = dict(doc)
                item["rrf_score"] = score
                results.append(item)

        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return results[:top_k]


class TestClaimConditionedRetrieval(unittest.TestCase):
    def setUp(self):
        self.mock_hybrid = MockHybridRetriever()
        self.gate = EntityAwareEvidenceGate()
        self.retriever = ClaimConditionedRetriever(
            hybrid_retriever=self.mock_hybrid,
            gate=self.gate,
            max_per_publisher=2,
        )

    def test_01_sentence_cooccurrence_guarantees_100pct_gate_pass(self):
        """Stage 1: All candidates returned by claim-conditioned retrieval pass the gate."""
        results = self.retriever.retrieve_for_claim(
            subject="APT28",
            predicate="USES",
            obj="Zebrocy",
            natural_query="Does APT28 use Zebrocy?",
            top_k=5,
        )
        self.assertGreaterEqual(len(results), 2)
        # Verify 100% gate pass rate
        for r in results:
            is_prob, reason = self.gate.is_probative(r["text"], "APT28", "Zebrocy")
            self.assertTrue(is_prob, f"Failed gate check: {reason}")
            self.assertEqual(reason, "PROBATIVE_COOCCURRENCE")

    def test_02_dual_query_retrieves_counter_evidence(self):
        """Stage 2: Counter query retrieves cessation/takedown evidence."""
        results = self.retriever.retrieve_for_claim(
            subject="Emotet",
            predicate="USES",
            obj="TrickBot",
            natural_query="Does Emotet distribute TrickBot?",
            top_k=5,
        )
        # Verify the takedown counter evidence is successfully retrieved
        takedown_hits = [r for r in results if "no longer" in r["text"].lower() or "disruption" in r["text"].lower()]
        self.assertGreaterEqual(len(takedown_hits), 1)
        self.assertEqual(takedown_hits[0]["publisher"], "Europol / CISA")

    def test_03_publisher_capping_enforced(self):
        """Stage 3: Max per publisher limit prevents single vendor dominance."""
        results = self.retriever.retrieve_for_claim(
            subject="APT28",
            predicate="USES",
            obj="Zebrocy",
            natural_query="Does APT28 use Zebrocy?",
            top_k=10,
        )
        pub_counts: Dict[str, int] = {}
        for r in results:
            pub = r["publisher"]
            pub_counts[pub] = pub_counts.get(pub, 0) + 1

        # VendorA had 3 matching documents in corpus, but must be capped at 2
        self.assertLessEqual(pub_counts.get("VendorA", 0), 2)
        # VendorB must also appear (diversification)
        self.assertIn("VendorB", pub_counts)

    def test_04_ungrounded_entity_returns_empty_without_noise(self):
        """Ungrounded entities yield 0 candidates instead of hallucinated irrelevant passages."""
        results = self.retriever.retrieve_for_claim(
            subject="Sandworm",
            predicate="USES",
            obj="Cyclops Blink",
            natural_query="Does Sandworm use Cyclops Blink?",
            top_k=5,
        )
        self.assertEqual(len(results), 0)

    def test_05_agent_toggle_disabled_by_default(self):
        """Default agent has enable_claim_conditioned_retrieval=False for backwards compatibility."""
        agent = CyberRAGAgent()
        self.assertFalse(agent.enable_claim_conditioned_retrieval)

    def test_06_agent_toggle_enabled_explicitly(self):
        """Agent accepts explicit enable_claim_conditioned_retrieval=True."""
        agent = CyberRAGAgent(enable_claim_conditioned_retrieval=True)
        self.assertTrue(agent.enable_claim_conditioned_retrieval)


if __name__ == "__main__":
    unittest.main()
