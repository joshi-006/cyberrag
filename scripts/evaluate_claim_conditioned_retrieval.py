"""
Empirical Evaluation of Claim-Conditioned Retrieval Upgrade.

Measures Before (Standard Hybrid RRF) vs. After (3-Stage Claim-Conditioned Retrieval):
1. Gate Pass Rate (% of retrieved top-k passages passing EntityAwareEvidenceGate).
2. Contradiction Recall & Cessation Accuracy on temporal conflicts (Emotet/TrickBot takedown).
3. Publisher Diversity (max passages per vendor).
"""
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from src.hybrid_graph_rag import HybridGraphRAG
from src.agent.decision_engine import EntityAwareEvidenceGate
from src.verification.evidence_verifier import CTIEvidenceVerifier

def run_evaluation():
    print("=" * 80)
    print("CYBERRAG RETRIEVAL UPGRADE EMPIRICAL EVALUATION")
    print("=" * 80)
    
    # Load demo overlay so post-takedown 2021 Europol/CISA record is available
    os.environ["CTI_LOAD_DEMO_OVERLAY"] = "1"
    
    rag = HybridGraphRAG()
    gate = EntityAwareEvidenceGate()
    verifier = CTIEvidenceVerifier()
    
    benchmark_queries = [
        {
            "query": "Does APT28 use Zebrocy?",
            "subject": "APT28",
            "predicate": "USES",
            "object": "Zebrocy",
            "type": "corroboration",
        },
        {
            "query": "Did Emotet stop distributing TrickBot?",
            "subject": "Emotet",
            "predicate": "USES",
            "object": "TrickBot",
            "type": "cessation",
        },
        {
            "query": "Does Emotet distribute TrickBot?",
            "subject": "Emotet",
            "predicate": "USES",
            "object": "TrickBot",
            "type": "temporal_conflict",
        },
        {
            "query": "Does Lazarus target banks?",
            "subject": "Lazarus",
            "predicate": "TARGETS",
            "object": "banks",
            "type": "sector_targeting",
        },
        {
            "query": "Does Sandworm use Cyclops Blink?",
            "subject": "Sandworm",
            "predicate": "USES",
            "object": "Cyclops Blink",
            "type": "ungrounded_negative",
        },
    ]

    print("\n" + "-" * 80)
    print("STAGE 1 & 2 EVALUATION: GATE PASS RATE (% of top-k passages passing gate)")
    print("-" * 80)
    print(f"{'Query':<42} | {'Before (Standard)':<17} | {'After (Claim-Cond)':<17} | {'Delta'}")
    print("-" * 80)

    before_total_retrieved = 0
    before_total_passed = 0
    after_total_retrieved = 0
    after_total_passed = 0

    for b in benchmark_queries:
        q = b["query"]
        s = b["subject"]
        p = b["predicate"]
        o = b["object"]

        # 1. Before: Standard hybrid retrieval (top 15)
        raw_before = rag.retrieve(q, top_k=15)
        cands_before = []
        for k in ("dense", "hybrid"):
            cands_before.extend(raw_before.get(k, []))
        # Deduplicate
        seen_b = set()
        deduped_before = []
        for c in cands_before:
            t = c.get("text", "")
            if t not in seen_b:
                seen_b.add(t)
                deduped_before.append(c)

        prob_before, _ = gate.filter_evidence_list(deduped_before, s, o)
        b_ret = len(deduped_before)
        b_pass = len(prob_before)
        b_pct = (b_pass / b_ret * 100) if b_ret else 0.0

        # 2. After: Claim-conditioned retrieval (top 10)
        cands_after = rag.claim_conditioned.retrieve_for_claim(
            subject=s, predicate=p, obj=o, natural_query=q, top_k=10
        )
        prob_after, _ = gate.filter_evidence_list(cands_after, s, o)
        a_ret = len(cands_after)
        a_pass = len(prob_after)
        a_pct = (a_pass / a_ret * 100) if a_ret else 0.0

        before_total_retrieved += b_ret
        before_total_passed += b_pass
        after_total_retrieved += a_ret
        after_total_passed += a_pass

        delta_str = f"+{a_pct - b_pct:.1f}%" if a_pct >= b_pct else f"{a_pct - b_pct:.1f}%"
        print(f"{q:<42} | {b_pass:>2}/{b_ret:<2} ({b_pct:>5.1f}%)    | {a_pass:>2}/{a_ret:<2} ({a_pct:>5.1f}%)    | {delta_str}")

    b_overall_pct = (before_total_passed / before_total_retrieved * 100) if before_total_retrieved else 0.0
    a_overall_pct = (after_total_passed / after_total_retrieved * 100) if after_total_retrieved else 0.0
    print("-" * 80)
    print(f"{'OVERALL AVERAGE PASS RATE':<42} | {before_total_passed:>2}/{before_total_retrieved:<2} ({b_overall_pct:>5.1f}%)    | {after_total_passed:>2}/{after_total_retrieved:<2} ({a_overall_pct:>5.1f}%)    | +{a_overall_pct - b_overall_pct:.1f}%")
    print("-" * 80)

    # 3. Cessation and Contradiction Retrieval Evaluation
    print("\n" + "-" * 80)
    print("STAGE 2: CONTRADICTION & CESSATION RECALL EVALUATION")
    print("-" * 80)
    emotet_query = "Did Emotet stop distributing TrickBot?"
    raw_before_emotet = rag.retrieve(emotet_query, top_k=15)
    cands_b_emotet = [c.get("text", "") for c in raw_before_emotet.get("dense", []) + raw_before_emotet.get("hybrid", [])]
    has_counter_before = any("no longer" in t.lower() or "disruption" in t.lower() or "takedown" in t.lower() for t in cands_b_emotet)

    cands_a_emotet = rag.claim_conditioned.retrieve_for_claim(
        subject="Emotet", predicate="USES", obj="TrickBot", natural_query=emotet_query, top_k=10
    )
    counter_after_items = [
        c for c in cands_a_emotet
        if "no longer" in c.get("text", "").lower() or "disruption" in c.get("text", "").lower() or "takedown" in c.get("text", "").lower()
    ]
    has_counter_after = len(counter_after_items) > 0

    print(f"Query: '{emotet_query}'")
    print(f"  - Standard Retrieval Counter-Evidence Retrieved: {has_counter_before}")
    print(f"  - Claim-Conditioned Counter-Evidence Retrieved: {has_counter_after} ({len(counter_after_items)} items)")
    if counter_after_items:
        first_counter = counter_after_items[0]
        print(f"    Publisher: {first_counter.get('publisher') or first_counter.get('source')}")
        print(f"    Date: {first_counter.get('publication_date')}")
        print(f"    Excerpt: {first_counter.get('text')[:120]}")

    # 4. Publisher Capping & Diversification Evaluation
    print("\n" + "-" * 80)
    print("STAGE 3: PUBLISHER DIVERSIFICATION EVALUATION")
    print("-" * 80)
    pub_counts_before = {}
    for c in deduped_before:
        p = c.get("publisher") or c.get("source") or "Unknown"
        pub_counts_before[p] = pub_counts_before.get(p, 0) + 1

    pub_counts_after = {}
    for c in cands_a_emotet:
        p = c.get("publisher") or c.get("source") or "Unknown"
        pub_counts_after[p] = pub_counts_after.get(p, 0) + 1

    print("Emotet/TrickBot Publisher Distribution (Standard):", pub_counts_before)
    print("Emotet/TrickBot Publisher Distribution (Claim-Conditioned max=2):", pub_counts_after)
    max_b = max(pub_counts_before.values()) if pub_counts_before else 0
    max_a = max(pub_counts_after.values()) if pub_counts_after else 0
    print(f"Max single-publisher concentration: Before={max_b} passages -> After={max_a} passages (Capped)")

if __name__ == "__main__":
    run_evaluation()
