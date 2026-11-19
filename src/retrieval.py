"""Dense and hybrid retrieval over processed AnnoCTR segments."""
from __future__ import annotations
import hashlib
import os
import json
import re
import logging
from pathlib import Path
from typing import Any
import numpy as np

# Enforce offline cache usage to prevent socket DNS lookups in sandboxed/offline environments
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b[\w.-]+\b", (text or "").lower())


def embedding_cache_key(texts: list[str], model_name: str) -> str:
    """Hash of model name + ordered segment texts: any data or model change invalidates the cache."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(str(len(texts)).encode("ascii"))
    for t in texts:
        h.update(b"|")
        h.update(t.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


class StandardRAGRetriever:
    """Cosine-similarity retrieval using normalized sentence embeddings."""

    def __init__(
        self,
        data_path: str | Path = "data/processed_segments.jsonl",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        documents: list[dict[str, Any]] | None = None,
        model: Any | None = None,
        embeddings: np.ndarray | None = None,
        batch_size: int = 64,
        cache_path: str | Path | None = None,
        cache_meta_path: str | Path | None = None,
    ):
        self.data_path = Path(data_path)
        self.documents = documents if documents is not None else _read_jsonl(self.data_path)
        if not self.documents:
            raise ValueError(f"No documents found in {self.data_path}")
        self.texts = [
            str(d.get("clean_text") or d.get("text") or "")
            for d in self.documents
        ]
        self.model = model if model is not None else SentenceTransformer(model_name)
        if embeddings is None:
            cache = Path(cache_path) if cache_path else self.data_path.parent / "segment_embeddings_minilm.npy"
            meta = Path(cache_meta_path) if cache_meta_path else cache.with_suffix(".meta.json")
            key = embedding_cache_key(self.texts, model_name)
            self.embeddings = self._load_cache(cache, meta, key)
            if self.embeddings is None:
                logger.info("Embedding cache missing or stale; rebuilding...")
                self.embeddings = self.model.encode(
                    self.texts, convert_to_numpy=True, normalize_embeddings=True,
                    show_progress_bar=False, batch_size=batch_size
                )
                try:
                    np.save(cache, self.embeddings)
                    meta.write_text(json.dumps({"key": key, "model": model_name, "count": len(self.texts)}), encoding="utf-8")
                except Exception:
                    pass
        else:
            self.embeddings = np.asarray(embeddings, dtype=np.float32)
        if len(self.embeddings) != len(self.documents):
            raise ValueError("Embedding count does not match document count")
        logger.info("Loaded %d segments", len(self.documents))
        logger.info("Embeddings shape: %s", self.embeddings.shape)

    @staticmethod
    def _load_cache(cache: Path, meta: Path, key: str):
        """Return cached embeddings only if the meta hash matches; corrupt files mean rebuild."""
        try:
            if not (cache.exists() and meta.exists()):
                return None
            if json.loads(meta.read_text(encoding="utf-8")).get("key") != key:
                return None
            return np.load(cache)
        except Exception:
            return None

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = np.asarray(self.embeddings @ q).reshape(-1)
        indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in indices:
            doc = self.documents[int(idx)]
            results.append({
                "score": float(scores[idx]),
                "report_id": doc.get("report_id"),
                "segment_id": doc.get("segment_id"),
                "source": doc.get("source"),
                "publisher": doc.get("publisher"),
                "publication_date": doc.get("publication_date"),
                "text": doc.get("text") or doc.get("clean_text") or "",
                "entities": doc.get("entities", []),
                "synthetic": bool(doc.get("synthetic", False)),
                "retrieval_method": "dense",
            })
        return results


class HybridRAGRetriever:
    """Reciprocal Rank Fusion of dense retrieval and BM25 lexical retrieval."""

    def __init__(self, dense_retriever: StandardRAGRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.documents = dense_retriever.documents
        self.rrf_k = rrf_k
        self.tokenized_docs = [_tokens(t) for t in dense_retriever.texts]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info("BM25 index created!")
        logger.info("Hybrid retriever ready!")

    def retrieve(
        self, query: str, top_k: int = 5, candidate_k: int = 50
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        candidate_k = max(top_k, candidate_k)
        q = self.dense.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        dense_scores = np.asarray(self.dense.embeddings @ q).reshape(-1)
        dense_ids = np.argsort(-dense_scores)[:candidate_k]
        bm25_scores = np.asarray(self.bm25.get_scores(_tokens(query)))
        bm25_ids = np.argsort(-bm25_scores)[:candidate_k]

        fused_scores: dict[int, float] = {}
        for rank, idx in enumerate(dense_ids, 1):
            fused_scores[int(idx)] = fused_scores.get(int(idx), 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ids, 1):
            fused_scores[int(idx)] = fused_scores.get(int(idx), 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        results = []
        for idx in ranked:
            doc = self.documents[idx]
            results.append({
                "rrf_score": float(fused_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "bm25_score": float(bm25_scores[idx]),
                "report_id": doc.get("report_id"),
                "segment_id": doc.get("segment_id"),
                "source": doc.get("source"),
                "publisher": doc.get("publisher"),
                "publication_date": doc.get("publication_date"),
                "text": doc.get("text") or doc.get("clean_text") or "",
                "entities": doc.get("entities", []),
                "synthetic": bool(doc.get("synthetic", False)),
                "retrieval_method": "hybrid",
            })
        return results


class ClaimConditionedRetriever:
    """Claim-Conditioned Retrieval matching the evidence gate in 3 stages:

    1. Entity-Pair Sentence Candidates: Keep only sentences where the subject
       (with aliases) and the object co-occur, aligning retrieval directly with gate expectations.
    2. Dual Query (Support + Counter): Issue both the affirmative claim query and a negated variant
       ('no longer', 'stopped', 'ceased', 'takedown', 'disrupted', 'not attributed') so contradictory
       evidence is actively retrieved and not starved out.
    3. Publisher Capping & Diversification: Cap retrieved candidates per publisher (default 2)
       to ensure the publisher count driving verdicts is genuinely independent and not inflated
       by a single syndicated vendor.
    """

    COUNTER_TERMS = (
        "stopped",
        "ceased",
        "no longer",
        "disrupted",
        "takedown",
        "terminated",
        "inactive",
        "not attributed",
        "denied",
        "disputed",
    )

    def __init__(
        self,
        hybrid_retriever: HybridRAGRetriever,
        gate: Any | None = None,
        max_per_publisher: int = 2,
    ):
        self.hybrid = hybrid_retriever
        self.documents = hybrid_retriever.documents
        self.max_per_publisher = max_per_publisher
        if gate is None:
            from src.agent.decision_engine import EntityAwareEvidenceGate
            self.gate = EntityAwareEvidenceGate()
        else:
            self.gate = gate
        logger.info(
            "ClaimConditionedRetriever initialized (max_per_publisher=%d)",
            self.max_per_publisher,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Fallback generic retrieve calling the underlying hybrid retriever."""
        return self.hybrid.retrieve(query, top_k=top_k)

    def retrieve_for_claim(
        self,
        subject: str,
        predicate: str,
        obj: str,
        natural_query: str | None = None,
        top_k: int = 10,
        candidate_pool_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Execute 3-stage claim-conditioned retrieval for the given claim triple."""
        if not subject or not obj:
            return self.hybrid.retrieve(natural_query or f"{subject} {predicate} {obj}", top_k=top_k)

        # Stage 2: Dual Query Generation (Support + Counter)
        primary_query = natural_query or f"{subject} {predicate} {obj}"
        counter_query = f"{subject} {' '.join(self.COUNTER_TERMS[:5])} {obj}"

        pool_aff = self.hybrid.retrieve(primary_query, top_k=candidate_pool_k, candidate_k=candidate_pool_k)
        pool_cnt = self.hybrid.retrieve(counter_query, top_k=candidate_pool_k, candidate_k=candidate_pool_k)

        merged_candidates: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(pool_aff, 1):
            key = (item.get("report_id"), item.get("segment_id"), item.get("text", "")[:60])
            item_copy = dict(item)
            item_copy["aff_rank"] = rank
            item_copy["claim_conditioned_score"] = item_copy.get("rrf_score", 0.0)
            merged_candidates[key] = item_copy

        for rank, item in enumerate(pool_cnt, 1):
            key = (item.get("report_id"), item.get("segment_id"), item.get("text", "")[:60])
            if key in merged_candidates:
                merged_candidates[key]["cnt_rank"] = rank
                # Dual-query fusion bonus: items relevant to both affirmative and counter queries
                merged_candidates[key]["claim_conditioned_score"] = (
                    merged_candidates[key]["claim_conditioned_score"] + (1.0 / (60 + rank))
                )
                merged_candidates[key]["dual_query_hit"] = True
            else:
                item_copy = dict(item)
                item_copy["cnt_rank"] = rank
                item_copy["claim_conditioned_score"] = item.get("rrf_score", 0.0)
                item_copy["is_counter_candidate"] = True
                merged_candidates[key] = item_copy

        # Stage 1: Entity-Pair Sentence-Level Pre-Gating
        # Keep only candidates that satisfy the EntityAwareEvidenceGate same-sentence co-occurrence
        probative_candidates: list[dict[str, Any]] = []
        for cand in merged_candidates.values():
            text = cand.get("text", "")
            is_prob, reason = self.gate.is_probative(text, subject, obj)
            cand["gate_status"] = reason
            if is_prob:
                probative_candidates.append(cand)

        # If candidate pool from top-60 had fewer than top_k probative items,
        # perform corpus scan over documents for co-occurring segments
        if len(probative_candidates) < top_k:
            seen_texts = {c.get("text", "") for c in probative_candidates}
            for doc in self.documents:
                doc_text = doc.get("text") or doc.get("clean_text") or ""
                if doc_text in seen_texts:
                    continue
                is_prob, reason = self.gate.is_probative(doc_text, subject, obj)
                if is_prob:
                    item = {
                        "rrf_score": 0.01,
                        "claim_conditioned_score": 0.01,
                        "report_id": doc.get("report_id"),
                        "segment_id": doc.get("segment_id"),
                        "source": doc.get("source"),
                        "publisher": doc.get("publisher"),
                        "publication_date": doc.get("publication_date"),
                        "text": doc_text,
                        "entities": doc.get("entities", []),
                        "synthetic": bool(doc.get("synthetic", False)),
                        "retrieval_method": "claim_conditioned_scan",
                        "gate_status": reason,
                    }
                    probative_candidates.append(item)
                    seen_texts.add(doc_text)
                    if len(probative_candidates) >= candidate_pool_k:
                        break

        # Sort probative candidates by claim_conditioned_score descending
        probative_candidates.sort(
            key=lambda x: x.get("claim_conditioned_score", 0.0), reverse=True
        )

        # Stage 3: Publisher Capping & Diversification
        final_results: list[dict[str, Any]] = []
        publisher_counts: dict[str, int] = {}
        for cand in probative_candidates:
            pub = (cand.get("publisher") or cand.get("source") or "Unknown").strip().lower()
            current_count = publisher_counts.get(pub, 0)
            if current_count < self.max_per_publisher:
                publisher_counts[pub] = current_count + 1
                cand["retrieval_method"] = "claim_conditioned"
                final_results.append(cand)
                if len(final_results) >= top_k:
                    break

        return final_results
