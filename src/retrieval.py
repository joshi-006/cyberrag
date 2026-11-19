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
