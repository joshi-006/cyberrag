"""Startup checks (R2) and service construction.

Failures here are specific and actionable: the file or model that is wrong is
named, never a bare stack trace.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.aliases import AliasResolver
from src.core.config import Settings
from src.core.errors import DataIntegrityError

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1


def sha256_file(path: Path, chunk: int = 1 << 20, normalize_newlines: bool = False) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if normalize_newlines:
            content = f.read()
            h.update(content.replace(b"\r\n", b"\n"))
        else:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
    return h.hexdigest()


def required_files(settings: Settings) -> List[Path]:
    return [
        settings.segments_path,
        settings.graph_path,
        settings.assertions_path,
        settings.registry_path,
        settings.alias_overrides_path,
    ]


def build_manifest(settings: Settings) -> Dict[str, Any]:
    files = {}
    for p in required_files(settings):
        if not p.exists():
            raise DataIntegrityError(f"Cannot build manifest: required file missing: {p}", str(p))
        files[p.relative_to(settings.data_dir).as_posix()] = {
            "sha256": sha256_file(p),
            "sha256_lf": sha256_file(p, normalize_newlines=True),
            "bytes": p.stat().st_size,
        }
    return {"version": MANIFEST_VERSION, "files": files}


def manifest_hash(manifest: Dict[str, Any]) -> str:
    raw = json.dumps(manifest["files"], sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def check_data(settings: Settings) -> Optional[str]:
    """Verify required files exist and match data/MANIFEST.json. Returns the manifest hash."""
    for p in required_files(settings):
        if not p.exists():
            raise DataIntegrityError(
                f"Required data file is missing: {p}. Restore it from the release zip or git.", str(p)
            )
    if not settings.verify_manifest:
        return None
    mpath = settings.manifest_path
    if not mpath.exists():
        logger.warning("No %s; skipping hash verification (run scripts/build_manifest.py)", mpath.name)
        return None
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DataIntegrityError(f"{mpath} is not valid JSON: {exc}", str(mpath)) from exc
    for rel, meta in manifest.get("files", {}).items():
        p = settings.data_dir / rel
        if not p.exists():
            raise DataIntegrityError(f"Data file listed in the manifest is missing: {p}", str(p))
        actual = sha256_file(p)
        expected = meta.get("sha256")
        expected_lf = meta.get("sha256_lf")

        # Allow exact match, or cross-platform LF-normalized match (handling CRLF vs LF checkout)
        if actual == expected or (expected_lf and actual == expected_lf):
            continue

        actual_norm = sha256_file(p, normalize_newlines=True)
        if actual_norm == expected or (expected_lf and actual_norm == expected_lf):
            continue

        raise DataIntegrityError(
            f"Data file {p} does not match data/MANIFEST.json (corrupted or edited). "
            "Restore the file, or run scripts/build_manifest.py if the change is intentional.",
            str(p),
        )
    return manifest_hash(manifest)


def load_overlay(settings: Settings) -> List[Dict[str, Any]]:
    """Demo overlay only when explicitly enabled (H2). Never in eval mode."""
    if not settings.load_demo_overlay:
        return []
    p = settings.demo_overlay_path
    if not p.exists():
        raise DataIntegrityError(f"load_demo_overlay is set but {p} does not exist", str(p))
    docs = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            rec["synthetic"] = True
            docs.append(rec)
    return docs


def build_service(settings: Settings):
    """Construct the full production service (loads models once)."""
    from src.core.service import VerificationService

    mhash = check_data(settings)
    aliases = AliasResolver(settings.registry_path, settings.alias_overrides_path)

    try:
        from src.graph_retrieval import GraphRAGRetriever
        from src.hybrid_graph_rag import HybridGraphRAG
        from src.retrieval import StandardRAGRetriever

        dense = StandardRAGRetriever(
            data_path=settings.segments_path,
            model_name=settings.embedding_model,
            cache_path=settings.embedding_cache_path,
            cache_meta_path=settings.embedding_cache_meta_path,
        )
        graph = GraphRAGRetriever(graph_path=settings.graph_path, assertions_path=settings.assertions_path)
        retriever = HybridGraphRAG(dense_retriever=dense, graph_retriever=graph)
    except (OSError, ImportError) as exc:
        raise DataIntegrityError(
            f"Could not load the embedding model {settings.embedding_model!r} ({exc}). "
            "Run `python scripts/download_models.py` once while online.",
        ) from exc

    verifier = None
    try:
        from src.verification.evidence_verifier import CTIEvidenceVerifier

        verifier = CTIEvidenceVerifier(model_name=settings.nli_model)
    except Exception as exc:
        logger.error("NLI verifier unavailable (%s). verify_* will return VERIFIER_UNAVAILABLE.", exc)

    llm = None
    if settings.llm_enabled:
        from src.llm.llm_client import LocalLLMClient

        llm = LocalLLMClient(base_url=settings.llm_url, model=settings.llm_model, timeout_secs=settings.llm_timeout_secs)

    svc = VerificationService(
        settings, retriever=retriever, verifier=verifier, aliases=aliases, llm=llm,
        overlay_docs=load_overlay(settings) or None,
    )
    svc.manifest_hash = mhash
    return svc
