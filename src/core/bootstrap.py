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


TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".csv", ".md", ".yaml", ".yml"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Raw byte hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text_normalized(path: Path, chunk: int = 1 << 20) -> str:
    """Hash with CRLF folded to LF, so a Windows and a Unix checkout of the same file agree."""
    h = hashlib.sha256()
    pending = b""
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            block = pending + block
            # Hold back a trailing CR: its LF may start the next block.
            if block.endswith(b"\r"):
                block, pending = block[:-1], b"\r"
            else:
                pending = b""
            h.update(block.replace(b"\r\n", b"\n"))
    h.update(pending)
    return h.hexdigest()


def content_hash(path: Path) -> str:
    """Manifest hash: line-ending independent for text files, raw bytes for everything else."""
    if path.suffix.lower() in TEXT_SUFFIXES:
        return sha256_text_normalized(path)
    return sha256_file(path)


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
        files[p.relative_to(settings.data_dir).as_posix()] = {"sha256": content_hash(p), "bytes": p.stat().st_size}
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
        # Accept either the line-ending independent hash (current manifests) or the
        # raw byte hash (manifests built before normalisation, on the same OS).
        expected = meta.get("sha256")
        if expected not in (content_hash(p), sha256_file(p)):
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


def build_service(settings: Settings, verifier=None):
    """Construct the full production service (loads models once).

    ``verifier`` lets a host that already loaded the NLI model (the workbench)
    share it instead of holding a second copy in memory.
    """
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

    if verifier is None:
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
