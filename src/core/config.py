"""Single validated Settings object (env ``CTI_*`` + optional ``config.yaml``).

Invalid configuration raises :class:`ConfigError` with a readable message;
``load_settings_or_exit`` turns that into exit code 2 for CLI entry points.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CTI_", extra="forbid")

    mode: Literal["demo", "eval", "prod"] = "prod"

    # Data
    data_dir: Path = REPO_ROOT / "data"
    segments_file: str = "processed_segments.jsonl"
    graph_file: str = "cti_knowledge_graph_trusted.json"
    assertions_file: str = "assertions.jsonl"
    registry_file: str = "knowledge/mitre_attack_registry.json"
    alias_overrides_file: str = "knowledge/alias_overrides.json"
    manifest_file: str = "MANIFEST.json"
    demo_overlay_file: str = "demo_overlay.jsonl"
    load_demo_overlay: bool = False
    verify_manifest: bool = True

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    offline: bool = True

    # LLM (optional)
    llm_enabled: bool = True
    llm_url: Optional[str] = None  # None -> auto-detect on common local ports
    llm_model: Optional[str] = None  # None -> read from /v1/models
    llm_timeout_secs: float = Field(12.0, gt=0, le=300)
    llm_retries: int = Field(1, ge=0, le=3)

    # Retrieval / verification
    top_k: int = Field(10, ge=1, le=50)
    retrieval_pool_k: int = Field(100, ge=5, le=200)
    entail_threshold: float = Field(0.50, gt=0, lt=1)
    contra_threshold: float = Field(0.50, gt=0, lt=1)
    temporal_threshold_days: int = Field(90, ge=1, le=3650)
    max_verify_passages: int = Field(15, ge=1, le=100)

    # Sessions / uploads
    session_ttl_secs: int = Field(7200, ge=60)
    confirm_token_ttl_secs: int = Field(600, ge=30)
    max_upload_bytes: int = Field(20 * 1024 * 1024, ge=1024)
    max_upload_pages: int = Field(200, ge=1)
    max_sessions: int = Field(256, ge=1)

    # MCP / HTTP
    mcp_host: str = "127.0.0.1"
    mcp_port: int = Field(8765, ge=1, le=65535)
    mcp_token: Optional[str] = None
    rate_limit_per_min: int = Field(60, ge=1)
    max_response_bytes: int = Field(200 * 1024, ge=1024)
    max_passage_chars: int = Field(2000, ge=100)

    log_level: str = "INFO"
    log_json: bool = True

    @field_validator("log_level")
    @classmethod
    def _level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("log_level must be DEBUG/INFO/WARNING/ERROR/CRITICAL")
        return v

    # Convenience paths -------------------------------------------------
    @property
    def segments_path(self) -> Path:
        return self.data_dir / self.segments_file

    @property
    def graph_path(self) -> Path:
        return self.data_dir / self.graph_file

    @property
    def assertions_path(self) -> Path:
        return self.data_dir / self.assertions_file

    @property
    def registry_path(self) -> Path:
        return self.data_dir / self.registry_file

    @property
    def alias_overrides_path(self) -> Path:
        return self.data_dir / self.alias_overrides_file

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / self.manifest_file

    @property
    def demo_overlay_path(self) -> Path:
        return self.data_dir / self.demo_overlay_file

    @property
    def embedding_cache_path(self) -> Path:
        return self.data_dir / "segment_embeddings_minilm.npy"

    @property
    def embedding_cache_meta_path(self) -> Path:
        return self.data_dir / "segment_embeddings_minilm.meta.json"


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ConfigError(f"{path} exists but PyYAML is not installed") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ConfigError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping of setting names to values")
    return data


def load_settings(config_file: Optional[os.PathLike] = None, **overrides: Any) -> Settings:
    """Build Settings. Precedence: explicit overrides > env > config.yaml > defaults."""
    cfg_path = Path(config_file or os.environ.get("CTI_CONFIG", REPO_ROOT / "config.yaml"))
    file_values: Dict[str, Any] = _read_yaml(cfg_path) if cfg_path.exists() else {}
    # env must beat the file: drop file keys that the environment sets explicitly
    for key in list(file_values):
        if f"CTI_{key.upper()}" in os.environ:
            file_values.pop(key)
    merged = {**file_values, **overrides}
    try:
        settings = Settings(**merged)
    except ValidationError as exc:
        lines = []
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "settings"
            lines.append(f"  - {loc}: {err['msg']}")
        raise ConfigError("Invalid configuration:\n" + "\n".join(lines)) from exc
    if settings.mode == "eval" and settings.load_demo_overlay:
        raise ConfigError("Invalid configuration:\n  - load_demo_overlay is refused when mode=eval")
    if settings.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    return settings


def load_settings_or_exit(**overrides: Any) -> Settings:
    try:
        return load_settings(**overrides)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(ConfigError.exit_code)
