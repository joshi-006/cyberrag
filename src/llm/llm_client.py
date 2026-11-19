"""
Local LLM Client for CyberRAG.
Connects to locally hosted Qwen2.5-3B-Instruct via LM Studio / llama.cpp / Ollama OpenAI API.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LocalLLMClient:
    """
    OpenAI-compatible client for locally served Qwen2.5-3B-Instruct.
    Auto-detects active port (default 1235 in LM Studio / Bionic, fallback 1234).
    Fails gracefully if local server is offline.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "qwen2.5-3b-instruct",
        timeout_secs: float = 12.0,
    ):
        if base_url:
            self.base_url = base_url.rstrip("/")
        else:
            self.base_url = self._auto_detect_url()
        self.model = model
        self.timeout_secs = timeout_secs

    @staticmethod
    def _list_models(base_url: str, timeout: float) -> List[str]:
        """Model ids from an OpenAI-compatible ``/models`` endpoint, or [] if the
        server is not one (a web app answering 200 with HTML does not count)."""
        try:
            req = urllib.request.Request(f"{base_url}/models", method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    return []
                body = json.loads(resp.read(1_000_000).decode("utf-8"))
        except Exception:
            return []
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            return []
        return [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]

    def _auto_detect_url(self) -> str:
        for port in [1235, 1234, 8000, 8080]:
            url = f"http://127.0.0.1:{port}/v1"
            if self._list_models(url, timeout=0.8):
                return url
        return "http://127.0.0.1:1235/v1"

    def is_available(self) -> bool:
        return bool(self._list_models(self.base_url, timeout=1.0))

    def chat_complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> Optional[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_secs) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.debug(f"Local LLM completion failed (offline or timeout): {e}")
            return None
