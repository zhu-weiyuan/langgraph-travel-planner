"""
LLM Client abstraction layer.

Provides a pluggable interface for LLM calls with built-in retry,
timeout, and robust JSON parsing.  API keys are read from environment
variables by default — **no hardcoding**.
"""

import json
import logging
import os
import re
import time
import urllib.request
from abc import ABC, abstractmethod
from typing import List, Dict

logger = logging.getLogger(__name__)


# ============================================================
# Abstract base
# ============================================================

class LLMClient(ABC):
    """Abstract base class for LLM clients.

    Subclasses must implement :meth:`chat` and :meth:`chat_json`.
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        system: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat request and return the plain-text response."""
        ...

    @abstractmethod
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: str,
        max_tokens: int = 256,
    ) -> dict:
        """Send a chat request and parse the response as JSON."""
        ...


# ============================================================
# Local / OpenAI-compatible implementation
# ============================================================

class LocalLLMClient(LLMClient):
    """Local LLM client (llama.cpp / any OpenAI-compatible HTTP endpoint).

    Features
    --------
    * Configurable **timeout** (per-request) and **retry** with
      exponential back-off.
    * Robust JSON extraction from free-form LLM outputs (3 strategies).
    * API key read from ``LLM_API_KEY`` env-var by default.
    * Endpoint URL read from ``LLM_API_URL`` env-var by default.

    Parameters
    ----------
    api_url : str, optional
        Chat-completions endpoint. Falls back to env-var ``LLM_API_URL``
        and then to ``http://127.0.0.1:8080/v1/chat/completions``.
    api_key : str, optional
        Bearer token. Falls back to env-var ``LLM_API_KEY``.
    timeout : int
        HTTP timeout in seconds (default 180).
    max_retries : int
        Maximum number of attempts before giving up (default 3).
    retry_base_delay : float
        Base delay (seconds) for exponential back-off (default 2.0).
    """

    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        timeout: int = 180,
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
    ):
        self.api_url = (
            api_url
            or os.environ.get("LLM_API_URL", "http://127.0.0.1:8080/v1/chat/completions")
        )
        self.api_key = (
            api_key or os.environ.get("LLM_API_KEY", "")
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

    # ------------------------------------------------------------------
    # Internal: HTTP request with retry
    # ------------------------------------------------------------------

    def _request(
        self,
        messages: List[Dict[str, str]],
        system: str,
        max_tokens: int,
        temperature: float,
        timeout: int = None,
    ) -> dict:
        """Send the request and return the raw JSON response.

        Retries up to ``self.max_retries`` times with exponential
        back-off on failure.
        """
        payload = {
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        effective_timeout = timeout or self.timeout

        last_error = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    self.api_url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    delay = self.retry_base_delay ** attempt
                    logger.warning(
                        "[LLM] Attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1, self.max_retries, exc, delay,
                    )
                    time.sleep(delay)

        raise last_error  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Call the LLM and return plain-text response."""
        try:
            result = self._request(messages, system, max_tokens, temperature)
            return result["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("[LLM] chat error: %s", exc)
            return "抱歉，我暂时无法处理您的请求，请稍后再试。"

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        system: str,
        max_tokens: int = 256,
    ) -> dict:
        """Call the LLM and parse the response as JSON (best-effort)."""
        try:
            result = self._request(messages, system, max_tokens, temperature=0.3)
            text = result["choices"][0]["message"]["content"].strip()
            return self._parse_json(text)
        except Exception as exc:
            logger.error("[LLM] chat_json error: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # JSON extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract a JSON object from *text* using three fallback strategies.

        1. Line-by-line scan for single-line JSON objects.
        2. Regex match for first ``{…}`` block (handles multi-line).
        3. Try the entire text (stripping trailing back-ticks).
        """
        # Strategy 1: line-by-line
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    pass

        # Strategy 2: regex for first JSON-like block
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Strategy 3: whole text
        try:
            cleaned = text.strip().rstrip("`").strip()
            if cleaned.startswith("{") and cleaned.endswith("}"):
                return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        logger.warning("[LLM] JSON parse failed — raw output: %.200s", text)
        return {}
