"""Thin Gemini wrapper: pacing, bounded retries with rate-limit handling, and token accounting."""
from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import errors, types

from .config import Settings


class QuotaExhausted(RuntimeError):
    """Daily/free-tier quota is used up; retrying now is pointless."""


class GeminiQuotaExhausted(QuotaExhausted):
    """Gemini free-tier quota for a specific model is used up."""


class ModelError(RuntimeError):
    pass


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    by_model: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    web_searches: int = 0

    def add_openai(self, model: str, u, cost: float, searches: int) -> None:
        self.requests += 1
        self.cost_usd += cost
        self.web_searches += searches
        i = (u.input_tokens or 0) if u is not None else 0
        o = (u.output_tokens or 0) if u is not None else 0
        r = (getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0) if u is not None else 0
        self.input_tokens += i
        self.output_tokens += o - r
        self.thinking_tokens += r
        m = self.by_model.setdefault(model, {"requests": 0, "input": 0, "output": 0, "thinking": 0, "cost_usd": 0.0})
        m["requests"] += 1
        m["input"] += i
        m["output"] += o - r
        m["thinking"] += r
        m["cost_usd"] = round(m.get("cost_usd", 0.0) + cost, 6)

    def add(self, model: str, meta) -> None:
        self.requests += 1
        if meta is None:
            return
        i = meta.prompt_token_count or 0
        o = meta.candidates_token_count or 0
        t = meta.thoughts_token_count or 0
        self.input_tokens += i
        self.output_tokens += o
        self.thinking_tokens += t
        m = self.by_model.setdefault(model, {"requests": 0, "input": 0, "output": 0, "thinking": 0})
        m["requests"] += 1
        m["input"] += i
        m["output"] += o
        m["thinking"] += t

    def as_dict(self, settings: Settings | None = None) -> dict:
        d = {"requests": self.requests, "input_tokens": self.input_tokens,
             "output_tokens": self.output_tokens, "thinking_tokens": self.thinking_tokens,
             "by_model": self.by_model}
        d["web_searches"] = self.web_searches
        d["estimated_cost_usd"] = round(self.cost_usd, 6)
        return d


_RETRY_DELAY = re.compile(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(\d+(?:\.\d+)?)s", re.I)


def _retry_delay(err: Exception) -> float | None:
    m = _RETRY_DELAY.search(str(err))
    return float(m.group(1)) if m else None


def _is_daily_quota(err: Exception) -> bool:
    s = str(err)
    return "PerDay" in s or "per day" in s.lower() or "limit: 0" in s


class Gemini:
    _lock = threading.Lock()
    _last_request = 0.0

    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ModelError("GEMINI_API_KEY is not set (add it to .env)")
        self.settings = settings
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=180_000),  # ms
        )

    def _pace(self) -> None:
        with Gemini._lock:
            wait = Gemini._last_request + self.settings.min_request_interval_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            Gemini._last_request = time.monotonic()

    def generate(self, model: str, contents, config: types.GenerateContentConfig, usage: Usage,
                 deadline: float | None = None):
        attempt = 0
        while True:
            self._pace()
            try:
                resp = self.client.models.generate_content(model=model, contents=contents, config=config)
                usage.add(model, resp.usage_metadata)
                return resp
            except errors.APIError as e:
                code = getattr(e, "code", None)
                retryable = code in (429, 500, 502, 503, 504) or code is None
                if code == 429 and _is_daily_quota(e):
                    raise GeminiQuotaExhausted(f"{model}: {str(e)[:500]}") from e
                attempt += 1
                if not retryable or attempt > self.settings.max_retries:
                    raise ModelError(f"{model} request failed (HTTP {code}): {str(e)[:500]}") from e
                delay = _retry_delay(e) or min(60.0, 2.0 * 2 ** attempt)
                delay += random.uniform(0, 1.5)
                if deadline is not None and time.monotonic() + delay > deadline:
                    raise ModelError(f"{model} request failed and question deadline reached: {str(e)[:300]}") from e
                print(f"  [llm] HTTP {code}; retry {attempt}/{self.settings.max_retries} in {delay:.0f}s")
                time.sleep(delay)
            except Exception as e:  # network-level errors from httpx etc.
                attempt += 1
                if attempt > self.settings.max_retries:
                    raise ModelError(f"{model} request failed: {e!r}") from e
                delay = min(60.0, 2.0 * 2 ** attempt)
                print(f"  [llm] {type(e).__name__}; retry {attempt}/{self.settings.max_retries} in {delay:.0f}s")
                time.sleep(delay)
