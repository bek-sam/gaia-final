"""OpenAI Responses API wrapper: bounded retries, token/cost accounting and a hard spend cap."""
from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path

from .config import Settings
from .llm import ModelError, QuotaExhausted, Usage


class BudgetExceeded(QuotaExhausted):
    """The configured AGENT_BUDGET_USD would be exceeded; stop before spending more."""


class SpendLedger:
    """Cumulative estimated spend across runs, persisted in runs/spend.json."""

    _lock = threading.Lock()

    def __init__(self, path: Path, budget_usd: float):
        self.path = Path(path)
        self.budget = budget_usd

    def total(self) -> float:
        if not self.path.exists():
            return 0.0
        return float(json.loads(self.path.read_text()).get("total_usd", 0.0))

    def add(self, usd: float, what: str) -> float:
        with SpendLedger._lock:
            data = json.loads(self.path.read_text()) if self.path.exists() else {"total_usd": 0.0, "by_item": {}}
            data["total_usd"] = round(data["total_usd"] + usd, 6)
            data["by_item"][what] = round(data["by_item"].get(what, 0.0) + usd, 6)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2))
            return data["total_usd"]

    def check(self) -> None:
        spent = self.total()
        if spent >= self.budget:
            raise BudgetExceeded(f"Estimated spend ${spent:.4f} reached AGENT_BUDGET_USD=${self.budget:.2f}. "
                                 "Raise the budget in .env to continue.")


class OpenAILLM:
    def __init__(self, settings: Settings):
        import openai

        if not settings.openai_api_key:
            raise ModelError("OPENAI_API_KEY is not set (add it to .env)")
        self.settings = settings
        self.openai = openai
        self.client = openai.OpenAI(api_key=settings.openai_api_key, timeout=240, max_retries=0)
        self.ledger = SpendLedger(settings.runs_dir / "spend.json", settings.budget_usd)

    def _cost(self, usage) -> float:
        s = self.settings
        if usage is None:
            return 0.0
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        fresh = (usage.input_tokens or 0) - cached
        return (fresh * s.price_in_per_m + cached * s.price_cached_in_per_m + (usage.output_tokens or 0) * s.price_out_per_m) / 1e6

    def _retrying(self, fn, deadline: float | None):
        attempt = 0
        while True:
            self.ledger.check()
            try:
                return fn()
            except self.openai.RateLimitError as e:
                if "insufficient_quota" in str(e):
                    raise QuotaExhausted(f"OpenAI account has insufficient quota/balance: {str(e)[:300]}") from e
                err = e
            except (self.openai.APIConnectionError, self.openai.APITimeoutError, self.openai.InternalServerError) as e:
                err = e
            except self.openai.APIStatusError as e:
                raise ModelError(f"OpenAI request failed (HTTP {e.status_code}): {str(e)[:500]}") from e
            attempt += 1
            if attempt > self.settings.max_retries:
                raise ModelError(f"OpenAI request failed after retries: {str(err)[:400]}") from err
            delay = min(60.0, 2.0 * 2 ** attempt) + random.uniform(0, 1.5)
            if deadline is not None and time.monotonic() + delay > deadline:
                raise ModelError(f"OpenAI request failed and question deadline reached: {str(err)[:300]}") from err
            print(f"  [llm] {type(err).__name__}; retry {attempt}/{self.settings.max_retries} in {delay:.0f}s")
            time.sleep(delay)

    def respond(self, usage: Usage, deadline: float | None = None, **kwargs):
        model = kwargs.setdefault("model", self.settings.model)
        resp = self._retrying(lambda: self.client.responses.create(**kwargs), deadline)
        u = resp.usage
        cost = self._cost(u)
        searches = sum(1 for item in resp.output if item.type == "web_search_call")
        cost += searches * self.settings.price_web_search
        usage.add_openai(model, u, cost, searches)
        self.ledger.add(cost, model)
        return resp

    def transcribe(self, path: Path, usage: Usage, deadline: float | None = None) -> str:
        model = self.settings.transcribe_model

        def call():
            with open(path, "rb") as f:
                return self.client.audio.transcriptions.create(model=model, file=f)

        result = self._retrying(call, deadline)
        # Priced per audio minute; estimated from file size (~1 MB per minute of typical MP3), rounded up generously.
        minutes = max(0.1, path.stat().st_size / 1_000_000)
        cost = minutes * self.settings.price_transcribe_per_min
        usage.add_openai(model, None, cost, 0)
        self.ledger.add(cost, model)
        return result.text
