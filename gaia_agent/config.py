"""Runtime settings, loaded from environment variables (and a local, git-ignored .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional (e.g. on a Space where secrets are real env vars)
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

API_URL = "https://agents-course-unit4-scoring.hf.space"


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _default_provider() -> str:
    explicit = os.getenv("AGENT_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    return "openai" if os.getenv("OPENAI_API_KEY") else "gemini"


DEFAULT_MODELS = {"openai": "gpt-6.1-sol", "gemini": "gemini-3.8-flash"}


@dataclass
class Settings:
    api_url: str = field(default_factory=lambda: os.getenv("SCORING_API_URL", API_URL))
    # "openai" (Responses API) or "gemini". Defaults to openai when OPENAI_API_KEY is set.
    provider: str = field(default_factory=_default_provider)
    model: str = field(default_factory=lambda: os.getenv("AGENT_MODEL", ""))
    # Gemini model used inside tools (Google-grounded search, YouTube video analysis) when GEMINI_API_KEY is set.
    tool_model: str = field(default_factory=lambda: os.getenv("AGENT_TOOL_MODEL", "gemini-3.8-flash"))
    tool_model_fallbacks: list = field(default_factory=lambda: [m.strip() for m in os.getenv(
        "AGENT_TOOL_MODEL_FALLBACKS", "gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash").split(",") if m.strip()])
    transcribe_model: str = field(default_factory=lambda: os.getenv("AGENT_TRANSCRIBE_MODEL", "gpt-4o-transcribe"))
    reasoning_effort: str = field(default_factory=lambda: os.getenv("AGENT_REASONING_EFFORT", "medium"))
    openai_web_search: bool = field(default_factory=lambda: os.getenv("AGENT_OPENAI_WEB_SEARCH", "1") not in ("0", "false", "off"))
    runs_dir: Path = field(default_factory=lambda: Path(os.getenv("AGENT_RUNS_DIR", PROJECT_ROOT / "runs")))
    max_steps: int = field(default_factory=lambda: _int("AGENT_MAX_STEPS", 16))
    question_timeout_s: int = field(default_factory=lambda: _int("AGENT_QUESTION_TIMEOUT", 600))
    # Minimum seconds between Gemini requests (free tier has low requests-per-minute caps).
    min_request_interval_s: float = field(default_factory=lambda: _float("AGENT_MIN_INTERVAL", 4.0))
    max_retries: int = field(default_factory=lambda: _int("AGENT_MAX_RETRIES", 6))
    # Hard cap on cumulative estimated OpenAI spend recorded in runs/spend.json.
    budget_usd: float = field(default_factory=lambda: _float("AGENT_BUDGET_USD", 5.0))
    # Prices in USD (per 1M tokens / per web search call / per audio minute) used for the cost estimate.
    price_in_per_m: float = field(default_factory=lambda: _float("AGENT_PRICE_IN_PER_M", -1))
    price_cached_in_per_m: float = field(default_factory=lambda: _float("AGENT_PRICE_CACHED_IN_PER_M", -1))
    price_out_per_m: float = field(default_factory=lambda: _float("AGENT_PRICE_OUT_PER_M", -1))
    price_web_search: float = field(default_factory=lambda: _float("AGENT_PRICE_WEB_SEARCH", 0.01))
    price_transcribe_per_min: float = field(default_factory=lambda: _float("AGENT_PRICE_TRANSCRIBE_PER_MIN", 0.006))

    def __post_init__(self) -> None:
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"AGENT_PROVIDER must be one of {list(DEFAULT_MODELS)}")
        if not self.model:
            self.model = DEFAULT_MODELS[self.provider]
        known = MODEL_PRICES.get(self.model, (0.0, 0.0, 0.0) if self.provider == "gemini" else None)
        if known is None and min(self.price_in_per_m, self.price_out_per_m) < 0:
            raise ValueError(f"No price known for {self.model}; set AGENT_PRICE_IN_PER_M / AGENT_PRICE_OUT_PER_M")
        if known is not None:
            if self.price_in_per_m < 0:
                self.price_in_per_m = known[0]
            if self.price_cached_in_per_m < 0:
                self.price_cached_in_per_m = known[1]
            if self.price_out_per_m < 0:
                self.price_out_per_m = known[2]
        if self.price_cached_in_per_m < 0:
            self.price_cached_in_per_m = self.price_in_per_m
        self.runs_dir = Path(self.runs_dir)

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or None

    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY") or None


# (input, cached input, output) USD per 1M tokens, Standard tier, from https://developers.openai.com/api/docs/pricing
# (checked 2026-10-10). Override with AGENT_PRICE_* if prices change.
MODEL_PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-sol": (2.00, 0.20, 10.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "gpt-5.6-luna": (0.20, 0.02, 1.20),
    "gpt-5.5": (5.00, 0.50, 30.00),
    "gpt-5.4": (2.50, 0.25, 15.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5-mini": (0.25, 0.025, 2.00),
}
