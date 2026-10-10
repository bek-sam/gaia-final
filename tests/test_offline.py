"""Offline tests: API parsing, payloads, attachments, tools, persistence, and the agent loop with a fake model."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from google.genai import types

from gaia_agent import scoring_api, tools
from gaia_agent.agent import GaiaAgent, clean_answer, extract_final_answer
from gaia_agent.config import PROJECT_ROOT, Settings
from gaia_agent.llm import Usage
from gaia_agent.store import RunStore

SPACE = "https://huggingface.co/spaces/someone/my-agent/tree/main"


# ---------------------------------------------------------------- scoring API
def test_parse_questions_keeps_fields():
    qs = scoring_api.parse_questions([{"task_id": "a", "question": "q?", "Level": "1", "file_name": "x.png", "extra": 1}])
    assert qs == [{"task_id": "a", "question": "q?", "Level": "1", "file_name": "x.png"}]


@pytest.mark.parametrize("bad", [[], {}, [{"question": "no id"}], [{"task_id": "a"}]])
def test_parse_questions_rejects_malformed(bad):
    with pytest.raises(scoring_api.ScoringAPIError):
        scoring_api.parse_questions(bad)


def test_build_payload_matches_schema_and_order():
    p = scoring_api.build_payload(" user ", SPACE, {"b": "2", "a": "Paris"}, ["a", "b", "c"])
    assert p == {"username": "user", "agent_code": SPACE,
                 "answers": [{"task_id": "a", "submitted_answer": "Paris"}, {"task_id": "b", "submitted_answer": "2"}]}
    json.dumps(p)  # serialisable


@pytest.mark.parametrize("kwargs", [
    dict(username="", agent_code=SPACE, answers={"a": "x"}),
    dict(username="u", agent_code="https://github.com/u/r", answers={"a": "x"}),
    dict(username="u", agent_code=SPACE, answers={}),
    dict(username="u", agent_code=SPACE, answers={"a": ["list"]}),
])
def test_build_payload_validation(kwargs):
    with pytest.raises(ValueError):
        scoring_api.build_payload(task_ids=["a"], **kwargs)


def test_download_refuses_unsafe_names(tmp_path):
    for name in ["../x.py", ".env", "a/b.txt"]:
        with pytest.raises(scoring_api.ScoringAPIError):
            scoring_api.download_attachment("http://x", "t", name, tmp_path)


class _Resp:
    def __init__(self, status, data=b""):
        self.status_code, self._data = status, data

    def iter_content(self, n):
        yield self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def close(self):
        pass


def test_download_uses_endpoint_then_cache(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(scoring_api.requests, "get", lambda url, **kw: calls.append(url) or _Resp(200, b"abc"))
    p = scoring_api.download_attachment("http://api", "tid", "f.xlsx", tmp_path)
    assert p.read_bytes() == b"abc" and calls == ["http://api/files/tid"]
    scoring_api.download_attachment("http://api", "tid", "f.xlsx", tmp_path)
    assert len(calls) == 1  # cached


def test_download_falls_back_to_exact_gaia_file(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring_api.requests, "get", lambda url, **kw: _Resp(404))
    src = tmp_path / "src.mp3"
    src.write_bytes(b"ID3data")
    requested = {}

    def fake_dl(repo, filename, repo_type):
        requested.update(repo=repo, filename=filename, repo_type=repo_type)
        return str(src)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    p = scoring_api.download_attachment("http://api", "tid", "abc.mp3", tmp_path / "out")
    assert p.read_bytes() == b"ID3data"
    assert requested == {"repo": "gaia-benchmark/GAIA", "filename": "2023/validation/abc.mp3", "repo_type": "dataset"}


# ---------------------------------------------------------------- answer extraction
@pytest.mark.parametrize("text,expected", [
    ("reasoning...\nFINAL ANSWER: Right", "Right"),
    ("FINAL ANSWER: **3**", "3"),
    ("Final answer: `Rd5`", "Rd5"),
    ("FINAL ANSWER: \"Paris\"", "Paris"),
    ("FINAL ANSWER: Paris.", "Paris"),
    ("FINAL ANSWER: St. Louis", "St. Louis"),
    ("FINAL ANSWER: 89706.00", "89706.00"),
    ("FINAL ANSWER: broccoli, celery, fresh basil", "broccoli, celery, fresh basil"),
    ("FINAL ANSWER: Yoshida, Uehara\nThis is because...", "Yoshida, Uehara"),
    ("first FINAL ANSWER: x\nthen FINAL ANSWER: y", "y"),
    ("no marker here", None),
])
def test_extract_final_answer(text, expected):
    assert extract_final_answer(text) == expected


def test_clean_answer_preserves_case_and_inner_punctuation():
    assert clean_answer("e.g. U.S.A") == "e.g. U.S.A"
    assert clean_answer("b, e") == "b, e"
    assert clean_answer("$\\boxed{42}$") == "42"


# ---------------------------------------------------------------- tools
@pytest.fixture
def ctx(tmp_path):
    return tools.ToolContext(settings=Settings(provider="gemini", runs_dir=tmp_path), llm=None, usage=Usage())


def test_run_python_basic(ctx):
    out = tools.run_python(ctx, "print(sum(range(10)))")
    assert "exit code 0" in out and "45" in out


def test_run_python_timeout_and_errors(ctx, monkeypatch):
    monkeypatch.setattr(tools, "PY_TIMEOUT_S", 2)
    assert "Timed out" in tools.run_python(ctx, "while True: pass")
    assert "ZeroDivisionError" in tools.run_python(ctx, "1/0")


def test_run_python_env_is_scrubbed(ctx, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value-123")
    out = tools.run_python(ctx, "import os; print(sorted(os.environ))")
    assert "GEMINI_API_KEY" not in out and "secret-value-123" not in out


@pytest.mark.skipif(not Path("/usr/bin/sandbox-exec").exists(), reason="macOS sandbox only")
def test_run_python_sandbox_blocks_network_writes_and_secrets(ctx):
    net = tools.run_python(ctx, "import socket; socket.create_connection(('1.1.1.1', 80), timeout=3); print('CONNECTED')")
    assert "stdout ---\nCONNECTED" not in net and "not permitted" in net
    w = tools.run_python(ctx, f"open({str(Path.home() / 'agent_sandbox_probe.txt')!r}, 'w').write('x'); print('WROTE')")
    assert "stdout ---\nWROTE" not in w and not (Path.home() / "agent_sandbox_probe.txt").exists()
    if (PROJECT_ROOT / ".env").exists():
        r = tools.run_python(ctx, f"print(open({str(PROJECT_ROOT / '.env')!r}).read())")
        assert "exit code 0" not in r


def test_run_python_sees_attachment(ctx, tmp_path):
    f = tmp_path / "code.py"
    f.write_text("print(6*7)")
    ctx.attachment = f
    out = tools.run_python(ctx, "import runpy; runpy.run_path('code.py')")
    assert "42" in out


def test_read_attachment_spreadsheet(ctx, tmp_path):
    f = tmp_path / "sales.xlsx"
    pd.DataFrame({"Item": ["Burger", "Soda"], "Sales": [10.5, 2.25]}).to_excel(f, index=False)
    ctx.attachment = f
    out = tools.read_attachment(ctx)
    assert "Burger" in out and "10.5" in out and "2 rows x 2 cols" in out


def test_read_attachment_none(ctx):
    assert "no attachment" in tools.read_attachment(ctx)


@pytest.mark.parametrize("url", ["http://localhost/x", "http://127.0.0.1/", "file:///etc/passwd", "ftp://x.org/a", "http://169.254.169.254/latest"])
def test_check_url_blocks_unsafe(url):
    with pytest.raises(ValueError):
        tools._check_url(url)


def test_html_to_text_keeps_tables_and_drops_scripts():
    html = "<html><head><title>T</title><script>evil()</script></head><body><table><tr><th>Year</th><th>Album</th></tr><tr><td>2005</td><td>Corazón Libre</td></tr></table></body></html>"
    text = tools.html_to_text(html)
    assert "Corazón Libre" in text and "| Year" in text and "evil" not in text


def test_find_windows_and_truncate():
    text = "a" * 5000 + "NEEDLE here" + "b" * 5000
    assert "NEEDLE here" in tools._find_windows(text, "needle")
    assert "No matches" in tools._find_windows(text, "zzz")
    assert "start=100" in tools._truncate(text, 0, 100)


def test_call_tool_isolates_errors(ctx):
    assert "Unknown tool" in tools.call_tool(ctx, "nope", {})
    assert "Bad arguments" in tools.call_tool(ctx, "run_python", {"wrong": 1})
    assert tools.call_tool(ctx, "read_attachment", {}).startswith(tools.UNTRUSTED_NOTE)


# ---------------------------------------------------------------- persistence
def test_store_roundtrip(tmp_path):
    store = RunStore(tmp_path)
    store.save_questions([{"task_id": "a", "question": "q"}])
    store.save_answer("a", {"answer": "1", "status": "ok"})
    store.save_answer("b", {"answer": "", "status": "error"})
    assert store.load_answers()["a"]["answer"] == "1" and store.load_questions()[0]["task_id"] == "a"
    store.save_trace("a", {"trace": []})
    assert store.load_trace("a") == {"trace": []}
    p = store.save_submission({"x": 1}, 200, {"score": 50})
    assert json.loads(p.read_text())["response"]["score"] == 50


# ---------------------------------------------------------------- agent loop with a fake model
class FakeLLM:
    """Scripted model: first asks for a tool, then answers using the tool result."""

    def __init__(self):
        self.calls = []

    def generate(self, model, contents, config, usage, deadline=None):
        self.calls.append(contents)
        usage.add(model, None)
        if len(self.calls) == 1:
            part = types.Part(function_call=types.FunctionCall(name="run_python", args={"code": "print(2**10)"}))
        else:
            last = contents[-1].parts[0].function_response.response["result"]
            assert "1024" in last
            part = types.Part.from_text(text="The result is 1024.\nFINAL ANSWER: 1024")
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))])


def test_agent_loop_tool_then_answer(tmp_path):
    agent = GaiaAgent(Settings(provider="gemini", runs_dir=tmp_path, max_steps=5), llm=FakeLLM())
    res = agent.run("What is 2 to the power 10?", None, log=lambda *a: None)
    assert res["answer"] == "1024" and res["status"] == "ok"
    assert [e["type"] for e in res["trace"]] == ["tool_call", "model_text"]
    assert res["usage"]["requests"] == 2


def test_agent_attaches_media_inline(tmp_path):
    img = tmp_path / "board.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    seen = {}

    class OneShot:
        def generate(self, model, contents, config, usage, deadline=None):
            seen["parts"] = contents[0].parts
            return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
                role="model", parts=[types.Part.from_text(text="FINAL ANSWER: Rd5")]))])

    res = GaiaAgent(Settings(provider="gemini", runs_dir=tmp_path), llm=OneShot()).run("Best move?", img, log=lambda *a: None)
    assert res["answer"] == "Rd5"
    assert any(p.inline_data and p.inline_data.mime_type == "image/png" for p in seen["parts"])


# ---------------------------------------------------------------- OpenAI provider (fake client, no network)
from types import SimpleNamespace

from gaia_agent.llm_openai import BudgetExceeded, OpenAILLM, SpendLedger


class _Item(SimpleNamespace):
    def model_dump(self, **kw):
        return {k: v for k, v in vars(self).items()}


class FakeOAI:
    """Scripted Responses API: a function call, then a final answer. Records requests."""

    def __init__(self):
        self.requests = []

    def respond(self, usage, deadline=None, **kw):
        self.requests.append(kw)
        usage.add_openai(kw["model"], None, 0.01, 0)
        if len(self.requests) == 1:
            out = [_Item(type="function_call", name="run_python", arguments='{"code": "print(7*6)"}', call_id="c1")]
            return SimpleNamespace(output=out, output_text="")
        outputs = [i for i in kw["input"] if isinstance(i, dict) and i.get("type") == "function_call_output"]
        assert outputs and "42" in outputs[0]["output"] and outputs[0]["call_id"] == "c1"
        return SimpleNamespace(output=[_Item(type="message")], output_text="Computed.\nFINAL ANSWER: 42")

    def transcribe(self, path, usage, deadline=None):
        return "please buy cornstarch and ripe strawberries"


def _oai_settings(tmp_path, **kw):
    return Settings(provider="openai", model="gpt-6.1-sol", runs_dir=tmp_path, **kw)


def test_openai_loop_tool_then_answer(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    fake = FakeOAI()
    res = GaiaAgent(_oai_settings(tmp_path, max_steps=5), llm=None, oai=fake).run("6*7?", None, log=lambda *a: None)
    assert res["answer"] == "42" and res["provider"] == "openai"
    assert res["usage"]["estimated_cost_usd"] == pytest.approx(0.02)
    tools_sent = fake.requests[0]["tools"]
    names = {t.get("name") for t in tools_sent}
    assert "run_python" in names and "google_search" not in names  # no Gemini key -> no google_search
    assert {"type": "web_search"} in tools_sent
    assert fake.requests[0]["store"] is False


def test_openai_audio_attachment_is_transcribed(tmp_path):
    mp3 = tmp_path / "a.mp3"
    mp3.write_bytes(b"ID3fake")

    class Once(FakeOAI):
        def respond(self, usage, deadline=None, **kw):
            self.requests.append(kw)
            return SimpleNamespace(output=[], output_text="FINAL ANSWER: cornstarch, ripe strawberries")

    fake = Once()
    res = GaiaAgent(_oai_settings(tmp_path), llm=None, oai=fake).run("Ingredients?", mp3, log=lambda *a: None)
    assert "cornstarch" in fake.requests[0]["input"][0]["content"][0]["text"]
    assert res["answer"] == "cornstarch, ripe strawberries"
    assert res["trace"][0]["type"] == "transcription"


def test_openai_image_attachment_inline(tmp_path):
    img = tmp_path / "b.png"
    img.write_bytes(b"\x89PNGfake")

    class Once(FakeOAI):
        def respond(self, usage, deadline=None, **kw):
            self.requests.append(kw)
            return SimpleNamespace(output=[], output_text="FINAL ANSWER: Rd5")

    fake = Once()
    GaiaAgent(_oai_settings(tmp_path), llm=None, oai=fake).run("Move?", img, log=lambda *a: None)
    parts = fake.requests[0]["input"][0]["content"]
    assert parts[1]["type"] == "input_image" and parts[1]["image_url"].startswith("data:image/png;base64,")


def test_spend_ledger_enforces_budget(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", budget_usd=0.05)
    ledger.check()
    ledger.add(0.03, "m")
    ledger.check()
    ledger.add(0.03, "m")
    with pytest.raises(BudgetExceeded):
        ledger.check()


def test_openai_cost_estimate(tmp_path):
    s = _oai_settings(tmp_path)
    llm = OpenAILLM.__new__(OpenAILLM)
    llm.settings = s
    u = SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000,
                        input_tokens_details=SimpleNamespace(cached_tokens=500_000))
    # 0.5M fresh * $2 + 0.5M cached * $0.10 + 0.1M out * $10
    assert llm._cost(u) == pytest.approx(1.0 + 0.05 + 1.0)


def test_unknown_openai_model_requires_prices(tmp_path, monkeypatch):
    for k in ("AGENT_PRICE_IN_PER_M", "AGENT_PRICE_OUT_PER_M"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError):
        Settings(provider="openai", model="some-new-model", runs_dir=tmp_path)


def test_gemini_helper_quota_is_not_fatal_on_openai(tmp_path, monkeypatch):
    from gaia_agent.llm import GeminiQuotaExhausted

    class Exhausted:
        calls = []

        def generate(self, model, *a, **k):
            Exhausted.calls.append(model)
            raise GeminiQuotaExhausted(f"{model}: PerDay")

    monkeypatch.setattr(tools, "_EXHAUSTED_GEMINI_MODELS", set())
    ctx = tools.ToolContext(settings=_oai_settings(tmp_path), llm=Exhausted(), usage=Usage())
    out = tools.call_tool(ctx, "google_search", {"query": "x"})
    assert "unavailable" in out
    assert Exhausted.calls == [ctx.settings.tool_model] + [m for m in ctx.settings.tool_model_fallbacks if m != ctx.settings.tool_model]
    gctx = tools.ToolContext(settings=Settings(provider="gemini", runs_dir=tmp_path), llm=Exhausted(), usage=Usage())
    monkeypatch.setattr(tools, "_EXHAUSTED_GEMINI_MODELS", set())
    with pytest.raises(GeminiQuotaExhausted):
        tools.call_tool(gctx, "google_search", {"query": "x"})


def test_gemini_helper_falls_back_to_next_model(tmp_path, monkeypatch):
    from gaia_agent.llm import GeminiQuotaExhausted

    class FirstExhausted:
        def generate(self, model, contents, cfg, usage, deadline=None):
            if model == "gemini-3.8-flash":
                raise GeminiQuotaExhausted("PerDay")
            return SimpleNamespace(text=f"answer from {model}", candidates=[])

    monkeypatch.setattr(tools, "_EXHAUSTED_GEMINI_MODELS", set())
    ctx = tools.ToolContext(settings=_oai_settings(tmp_path), llm=FirstExhausted(), usage=Usage())
    assert "answer from gemini-3.7-flash" in tools.google_search(ctx, "x")
