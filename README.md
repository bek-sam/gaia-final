---
title: GAIA Agent - Agents Course Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
---

# GAIA agent for the Hugging Face Agents Course (Unit 4)

A tool-calling research agent that answers the 20 GAIA level-1 questions served by the course scoring API
(<https://agents-course-unit4-scoring.hf.space/docs>). It runs locally (CLI) and the same code is published
as this Space so the leaderboard can link to it.

## Architecture

- **Model**: OpenAI via the Responses API (default `gpt-6.1-sol`, with OpenAI's hosted web search) when
  `OPENAI_API_KEY` is set; otherwise Google Gemini via `google-genai` (default `gemini-3.8-flash`, free tier).
  If a `GEMINI_API_KEY` is also present, Gemini is used as a free helper inside two tools: Google-grounded
  search and watching YouTube videos (OpenAI models cannot take video). Audio is transcribed with
  `gpt-4o-transcribe` on OpenAI.
- **Loop** (`gaia_agent/agent.py`): a plain function-calling loop. The model reasons, calls tools and must finish
  with `FINAL ANSWER: ...`. Only the text after that marker (stripped of wrappers like quotes, `**`, or a lone
  trailing period) is submitted; case, inner punctuation, units and ordering are preserved. Bounded by
  `AGENT_MAX_STEPS` (16) and `AGENT_QUESTION_TIMEOUT` (600 s); the last step forces an answer without tools.
- **Tools** (`gaia_agent/tools.py`):
  `google_search` (Gemini Google-Search grounding, sourced), `web_search` (DuckDuckGo/Bing via `ddgs`),
  `fetch_webpage` (HTML/PDF → Markdown with tables, `find`/`start` paging), `wikipedia_search`,
  `wikipedia_page` (rendered article incl. tables, `as_of` date → historical revision), `run_python`
  (sandboxed subprocess), `read_attachment` (xlsx/csv/pdf/docx/code/text), `analyze_media` (image/audio/video
  file or YouTube URL via Gemini), `youtube_transcript`.
- **Attachments**: downloaded via `GET /files/{task_id}`. That endpoint currently returns 404 for every file
  ([discussion #155](https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate/discussions/155)), so the
  client falls back to downloading **only that exact file** (`2023/validation/<file_name>`) from the gated
  official dataset `gaia-benchmark/GAIA`. The metadata file with answers is never requested. Images and audio are
  passed to the model inline; other files through `read_attachment` / `run_python`.
- **Reliability** (`gaia_agent/llm.py`, `gaia_agent/llm_openai.py`, `cli.py`): request pacing for free-tier rate
  limits, bounded retries honouring the server's `retryDelay`, quota/balance detection (stops cleanly, resume
  later), per-question failure isolation, token usage and estimated cost recorded per question.
- **Spend cap**: every OpenAI call is priced (token prices from the official pricing page, hosted web search
  $0.01/call, transcription $0.006/min) and added to `runs/spend.json`; no request is sent once the cumulative
  estimate reaches `AGENT_BUDGET_USD` (default $5).
- **Safety**: web pages, files and transcripts are treated as untrusted data (labelled as such, and the prompt
  forbids following instructions in them). Fetching is http(s) only, blocks private/loopback addresses on every
  redirect hop, and caps size/time. `run_python` runs with a 30 s timeout, an environment without API keys, a
  throwaway working directory and, on macOS, a `sandbox-exec` profile denying network, writes outside the temp
  dir, and reads of `.env`/HF tokens/SSH keys.

## Result

Submitted 2026-10-10 as `bekbolsun446` with `gpt-6.1-sol` (OpenAI Responses API, medium reasoning) plus the
free Gemini helper for YouTube: the scoring API returned **score 100.0 (20/20 correct)**. That run made 48
model calls, about 435k input tokens and an estimated $0.78 in OpenAI usage. Web content changes over time,
so a rerun can score differently.

## Setup (macOS, local)

```bash
cd Final_Assignment_Template
python3 -m venv .venv                     # already present in this checkout
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env && chmod 600 .env    # then edit .env: OPENAI_API_KEY=... and/or GEMINI_API_KEY=...
.venv/bin/hf auth login                   # write token; used for the attachment fallback, publishing, and your username
```

- OpenAI key: <https://platform.openai.com/api-keys> (uses your prepaid API balance; capped by `AGENT_BUDGET_USD`).
- Gemini key (optional with OpenAI, required without): <https://aistudio.google.com/apikey> (free tier).
- Attachment fallback: once, open <https://huggingface.co/datasets/gaia-benchmark/GAIA> while logged in and
  accept the terms (auto-approved).
- Optional settings in `.env`: `AGENT_PROVIDER` (`openai`/`gemini`), `AGENT_MODEL`, `AGENT_REASONING_EFFORT`
  (default `medium`), `AGENT_OPENAI_WEB_SEARCH=0`, `AGENT_BUDGET_USD`, `AGENT_TOOL_MODEL` (Gemini helper),
  `AGENT_MIN_INTERVAL` (seconds between Gemini calls), `AGENT_MAX_STEPS`, `AGENT_QUESTION_TIMEOUT`,
  `AGENT_PRICE_IN_PER_M`/`AGENT_PRICE_CACHED_IN_PER_M`/`AGENT_PRICE_OUT_PER_M` (required for models not in the
  built-in price table), `HF_USERNAME`, `ATTACHMENT_FALLBACK=off`.

## Commands

```bash
.venv/bin/python -m pytest -q tests          # offline checks (API parsing, payload, tools, sandbox, loop)
.venv/bin/python cli.py fetch                 # questions -> runs/questions.json, attachments -> runs/files/
.venv/bin/python cli.py models                # models available to your key (current provider)
.venv/bin/python cli.py smoke                 # 3 representative questions end-to-end
.venv/bin/python cli.py run                   # all questions; skips ones already answered (resume)
.venv/bin/python cli.py run --only cca5 99c9 --force   # rerun specific tasks by task_id prefix
.venv/bin/python cli.py review                # answers table + usage totals
.venv/bin/python cli.py review --task 8e86    # full reasoning/tool trace of one task
.venv/bin/python cli.py validate --space USER/SPACE      # build + validate the payload (no network)
.venv/bin/python cli.py submit --space USER/SPACE --confirm  # submit saved answers; does not rerun the agent
```

Everything generated lives in `runs/` (git-ignored): `answers.json` (the only input to submission),
`traces/<task_id>.json` (reasoning and tool calls, never submitted), `submissions/<timestamp>.json`
(exact payload and the server's real response).

## Publishing

The course asks for a public link to the agent's code on a Hugging Face Space
(`https://huggingface.co/spaces/<user>/<space>/tree/main`). Hugging Face now requires a PRO subscription for
Gradio/Docker Spaces on free CPU hardware, while static Spaces are free, so the code is published as a **static
Space** (`index.html` is just a landing page; the agent runs locally). `app.py` still provides the template's
Gradio login-and-submit UI for local use (`.venv/bin/python app.py`) or for a Gradio Space if you have PRO. Only source files are uploaded; `.env`, `.venv`,
`runs/` and caches are excluded.

## Limitations

- Gemini free-tier quotas (requests per minute/day) can stop a run midway; rerun `cli.py run` later to resume.
- Cost figures are estimates from token counts and list prices; your OpenAI usage dashboard is authoritative.
- Without a Gemini key, YouTube questions rely on transcripts and web search only (no visual analysis).
- Answers are exact-match graded; the agent can still be wrong on hard retrieval, video counting or vision
  (e.g. chess positions).
- The `sandbox-exec` restrictions apply on macOS only; elsewhere `run_python` relies on timeout, scrubbed
  environment and a temp directory.
- Web results change over time, so scores are not perfectly reproducible.
