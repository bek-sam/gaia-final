#!/usr/bin/env python
"""Local evaluation runner for the Agents Course Unit 4 final assignment.

  python cli.py fetch                 # download questions + attachments to runs/
  python cli.py models                # list Gemini models available to your key
  python cli.py smoke                 # run the agent on 3 representative questions
  python cli.py run                   # run all unanswered questions (resumable)
  python cli.py run --only 8e86 cca5 --force   # rerun specific tasks (task_id prefixes)
  python cli.py review [--task ID]    # table of answers / full trace of one task
  python cli.py validate --space USER/SPACE    # build + validate the payload, no network
  python cli.py submit --space USER/SPACE --confirm  # submit saved answers (does not rerun the agent)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from gaia_agent.config import Settings
from gaia_agent.scoring_api import (ScoringAPIError, build_payload, download_attachment, fetch_questions, submit)
from gaia_agent.store import RunStore, now_iso

# Smoke-test set: one plain web question, one attachment (spreadsheet), one text-trick question.
SMOKE_PREFIXES = ["2d83110e", "7bd855d8", "8e867cd7"]


def _select(questions: list[dict], only: list[str] | None, limit: int | None) -> list[dict]:
    if only:
        questions = [q for q in questions if any(q["task_id"].startswith(p) for p in only)]
    return questions[:limit] if limit else questions


def _attachment(settings: Settings, store: RunStore, q: dict):
    if not q.get("file_name"):
        return None
    return download_attachment(settings.api_url, q["task_id"], q["file_name"], store.files_dir / q["task_id"])


def cmd_fetch(args, settings: Settings, store: RunStore) -> int:
    questions = fetch_questions(settings.api_url)
    store.save_questions(questions)
    print(f"Saved {len(questions)} questions to {store.questions_path}")
    failed = 0
    for q in questions:
        if q["file_name"]:
            try:
                path = _attachment(settings, store, q)
                print(f"  {q['task_id'][:8]} attachment -> {path} ({path.stat().st_size} bytes)")
            except ScoringAPIError as e:
                failed += 1
                print(f"  {q['task_id'][:8]} attachment FAILED: {e}")
    return 1 if failed else 0


def cmd_models(args, settings: Settings, store: RunStore) -> int:
    from gaia_agent.llm import Gemini

    if settings.provider == "openai":
        from gaia_agent.llm_openai import OpenAILLM

        ids = sorted(m.id for m in OpenAILLM(settings).client.models.list())
        print("\n".join(i for i in ids if i.startswith(("gpt", "o"))))
        return 0

    llm = Gemini(settings)
    for m in llm.client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            print(f"{m.name.removeprefix('models/'):45s} in={m.input_token_limit} out={m.output_token_limit}")
    return 0


def cmd_run(args, settings: Settings, store: RunStore, only=None) -> int:
    from gaia_agent.agent import GaiaAgent
    from gaia_agent.llm import QuotaExhausted

    if args.model:
        settings = Settings(model=args.model)
    questions = _select(store.load_questions(), only or args.only, args.limit)
    agent = GaiaAgent(settings)
    answers = store.load_answers()
    totals = {"ok": 0, "skipped": 0, "error": 0, "no_answer": 0}
    for i, q in enumerate(questions, 1):
        tid = q["task_id"]
        prev = answers.get(tid)
        if prev and prev.get("status") == "ok" and not args.force:
            totals["skipped"] += 1
            continue
        print(f"\n[{i}/{len(questions)}] {tid} {q['question'][:90]!r}")
        record = {"task_id": tid, "question": q["question"], "file_name": q.get("file_name", ""), "updated_at": now_iso()}
        try:
            path = _attachment(settings, store, q)
            result = agent.run(q["question"], path)
            trace_path = store.save_trace(tid, {"task_id": tid, "question": q["question"], **result})
            record.update({k: result[k] for k in ("answer", "status", "steps", "seconds", "model", "usage")})
            record["trace_file"] = str(trace_path.relative_to(store.root))
            print(f"  -> {record['status']}: {record['answer']!r}  ({result['steps']} steps, {result['seconds']}s, "
                  f"{result['usage']['requests']} model calls, ~${result['usage'].get('estimated_cost_usd', 0):.4f})")
        except QuotaExhausted as e:
            record.update({"status": "error", "error": f"QuotaExhausted: {str(e)[:300]}", "answer": ""})
            store.save_answer(tid, record)
            print(f"  !! Model quota exhausted; stopping. Resume later with the same command.\n  {str(e)[:300]}")
            totals["error"] += 1
            break
        except (Exception, KeyboardInterrupt) as e:
            record.update({"status": "error", "error": f"{type(e).__name__}: {str(e)[:500]}", "answer": "",
                           "traceback": traceback.format_exc()[-3000:]})
            print(f"  !! infrastructure error: {record['error']}")
            if isinstance(e, KeyboardInterrupt):
                store.save_answer(tid, record)
                raise
        store.save_answer(tid, record)
        totals[record["status"]] = totals.get(record["status"], 0) + 1
    spend = settings.runs_dir / "spend.json"
    if spend.exists():
        print(f"Cumulative estimated OpenAI spend: ${json.loads(spend.read_text())['total_usd']:.4f} "
              f"(budget ${settings.budget_usd:.2f})")
    print(f"\nDone: {totals}. Review with `python cli.py review`.")
    return 0


def cmd_smoke(args, settings: Settings, store: RunStore) -> int:
    args.limit = None
    return cmd_run(args, settings, store, only=args.only or SMOKE_PREFIXES)


def cmd_review(args, settings: Settings, store: RunStore) -> int:
    answers = store.load_answers()
    if args.task:
        tid = next((t for t in answers if t.startswith(args.task)), args.task)
        trace = store.load_trace(tid)
        if not trace:
            print("No trace for", args.task)
            return 1
        print(f"Q: {trace['question']}\n")
        for ev in trace["trace"]:
            if ev["type"] == "model_text":
                print(f"[{ev['step']}] MODEL: {ev['text'][:3000]}\n")
            elif ev["type"] == "tool_call":
                print(f"[{ev['step']}] TOOL {ev['tool']}({json.dumps(ev['args'], ensure_ascii=False)[:400]}) "
                      f"{ev['seconds']}s\n    -> {ev['result'][:args.chars]}\n")
            else:
                print(f"[{ev['step']}] {ev}")
        print(f"ANSWER: {trace['answer']!r}  status={trace['status']}  usage={trace['usage']}")
        return 0
    questions = {q["task_id"]: q for q in store.load_questions()}
    tot_req = tot_in = tot_out = 0
    tot_cost = 0.0
    print(f"{'task':8s}  {'status':9s} {'file':5s} {'steps':>5s} {'calls':>5s}  answer")
    for tid, q in questions.items():
        r = answers.get(tid, {})
        u = r.get("usage") or {}
        tot_req += u.get("requests", 0)
        tot_in += u.get("input_tokens", 0)
        tot_out += u.get("output_tokens", 0) + u.get("thinking_tokens", 0)
        tot_cost += u.get("estimated_cost_usd", 0) or 0
        ext = (q.get("file_name") or "").rsplit(".", 1)[-1] if q.get("file_name") else ""
        shown = r.get("answer", "") if r.get("status") != "error" else r.get("error", "")[:70]
        print(f"{tid[:8]}  {r.get('status', '-'):9s} {ext:5s} {r.get('steps', ''):>5} {u.get('requests', ''):>5}  {shown}")
    n_ok = sum(1 for r in answers.values() if r.get("status") == "ok")
    print(f"\n{n_ok}/{len(questions)} answered. Model calls: {tot_req}, input tokens: {tot_in}, output+thinking tokens: {tot_out}, "
          f"est. cost of latest attempts: ${tot_cost:.4f}")
    return 0


def _username(args) -> str:
    if args.username:
        return args.username
    if os.getenv("HF_USERNAME"):
        return os.environ["HF_USERNAME"]
    from huggingface_hub import whoami

    return whoami()["name"]


def _payload(args, store: RunStore) -> dict:
    space = args.space or os.getenv("AGENT_SPACE")
    if not space or space.count("/") != 1:
        raise SystemExit("Pass --space USER/SPACE_NAME (the public Space hosting this code)")
    agent_code = f"https://huggingface.co/spaces/{space}/tree/main"
    answers = {tid: r["answer"] for tid, r in store.load_answers().items() if r.get("status") == "ok" and r.get("answer")}
    task_ids = [q["task_id"] for q in store.load_questions()]
    return build_payload(_username(args), agent_code, answers, task_ids)


def cmd_validate(args, settings: Settings, store: RunStore) -> int:
    payload = _payload(args, store)
    missing = len(store.load_questions()) - len(payload["answers"])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nPayload valid: {len(payload['answers'])} answers ({missing} questions without an answer).")
    return 0


def cmd_submit(args, settings: Settings, store: RunStore) -> int:
    payload = _payload(args, store)
    print(f"Submitting {len(payload['answers'])} answers as {payload['username']!r} with agent_code {payload['agent_code']}")
    if not args.confirm:
        print("Dry run. Re-run with --confirm to actually submit.")
        return 0
    status, body = submit(settings.api_url, payload)
    path = store.save_submission(payload, status, body)
    print(f"HTTP {status}\n{json.dumps(body, indent=2, ensure_ascii=False)}\nSaved to {path}")
    return 0 if status == 200 else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("models")
    for name in ("run", "smoke"):
        sp = sub.add_parser(name)
        sp.add_argument("--only", nargs="*", help="task_id prefixes")
        sp.add_argument("--limit", type=int)
        sp.add_argument("--force", action="store_true", help="rerun even if an answer exists")
        sp.add_argument("--model", help="override AGENT_MODEL")
    rp = sub.add_parser("review")
    rp.add_argument("--task", help="task_id prefix to show the full trace")
    rp.add_argument("--chars", type=int, default=800, help="tool result chars to show")
    for name in ("validate", "submit"):
        sp = sub.add_parser(name)
        sp.add_argument("--space", help="USER/SPACE_NAME of the public Space with this code")
        sp.add_argument("--username", help="HF username (default: HF_USERNAME or `hf auth whoami`)")
        if name == "submit":
            sp.add_argument("--confirm", action="store_true")
    args = p.parse_args(argv)
    settings = Settings()
    store = RunStore(settings.runs_dir)
    handler = {"fetch": cmd_fetch, "models": cmd_models, "run": cmd_run, "smoke": cmd_smoke,
               "review": cmd_review, "validate": cmd_validate, "submit": cmd_submit}[args.cmd]
    try:
        return handler(args, settings, store)
    except (ScoringAPIError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
