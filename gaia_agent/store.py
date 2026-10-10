"""Local persistence of questions, answers and traces under runs/ (git-ignored).

Layout:
  runs/questions.json          cached /questions response
  runs/files/<file_name>       downloaded attachments
  runs/answers.json            {task_id: record}; the only source used for submission
  runs/traces/<task_id>.json   full reasoning/tool trace of the latest attempt (never submitted)
  runs/submissions/<ts>.json   payload sent and the server's actual response
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.files_dir = self.root / "files"
        self.traces_dir = self.root / "traces"
        self.submissions_dir = self.root / "submissions"
        self.questions_path = self.root / "questions.json"
        self.answers_path = self.root / "answers.json"

    # questions
    def save_questions(self, questions: list[dict]) -> None:
        _atomic_write_json(self.questions_path, questions)

    def load_questions(self) -> list[dict]:
        if not self.questions_path.exists():
            raise FileNotFoundError("No cached questions; run `python cli.py fetch` first")
        return json.loads(self.questions_path.read_text(encoding="utf-8"))

    # answers
    def load_answers(self) -> dict[str, dict]:
        if not self.answers_path.exists():
            return {}
        return json.loads(self.answers_path.read_text(encoding="utf-8"))

    def save_answer(self, task_id: str, record: dict) -> None:
        answers = self.load_answers()
        answers[task_id] = record
        _atomic_write_json(self.answers_path, answers)

    # traces
    def save_trace(self, task_id: str, trace: dict) -> Path:
        path = self.traces_dir / f"{task_id}.json"
        _atomic_write_json(path, trace)
        return path

    def load_trace(self, task_id: str) -> dict | None:
        path = self.traces_dir / f"{task_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    # submissions
    def save_submission(self, payload: dict, status: int, response: Any) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.submissions_dir / f"{ts}.json"
        _atomic_write_json(path, {"submitted_at": now_iso(), "payload": payload,
                                  "http_status": status, "response": response})
        return path
