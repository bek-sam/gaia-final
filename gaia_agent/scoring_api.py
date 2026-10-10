"""Client for the course scoring API (https://agents-course-unit4-scoring.hf.space/docs)."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import requests

MAX_ATTACHMENT_BYTES = 50 * 1024 * 1024
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,200}$")


class ScoringAPIError(RuntimeError):
    pass


def _get(url: str, timeout: int = 30, attempts: int = 3, **kw) -> requests.Response:
    last: Exception | None = None
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=timeout, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise ScoringAPIError(f"HTTP {r.status_code} from {url}")
            return r
        except (requests.RequestException, ScoringAPIError) as e:
            last = e
            time.sleep(2 * (i + 1))
    raise ScoringAPIError(f"GET {url} failed after {attempts} attempts: {last}")


def parse_questions(data: Any) -> list[dict]:
    """Validate the /questions payload and keep the documented fields."""
    if not isinstance(data, list) or not data:
        raise ScoringAPIError("Questions payload is empty or not a list")
    out = []
    for item in data:
        if not isinstance(item, dict) or not item.get("task_id") or not isinstance(item.get("question"), str):
            raise ScoringAPIError(f"Malformed question item: {str(item)[:200]}")
        out.append({
            "task_id": str(item["task_id"]),
            "question": item["question"],
            "Level": item.get("Level"),
            "file_name": item.get("file_name") or "",
        })
    return out


def fetch_questions(api_url: str) -> list[dict]:
    r = _get(f"{api_url}/questions")
    r.raise_for_status()
    return parse_questions(r.json())


def download_attachment(api_url: str, task_id: str, file_name: str, dest_dir: Path) -> Path:
    """Download a task attachment via GET /files/{task_id}. Returns the local path (cached)."""
    if not _SAFE_NAME.match(file_name) or file_name.startswith("."):
        raise ScoringAPIError(f"Refusing unsafe attachment name: {file_name!r}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = _get(f"{api_url}/files/{task_id}", timeout=60, stream=True)
    if r.status_code == 404:
        # The documented endpoint has been returning 404 for every attachment since late 2025
        # (https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate/discussions/155).
        r.close()
        return _download_from_gaia(file_name, dest)
    r.raise_for_status()
    size = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(64 * 1024):
            size += len(chunk)
            if size > MAX_ATTACHMENT_BYTES:
                f.close()
                tmp.unlink(missing_ok=True)
                raise ScoringAPIError(f"Attachment for {task_id} exceeds {MAX_ATTACHMENT_BYTES} bytes")
            f.write(chunk)
    tmp.replace(dest)
    return dest


def build_payload(username: str, agent_code: str, answers: dict[str, str], task_ids: list[str]) -> dict:
    """Build and validate a POST /submit body against the documented Submission schema."""
    username = (username or "").strip()
    if not username:
        raise ValueError("username must be non-empty")
    if len(agent_code or "") < 10:
        raise ValueError("agent_code must be at least 10 characters")
    if not re.match(r"^https://huggingface\.co/spaces/[^/]+/[^/]+/tree/main$", agent_code):
        raise ValueError(f"agent_code should be a public Space tree URL, got {agent_code!r}")
    items = []
    for tid in task_ids:
        if tid not in answers:
            continue
        ans = answers[tid]
        if not isinstance(ans, (str, int, float)) or isinstance(ans, bool):
            raise ValueError(f"answer for {tid} has unsupported type {type(ans).__name__}")
        items.append({"task_id": tid, "submitted_answer": ans})
    if not items:
        raise ValueError("no answers to submit")
    return {"username": username, "agent_code": agent_code, "answers": items}


def submit(api_url: str, payload: dict) -> tuple[int, Any]:
    """POST the payload once (no automatic retry, to avoid duplicate submissions)."""
    r = requests.post(f"{api_url}/submit", json=payload, timeout=120)
    try:
        body = r.json()
    except ValueError:
        body = {"raw": r.text[:2000]}
    return r.status_code, body


GAIA_REPO = "gaia-benchmark/GAIA"


def _download_from_gaia(file_name: str, dest: Path) -> Path:
    """Fallback: fetch exactly this attachment from the official GAIA validation files on the Hub.

    Only `2023/validation/<file_name>` is downloaded; the metadata file containing answers is never
    requested. Requires an HF login (`hf auth login`) and having accepted the dataset's terms at
    https://huggingface.co/datasets/gaia-benchmark/GAIA. Disable with ATTACHMENT_FALLBACK=off.
    """
    import os
    import shutil

    if os.getenv("ATTACHMENT_FALLBACK", "gaia").lower() in ("off", "0", "false", "no"):
        raise ScoringAPIError(f"/files returned 404 for {file_name} and the GAIA fallback is disabled")
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        cached = hf_hub_download(GAIA_REPO, f"2023/validation/{file_name}", repo_type="dataset")
    except GatedRepoError as e:
        raise ScoringAPIError(
            "Attachment endpoint returned 404; GAIA fallback needs access: log in with `hf auth login` and accept "
            f"the terms at https://huggingface.co/datasets/{GAIA_REPO} ({type(e).__name__})") from e
    except HfHubHTTPError as e:
        raise ScoringAPIError(f"GAIA fallback download failed for {file_name}: {e}") from e
    if Path(cached).stat().st_size > MAX_ATTACHMENT_BYTES:
        raise ScoringAPIError(f"Attachment {file_name} too large")
    shutil.copyfile(cached, dest)
    (dest.parent / (dest.name + ".source")).write_text(f"hf://datasets/{GAIA_REPO}/2023/validation/{file_name}\n")
    return dest
