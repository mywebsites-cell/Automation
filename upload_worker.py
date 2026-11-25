"""Run one upload-only story publish in a detached process.

Usage:
    python upload_worker.py <story_id>

Writes its own status to cache/autopilot/uploads/<story_id>.json so the
Streamlit panel can show progress, and registers its real PID so the controller
can tell whether the task is still running (the venv python.exe is a redirector,
so we never trust the launcher PID).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from modules.story_uploader import (
    register_upload_pid,
    upload_story,
    _pid_path,
    _read_status,
    _write_status,
)
from modules.utils import (
    ensure_core_directories,
    get_project_paths,
    load_json,
    setup_logging,
)

BASE_DIR = Path(__file__).resolve().parent
PATHS = get_project_paths(BASE_DIR)
ensure_core_directories(PATHS)


def main() -> None:
    story_id = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not story_id:
        print("usage: python upload_worker.py <story_id>", file=sys.stderr)
        raise SystemExit(2)

    setup_logging(PATHS.logs_dir, f"upload_{story_id}")
    register_upload_pid(PATHS.cache_dir, story_id)

    settings = load_json(PATHS.config_dir / "settings.json")
    secrets_path = PATHS.config_dir / "secrets.json"
    secrets = load_json(secrets_path) if secrets_path.exists() else {"llm": {}, "image": {}}
    ap_path = PATHS.config_dir / "autopilot.json"
    ap = load_json(ap_path) if ap_path.exists() else {}

    try:
        summary = upload_story(PATHS, settings, secrets, ap, story_id)
        started_iso = _read_status(PATHS.cache_dir, story_id).get("started_iso")
        _write_status(
            PATHS.cache_dir,
            story_id,
            {
                "story_id": story_id,
                "state": "done",
                "videos": summary.get("videos", []),
                "errors": summary.get("errors", []),
                "started_iso": started_iso,
                "finished_iso": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print("UPLOAD_DONE", summary)
    except Exception as exc:
        started_iso = _read_status(PATHS.cache_dir, story_id).get("started_iso")
        _write_status(
            PATHS.cache_dir,
            story_id,
            {
                "story_id": story_id,
                "state": "error",
                "error": str(exc),
                "videos": [],
                "started_iso": started_iso,
                "finished_iso": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print("UPLOAD_ERROR", str(exc), file=sys.stderr)
        raise SystemExit(1)
    finally:
        _pid_path(PATHS.cache_dir, story_id).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
