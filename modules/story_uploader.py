"""Upload-only story publisher.

Lists every created story with its YouTube upload status and uploads a single
story's already-rendered videos — without touching the pipeline (no TTS, no
whisper, no rendering). It is meant to run as a *detached parallel task* so it
never disturbs a story that is currently being created by the autopilot.

The scheduler writes an "active story" marker while it is producing a story;
the UI and the upload worker both honour that marker so an upload never races
the scheduler on the same story.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.autopilot_controller import process_alive
from modules.llm_client import LLMClient
from modules.story_generator import StoryGenerator
from modules.utils import (
    ProjectPaths,
    ensure_story_output_structure,
    ffprobe_duration_seconds,
    load_json,
    write_json,
)
from modules.youtube_accounts import YouTubeAccount, YouTubeAccountManager
from modules.youtube_uploader import YouTubeUploader

logger = logging.getLogger(__name__)

_YT_CATEGORIES = {
    "Film & Animation": "1",
    "Autos & Vehicles": "2",
    "Music": "10",
    "Pets & Animals": "15",
    "Sports": "17",
    "Travel & Events": "19",
    "Gaming": "20",
    "People & Blogs": "22",
    "Comedy": "23",
    "Entertainment": "24",
    "News & Politics": "25",
    "Howto & Style": "26",
    "Education": "27",
    "Science & Technology": "28",
}


# ── active-story marker (written by the scheduler) ──────────────────────────
def active_story_path(cache_dir: Path) -> Path:
    return cache_dir / "autopilot" / "active_story.json"


def read_active_story(cache_dir: Path) -> str | None:
    p = active_story_path(cache_dir)
    if p.exists():
        try:
            data = load_json(p)
            return str(data.get("story_id") or "") or None
        except Exception:
            return None
    return None


def read_active_status(cache_dir: Path) -> dict[str, Any]:
    p = active_story_path(cache_dir)
    if p.exists():
        try:
            data = load_json(p)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def set_active_story(
    cache_dir: Path,
    story_id: str | None,
    stage: str = "Starting",
    part_index: int | None = None,
    part_count: int | None = None,
) -> None:
    p = active_story_path(cache_dir)
    if story_id is None:
        p.unlink(missing_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            p,
            {
                "story_id": story_id,
                "stage": stage,
                "part_index": part_index,
                "part_count": part_count,
                "iso": datetime.now().isoformat(timespec="seconds"),
            },
        )


def update_active_story(
    cache_dir: Path,
    stage: str,
    part_index: int | None = None,
    part_count: int | None = None,
) -> None:
    active = read_active_status(cache_dir)
    story_id = active.get("story_id")
    if story_id:
        set_active_story(cache_dir, str(story_id), stage, part_index, part_count)


# ── upload-task status files ────────────────────────────────────────────────
def _upload_dir(cache_dir: Path) -> Path:
    return cache_dir / "autopilot" / "uploads"


def _status_path(cache_dir: Path, story_id: str) -> Path:
    return _upload_dir(cache_dir) / f"{story_id}.json"


def _pid_path(cache_dir: Path, story_id: str) -> Path:
    return _upload_dir(cache_dir) / f"{story_id}.pid"


def _write_status(cache_dir: Path, story_id: str, payload: dict[str, Any]) -> None:
    _upload_dir(cache_dir).mkdir(parents=True, exist_ok=True)
    write_json(_status_path(cache_dir, story_id), payload)


def _read_status(cache_dir: Path, story_id: str) -> dict[str, Any]:
    p = _status_path(cache_dir, story_id)
    if p.exists():
        try:
            return load_json(p)
        except Exception:
            return {}
    return {}


def register_upload_pid(cache_dir: Path, story_id: str) -> None:
    """Record the worker's real PID (called by upload_worker at startup)."""
    _upload_dir(cache_dir).mkdir(parents=True, exist_ok=True)
    _pid_path(cache_dir, story_id).write_text(str(_current_pid()), encoding="utf-8")


def _current_pid() -> int:
    import os

    return os.getpid()


def upload_status(cache_dir: Path, story_id: str) -> dict[str, Any]:
    """Current state of a story's upload task (running / done / error / idle)."""
    pid = None
    pid_file = _pid_path(cache_dir, story_id)
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            pid = None
    status = _read_status(cache_dir, story_id)
    state = status.get("state", "idle")
    running = pid is not None and process_alive(pid)
    if state == "running" and not running:
        # Worker died without reporting back.
        state = "error"
        status = dict(status, state="error", error=status.get("error") or "Upload task terminated unexpectedly.")
    return {
        "running": running,
        "state": state,
        "videos": status.get("videos", []),
        "error": status.get("error"),
        "started_iso": status.get("started_iso"),
        "finished_iso": status.get("finished_iso"),
    }


def spawn_upload(base_dir: Path, cache_dir: Path, story_id: str) -> int | None:
    """Launch upload_worker.py detached for one story. Returns its PID or None.

    Idempotent: if an upload for this story is already running, returns its PID.
    """
    if upload_status(cache_dir, story_id)["running"]:
        pid_file = _pid_path(cache_dir, story_id)
        try:
            return int(pid_file.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    _write_status(
        cache_dir,
        story_id,
        {"story_id": story_id, "state": "running",
         "started_iso": datetime.now().isoformat(timespec="seconds"), "videos": []},
    )
    # Clear any stale pid file.
    _pid_path(cache_dir, story_id).unlink(missing_ok=True)

    worker = base_dir / "upload_worker.py"
    kwargs: dict = {
        "cwd": str(base_dir),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED | NEW_PROCESS_GROUP

    logger.info("Launching upload worker for %s", story_id)
    proc = subprocess.Popen([sys.executable, str(worker), story_id], **kwargs)

    # venv python.exe is a redirector; wait for the worker to self-register.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        pid_file = _pid_path(cache_dir, story_id)
        if pid_file.exists():
            try:
                return int(pid_file.read_text(encoding="utf-8").strip())
            except Exception:
                pass
        if proc.poll() is not None:
            return None
        time.sleep(0.2)
    return None


# ── story listing (for the UI) ──────────────────────────────────────────────
def _progress(paths: ProjectPaths, story_id: str) -> dict[str, Any]:
    p = paths.output_dir / story_id / "progress.json"
    if p.exists():
        try:
            return load_json(p)
        except Exception:
            return {}
    return {}


def _video_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return ffprobe_duration_seconds(path) > 0
    except Exception:
        return False


def _metadata_title(paths: ProjectPaths, story_id: str) -> str | None:
    p = paths.output_dir / story_id / "scripts" / "metadata.json"
    if p.exists():
        try:
            data = load_json(p)
            return str(data.get("story_title") or "") or None
        except Exception:
            return None
    return None


def _rendered_part_indices(story_output: Any) -> list[int]:
    final_dir = story_output.final
    indices: list[int] = []
    if final_dir.exists():
        for f in final_dir.glob("part*.mp4"):
            suffix = f.name[len("part"):-len(".mp4")]
            if suffix.isdigit() and _video_ok(f):
                indices.append(int(suffix))
    return sorted(indices)


def story_summary(paths: ProjectPaths, story_id: str) -> dict[str, Any]:
    """One dict describing a story's upload status for the UI."""
    prog = _progress(paths, story_id)
    parts = prog.get("parts", {})
    long_video = prog.get("long_video", {})

    part_statuses = []
    total = 0
    uploaded = 0
    for key in sorted(parts.keys(), key=lambda k: int(k) if k.isdigit() else 0):
        total += 1
        e = parts[key]
        ok = bool(e.get("uploaded"))
        if ok:
            uploaded += 1
        part_statuses.append({
            "index": int(key),
            "uploaded": ok,
            "url": e.get("url"),
            "failed": bool(e.get("failed")),
        })

    story_output = ensure_story_output_structure(paths, story_id)
    rendered_parts = _rendered_part_indices(story_output)
    full_ok = _video_ok(story_output.final / "full.mp4")
    long_uploaded = bool(long_video.get("uploaded"))

    return {
        "story_id": story_id,
        "title": prog.get("title") or _metadata_title(paths, story_id) or story_id,
        "topic": prog.get("topic") or "",
        "state": prog.get("state") or "unknown",
        "account": prog.get("account"),
        "parts": part_statuses,
        "parts_total": total,
        "parts_uploaded": uploaded,
        "long_uploaded": long_uploaded,
        "long_url": long_video.get("url"),
        "full_rendered": full_ok,
        "rendered_parts": rendered_parts,
        "updated_iso": prog.get("updated_iso"),
    }


def list_stories(paths: ProjectPaths) -> list[dict[str, Any]]:
    dirs = sorted([p for p in paths.output_dir.glob("story_*") if p.is_dir()], reverse=True)
    return [story_summary(paths, d.name) for d in dirs]


def delete_story(paths: ProjectPaths, story_id: str) -> None:
    """Delete a story and all related output and transient state."""
    if not story_id or not story_id.replace("_", "").isalnum():
        raise ValueError("Invalid story ID")

    story_dir = paths.output_dir / story_id
    if story_dir.parent.resolve() != paths.output_dir.resolve():
        raise ValueError("Invalid story directory")

    import shutil
    if story_dir.exists():
        shutil.rmtree(story_dir)

    narrative = paths.stories_dir / f"{story_id}.txt"
    narrative.unlink(missing_ok=True)

    upload_status_dir = paths.cache_dir / "autopilot" / "uploads"
    (upload_status_dir / f"{story_id}.json").unlink(missing_ok=True)
    (upload_status_dir / f"{story_id}.pid").unlink(missing_ok=True)


# ── the upload-only job ─────────────────────────────────────────────────────
def upload_story(
    paths: ProjectPaths,
    settings: dict[str, Any],
    secrets: dict[str, Any],
    ap: dict[str, Any],
    story_id: str,
) -> dict[str, Any]:
    """Upload one story's already-rendered videos to YouTube.

    Never renders and never touches another story. Returns a summary dict.
    """
    # Refuse to upload the story the scheduler is actively producing.
    if read_active_story(paths.cache_dir) == story_id:
        raise RuntimeError("This story is currently being created; upload skipped to avoid a conflict.")

    account_manager = YouTubeAccountManager(paths.cache_dir)
    accounts = account_manager.list_accounts()
    if not accounts:
        raise RuntimeError("No YouTube accounts configured.")

    uploaders: dict[str, YouTubeUploader] = {}
    for acc in accounts:
        try:
            u = account_manager.get_uploader(acc)
            u.authenticate()
            uploaders[acc.name] = u
        except Exception as exc:
            logger.warning("Could not authenticate account '%s': %s", acc.name, exc)

    if not uploaders:
        raise RuntimeError("Could not authenticate any YouTube account.")

    story_output = ensure_story_output_structure(paths, story_id)
    prog = _progress(paths, story_id)

    meta_path = story_output.scripts / "metadata.json"
    metadata = load_json(meta_path) if meta_path.exists() else {}
    story_title = prog.get("title") or metadata.get("story_title") or story_id
    topic = prog.get("topic") or metadata.get("topic") or ""

    narrative_path = paths.stories_dir / f"{story_id}.txt"
    narrative = narrative_path.read_text(encoding="utf-8").strip() if narrative_path.exists() else ""

    # Resolve account: recorded one first, then keyword match, then default.
    account: YouTubeAccount | None = None
    account_name = prog.get("account")
    if account_name:
        account = next((a for a in accounts if a.name == account_name), None)
    if account is None:
        account = account_manager.resolve_account(f"{story_title} {topic}", accounts)
    if account is None or account.name not in uploaders:
        raise RuntimeError("No uploadable YouTube account available for this story.")
    uploader = uploaders[account.name]

    # Optional LLM for long-video metadata (absent if no API key configured).
    story_generator = None
    if secrets.get("llm", {}).get("api_key", "").strip():
        try:
            story_generator = StoryGenerator(
                LLMClient(
                    provider=secrets["llm"]["provider"],
                    api_key=secrets["llm"]["api_key"].strip(),
                    model=secrets["llm"]["model"],
                    base_url=secrets["llm"].get("base_url", "").strip() or None,
                    temperature=float(settings.get("gemini", {}).get("temperature", 0.9)),
                )
            )
        except Exception:
            story_generator = None

    category_id = _YT_CATEGORIES.get(ap.get("youtube_category", "Entertainment"), "24")
    privacy = ap.get("youtube_privacy", "private")
    default_tags = [t.strip() for t in str(ap.get("youtube_tags", "")).split(",") if t.strip()]

    thumbnail_path = story_output.thumbnails / "cover.png"
    if not thumbnail_path.exists():
        thumbnail_path = None

    result: dict[str, Any] = {"story_id": story_id, "videos": [], "errors": []}
    prog_parts = dict(prog.get("parts", {}))

    # Parts come from metadata; fall back to rendered part files if absent.
    parts = metadata.get("parts", [])
    if not parts:
        parts = [{"part_index": idx} for idx in _rendered_part_indices(story_output)]

    for part_meta in parts:
        idx = int(part_meta.get("part_index"))
        key = str(idx)
        entry = prog_parts.get(key, {})
        if entry.get("uploaded"):
            if entry.get("url"):
                result["videos"].append(entry["url"])
            continue

        final_path = story_output.final / f"part{idx}.mp4"
        if not _video_ok(final_path):
            continue  # nothing rendered to upload

        title = part_meta.get("youtube_title") or f"{story_title} - Part {idx}"
        description = part_meta.get("youtube_description") or ""
        if not description:
            caption = part_meta.get("caption", "")
            hashtags = " ".join(part_meta.get("hashtags", []))
            description = f"{caption}\n\n{hashtags}".strip()
        tags = list(part_meta.get("youtube_tags", [])) or default_tags

        try:
            res = uploader.upload_video(
                video_path=final_path,
                title=title,
                description=description,
                tags=tags,
                category_id=category_id,
                privacy_status=privacy,
            )
            if thumbnail_path:
                uploader.set_thumbnail(res.video_id, thumbnail_path)
            logger.info("Uploaded %s part%d -> %s", story_id, idx, res.video_url)
            result["videos"].append(res.video_url)
            prog_parts[key] = {"uploaded": True, "url": res.video_url, "failed": False}
        except Exception as exc:
            logger.exception("Upload failed for %s part%d", story_id, idx)
            result["errors"].append(f"part{idx}: {exc}")

    # Long video (full story).
    long_entry = dict(prog.get("long_video", {}))
    if bool(ap.get("long_video_enabled", True)) and not long_entry.get("uploaded"):
        full_path = story_output.final / "full.mp4"
        if _video_ok(full_path):
            title = story_title
            description = f"{topic} — full story."
            tags = default_tags
            if story_generator is not None:
                try:
                    md = story_generator.generate_youtube_metadata(
                        story_title, topic, 0, "Full Story", narrative
                    )
                    title = md.title
                    description = md.description
                    tags = md.tags
                except Exception as exc:
                    logger.warning("Long-video metadata generation failed: %s", exc)
            try:
                res = uploader.upload_video(
                    video_path=full_path,
                    title=title,
                    description=description,
                    tags=tags,
                    category_id=category_id,
                    privacy_status=privacy,
                )
                if thumbnail_path:
                    uploader.set_thumbnail(res.video_id, thumbnail_path)
                logger.info("Uploaded %s full -> %s", story_id, res.video_url)
                result["videos"].append(res.video_url)
                long_entry = {"uploaded": True, "url": res.video_url, "failed": False}
            except Exception as exc:
                logger.exception("Upload failed for %s full", story_id)
                result["errors"].append(f"full: {exc}")

    # Merge back into progress.json (re-read to avoid clobbering a concurrent write).
    fresh = _progress(paths, story_id)
    fresh.setdefault("story_id", story_id)
    fresh["title"] = story_title
    fresh["topic"] = topic
    fresh["account"] = account.name
    if "channel" not in fresh:
        fresh["channel"] = account.channel_title
    fresh_parts = dict(fresh.get("parts", {}))
    fresh_parts.update(prog_parts)
    fresh["parts"] = fresh_parts
    fresh["long_video"] = long_entry

    parts_ok = bool(fresh_parts) and all(v.get("uploaded") for v in fresh_parts.values())
    long_ok = long_entry.get("uploaded", False)
    fresh["state"] = "completed" if (parts_ok and long_ok) else "in_progress"
    write_json(story_output.root / "progress.json", fresh)

    result["state"] = fresh["state"]
    result["account"] = account.name
    return result
