"""Autopilot scheduler: continuously generates stories and uploads them to the
correct YouTube account(s) without any human intervention.

Run:
    python scheduler.py              # run forever
    python scheduler.py --once       # produce + publish exactly one story, then exit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

from modules.autopilot import AutopilotConfig, AutopilotEngine, load_secrets
from modules.autopilot_controller import process_alive, read_pid, register
from modules.story_uploader import set_active_story
from modules.utils import (
    ensure_core_directories,
    get_project_paths,
    has_ffmpeg_with_local_fallback,
    load_json,
    read_autopilot_control,
    setup_logging,
    write_json,
)
from modules.youtube_accounts import YouTubeAccount

BASE_DIR = Path(__file__).resolve().parent
PATHS = get_project_paths(BASE_DIR)
ensure_core_directories(PATHS)

AUTOPILOT_CONFIG_PATH = PATHS.config_dir / "autopilot.json"
STATE_DIR = PATHS.cache_dir / "autopilot"
STATE_PATH = STATE_DIR / "state.json"
USED_TOPICS_PATH = STATE_DIR / "used_topics.json"


def _default_autopilot_config() -> dict:
    return {
        "topics": [
            "Scary historical shipwrecks",
            "True crime cold cases",
            "Creepy abandoned places",
            "Unexplained disappearances",
            "Dark folklore legends",
        ],
        "seed_topics": [
            "Scary historical shipwrecks",
            "True crime cold cases",
            "Creepy abandoned places",
        ],
        "auto_generate_topics": True,
        "interval_minutes": 60,
        "jitter_minutes": 15,
        "max_per_day": 4,
        "quiet_hours_start": 1,
        "quiet_hours_end": 5,
        "shorts_per_story": 5,
        "total_story_minutes": 10,
        "voice": "af_heart",
        "auto_voice": True,
        "quality": "Balanced",
        "enable_subtitles": True,
        "subtitle_position": "Bottom",
        "subtitle_font_size": 16,
        "background_category": "minecraft",
        "background_video": "",
        "long_video_enabled": True,
        "enable_thumbnail": False,
        "enable_youtube": True,
        "youtube_category": "Entertainment",
        "youtube_privacy": "private",
        "youtube_tags": "shorts, storytime, viral, fyp",
    }


def load_autopilot_config() -> dict:
    cfg = _default_autopilot_config()
    if AUTOPILOT_CONFIG_PATH.exists():
        try:
            saved = load_json(AUTOPILOT_CONFIG_PATH)
        except Exception:
            saved = {}
        if isinstance(saved, dict):
            cfg.update(saved)
    return cfg


def build_engine(ap: dict, settings: dict, secrets: dict) -> AutopilotEngine:
    cfg = AutopilotConfig(
        llm_provider=secrets["llm"]["provider"],
        llm_api_key=secrets["llm"]["api_key"],
        llm_model=secrets["llm"]["model"],
        llm_base_url=secrets["llm"].get("base_url", ""),
        temperature=float(settings.get("gemini", {}).get("temperature", 0.9)),
        enable_thumbnail=bool(ap.get("enable_thumbnail")),
        image_provider=secrets["image"]["provider"],
        image_api_key=secrets["image"]["api_key"],
        image_model=secrets["image"]["model"],
        image_base_url=secrets["image"].get("base_url", ""),
        image_size=secrets["image"].get("size", "1024x1792"),
        shorts_per_story=int(ap.get("shorts_per_story", 5)),
        total_story_minutes=int(ap.get("total_story_minutes", 10)),
        voice=ap.get("voice", "af_heart"),
        auto_voice=bool(ap.get("auto_voice", True)),
        quality=ap.get("quality", "Balanced"),
        enable_subtitles=bool(ap.get("enable_subtitles", True)),
        subtitle_position=ap.get("subtitle_position", "Bottom"),
        subtitle_font_size=int(ap.get("subtitle_font_size", 16)),
        background_category=ap.get("background_category", "minecraft"),
        background_video=ap.get("background_video", ""),
        long_video_enabled=bool(ap.get("long_video_enabled", True)),
        enable_youtube=bool(ap.get("enable_youtube", True)),
        youtube_category=ap.get("youtube_category", "Entertainment"),
        youtube_privacy=ap.get("youtube_privacy", "private"),
        youtube_tags=ap.get("youtube_tags", "shorts, storytime, viral, fyp"),
    )
    return AutopilotEngine(cfg, PATHS, settings)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return load_json(STATE_PATH)
        except Exception:
            pass
    return {"history": [], "last_run_iso": None}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(STATE_PATH, state)


def load_used_topics() -> list[str]:
    if USED_TOPICS_PATH.exists():
        try:
            return load_json(USED_TOPICS_PATH)
        except Exception:
            pass
    return []


def save_used_topics(topics: list[str]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(USED_TOPICS_PATH, topics)


def _already_used_today(state: dict) -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    return sum(1 for h in state["history"] if h.get("date") == today)


def _cycle_accounts(engine: AutopilotEngine) -> list[YouTubeAccount]:
    """Order accounts so niche accounts (with keywords) go first, then the catch-all."""
    niche = [a for a in engine.accounts if a.keywords]
    catchall = [a for a in engine.accounts if not a.keywords]
    return niche + catchall


def _topic_for_account(engine: AutopilotEngine, account: YouTubeAccount, ap: dict, used: list[str]) -> str:
    """Pick/generate a topic that belongs to this account's niche."""
    static = list(ap.get("topics", []))

    if account.keywords:
        # Niche account: ONLY topics matching its keywords (never unrelated ones).
        matching = [t for t in static if any(k.lower() in t.lower() for k in account.keywords)]
        pool = matching
    else:
        # Catch-all account: anything in the pool.
        pool = static

    available = [t for t in pool if t.strip() and t not in used]
    if available:
        return random.choice(available)

    # Refill via the LLM, seeded with this account's niche.
    seed = account.keywords if account.keywords else list(ap.get("seed_topics", []))
    if ap.get("auto_generate_topics", True):
        try:
            fresh = engine.generate_topics(1, seed)
            if fresh:
                return fresh[0]
        except Exception as exc:
            logger.warning("Topic generation failed for '%s': %s", account.name, exc)

    # Last resort: use the keywords themselves as a topic (or a generic one).
    if account.keywords:
        return " ".join(account.keywords) + " story"
    return random.choice(static or ["viral short-form story"])


def _in_quiet_hours(ap: dict) -> bool:
    now = datetime.now()
    start = int(ap.get("quiet_hours_start", 1))
    end = int(ap.get("quiet_hours_end", 5))
    if start == end:
        return False
    if start < end:
        return start <= now.hour < end
    return now.hour >= start or now.hour < end  # wraps midnight


def _wait_until_allowed(ap: dict, state: dict) -> None:
    interval = max(1, int(ap.get("interval_minutes", 60)))
    jitter = int(ap.get("jitter_minutes", 15))
    delay = interval * 60 + random.randint(0, jitter * 60)
    # Never wait through the quiet hours window.
    while _in_quiet_hours(ap):
        logger.info("In quiet hours (%s-%s). Sleeping until clear…",
                    ap.get("quiet_hours_start"), ap.get("quiet_hours_end"))
        time.sleep(60)
    logger.info("Waiting %d minutes before the next story…", round(delay / 60))
    time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Autopilot story scheduler")
    parser.add_argument("--once", action="store_true", help="run a single story then exit")
    args = parser.parse_args()

    log_path = setup_logging(PATHS.logs_dir, "autopilot")
    global logger
    logger = logging.getLogger("autopilot")

    # Prevent a second instance from uploading the same stories twice.
    existing = read_pid(PATHS.cache_dir)
    if existing is not None and existing != os.getpid() and process_alive(existing):
        logger.error("Another autopilot instance is already running (PID %s). Exiting.", existing)
        raise SystemExit(1)
    register(PATHS.cache_dir)

    settings = load_json(PATHS.config_dir / "settings.json")
    secrets = load_secrets(PATHS)
    ap = load_autopilot_config()

    if not has_ffmpeg_with_local_fallback(PATHS.base):
        logger.error("FFmpeg is not available. Install it before running autopilot.")
        raise SystemExit(1)

    if ap.get("enable_youtube") and not secrets["llm"]["api_key"]:
        logger.error("No LLM API key configured (config/secrets.json). Set it in the Streamlit sidebar first.")
        raise SystemExit(1)

    engine = build_engine(ap, settings, secrets)
    state = load_state()
    used = load_used_topics()
    topics_pool = list(ap.get("topics", []))

    target_accounts = _cycle_accounts(engine)
    if not target_accounts and ap.get("enable_youtube"):
        logger.error("No YouTube accounts configured. Connect at least one first.")
        raise SystemExit(1)

    logger.info(
        "Autopilot ready. %d account(s), %d topics in pool.",
        len(engine.accounts), len(topics_pool),
    )

    account_index = 0
    while True:
        # ── Pause / resume via the Streamlit toggle ─────────────────────────
        control = read_autopilot_control(PATHS.cache_dir)
        if not control.get("enabled", True):
            logger.info("Autopilot is PAUSED (control file). Checking again in 30s…")
            time.sleep(30)
            continue

        max_per_day = int(ap.get("max_per_day", 4))
        if _already_used_today(state) >= max_per_day:
            logger.info("Daily cap of %d reached. Exiting for today.", max_per_day)
            if args.once:
                break
            # Sleep a couple hours then re-check (new day may have started).
            time.sleep(7200)
            continue

        # ── First: resume any interrupted story from a previous run ─────────
        try:
            resumed = engine.resume_pending_story()
        except Exception as exc:
            logger.exception("Resume failed: %s", exc)
            resumed = None
        finally:
            set_active_story(PATHS.cache_dir, None)
        if resumed is not None:
            state["history"].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "iso": datetime.now().isoformat(timespec="seconds"),
                **resumed,
            })
            state["last_run_iso"] = datetime.now().isoformat(timespec="seconds")
            save_state(state)
            logger.info("Resumed story. Videos: %s", resumed["videos"])
            if args.once:
                break
            _wait_until_allowed(ap, state)
            continue

        # Never start a brand-new story while another is still in progress.
        if engine.has_pending_story():
            logger.info("A story is still in progress; waiting before creating a new one.")
            if args.once:
                time.sleep(30)
                continue
            _wait_until_allowed(ap, state)
            continue

        # ── Pick the next account (round-robin across niches) ───────────────
        if target_accounts:
            account = target_accounts[account_index % len(target_accounts)]
            account_index += 1
        else:
            account = None

        topic = _topic_for_account(engine, account, ap, used) if account else random.choice(topics_pool or ["viral story"])
        used.append(topic)
        save_used_topics(used)

        try:
            summary = engine.run_story(topic, force_account=account)
            state["history"].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "iso": datetime.now().isoformat(timespec="seconds"),
                **summary,
            })
            state["last_run_iso"] = datetime.now().isoformat(timespec="seconds")
            save_state(state)
            logger.info("Story published. Videos: %s", summary["videos"])
        except Exception as exc:
            logger.exception("Story failed: %s", exc)
            state["history"].append({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "iso": datetime.now().isoformat(timespec="seconds"),
                "error": str(exc),
                "topic": topic,
                "account": account.name if account else None,
            })
            save_state(state)
        finally:
            set_active_story(PATHS.cache_dir, None)

        if args.once:
            break

        _wait_until_allowed(ap, state)


if __name__ == "__main__":
    main()
