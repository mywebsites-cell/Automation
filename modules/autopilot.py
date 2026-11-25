"""Headless autopilot pipeline.

Produces one complete story (shorts + optional long video + thumbnail) and
publishes it to the correct YouTube account — with no Streamlit dependency, so
it can run inside a long-lived scheduler/daemon.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from modules.background_manager import BackgroundManager
from modules.cleanup_manager import force_memory_cleanup
from modules.ffmpeg_renderer import FFmpegRenderer, RenderConfig
from modules.llm_client import ImageGenerator, LLMClient
from modules.metadata_generator import generate_part_metadata
from modules.story_generator import StoryGenerator
from modules.story_splitter import StoryPart, split_story_into_parts
from modules.subtitle_engine import SubtitleEngine
from modules.tts_engine import KokoroTTSEngine
from modules.utils import (
    ProjectPaths,
    ensure_story_output_structure,
    ffprobe_duration_seconds,
    load_json,
    next_story_id,
    write_json,
)
from modules.video_engine import RenderedPartArtifact, VideoPipelineEngine
from modules.youtube_accounts import YouTubeAccount, YouTubeAccountManager
from modules.youtube_uploader import YouTubeUploadResult, YouTubeUploader
from modules.story_uploader import set_active_story

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

_SUBTITLE_POS = {
    "Top": (8, 60),
    "Middle": (5, 0),
    "Bottom": (2, 60),
}

# Narrator profile (gender, tone) -> Kokoro voice id. Kept in code so the LLM
# can never select an invalid voice; every combination maps to a real voice.
_VOICE_PROFILES: dict[str, dict[str, str]] = {
    "female": {
        "deep": "af_heart",
        "warm": "af_heart",
        "bright": "af_sky",
        "clear": "af_sarah",
        "calm": "af_sarah",
    },
    "male": {
        "deep": "am_michael",
        "warm": "am_adam",
        "bright": "am_adam",
        "clear": "am_adam",
        "calm": "am_adam",
    },
}


@dataclass
class AutopilotConfig:
    # LLM
    llm_provider: str = "gemini"
    llm_api_key: str = ""
    llm_model: str = "gemini-1.5-flash"
    llm_base_url: str = ""
    temperature: float = 0.9
    # Image
    enable_thumbnail: bool = False
    image_provider: str = "openai"
    image_api_key: str = ""
    image_model: str = "gpt-image-1"
    image_base_url: str = ""
    image_size: str = "1024x1792"
    # Story shape
    shorts_per_story: int = 5
    total_story_minutes: int = 10
    # Render
    voice: str = "af_heart"
    auto_voice: bool = True
    quality: str = "Balanced"
    enable_subtitles: bool = True
    subtitle_position: str = "Bottom"
    subtitle_font_size: int = 16
    background_category: str = "minecraft"
    background_video: str = ""  # optional path to a single background source
    long_video_enabled: bool = True
    # YouTube
    enable_youtube: bool = True
    youtube_category: str = "Entertainment"
    youtube_privacy: str = "private"
    youtube_tags: str = "shorts, storytime, viral, fyp"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AutopilotConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_secrets(paths: ProjectPaths) -> dict[str, Any]:
    """Merge persisted credentials over defaults (same shape as app.py)."""
    defaults = {
        "llm": {
            "provider": "gemini",
            "api_key": "",
            "model": "gemini-1.5-flash",
            "base_url": "",
        },
        "image": {
            "provider": "openai",
            "api_key": "",
            "model": "gpt-image-1",
            "base_url": "",
            "size": "1024x1792",
        },
    }
    secrets_path = paths.config_dir / "secrets.json"
    if secrets_path.exists():
        try:
            saved = load_json(secrets_path)
        except Exception:
            saved = {}
        for section in ("llm", "image"):
            if isinstance(saved.get(section), dict):
                defaults[section].update(saved[section])
    return defaults


class AutopilotEngine:
    """Builds the pipeline once and runs any number of stories headlessly."""

    def __init__(self, cfg: AutopilotConfig, paths: ProjectPaths, settings: dict[str, Any]):
        self.cfg = cfg
        self.paths = paths
        self.settings = settings

        # LLM
        self.llm_client = LLMClient(
            provider=cfg.llm_provider,
            api_key=cfg.llm_api_key.strip(),
            model=cfg.llm_model.strip(),
            base_url=cfg.llm_base_url.strip() or None,
            temperature=cfg.temperature,
        )
        self.story_generator = StoryGenerator(self.llm_client)

        # TTS / subtitles / render
        tts_cfg = settings["tts"]
        subtitle_cfg = settings["subtitle"]
        self.tts_engine = KokoroTTSEngine(
            model_path=paths.base / tts_cfg["model_path"],
            voices_path=paths.base / tts_cfg["voices_path"],
            model_url=tts_cfg["model_url"],
            voices_url=tts_cfg["voices_url"],
            language=tts_cfg["language"],
            speed=float(tts_cfg["speed"]),
        )
        self.subtitle_engine = SubtitleEngine(
            model_size=subtitle_cfg["model_size"],
            compute_type="float32",
            device="cpu",
        )

        render_cfg = self._build_render_config(settings)
        self.background_manager = BackgroundManager(paths.backgrounds_dir)
        self.ffmpeg_renderer = FFmpegRenderer(render_cfg)
        self.pipeline = VideoPipelineEngine(
            tts_engine=self.tts_engine,
            subtitle_engine=self.subtitle_engine,
            background_manager=self.background_manager,
            ffmpeg_renderer=self.ffmpeg_renderer,
        )

        # YouTube accounts
        self.account_manager = YouTubeAccountManager(paths.cache_dir)
        self.accounts: list[YouTubeAccount] = []
        self._account_uploaders: dict[str, tuple[YouTubeAccount, YouTubeUploader]] = {}
        if cfg.enable_youtube:
            self._prepare_accounts()

    def _build_render_config(self, settings: dict[str, Any]) -> RenderConfig:
        render = settings["render"]
        cfg = RenderConfig(
            width=int(render["width"]),
            height=int(render["height"]),
            fps=int(render["fps"]),
            video_codec=render["video_codec"],
            audio_codec=render["audio_codec"],
            video_bitrate=render["video_bitrate"],
            audio_bitrate=render["audio_bitrate"],
            threads=max(1, min(2, int(render["threads"]))),
            subtitle_font_size=self.cfg.subtitle_font_size or int(render["subtitle_font_size"]),
            subtitle_alignment=int(render["subtitle_alignment"]),
            subtitle_margin_v=int(render["subtitle_margin_v"]),
            subtitle_primary_color=render["subtitle_primary_color"],
            subtitle_outline_color=render["subtitle_outline_color"],
            prefer_nvenc=True,
        )
        align, margin = _SUBTITLE_POS.get(self.cfg.subtitle_position, (2, 60))
        cfg.subtitle_alignment = align
        cfg.subtitle_margin_v = margin
        cfg.subtitle_font_size = self.cfg.subtitle_font_size
        return cfg

    def _prepare_accounts(self) -> None:
        if not self.account_manager.has_client_secret():
            raise RuntimeError("YouTube is enabled but no client_secret.json was uploaded.")
        self.accounts = self.account_manager.list_accounts()
        if not self.accounts:
            raise RuntimeError("No YouTube account connected. Connect one in the Streamlit sidebar first.")
        for acc in self.accounts:
            uploader = self.account_manager.get_uploader(acc)
            uploader.authenticate()  # uses cached token; opens browser only on first ever use
            self._account_uploaders[acc.name] = (acc, uploader)
            logger.info("Ready: account '%s' (%s)", acc.name, acc.channel_title or "unknown channel")

    def generate_topics(self, count: int, seed_topics: list[str]) -> list[str]:
        """Ask the LLM for fresh story topics."""
        seed_txt = ", ".join(seed_topics) if seed_topics else "viral short-form stories"
        prompt = (
            "You are a viral content strategist. "
            "Return only valid JSON with key 'topics' (an array of strings). "
            "No markdown, no commentary. "
            f"Generate {count} distinct, highly engaging short-form story topics "
            f"similar in spirit to: {seed_txt}. "
            "Each topic must be a short, specific prompt (under 15 words)."
        )
        payload = self.llm_client.generate_json(prompt)
        topics = [str(t).strip() for t in payload.get("topics", []) if str(t).strip()]
        return topics[:count]

    def resolve_voice(self, narrator_gender: str, narrator_tone: str) -> str:
        """Map an LLM-chosen narrator profile to a concrete Kokoro voice.

        Falls back to cfg.voice if the profile is unrecognized, so a bad LLM
        answer can never crash the pipeline.
        """
        if not self.cfg.auto_voice:
            return self.cfg.voice
        gender = (narrator_gender or "").strip().lower()
        tone = (narrator_tone or "").strip().lower()
        group = _VOICE_PROFILES.get(gender)
        if group:
            return group.get(tone) or next(iter(group.values()))
        return self.cfg.voice

    def run_story(self, topic: str, force_account: YouTubeAccount | None = None) -> dict[str, Any]:
        """Produce + publish one story. Returns a summary dict.

        When `force_account` is given, the story is routed to that account
        directly instead of being matched by keywords.
        """
        cfg = self.cfg
        story_id = next_story_id(self.paths.output_dir)
        story_output = ensure_story_output_structure(self.paths, story_id)
        set_active_story(self.paths.cache_dir, story_id)
        logger.info("=== Starting %s | topic=%r ===", story_id, topic)

        target_wpm = 145
        total_target_words = max(200, int(cfg.total_story_minutes * target_wpm))
        words_per_part = max(80, int(total_target_words / max(1, cfg.shorts_per_story)))
        min_words = max(60, int(words_per_part * 0.9))
        max_words = max(min_words + 10, int(words_per_part * 1.1))

        generated = self.story_generator.generate_story(
            topic.strip(),
            requested_parts=cfg.shorts_per_story,
            target_total_words=total_target_words,
        )
        narrative = generated.narrative
        story_title = generated.title

        # One fixed voice for the WHOLE story (all shorts + long video).
        voice = self.resolve_voice(generated.narrator_gender, generated.narrator_tone)
        logger.info("Voice selected: %s (gender=%r, tone=%r)", voice, generated.narrator_gender, generated.narrator_tone)

        (self.paths.stories_dir / f"{story_id}.txt").write_text(narrative + "\n", encoding="utf-8")

        parts = split_story_into_parts(narrative, min_words=min_words, max_words=max_words, max_parts=cfg.shorts_per_story)
        if not parts:
            raise RuntimeError("No story parts were generated from the script.")

        for part in parts:
            text = part.text.strip()
            if not text.lower().endswith("full story in profile."):
                part.text = f"{text} Full story in profile."

        metadata_payload = self._story_to_metadata(story_title, topic, parts)
        if cfg.enable_youtube:
            self._enrich_youtube_metadata(story_title, topic, parts, metadata_payload)
        write_json(story_output.scripts / "metadata.json", metadata_payload)

        thumbnail_path: Path | None = None
        if cfg.enable_thumbnail:
            thumbnail_path = self._generate_thumbnail(story_title, topic, story_output)

        source_bg_video: Path | None = None
        if cfg.background_video:
            p = Path(cfg.background_video)
            if p.exists():
                source_bg_video = p

        # Resolve target account for this story.
        youtube_uploader: YouTubeUploader | None = None
        youtube_account: YouTubeAccount | None = None
        if cfg.enable_youtube:
            if force_account is not None:
                youtube_account = force_account
            else:
                youtube_account = self.account_manager.resolve_account(
                    f"{story_title} {topic}", self.accounts
                )
            if youtube_account is None:
                raise RuntimeError("No YouTube account available for routing.")
            youtube_uploader = self._account_uploaders[youtube_account.name][1]
            logger.info("Routing '%s' -> account '%s'", story_title, youtube_account.name)

        meta_by_index = {p["part_index"]: p for p in metadata_payload["parts"]}
        artifacts = []
        bg_position = 0.0
        uploaded_videos: list[str] = []

        # ── Progress tracking (for crash/internet-failure resume) ──────────────
        progress = self._init_progress(
            story_output, story_id, story_title, topic,
            youtube_account, voice, parts,
        )
        self._save_progress(story_output, progress)

        for part in parts:
            artifact, bg_position = self.pipeline.process_part(
                story_output=story_output,
                part_index=part.index,
                part_text=part.text,
                voice=voice,
                background_video=source_bg_video,
                category=cfg.background_category if not source_bg_video else None,
                enable_subtitles=cfg.enable_subtitles,
                quality=cfg.quality,
                background_position=bg_position,
            )
            artifacts.append(artifact)

            if youtube_uploader is not None:
                url = self._upload_part(youtube_uploader, artifact, part, meta_by_index.get(part.index), story_title, thumbnail_path)
                if url:
                    uploaded_videos.append(url)
                    self._mark_part_uploaded(progress, part.index, url)
                else:
                    self._mark_part_failed(progress, part.index)
                self._save_progress(story_output, progress)

        long_artifact = None
        if cfg.long_video_enabled:
            long_artifact = self.pipeline.process_full_story(
                story_output=story_output,
                narrative=narrative,
                voice=voice,
                background_video=source_bg_video,
                category=cfg.background_category if not source_bg_video else None,
                enable_subtitles=cfg.enable_subtitles,
                quality=cfg.quality,
            )
            if youtube_uploader is not None:
                url = self._upload_long(youtube_uploader, long_artifact, story_title, topic, narrative, thumbnail_path)
                if url:
                    uploaded_videos.append(url)
                    self._mark_long_uploaded(progress, url)
                else:
                    self._mark_long_failed(progress)
                self._save_progress(story_output, progress)

        # Finalize: mark completed only if everything that should upload did.
        self._finalize_progress(progress)
        self._save_progress(story_output, progress)

        force_memory_cleanup()
        set_active_story(self.paths.cache_dir, None)
        logger.info("=== Completed %s | %s | %d upload(s) ===", story_id, story_title, len(uploaded_videos))
        return {
            "story_id": story_id,
            "title": story_title,
            "topic": topic,
            "account": youtube_account.name if youtube_account else None,
            "channel": youtube_account.channel_title if youtube_account else None,
            "parts": len(parts),
            "long_video": long_artifact is not None,
            "thumbnail": bool(thumbnail_path),
            "videos": uploaded_videos,
            "output_dir": str(story_output.root),
        }

    # ── progress tracking ────────────────────────────────────────────────────
    @staticmethod
    def _progress_path(story_output: Any) -> Path:
        return story_output.root / "progress.json"

    def _load_progress(self, story_output: Any) -> dict[str, Any]:
        p = self._progress_path(story_output)
        if p.exists():
            try:
                return load_json(p)
            except Exception:
                return {}
        return {}

    def _save_progress(self, story_output: Any, progress: dict[str, Any]) -> None:
        progress["updated_iso"] = datetime.now().isoformat(timespec="seconds")
        write_json(self._progress_path(story_output), progress)

    @staticmethod
    def _init_progress(
        story_output: Any,
        story_id: str,
        story_title: str,
        topic: str,
        account: YouTubeAccount | None,
        voice: str,
        parts: list[StoryPart],
    ) -> dict[str, Any]:
        now = datetime.now().isoformat(timespec="seconds")
        return {
            "story_id": story_id,
            "state": "in_progress",
            "title": story_title,
            "topic": topic,
            "account": account.name if account else None,
            "channel": account.channel_title if account else None,
            "voice": voice,
            "created_iso": now,
            "updated_iso": now,
            "parts": {str(p.index): {"uploaded": False, "url": None, "failed": False} for p in parts},
            "long_video": {"uploaded": False, "url": None, "failed": False},
        }

    @staticmethod
    def _mark_part_uploaded(progress: dict[str, Any], index: int, url: str) -> None:
        progress["parts"][str(index)] = {"uploaded": True, "url": url, "failed": False}

    @staticmethod
    def _mark_part_failed(progress: dict[str, Any], index: int) -> None:
        progress["parts"][str(index)] = {"uploaded": False, "url": None, "failed": True}

    @staticmethod
    def _mark_long_uploaded(progress: dict[str, Any], url: str) -> None:
        progress["long_video"] = {"uploaded": True, "url": url, "failed": False}

    @staticmethod
    def _mark_long_failed(progress: dict[str, Any]) -> None:
        progress["long_video"] = {"uploaded": False, "url": None, "failed": True}

    def _finalize_progress(self, progress: dict[str, Any]) -> None:
        if not self.cfg.enable_youtube:
            # YouTube disabled: rendering is the whole job; there is nothing to upload.
            progress["state"] = "completed"
            return
        parts_ok = all(v.get("uploaded") for v in progress["parts"].values())
        long_ok = progress["long_video"].get("uploaded")
        progress["state"] = "completed" if (parts_ok and long_ok) else "in_progress"

    def _find_pending_story(self) -> tuple[str, Any, dict[str, Any]] | None:
        """Return (story_id, story_output, progress) for the oldest unfinished story."""
        dirs = sorted([p for p in self.paths.output_dir.glob("story_*") if p.is_dir()])
        for story_dir in dirs:
            prog_path = story_dir / "progress.json"
            if not prog_path.exists():
                continue
            try:
                progress = load_json(prog_path)
            except Exception:
                continue
            if progress.get("state") == "in_progress":
                story_output = ensure_story_output_structure(self.paths, story_dir.name)
                return story_dir.name, story_output, progress
        return None

    def has_pending_story(self) -> bool:
        """Return True if any story is still in progress (not fully completed)."""
        return self._find_pending_story() is not None

    @staticmethod
    def _final_video_ok(path: Path) -> bool:
        """Return True if path is a complete, playable video.

        An interrupted render (killed mid-encode with +faststart) leaves an mp4
        with no valid moov atom — ffprobe can't read its duration.
        """
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            return ffprobe_duration_seconds(path) > 0
        except Exception:
            return False

    def resume_pending_story(self) -> dict[str, Any] | None:
        """Continue the oldest interrupted story (retry failed/missing uploads).

        Returns a summary dict on success/progress, or None if there is nothing
        pending. Already-uploaded parts are skipped (never re-uploaded).
        """
        cfg = self.cfg
        found = self._find_pending_story()
        if found is None:
            return None
        story_id, story_output, progress = found
        set_active_story(self.paths.cache_dir, story_id)
        logger.info("=== Resuming %s ===", story_id)

        narrative_path = self.paths.stories_dir / f"{story_id}.txt"
        if not narrative_path.exists():
            logger.warning("Resume for %s: narrative file missing; marking failed.", story_id)
            progress["state"] = "failed"
            self._save_progress(story_output, progress)
            return None
        narrative = narrative_path.read_text(encoding="utf-8").strip()

        # Rebuild the split parts identically to the original run.
        target_wpm = 145
        total_target_words = max(200, int(cfg.total_story_minutes * target_wpm))
        words_per_part = max(80, int(total_target_words / max(1, cfg.shorts_per_story)))
        min_words = max(60, int(words_per_part * 0.9))
        max_words = max(min_words + 10, int(words_per_part * 1.1))
        parts = split_story_into_parts(narrative, min_words=min_words, max_words=max_words, max_parts=cfg.shorts_per_story)
        for part in parts:
            text = part.text.strip()
            if not text.lower().endswith("full story in profile."):
                part.text = f"{text} Full story in profile."

        story_title = progress.get("title") or story_id
        topic = progress.get("topic") or ""

        # Reload enriched metadata (YouTube titles/descriptions/tags).
        metadata_payload = self._load_progress_metadata(story_output)
        meta_by_index = {p["part_index"]: p for p in metadata_payload.get("parts", [])}

        voice = progress.get("voice") or cfg.voice

        # Resolve uploader for the recorded account.
        account_name = progress.get("account")
        youtube_account = None
        youtube_uploader = None
        if cfg.enable_youtube and account_name:
            youtube_account = next((a for a in self.accounts if a.name == account_name), None)
            if youtube_account is not None and youtube_account.name in self._account_uploaders:
                youtube_uploader = self._account_uploaders[youtube_account.name][1]

        thumbnail_path = story_output.thumbnails / "cover.png"
        if not thumbnail_path.exists():
            thumbnail_path = None

        source_bg_video: Path | None = None
        if cfg.background_video:
            p = Path(cfg.background_video)
            if p.exists():
                source_bg_video = p

        uploaded_videos: list[str] = []

        # ── Retry each pending part ───────────────────────────────────────────
        for part in parts:
            entry = progress["parts"].get(str(part.index), {})
            if entry.get("uploaded"):
                if entry.get("url"):
                    uploaded_videos.append(entry["url"])
                continue

            final_path = story_output.final / f"part{part.index}.mp4"
            if not self._final_video_ok(final_path):
                if final_path.exists():
                    logger.warning("Removing corrupt/incomplete %s; re-rendering.", final_path.name)
                    final_path.unlink(missing_ok=True)
                _, _bg = self.pipeline.process_part(
                    story_output=story_output,
                    part_index=part.index,
                    part_text=part.text,
                    voice=voice,
                    background_video=source_bg_video,
                    category=cfg.background_category if not source_bg_video else None,
                    enable_subtitles=cfg.enable_subtitles,
                    quality=cfg.quality,
                    background_position=0.0,
                )
            artifact = RenderedPartArtifact(
                index=part.index,
                script_path=story_output.scripts / f"part{part.index}.txt",
                audio_path=story_output.audio / f"part{part.index}.wav",
                subtitle_path=story_output.subtitles / f"part{part.index}.srt",
                final_video_path=final_path,
            )

            if youtube_uploader is not None:
                url = self._upload_part(youtube_uploader, artifact, part, meta_by_index.get(part.index), story_title, thumbnail_path)
                if url:
                    uploaded_videos.append(url)
                    self._mark_part_uploaded(progress, part.index, url)
                else:
                    self._mark_part_failed(progress, part.index)
                self._save_progress(story_output, progress)

        # ── Long video ────────────────────────────────────────────────────────
        if cfg.long_video_enabled and not progress["long_video"].get("uploaded"):
            full_path = story_output.final / "full.mp4"
            if not self._final_video_ok(full_path):
                if full_path.exists():
                    full_path.unlink(missing_ok=True)
                self.pipeline.process_full_story(
                    story_output=story_output,
                    narrative=narrative,
                    voice=voice,
                    background_video=source_bg_video,
                    category=cfg.background_category if not source_bg_video else None,
                    enable_subtitles=cfg.enable_subtitles,
                    quality=cfg.quality,
                )
            long_artifact = RenderedPartArtifact(
                index=0,
                script_path=story_output.scripts / "full.txt",
                audio_path=story_output.audio / "full.wav",
                subtitle_path=story_output.subtitles / "full.srt",
                final_video_path=full_path,
            )
            if youtube_uploader is not None:
                url = self._upload_long(youtube_uploader, long_artifact, story_title, topic, narrative, thumbnail_path)
                if url:
                    uploaded_videos.append(url)
                    self._mark_long_uploaded(progress, url)
                else:
                    self._mark_long_failed(progress)
                self._save_progress(story_output, progress)

        self._finalize_progress(progress)
        self._save_progress(story_output, progress)

        force_memory_cleanup()
        set_active_story(self.paths.cache_dir, None)
        logger.info("=== Resumed %s | state=%s | %d upload(s) ===", story_id, progress["state"], len(uploaded_videos))
        return {
            "story_id": story_id,
            "title": story_title,
            "topic": topic,
            "account": youtube_account.name if youtube_account else account_name,
            "channel": youtube_account.channel_title if youtube_account else None,
            "parts": len(parts),
            "long_video": cfg.long_video_enabled and progress["long_video"].get("uploaded", False),
            "thumbnail": bool(thumbnail_path),
            "videos": uploaded_videos,
            "output_dir": str(story_output.root),
            "resumed": True,
        }

    def _load_progress_metadata(self, story_output: Any) -> dict[str, Any]:
        p = story_output.scripts / "metadata.json"
        if p.exists():
            try:
                return load_json(p)
            except Exception:
                return {}
        return {}

    # ── helpers ──────────────────────────────────────────────────────────────
    def _story_to_metadata(self, story_title: str, topic: str, parts: list[StoryPart]) -> dict[str, Any]:
        part_payload = []
        for part in parts:
            md = generate_part_metadata(story_title, topic, part)
            part_payload.append({
                "part_index": part.index,
                "part_title": part.part_title,
                "caption": md.caption,
                "hashtags": md.hashtags,
                "script": part.text,
            })
        return {
            "story_title": story_title,
            "topic": topic,
            "platform_targets": ["tiktok", "instagram_reels", "facebook_reels", "youtube_shorts"],
            "parts": part_payload,
        }

    def _enrich_youtube_metadata(self, story_title: str, topic: str, parts: list[StoryPart], payload: dict[str, Any]) -> None:
        for part in parts:
            meta = self.story_generator.generate_youtube_metadata(
                story_title=story_title, topic=topic,
                part_index=part.index, part_title=part.part_title, part_text=part.text,
            )
            for p in payload["parts"]:
                if p["part_index"] == part.index:
                    p["youtube_title"] = meta.title
                    p["youtube_description"] = meta.description
                    p["youtube_tags"] = meta.tags
                    break

    def _generate_thumbnail(self, story_title: str, topic: str, story_output: Any) -> Path | None:
        try:
            image_gen = ImageGenerator(
                provider=self.cfg.image_provider,
                api_key=self.cfg.image_api_key.strip(),
                model=self.cfg.image_model.strip(),
                base_url=self.cfg.image_base_url.strip() or None,
            )
            prompt = self.story_generator.generate_thumbnail_prompt(story_title, topic)
            image_bytes = image_gen.generate(prompt, size=self.cfg.image_size)
            thumbnail_path = story_output.thumbnails / "cover.png"
            thumbnail_path.write_bytes(image_bytes)
            return thumbnail_path
        except Exception as exc:
            logger.warning("Thumbnail generation failed (continuing): %s", exc)
            return None

    def _upload_part(self, uploader, artifact, part, meta, story_title, thumbnail_path) -> str | None:
        cfg = self.cfg
        try:
            if meta and meta.get("youtube_title"):
                title = meta["youtube_title"]
            else:
                title = f"{story_title} - Part {part.index}"
            if meta and meta.get("youtube_description"):
                description = meta["youtube_description"]
            else:
                caption = meta.get("caption", "") if meta else part.cliffhanger
                hashtags = " ".join(meta.get("hashtags", [])) if meta else ""
                description = f"{caption}\n\n{hashtags}".strip()
            tags = list(meta["youtube_tags"]) if meta and meta.get("youtube_tags") else [t.strip() for t in cfg.youtube_tags.split(",") if t.strip()]

            result = uploader.upload_video(
                video_path=artifact.final_video_path,
                title=title,
                description=description,
                tags=tags,
                category_id=_YT_CATEGORIES.get(cfg.youtube_category, "24"),
                privacy_status=cfg.youtube_privacy,
            )
            if thumbnail_path and thumbnail_path.exists():
                uploader.set_thumbnail(result.video_id, thumbnail_path)
            logger.info("Uploaded Part %s -> %s", part.index, result.video_url)
            return result.video_url
        except Exception as exc:
            logger.exception("Upload failed for part %s", part.index)
            return None

    def _upload_long(self, uploader, artifact, story_title, topic, narrative, thumbnail_path) -> str | None:
        cfg = self.cfg
        try:
            md = self.story_generator.generate_youtube_metadata(story_title, topic, 0, "Full Story", narrative)
            title = md.title
            description = md.description
            tags = md.tags
        except Exception:
            title = story_title
            description = f"{topic} — full story."
            tags = [t.strip() for t in cfg.youtube_tags.split(",") if t.strip()]
        try:
            result = uploader.upload_video(
                video_path=artifact.final_video_path,
                title=title,
                description=description,
                tags=tags,
                category_id=_YT_CATEGORIES.get(cfg.youtube_category, "24"),
                privacy_status=cfg.youtube_privacy,
            )
            if thumbnail_path and thumbnail_path.exists():
                uploader.set_thumbnail(result.video_id, thumbnail_path)
            logger.info("Uploaded full story -> %s", result.video_url)
            return result.video_url
        except Exception as exc:
            logger.exception("Upload failed for full story")
            return None
