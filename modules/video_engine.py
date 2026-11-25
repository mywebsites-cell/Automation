from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from modules.background_manager import BackgroundManager
from modules.cleanup_manager import cleanup_temp_directory, force_memory_cleanup
from modules.ffmpeg_renderer import FFmpegRenderer
from modules.subtitle_engine import SubtitleEngine
from modules.tts_engine import KokoroTTSEngine
from modules.utils import StoryOutputPaths, load_json

logger = logging.getLogger(__name__)


@dataclass
class RenderedPartArtifact:
    index: int
    script_path: Path
    audio_path: Path
    subtitle_path: Path | None
    final_video_path: Path


class VideoPipelineEngine:
    def __init__(
        self,
        tts_engine: KokoroTTSEngine,
        subtitle_engine: SubtitleEngine,
        background_manager: BackgroundManager,
        ffmpeg_renderer: FFmpegRenderer,
    ):
        self.tts_engine = tts_engine
        self.subtitle_engine = subtitle_engine
        self.background_manager = background_manager
        self.ffmpeg_renderer = ffmpeg_renderer

    def _produce(
        self,
        story_output: StoryOutputPaths,
        name: str,
        index: int,
        text: str,
        voice: str,
        background_video: Path | None,
        category: str | None,
        enable_subtitles: bool,
        quality: str,
        background_start_time: float = 0.0,
    ) -> tuple[RenderedPartArtifact, float]:
        script_path = story_output.scripts / f"{name}.txt"
        audio_path = story_output.audio / f"{name}.wav"
        subtitle_path = story_output.subtitles / f"{name}.srt"
        final_path = story_output.final / f"{name}.mp4"

        script_path.write_text(text.strip() + "\n", encoding="utf-8")

        logger.info("TTS started for %s", name)
        tts_result = self.tts_engine.synthesize_to_file(text, voice, audio_path)
        logger.info("TTS complete for %s, duration=%.2fs", name, tts_result.duration_seconds)

        subtitle_result = None
        if enable_subtitles:
            logger.info("Subtitle transcription started for %s", name)
            subtitle_result = self.subtitle_engine.transcribe_to_srt(audio_path, subtitle_path)
            logger.info("Subtitle generated for %s with %d segments", name, subtitle_result.segment_count)

        # Use provided background video directly (continuous playback from single source)
        if background_video and background_video.exists():
            render_bg = background_video
        else:
            # Fallback: concatenate clips if no source video provided
            logger.info("Planning backgrounds for %s from category %s", name, category)
            clip_plan = self.background_manager.plan_clips_for_duration(
                category=category or "minecraft",
                duration_seconds=tts_result.duration_seconds,
                temp_dir=story_output.temp,
            )
            render_bg = story_output.temp / f"{name}_background_concat.mp4"
            self.ffmpeg_renderer.concat_backgrounds(
                clip_plan.concat_file,
                render_bg,
                duration_seconds=tts_result.duration_seconds,
            )
            background_start_time = 0.0  # Reset position for clip-based rendering

        logger.info("Rendering final video for %s", name)
        self.ffmpeg_renderer.render_vertical_short(
            background_video=render_bg,
            audio_path=audio_path,
            subtitle_path=subtitle_path if enable_subtitles else None,
            output_path=final_path,
            quality=quality,
            background_start_time=background_start_time,
        )

        cleanup_temp_directory(story_output.temp)
        force_memory_cleanup()

        return (
            RenderedPartArtifact(
                index=index,
                script_path=script_path,
                audio_path=audio_path,
                subtitle_path=subtitle_path if enable_subtitles else None,
                final_video_path=final_path,
            ),
            tts_result.duration_seconds,
        )

    def process_part(
        self,
        story_output: StoryOutputPaths,
        part_index: int,
        part_text: str,
        voice: str,
        background_video: Path | None,
        category: str | None,
        enable_subtitles: bool,
        quality: str,
        background_position: float = 0.0,
    ) -> tuple[RenderedPartArtifact, float]:
        artifact, duration = self._produce(
            story_output=story_output,
            name=f"part{part_index}",
            index=part_index,
            text=part_text,
            voice=voice,
            background_video=background_video,
            category=category,
            enable_subtitles=enable_subtitles,
            quality=quality,
            background_start_time=background_position,
        )
        return artifact, background_position + duration

    def process_full_story(
        self,
        story_output: StoryOutputPaths,
        narrative: str,
        voice: str,
        background_video: Path | None,
        category: str | None,
        enable_subtitles: bool,
        quality: str,
    ) -> RenderedPartArtifact:
        """Render the complete story as one long vertical video."""
        artifact, _ = self._produce(
            story_output=story_output,
            name="full",
            index=0,
            text=narrative,
            voice=voice,
            background_video=background_video,
            category=category,
            enable_subtitles=enable_subtitles,
            quality=quality,
            background_start_time=0.0,
        )
        return artifact
