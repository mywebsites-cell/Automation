from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from modules.utils import ffprobe_duration_seconds, get_ffmpeg_executable, run_command


@dataclass
class RenderConfig:
    width: int
    height: int
    fps: int
    video_codec: str
    audio_codec: str
    video_bitrate: str
    audio_bitrate: str
    threads: int
    subtitle_font_size: int
    subtitle_alignment: int
    subtitle_margin_v: int
    subtitle_primary_color: str
    subtitle_outline_color: str
    prefer_nvenc: bool = True


class FFmpegRenderer:
    def __init__(self, render_config: RenderConfig):
        self.cfg = render_config

    @staticmethod
    def is_available() -> bool:
        _ = shutil
        return get_ffmpeg_executable() is not None

    def concat_backgrounds(
        self,
        concat_file: Path,
        output_video: Path,
        duration_seconds: float | None = None,
    ) -> Path:
        ffmpeg_exe = get_ffmpeg_executable()
        if not ffmpeg_exe:
            raise RuntimeError("FFmpeg executable not found")
        output_video.parent.mkdir(parents=True, exist_ok=True)
        
        # Use GPU codec for concatenation too if available
        concat_codec = self._resolve_video_codec()
        concat_preset = self._quality_to_preset("Draft")  # Fast concatenation
        print(f"[CONCAT] Using codec: {concat_codec}, preset: {concat_preset}")
        
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:v",
            concat_codec,
            "-preset",
            concat_preset,
        ]
        if concat_codec.endswith("_nvenc"):
            # NVENC defaults to a ~2 Mbps bitrate; force constant quality instead.
            cmd += ["-cq", "18", "-b:v", "0"]
        cmd += ["-pix_fmt", "yuv420p"]
        # Only encode the portion we actually need. Without this, a long source
        # (e.g. a 1-hour background) is re-encoded in full, producing a multi-GB
        # intermediate and wasting minutes per part.
        if duration_seconds is not None:
            cmd += ["-t", str(duration_seconds)]
        cmd += [str(output_video)]
        code, _, err = run_command(cmd, timeout=3600)
        if code != 0:
            raise RuntimeError(f"Failed to concat backgrounds: {err.strip()}")
        return output_video

    def render_vertical_short(
        self,
        background_video: Path,
        audio_path: Path,
        subtitle_path: Path | None,
        output_path: Path,
        quality: str,
        background_start_time: float = 0.0,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preset = self._quality_to_preset(quality)
        video_codec = self._resolve_video_codec()
        ffmpeg_exe = get_ffmpeg_executable()
        print(f"[RENDER] Using codec: {video_codec}, preset: {preset}, quality: {quality}, prefer_nvenc: {self.cfg.prefer_nvenc}")
        print(f"[BITRATE] Video: {self.cfg.video_bitrate}, Audio: {self.cfg.audio_bitrate}")
        print(f"[SUBTITLE] FontSize={self.cfg.subtitle_font_size}, Alignment={self.cfg.subtitle_alignment}, MarginV={self.cfg.subtitle_margin_v}")
        if not ffmpeg_exe:
            raise RuntimeError("FFmpeg executable not found")

        vf_base = (
            f"scale={self.cfg.width}:{self.cfg.height}:flags=lanczos:force_original_aspect_ratio=increase,"
            f"crop={self.cfg.width}:{self.cfg.height},fps={self.cfg.fps}"
        )
        vf = vf_base
        if subtitle_path is not None and subtitle_path.exists():
            safe_srt = str(subtitle_path).replace('\\', '/').replace(':', '\\:').replace("'", "\\'")
            style = (
                f"FontName=Arial,"
                f"FontSize={self.cfg.subtitle_font_size},"
                f"PrimaryColour={self.cfg.subtitle_primary_color},"
                f"OutlineColour={self.cfg.subtitle_outline_color},"
                f"BorderStyle=1,"
                f"Outline=2,"
                f"Shadow=0,"
                f"Alignment={self.cfg.subtitle_alignment},"
                f"MarginL=0,"
                f"MarginR=0,"
                f"MarginV={self.cfg.subtitle_margin_v}"
            )
            vf = f"{vf_base},subtitles='{safe_srt}':force_style='{style}'"

        # Codec-specific quality args: NVENC uses constant-quality (-cq) with
        # -b:v 0 and does NOT accept -crf/-maxrate/-bufsize; libx264 uses -crf
        # with a bitrate cap.
        if video_codec.endswith("_nvenc"):
            quality_args = ["-cq", "18", "-b:v", "0"]
        else:
            quality_args = [
                "-crf", "12",
                "-maxrate", self.cfg.video_bitrate,
                "-bufsize", "80M",
            ]

        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss",
            str(background_start_time),  # Seek to this position in background video
            "-stream_loop",
            "-1",
            "-i",
            str(background_video),
            "-i",
            str(audio_path),
            "-t",
            self._probe_duration_for_trim(audio_path),
            "-map", "0:v:0",   # video from background only
            "-map", "1:a:0",   # audio from TTS only (background audio discarded)
            "-vf",
            vf,
            "-c:v",
            video_codec,
            "-preset",
            preset,
            *quality_args,
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            self.cfg.audio_codec,
            "-b:a",
            self.cfg.audio_bitrate,
            "-threads",
            str(self.cfg.threads),
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        code, _, err = run_command(cmd, timeout=3600)
        if code != 0:
            raise RuntimeError(f"FFmpeg render failed: {err.strip()}")
        return output_path

    def _resolve_video_codec(self) -> str:
        if self.cfg.prefer_nvenc and self._ffmpeg_has_encoder("h264_nvenc"):
            print("[CODEC] h264_nvenc is available and preferred - using GPU encoding")
            return "h264_nvenc"
        if self.cfg.prefer_nvenc:
            print("[CODEC] GPU preferred but h264_nvenc not available - falling back to CPU (libx264)")
        return self.cfg.video_codec

    @staticmethod
    def _ffmpeg_has_encoder(encoder_name: str) -> bool:
        ffmpeg_exe = get_ffmpeg_executable()
        if not ffmpeg_exe:
            return False
        cmd = [ffmpeg_exe, "-hide_banner", "-encoders"]
        code, out, _ = run_command(cmd, timeout=120)
        if code != 0:
            return False
        return encoder_name in out

    def _quality_to_preset(self, quality: str) -> str:
        """Map quality setting to codec-specific preset."""
        codec = self._resolve_video_codec()
        
        if codec == "h264_nvenc":
            # NVIDIA h264_nvenc presets: default, fast, medium, slow (slowest but still fast)
            mapping = {
                "Draft": "fast",
                "Balanced": "medium",
                "High": "slow",
            }
        else:
            # libx264 CPU presets: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
            mapping = {
                "Draft": "ultrafast",
                "Balanced": "slow",
                "High": "veryslow",
            }
        return mapping.get(quality, mapping["Balanced"])

    @staticmethod
    def _probe_duration_for_trim(audio_path: Path) -> str:
        return str(ffprobe_duration_seconds(audio_path))
