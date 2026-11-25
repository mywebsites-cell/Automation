import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ProjectPaths:
    base: Path
    config_dir: Path
    assets_dir: Path
    backgrounds_dir: Path
    music_dir: Path
    fonts_dir: Path
    stories_dir: Path
    output_dir: Path
    logs_dir: Path
    modules_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class StoryOutputPaths:
    root: Path
    scripts: Path
    audio: Path
    subtitles: Path
    temp: Path
    final: Path
    thumbnails: Path


def get_project_paths(base_dir: Path) -> ProjectPaths:
    return ProjectPaths(
        base=base_dir,
        config_dir=base_dir / "config",
        assets_dir=base_dir / "assets",
        backgrounds_dir=base_dir / "assets" / "backgrounds",
        music_dir=base_dir / "assets" / "music",
        fonts_dir=base_dir / "assets" / "fonts",
        stories_dir=base_dir / "stories",
        output_dir=base_dir / "output",
        logs_dir=base_dir / "output" / "logs",
        modules_dir=base_dir / "modules",
        cache_dir=base_dir / "cache",
    )


def ensure_core_directories(paths: ProjectPaths) -> None:
    required = [
        paths.config_dir,
        paths.assets_dir,
        paths.backgrounds_dir,
        paths.backgrounds_dir / "minecraft",
        paths.backgrounds_dir / "subway",
        paths.backgrounds_dir / "horror",
        paths.backgrounds_dir / "rain",
        paths.music_dir,
        paths.fonts_dir,
        paths.stories_dir,
        paths.output_dir,
        paths.logs_dir,
        paths.modules_dir,
        paths.cache_dir,
    ]
    for directory in required:
        directory.mkdir(parents=True, exist_ok=True)


def ensure_story_output_structure(paths: ProjectPaths, story_id: str) -> StoryOutputPaths:
    root = paths.output_dir / story_id
    output_paths = StoryOutputPaths(
        root=root,
        scripts=root / "scripts",
        audio=root / "audio",
        subtitles=root / "subtitles",
        temp=root / "temp",
        final=root / "final",
        thumbnails=root / "final" / "thumbnails",
    )
    for directory in [
        output_paths.root,
        output_paths.scripts,
        output_paths.audio,
        output_paths.subtitles,
        output_paths.temp,
        output_paths.final,
        output_paths.thumbnails,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    return output_paths


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def autopilot_control_path(cache_dir: Path) -> Path:
    return cache_dir / "autopilot" / "control.json"


def read_autopilot_control(cache_dir: Path) -> dict[str, Any]:
    """Read the autopilot pause/resume control file (missing file = enabled)."""
    path = autopilot_control_path(cache_dir)
    if path.exists():
        try:
            return load_json(path)
        except Exception:
            return {}
    return {}


def write_autopilot_control(cache_dir: Path, control: dict[str, Any]) -> None:
    path = autopilot_control_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, control)


def timestamp_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def next_story_id(output_dir: Path) -> str:
    story_dirs = sorted([p for p in output_dir.glob("story_*") if p.is_dir()])
    if not story_dirs:
        return "story_001"
    last = story_dirs[-1].name.split("_")[-1]
    try:
        next_idx = int(last) + 1
    except ValueError:
        next_idx = len(story_dirs) + 1
    return f"story_{next_idx:03d}"


def setup_logging(log_dir: Path, run_name: str = "run") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}_{timestamp_id()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def run_command(command: Iterable[str], timeout: int | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            list(command),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as exc:
        return 127, "", str(exc)


def get_ffmpeg_executable() -> str | None:
    # 1) Prefer system ffmpeg when available.
    code, _, _ = run_command(["ffmpeg", "-version"], timeout=30)
    if code == 0:
        return "ffmpeg"

    # 2) Fallback to imageio-ffmpeg bundled binary inside the venv.
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except Exception:
        pass

    return None


def ffprobe_duration_seconds(input_path: Path) -> float:
    ffmpeg_exe = get_ffmpeg_executable()
    if not ffmpeg_exe:
        raise RuntimeError("FFmpeg executable not found")

    # Use ffmpeg stderr metadata probe (works without ffprobe binary).
    command = [ffmpeg_exe, "-hide_banner", "-i", str(input_path)]
    code, _, stderr = run_command(command, timeout=120)
    _ = code  # ffmpeg returns non-zero for probe-only -i usage.

    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if not match:
        raise RuntimeError(f"Could not parse duration from ffmpeg output for {input_path}")

    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def has_ffmpeg() -> bool:
    return get_ffmpeg_executable() is not None


def register_local_ffmpeg(base_dir: Path) -> bool:
    """
    Register a project-local FFmpeg binary directory in PATH if present.

    Expected location pattern:
      <project>/cache/ffmpeg/**/bin/(ffmpeg.exe, ffprobe.exe)
    """
    ffmpeg_root = base_dir / "cache" / "ffmpeg"
    if not ffmpeg_root.exists():
        return False

    bin_candidates = sorted(ffmpeg_root.glob("**/bin"))
    for bin_dir in bin_candidates:
        ffmpeg_bin = bin_dir / "ffmpeg.exe"
        ffprobe_bin = bin_dir / "ffprobe.exe"
        if ffmpeg_bin.exists() and ffprobe_bin.exists():
            current_path = os.environ.get("PATH", "")
            if str(bin_dir) not in current_path:
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{current_path}"
            return True

    return False


def has_ffmpeg_with_local_fallback(base_dir: Path) -> bool:
    # Try system ffmpeg first.
    if has_ffmpeg():
        return True

    # Then register and test project-local ffmpeg binaries.
    if register_local_ffmpeg(base_dir):
        return has_ffmpeg()
    return False
