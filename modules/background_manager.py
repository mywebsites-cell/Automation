from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from modules.utils import ffprobe_duration_seconds, load_json, write_json


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv"}


@dataclass
class BackgroundClipPlan:
    selected_clips: list[Path]
    concat_file: Path


class BackgroundManager:
    def __init__(self, backgrounds_root: Path):
        self.backgrounds_root = backgrounds_root
        self._clip_index = {}  # Track current clip index per category for sequential playback

    def categories(self) -> list[str]:
        cats = [d.name for d in self.backgrounds_root.iterdir() if d.is_dir()]
        return sorted(cats)

    def list_category_files(self, category: str) -> list[Path]:
        category_dir = self.backgrounds_root / category
        if not category_dir.exists():
            return []
        return sorted(
            [
                p
                for p in category_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        )

    def plan_clips_for_duration(
        self,
        category: str,
        duration_seconds: float,
        temp_dir: Path,
    ) -> BackgroundClipPlan:
        files = self.list_category_files(category)
        if not files:
            raise RuntimeError(
                f"No background clips found in category '{category}'. "
                f"Add files to {self.backgrounds_root / category}"
            )

        # Initialize clip index for this category if not already done
        if category not in self._clip_index:
            self._clip_index[category] = 0

        selected: list[Path] = []
        total = 0.0
        safety_counter = 0
        
        # Use sequential cycling through files (not random) for continuous playback
        while total < duration_seconds and safety_counter < 200:
            safety_counter += 1
            clip = files[self._clip_index[category] % len(files)]
            selected.append(clip)
            total += ffprobe_duration_seconds(clip)
            self._clip_index[category] += 1

        if total < duration_seconds:
            raise RuntimeError("Unable to plan enough background duration from available clips")

        concat_path = temp_dir / "background_concat.txt"
        concat_lines = [f"file '{str(p).replace('\\', '/')}'" for p in selected]
        concat_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

        return BackgroundClipPlan(selected_clips=selected, concat_file=concat_path)

    def generate_continuous_playlist(
        self,
        category: str,
        total_duration_seconds: float,
        output_dir: Path,
    ) -> list[str]:
        """Generate a continuous clip sequence for the entire story.
        Returns a list of clip paths that will be used sequentially across all parts.
        """
        files = self.list_category_files(category)
        if not files:
            raise RuntimeError(
                f"No background clips found in category '{category}'. "
                f"Add files to {self.backgrounds_root / category}"
            )

        playlist: list[str] = []
        total = 0.0
        safety_counter = 0
        clip_idx = 0
        
        # Sequential cycling through files (not random) for continuous playback
        while total < total_duration_seconds and safety_counter < 500:
            safety_counter += 1
            clip = files[clip_idx % len(files)]
            playlist.append(str(clip))
            total += ffprobe_duration_seconds(clip)
            clip_idx += 1

        if total < total_duration_seconds:
            raise RuntimeError("Unable to plan enough background duration from available clips")

        # Save the playlist so all parts use the same sequence
        playlist_file = output_dir / "scripts" / "background_playlist.json"
        playlist_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(playlist_file, {"clips": playlist, "total_duration": total})
        return playlist

    def get_clips_for_part(
        self,
        part_index: int,
        part_duration_seconds: float,
        playlist: list[str],
    ) -> list[Path]:
        """Return the next sequential clips for this part from the continuous playlist."""
        # Calculate which clips this part needs based on its index
        cum_duration = 0.0
        start_idx = 0

        # Find approximately where this part starts in the timeline
        for i, clip_str in enumerate(playlist):
            clip_duration = ffprobe_duration_seconds(Path(clip_str))
            if cum_duration + clip_duration > (part_index - 1) * part_duration_seconds:
                start_idx = i
                break
            cum_duration += clip_duration

        # Collect clips until we have enough duration for this part
        part_clips: list[Path] = []
        part_duration = 0.0
        idx = start_idx
        safety = 0
        while part_duration < part_duration_seconds and idx < len(playlist) and safety < 100:
            safety += 1
            clip_path = Path(playlist[idx])
            part_clips.append(clip_path)
            part_duration += ffprobe_duration_seconds(clip_path)
            idx += 1

        return part_clips if part_clips else [Path(playlist[0])]
