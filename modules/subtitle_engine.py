from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubtitleResult:
    srt_path: Path
    segment_count: int


class SubtitleEngine:
    def __init__(self, model_size: str, compute_type: str, device: str):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device

    def transcribe_to_srt(self, audio_path: Path, srt_path: Path) -> SubtitleResult:
        """Run Whisper in an isolated subprocess so any ctranslate2 crash can't kill Streamlit."""
        worker = Path(__file__).parent / "whisper_worker.py"
        result = subprocess.run(
            [
                sys.executable,
                str(worker),
                self.model_size,
                self.compute_type,
                self.device,
                str(audio_path),
                str(srt_path),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "unknown error")[-600:]
            raise RuntimeError(f"Whisper transcription failed: {err}")

        seg_count = 0
        for line in result.stdout.splitlines():
            if line.startswith("SEGMENTS:"):
                seg_count = int(line.split(":")[1])

        return SubtitleResult(srt_path=srt_path, segment_count=seg_count)

    @staticmethod
    def _ts(seconds: float) -> str:
        total_ms = int(seconds * 1000)
        hours = total_ms // 3_600_000
        rem = total_ms % 3_600_000
        minutes = rem // 60_000
        rem = rem % 60_000
        secs = rem // 1000
        ms = rem % 1000
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"

