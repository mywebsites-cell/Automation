"""
Standalone Whisper transcription worker.
Run as a subprocess to isolate ctranslate2/Intel-Fortran crashes from Streamlit.
Usage: python whisper_worker.py <model_size> <compute_type> <device> <audio_path> <srt_path>
Exits 0 on success and prints SEGMENTS:<n>. Exits 1 on failure and prints error to stderr.
"""
from __future__ import annotations

import os

# Must be set before ctranslate2 / faster-whisper are imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path


def _ts(seconds: float) -> str:
    total_ms = int(seconds * 1000)
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    secs = rem // 1000
    ms = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def main() -> None:
    if len(sys.argv) != 6:
        print(
            "Usage: whisper_worker.py <model_size> <compute_type> <device> <audio_path> <srt_path>",
            file=sys.stderr,
        )
        sys.exit(1)

    model_size, compute_type, device, audio_path, srt_path = sys.argv[1:]

    # Force float32 on CPU — int8 requires cublas64_12.dll (CUDA 12) which is
    # not present on CPU-only machines even though ctranslate2 ships without it.
    if device == "cpu" and compute_type == "int8":
        compute_type = "float32"
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        print(f"faster-whisper not installed: {exc}", file=sys.stderr)
        sys.exit(1)

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, _ = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        language="en",
    )

    segment_list = list(segments)

    srt = Path(srt_path)
    srt.parent.mkdir(parents=True, exist_ok=True)
    with srt.open("w", encoding="utf-8") as f:
        # One subtitle entry per word, timed by Whisper's word timestamps so only
        # the currently-spoken word is shown (word-by-word / TikTok style).
        idx = 1
        for seg in segment_list:
            words = list(getattr(seg, "words", None) or [])
            if not words:
                text = (seg.text or "").strip()
                if text:
                    f.write(f"{idx}\n")
                    f.write(f"{_ts(seg.start)} --> {_ts(seg.end)}\n")
                    f.write(f"{text}\n\n")
                    idx += 1
                continue

            for i, w in enumerate(words):
                start = float(w.start)
                # End when the next word begins so there is no gap; last word
                # ends at its own end time.
                if i + 1 < len(words):
                    end = float(words[i + 1].start)
                else:
                    end = float(w.end)
                if end <= start:
                    end = start + 0.1

                f.write(f"{idx}\n")
                f.write(f"{_ts(start)} --> {_ts(end)}\n")
                f.write(f"{w.word.strip()}\n\n")
                idx += 1

    print(f"SEGMENTS:{len(segment_list)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
