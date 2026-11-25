from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import requests
import soundfile as sf


logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    audio_path: Path
    sample_rate: int
    duration_seconds: float


class KokoroTTSEngine:
    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        model_url: str,
        voices_url: str,
        language: str,
        speed: float,
    ):
        self.model_path = model_path
        self.voices_path = voices_path
        self.model_url = model_url
        self.voices_url = voices_url
        self.language = language
        self.speed = speed
        self._kokoro = None

    def ensure_model_files(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists():
            self._download(self.model_url, self.model_path)
        if not self.voices_path.exists():
            self._download(self.voices_url, self.voices_path)

    def available_voices(self) -> list[str]:
        engine = self._get_engine()
        return sorted(engine.get_voices())

    def synthesize_to_file(self, text: str, voice: str, output_path: Path) -> TTSResult:
        engine = self._get_engine()
        samples, sample_rate = engine.create(
            text,
            voice=voice,
            speed=self.speed,
            lang=self.language,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), samples, sample_rate)
        duration = float(len(samples)) / float(sample_rate)
        return TTSResult(audio_path=output_path, sample_rate=sample_rate, duration_seconds=duration)

    def _get_engine(self):
        if self._kokoro is not None:
            return self._kokoro
        try:
            from kokoro_onnx import Kokoro
        except Exception as exc:
            raise RuntimeError(
                "kokoro-onnx is not installed. Install requirements and restart."
            ) from exc

        self.ensure_model_files()
        self._kokoro = Kokoro(str(self.model_path), str(self.voices_path))
        return self._kokoro

    @staticmethod
    def _download(url: str, target: Path) -> None:
        import time
        from urllib3.exceptions import ProtocolError
        
        logger.info("Downloading %s -> %s (size ~325MB, may take 5-15 min)", url, target)
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Use session with retry strategy for better resilience
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(max_retries=3)
                session.mount("https://", adapter)
                
                with session.get(url, stream=True, timeout=900, allow_redirects=True) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    logger.info("Download size: %.1f MB", total_size / (1024*1024))
                    
                    downloaded = 0
                    with target.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=5 * 1024 * 1024):  # 5MB chunks
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                                if downloaded % (50 * 1024 * 1024) == 0:  # Log every 50MB
                                    logger.info("Downloaded: %.1f MB / %.1f MB (%.1f%%)", 
                                              downloaded / (1024*1024), total_size / (1024*1024), pct)
                logger.info("Download successful")
                return
            except (requests.exceptions.ChunkedEncodingError, ProtocolError, requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout) as exc:
                logger.warning("Download attempt %d/%d failed (%s): %s", attempt + 1, max_retries, type(exc).__name__, exc)
                if target.exists():
                    try:
                        target.unlink()
                    except:
                        pass
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # 5s, 10s, 20s, 40s, 80s
                    logger.info("Retrying in %d seconds...", wait_time)
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts") from exc
            except Exception as exc:
                logger.error("Unexpected error during download: %s", exc, exc_info=True)
                if target.exists():
                    try:
                        target.unlink()
                    except:
                        pass
                raise
