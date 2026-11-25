import gc
import shutil
from pathlib import Path


def cleanup_temp_directory(temp_dir: Path) -> None:
    if not temp_dir.exists():
        return
    for child in temp_dir.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
        except Exception:
            continue


def force_memory_cleanup() -> None:
    gc.collect()
