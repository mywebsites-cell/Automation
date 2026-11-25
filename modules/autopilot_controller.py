"""Start / stop the autopilot scheduler as a background process.

Used by the Streamlit panel so the "Enable Autopilot" toggle actually launches
(or terminates) the scheduler, instead of only writing a flag that nothing reads.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PID_FILE_NAME = "autopilot.pid"


def pid_path(cache_dir: Path) -> Path:
    return cache_dir / "autopilot" / PID_FILE_NAME


def read_pid(cache_dir: Path) -> int | None:
    p = pid_path(cache_dir)
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def process_alive(pid: int) -> bool:
    """Best-effort check whether a process with the given PID is still running."""
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return bool(ok) and exit_code.value == 259  # STILL_ACTIVE
        except Exception:
            return False
    else:
        try:
            import os

            os.kill(pid, 0)
            return True
        except OSError:
            return False


def is_running(cache_dir: Path) -> bool:
    pid = read_pid(cache_dir)
    return pid is not None and process_alive(pid)


def register(cache_dir: Path) -> None:
    """Record the current process PID (called by the scheduler at startup)."""
    p = pid_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(_current_pid()), encoding="utf-8")


def _current_pid() -> int:
    import os

    return os.getpid()


def start(cache_dir: Path, base_dir: Path) -> int | None:
    """Launch scheduler.py detached (idempotent + safe against concurrent calls).

    Returns the scheduler PID, or the existing PID if already running. A stale PID
    file (pointing at a dead process) is cleared before launch so the new scheduler's
    double-instance guard never sees a phantom "already running" PID.
    """
    if is_running(cache_dir):
        return read_pid(cache_dir)

    p = pid_path(cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.parent / "start.lock"

    # Atomically acquire the start lock so two rapid toggles can't double-launch.
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Another start is in flight — give it a moment, then report current state.
        time.sleep(1.0)
        return read_pid(cache_dir) if is_running(cache_dir) else None

    try:
        os.write(lock_fd, str(_current_pid()).encode())
        # Re-check now that we hold the lock.
        if is_running(cache_dir):
            return read_pid(cache_dir)

        # Clear any stale PID file *before* spawning, so the scheduler can never read
        # a dead value as "already running".
        p.unlink(missing_ok=True)

        scheduler = base_dir / "scheduler.py"
        kwargs: dict = {
            "cwd": str(base_dir),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "stdin": subprocess.DEVNULL,
            "close_fds": True,
        }
        if sys.platform == "win32":
            # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP — survives the Streamlit app closing.
            kwargs["creationflags"] = 0x00000008 | 0x00000200

        logger.info("Launching autopilot scheduler: %s", scheduler)
        proc = subprocess.Popen([sys.executable, str(scheduler)], **kwargs)

        # The venv's python.exe is a redirector: proc.pid is the *launcher*, not the
        # real interpreter. Never trust it. Instead wait for the scheduler to register
        # its own real PID via register().
        deadline = time.time() + 30.0
        while time.time() < deadline:
            pid = read_pid(cache_dir)
            if pid is not None:
                return pid
            if proc.poll() is not None:
                # Launcher exited before the scheduler registered → startup failed.
                return None
            time.sleep(0.2)
        return read_pid(cache_dir)
    finally:
        try:
            os.close(lock_fd)
        except Exception:
            pass
        lock_path.unlink(missing_ok=True)


def stop(cache_dir: Path) -> None:
    """Terminate the autopilot process (if any) and clear its PID file."""
    pid = read_pid(cache_dir)
    if pid is not None and process_alive(pid):
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                import signal

                os.kill(pid, signal.SIGTERM)
        except Exception as exc:
            logger.warning("Failed to stop autopilot PID %s: %s", pid, exc)
    try:
        pid_path(cache_dir).unlink(missing_ok=True)
    except Exception:
        pass
