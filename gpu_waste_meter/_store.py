"""Append-only JSONL evidence store.

Each event is written as one complete line with a single os.write() call, so a
crash (even SIGKILL) can at worst leave one partial trailing line. Data is in
the OS page cache immediately and survives process death; fsync (for
power-loss / node-crash durability) happens at most every `fsync_interval`
seconds and on close().
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Iterator

ROOT_ENV = "GWM_HOME"


def root_dir() -> Path:
    return Path(os.environ.get(ROOT_ENV) or Path.home() / ".gpu_waste")


def attempt_dir(job: str, attempt: int) -> Path:
    return root_dir() / _safe(job) / str(attempt)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:128]


class EventWriter:
    def __init__(self, path: Path | str, fsync_interval: float = 5.0):
        self.path = Path(path)
        self.fsync_interval = fsync_interval
        self._lock = threading.Lock()
        self._fd: int | None = None
        self._last_sync = time.monotonic()
        self.broken = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        except OSError:
            self.broken = True

    def write(self, event: dict) -> None:
        if self.broken or self._fd is None:
            return
        try:
            event.setdefault("ts", time.time())
            line = (json.dumps(event, separators=(",", ":"), default=str) + "\n").encode()
            with self._lock:
                os.write(self._fd, line)
                now = time.monotonic()
                if now - self._last_sync >= self.fsync_interval:
                    os.fsync(self._fd)
                    self._last_sync = now
        except Exception:
            self.broken = True

    def close(self) -> None:
        with self._lock:
            if self._fd is None:
                return
            try:
                os.fsync(self._fd)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


def read_events(path: Path | str) -> Iterator[dict]:
    """Yield parsed events, skipping blank or corrupt (e.g. truncated) lines."""
    try:
        with open(path, "rb") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except OSError:
        return


def iter_job_events(job: str) -> Iterator[dict]:
    """All events for a job across attempts and files."""
    base = root_dir() / _safe(job)
    if not base.is_dir():
        return
    for p in sorted(base.rglob("*.jsonl")):
        yield from read_events(p)
