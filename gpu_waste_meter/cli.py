"""`gpu-waste` command line.

    gpu-waste run -- <cmd...>     run a command, record attempt_start / child_exit
"""

from __future__ import annotations

import argparse
import collections
import os
import signal
import subprocess
import sys
import threading
import time

from . import _env
from ._store import EventWriter, attempt_dir, root_dir, _safe

STDERR_TAIL_LINES = 200
CGROUP_MEMORY_EVENTS = "/sys/fs/cgroup/memory.events"


# --------------------------------------------------------------------------- helpers

def _read_oom_kill(path: str = CGROUP_MEMORY_EVENTS) -> int | None:
    """Current cgroup `oom_kill` counter, or None if unreadable."""
    try:
        with open(path) as f:
            for line in f:
                key, _, val = line.partition(" ")
                if key == "oom_kill":
                    return int(val)
    except (OSError, ValueError):
        pass
    return None


def _next_attempt(job: str) -> int:
    """Wrapper-assigned attempt counter: one past the highest existing attempt dir."""
    base = root_dir() / _safe(job)
    try:
        nums = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    except OSError:
        nums = []
    return max(nums) + 1 if nums else 0


def _resolve_attempt(job: str) -> int:
    for var in ("GWM_ATTEMPT", "TORCHELASTIC_RESTART_COUNT"):
        val = os.environ.get(var)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return _next_attempt(job)


def _tee(src, dst, tail: collections.deque) -> None:
    """Copy child stderr to our stderr line by line, keeping the last N lines."""
    try:
        for raw in iter(src.readline, b""):
            try:
                dst.write(raw)
                dst.flush()
            except Exception:
                pass
            tail.append(raw.decode("utf-8", "replace").rstrip("\n"))
    except Exception:
        pass
    finally:
        try:
            src.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- run

def cmd_run(argv: list[str]) -> int:
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        print("usage: gpu-waste run -- <command> [args...]", file=sys.stderr)
        return 2

    job = _env.job_id()
    attempt = _resolve_attempt(job)
    ident = _env.get_identity()
    writer = EventWriter(attempt_dir(job, attempt) / "wrapper.jsonl")

    def emit(type_: str, confidence: str, data: dict) -> None:
        writer.write({
            "ts": time.time(), "job": job, "attempt": attempt, "rank": None,
            "node": ident["node"], "type": type_, "confidence": confidence,
            "source": "wrapper", "data": data,
        })

    child_env = dict(os.environ, GWM_JOB=job, GWM_ATTEMPT=str(attempt))
    oom_before = _read_oom_kill()
    t0 = time.time()

    try:
        proc = subprocess.Popen(argv, env=child_env, stderr=subprocess.PIPE)
    except OSError as e:
        emit("attempt_start", "context", {"argv": argv, "pid": None})
        emit("child_exit", "confirmed", {
            "exit_code": 127, "signal": None, "error": f"{type(e).__name__}: {e}",
            "stderr_tail": [], "oom_kill_delta": None, "duration_s": 0.0,
        })
        writer.close()
        print(f"gpu-waste: cannot run {argv[0]!r}: {e}", file=sys.stderr)
        return 127

    emit("attempt_start", "context", {"argv": argv, "pid": proc.pid})

    tail: collections.deque = collections.deque(maxlen=STDERR_TAIL_LINES)
    out = getattr(sys.stderr, "buffer", sys.stderr)
    tee = threading.Thread(target=_tee, args=(proc.stderr, out, tail), daemon=True)
    tee.start()

    # Forward SIGTERM / SIGINT to the child; restore our previous handlers after.
    def forward(signum, _frame):
        try:
            proc.send_signal(signum)
        except Exception:
            pass

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[sig] = signal.signal(sig, forward)
        except (ValueError, OSError):  # not main thread / unsupported
            pass

    try:
        while True:
            try:
                rc = proc.wait()
                break
            except KeyboardInterrupt:  # SIGINT raced the handler install
                forward(signal.SIGINT, None)
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    tee.join(timeout=5)
    oom_after = _read_oom_kill()
    oom_delta = (oom_after - oom_before) if None not in (oom_before, oom_after) else None

    sig = -rc if rc < 0 else None
    emit("child_exit", "context" if rc == 0 else "confirmed", {
        "exit_code": None if sig else rc,
        "signal": sig,
        "stderr_tail": list(tail),
        "oom_kill_delta": oom_delta,
        "duration_s": round(time.time() - t0, 3),
    })
    writer.close()
    return 128 + sig if sig else rc


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="gpu-waste")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run", help="run a command and record its lifecycle",
                   usage="gpu-waste run -- <command> [args...]")

    if argv and argv[0] == "run":
        return cmd_run(argv[1:])
    parser.parse_args(argv)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
