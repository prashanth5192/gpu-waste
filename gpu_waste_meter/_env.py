"""Job / attempt / rank identity from the environment. Never raises."""

from __future__ import annotations

import hashlib
import os
import socket
import sys


def _int(name: str, default: int | None = None) -> int | None:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def job_id() -> str:
    """GWM_JOB > TORCHELASTIC_RUN_ID > SLURM_JOB_ID > hash(argv + cwd)."""
    for var in ("GWM_JOB", "TORCHELASTIC_RUN_ID", "SLURM_JOB_ID"):
        val = os.environ.get(var)
        if val:
            return val
    try:
        seed = " ".join(sys.argv) + "|" + os.getcwd()
    except Exception:
        seed = "unknown"
    return "job-" + hashlib.sha1(seed.encode()).hexdigest()[:12]


def attempt_id() -> int:
    """GWM_ATTEMPT (set by `gpu-waste run`) > TORCHELASTIC_RESTART_COUNT > 0."""
    val = _int("GWM_ATTEMPT")
    if val is None:
        val = _int("TORCHELASTIC_RESTART_COUNT", 0)
    return val or 0


def get_identity() -> dict:
    rank = _int("RANK")
    if rank is None:
        rank = _int("SLURM_PROCID")
    local_rank = _int("LOCAL_RANK")
    if local_rank is None:
        local_rank = _int("SLURM_LOCALID")
    world = _int("WORLD_SIZE")
    if world is None:
        world = _int("SLURM_NTASKS")
    try:
        node = socket.gethostname()
    except Exception:
        node = "unknown"
    return {
        "job": job_id(),
        "attempt": attempt_id(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "node": node,
    }
