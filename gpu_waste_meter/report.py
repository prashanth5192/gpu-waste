"""Render gpu-waste.md from the JSONL evidence of one job.

The report is derived purely from evidence files and can always be regenerated.
Sections without data say `not measured` rather than guessing.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from ._store import iter_job_events, root_dir

NM = "not measured"
NM_ACCOUNTING = "not measured (accounting pending)"
NOT_MEASURED_SOURCES = [
    ("XID / GPU hardware errors", "no NVIDIA XID collection yet"),
    ("NIC errors", "no NIC counter collection yet"),
    ("Fabric / network", "no fabric telemetry yet"),
    ("Checkpoint save / load time", "no checkpoint hooks yet"),
    ("Dataloader stalls", "no dataloader hooks yet"),
]


# --------------------------------------------------------------------------- helpers

def latest_job() -> str | None:
    """Most recently modified job directory under the evidence root."""
    try:
        jobs = [p for p in root_dir().iterdir() if p.is_dir()]
    except OSError:
        return None
    if not jobs:
        return None
    return max(jobs, key=lambda p: p.stat().st_mtime).name


def _ts(t: float | None) -> str:
    if t is None:
        return "—"
    return _dt.datetime.fromtimestamp(t, _dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _dur(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def _cell(text: str, limit: int = 80) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _gpu_count(events: list[dict], override: int | None) -> tuple[int | None, str]:
    if override:
        return override, "--gpus"
    worlds = [e.get("world_size") for e in events if isinstance(e.get("world_size"), int)]
    if worlds:
        return max(worlds), "world_size"
    ranks = {e.get("rank") for e in events if isinstance(e.get("rank"), int)}
    if ranks:
        return len(ranks), "ranks reporting"
    return None, ""


# --------------------------------------------------------------------------- model

def _attempts(events: list[dict]) -> list[dict]:
    by: dict[int, dict] = {}
    for e in events:
        a = e.get("attempt")
        if not isinstance(a, int):
            continue
        row = by.setdefault(a, {"attempt": a, "start": None, "end": None, "exit": None})
        ts = e.get("ts")
        if e.get("type") == "attempt_start" and isinstance(ts, (int, float)):
            row["start"] = ts if row["start"] is None else min(row["start"], ts)
        elif e.get("type") == "child_exit":
            row["exit"] = e.get("data") or {}
            row["end"] = ts
    return [by[k] for k in sorted(by)]


def _outcome(row: dict) -> str:
    ex = row["exit"]
    if ex is None:
        return "unknown (no exit recorded)"
    if ex.get("signal"):
        return f"failed (signal {ex['signal']})"
    if ex.get("exit_code") == 0:
        return "ok"
    return f"failed (exit {ex.get('exit_code')})"


def _last_stderr(row: dict) -> str:
    ex = row["exit"] or {}
    if ex.get("error"):
        return ex["error"]
    tail = [l for l in ex.get("stderr_tail") or [] if l.strip()]
    return tail[-1] if tail else ""


# --------------------------------------------------------------------------- render

def render(job: str, gpus: int | None = None) -> str:
    events = [e for e in iter_job_events(job) if isinstance(e.get("ts"), (int, float))]
    events.sort(key=lambda e: e["ts"])
    out: list[str] = [f"# GPU waste report — `{job}`", ""]

    # Allocation
    out += ["## Allocation", ""]
    n_gpus, gpu_src = _gpu_count(events, gpus)
    if events:
        first, last = events[0]["ts"], events[-1]["ts"]
        wall = last - first
        out.append(f"- Window: {_ts(first)} → {_ts(last)} ({_dur(wall)})")
        if n_gpus:
            out.append(f"- GPUs: {n_gpus} (from {gpu_src})")
            out.append(f"- Allocated: **{n_gpus * wall / 3600:.2f} GPU-hours**")
        else:
            out.append(f"- GPUs: {NM} (pass `--gpus N`)")
            out.append(f"- Allocated GPU-hours: {NM}")
    else:
        out.append(f"{NM} (no evidence found)")
    out.append("")

    # Loss sections
    for title in ("Confirmed loss", "Attributed loss", "Potential (not counted as loss)"):
        out += [f"## {title}", "", NM_ACCOUNTING, ""]

    # Attempts
    out += ["## Attempts", ""]
    rows = _attempts(events)
    if rows:
        failed = sum(1 for r in rows if _outcome(r).startswith("failed"))
        out.append(f"{len(rows)} attempts, {failed} failed.")
        out.append("")
        out.append("| # | Start | Duration | Outcome | OOM kills | Last stderr line |")
        out.append("|---|---|---|---|---|---|")
        for r in rows:
            ex = r["exit"] or {}
            dur = ex.get("duration_s")
            if dur is None and r["start"] is not None and r["end"] is not None:
                dur = r["end"] - r["start"]
            oom = ex.get("oom_kill_delta")
            out.append("| {} | {} | {} | {} | {} | {} |".format(
                r["attempt"], _ts(r["start"]), _dur(dur), _outcome(r),
                "—" if oom is None else oom, _cell(_last_stderr(r)) or "—"))
    else:
        out.append(NM)
    out.append("")

    # Recurring patterns
    out += ["## Recurring patterns", "", NM_ACCOUNTING, ""]

    # Coverage
    ranks = sorted({e["rank"] for e in events if isinstance(e.get("rank"), int)})
    has_wrapper = any(e.get("source") == "wrapper" for e in events)
    has_step = any(e.get("type") == "step" for e in events)
    out += ["## Coverage", ""]
    out.append(f"- Wrapper (`gpu-waste run`): {'yes' if has_wrapper else 'no'}")
    out.append(f"- Ranks reporting: {len(ranks) if ranks else 'none (in-process hooks not active)'}")
    out.append(f"- Collectives observed: {NM}")
    out.append(f"- `step()` present: {'yes' if has_step else 'no'}")
    out.append("")

    # NOT MEASURED
    out += ["## NOT MEASURED", "", "> These sources of waste are **not** included in any number above:", ">"]
    for name, why in NOT_MEASURED_SOURCES:
        out.append(f"> - **{name}** — {why}")
    out.append("")

    return "\n".join(out)


def write_report(job: str, path: Path | str = "gpu-waste.md", gpus: int | None = None) -> Path:
    path = Path(path)
    path.write_text(render(job, gpus=gpus))
    return path
