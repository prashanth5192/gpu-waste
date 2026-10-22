"""Render gpu-waste.md from the JSONL evidence of one job.

The report is derived purely from evidence files and can always be regenerated.
Sections without data say `not measured` rather than guessing.
"""

from __future__ import annotations

import datetime as _dt
import re
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
    worlds = [w for e in events
              for w in (e.get("world_size"), (e.get("data") or {}).get("world_size"))
              if isinstance(w, int)]
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
        row = by.setdefault(a, {"attempt": a, "start": None, "end": None, "exit": None,
                                "exc": [], "oom": False, "sigterm": False, "ended": False,
                                "rank_started": set(), "rank_ended": set()})
        ts = e.get("ts")
        t = e.get("type")
        data = e.get("data") or {}
        rank = e.get("rank") if e.get("source") == "inproc" else None
        if t == "attempt_start" and isinstance(ts, (int, float)):
            row["start"] = ts if row["start"] is None else min(row["start"], ts)
            if isinstance(rank, int):
                row["rank_started"].add(rank)
        elif t == "child_exit":
            row["exit"] = data
            row["end"] = ts
        elif t == "attempt_end":
            row["ended"] = True
            if isinstance(rank, int):
                row["rank_ended"].add(rank)
            if row["exit"] is None:
                row["end"] = ts if row["end"] is None else max(row["end"], ts)
        elif t == "exception" and data.get("exc_type"):
            if data["exc_type"] not in row["exc"]:
                row["exc"].append(data["exc_type"])
        elif t == "cuda_oom":
            row["oom"] = True
        elif t == "sigterm":
            row["sigterm"] = True
    return [by[k] for k in sorted(by)]


def _cause(row: dict) -> str:
    """In-process evidence explaining a failure, if any.

    A rank that started but never wrote attempt_end died without running any
    Python cleanup (SIGKILL, kernel OOM killer, segfault). In a multi-rank
    attempt that is almost always the root cause; other ranks' errors are
    collateral (peers losing their connection).
    """
    if row["oom"]:
        return "CUDA OOM"
    vanished = sorted(row["rank_started"] - row["rank_ended"])
    if vanished and len(vanished) < len(row["rank_started"]):
        return "rank " + ",".join(map(str, vanished)) + " died without exit (SIGKILL / OOM-killer / crash?)"
    if row["exc"]:
        return row["exc"][0].rsplit(".", 1)[-1]
    if row["sigterm"]:
        return "SIGTERM"
    return ""


def _outcome(row: dict) -> str:
    ex = row["exit"]
    cause = _cause(row)
    if ex is None:  # no wrapper: fall back to in-process evidence
        if cause:
            return f"failed ({cause})"
        return "ok (in-process)" if row["ended"] else "unknown (no exit recorded)"
    if ex.get("signal"):
        base = f"failed (signal {ex['signal']})"
    elif ex.get("exit_code") == 0:
        return "ok"
    else:
        base = f"failed (exit {ex.get('exit_code')})"
    return f"{base} · {cause}" if cause else base


_RANK_PREFIX = re.compile(r"^\[rank\d+\]:\s?")
_EXC_LINE = re.compile(r"^[\w.]*(Error|Exception|Interrupt)\b")
_LAUNCHER_NOISE = ("ChildFailedError",)


def _key_stderr(row: dict) -> str:
    """Most informative stderr line: the FIRST real exception line in the tail
    (in distributed jobs later errors are usually collateral, e.g. peers losing
    their connection; launcher wrappers like torchrun's ChildFailedError are
    skipped), else the last non-decoration line."""
    ex = row["exit"] or {}
    if ex.get("error"):
        return ex["error"]
    tail = ex.get("stderr_tail") or []
    for line in tail:
        body = _RANK_PREFIX.sub("", line.strip())
        if _EXC_LINE.match(body) and not any(n in body for n in _LAUNCHER_NOISE):
            return line.strip()
    for line in reversed(tail):
        if re.search(r"[A-Za-z0-9]", line):
            return line.strip()
    return ""


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
        out.append("| # | Start | Duration | Outcome | OOM kills | First error (stderr) |")
        out.append("|---|---|---|---|---|---|")
        for r in rows:
            ex = r["exit"] or {}
            dur = ex.get("duration_s")
            if dur is None and r["start"] is not None and r["end"] is not None:
                dur = r["end"] - r["start"]
            oom = ex.get("oom_kill_delta")
            out.append("| {} | {} | {} | {} | {} | {} |".format(
                r["attempt"], _ts(r["start"]), _dur(dur), _outcome(r),
                "—" if oom is None else oom, _cell(_key_stderr(r)) or "—"))
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
