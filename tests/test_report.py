"""Snapshot test for gpu-waste.md.

Regenerate the snapshot after an intentional change:
    GWM_UPDATE_SNAPSHOT=1 pytest tests/test_report.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gpu_waste_meter import report

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
SNAPSHOT = FIXTURES / "fixture-job.expected.md"


@pytest.fixture
def home(tmp_path, monkeypatch):
    shutil.copytree(FIXTURES / "fixture-job", tmp_path / "fixture-job")
    monkeypatch.setenv("GWM_HOME", str(tmp_path))
    return tmp_path


def test_snapshot(home):
    md = report.render("fixture-job", gpus=8)
    if os.environ.get("GWM_UPDATE_SNAPSHOT"):
        SNAPSHOT.write_text(md)
    assert md == SNAPSHOT.read_text()


def test_every_spec_section_present(home):
    md = report.render("fixture-job")
    for heading in ("Allocation", "Confirmed loss", "Attributed loss", "Potential",
                    "Attempts", "Recurring patterns", "Coverage", "NOT MEASURED"):
        assert f"## {heading}" in md
    assert "GPUs: not measured" in md          # no --gpus, nothing to infer from


def test_missing_job_says_not_measured(home):
    md = report.render("no-such-job", gpus=8)
    assert "not measured (no evidence found)" in md


def test_inproc_events_feed_report(home):
    from gpu_waste_meter._store import EventWriter
    w = EventWriter(home / "ip-job" / "0" / "rank0.jsonl")
    base = {"job": "ip-job", "attempt": 0, "rank": 0, "node": "n", "source": "inproc"}
    w.write({**base, "ts": 100.0, "type": "attempt_start", "data": {"world_size": 4}})
    w.write({**base, "ts": 160.0, "type": "exception", "data": {"exc_type": "torch.cuda.OutOfMemoryError"}})
    w.write({**base, "ts": 160.0, "type": "cuda_oom", "data": {}})
    w.write({**base, "ts": 161.0, "type": "attempt_end", "data": {}})
    w.close()
    md = report.render("ip-job")
    assert "- GPUs: 4 (from world_size)" in md
    assert "| 0 | 1970-01-01 00:01:40 UTC | 1m 01s | failed (CUDA OOM) |" in md
    assert "- Ranks reporting: 1" in md


def test_key_stderr_line_skips_launcher_noise():
    row = {"exit": {"stderr_tail": [
        "[rank1]: Traceback (most recent call last):",
        "[rank1]: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "[rank0]: RuntimeError: Connection closed by peer",
        "torch.distributed.elastic.multiprocessing.errors.ChildFailedError: ",
        "============================================================",
        "  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html",
        "============================================================",
    ]}}
    assert report._key_stderr(row).startswith("[rank1]: torch.OutOfMemoryError")
    row = {"exit": {"stderr_tail": ["step 5 loss=1.2", "=====", ""]}}
    assert report._key_stderr(row) == "step 5 loss=1.2"


def test_vanished_rank_is_root_cause(home):
    from gpu_waste_meter._store import EventWriter
    base = {"job": "vk", "attempt": 0, "node": "n", "source": "inproc"}
    for rank in (0, 1):
        w = EventWriter(home / "vk" / "0" / f"rank{rank}.jsonl")
        w.write({**base, "rank": rank, "ts": 10.0, "type": "attempt_start", "data": {}})
        if rank == 1:  # rank 1 sees collateral error; rank 0 was SIGKILLed (no more events)
            w.write({**base, "rank": 1, "ts": 20.0, "type": "exception", "data": {"exc_type": "RuntimeError"}})
            w.write({**base, "rank": 1, "ts": 20.0, "type": "attempt_end", "data": {}})
        w.close()
    md = report.render("vk")
    assert "failed (rank 0 died without exit" in md


def test_cli_writes_file(home, tmp_path):
    out = tmp_path / "out.md"
    env = dict(os.environ, GWM_HOME=str(home), PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "gpu_waste_meter.cli", "report",
                        "--gpus", "8", "-o", str(out)], env=env, capture_output=True)
    assert r.returncode == 0, r.stderr
    assert out.read_text() == SNAPSHOT.read_text()   # picks the latest job by default
