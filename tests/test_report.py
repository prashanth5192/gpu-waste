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


def test_cli_writes_file(home, tmp_path):
    out = tmp_path / "out.md"
    env = dict(os.environ, GWM_HOME=str(home), PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-m", "gpu_waste_meter.cli", "report",
                        "--gpus", "8", "-o", str(out)], env=env, capture_output=True)
    assert r.returncode == 0, r.stderr
    assert out.read_text() == SNAPSHOT.read_text()   # picks the latest job by default
