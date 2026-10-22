import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from gpu_waste_meter._store import read_events

ROOT = Path(__file__).resolve().parents[1]


def gpu_waste(tmp_path, *cmd, **kw):
    env = dict(os.environ, GWM_HOME=str(tmp_path), GWM_JOB="testjob",
               PYTHONPATH=str(ROOT))
    for v in ("GWM_ATTEMPT", "TORCHELASTIC_RESTART_COUNT"):
        env.pop(v, None)
    return subprocess.Popen(
        [sys.executable, "-m", "gpu_waste_meter.cli", "run", "--", *cmd],
        env=env, stderr=subprocess.PIPE, **kw)


def events(tmp_path, attempt=0):
    return list(read_events(tmp_path / "testjob" / str(attempt) / "wrapper.jsonl"))


def exits(evs):
    return [e for e in evs if e["type"] == "child_exit"]


def test_exit_code_propagates(tmp_path):
    p = gpu_waste(tmp_path, sys.executable, "-c", "import sys; sys.exit(3)")
    p.communicate(timeout=30)
    assert p.returncode == 3
    evs = events(tmp_path)
    assert [e["type"] for e in evs] == ["attempt_start", "child_exit"]
    (ex,) = exits(evs)
    assert ex["data"]["exit_code"] == 3 and ex["data"]["signal"] is None
    assert ex["source"] == "wrapper" and ex["job"] == "testjob" and ex["attempt"] == 0


def test_sigkill_recorded(tmp_path):
    p = gpu_waste(tmp_path, sys.executable, "-c", "import os; os.kill(os.getpid(), 9)")
    p.communicate(timeout=30)
    assert p.returncode == 128 + 9
    (ex,) = exits(events(tmp_path))
    assert ex["data"]["signal"] == 9 and ex["data"]["exit_code"] is None


def test_stderr_tee_and_tail(tmp_path):
    code = "import sys\nfor i in range(300): print('line', i, file=sys.stderr)\nsys.exit(1)"
    p = gpu_waste(tmp_path, sys.executable, "-c", code)
    _, err = p.communicate(timeout=30)
    assert err.decode().count("line ") == 300          # still shown to the user
    tail = exits(events(tmp_path))[0]["data"]["stderr_tail"]
    assert len(tail) == 200 and tail[0] == "line 100" and tail[-1] == "line 299"


def test_sigterm_forwarded(tmp_path):
    code = ("import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *a: sys.exit(7))\n"
            "print('ready', file=sys.stderr, flush=True)\n"
            "time.sleep(30)")
    p = gpu_waste(tmp_path, sys.executable, "-c", code)
    assert p.stderr.readline().strip() == b"ready"
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=30)
    assert p.returncode == 7
    assert exits(events(tmp_path))[0]["data"]["exit_code"] == 7


def test_attempt_counter_and_child_env(tmp_path):
    code = "import os, sys; sys.exit(int(os.environ['GWM_ATTEMPT']) + 10)"
    for expected in (0, 1):
        p = gpu_waste(tmp_path, sys.executable, "-c", code)
        p.communicate(timeout=30)
        assert p.returncode == 10 + expected
        assert exits(events(tmp_path, expected))[0]["attempt"] == expected


def test_missing_command_never_crashes(tmp_path):
    p = gpu_waste(tmp_path, "definitely-not-a-real-command-xyz")
    p.communicate(timeout=30)
    assert p.returncode == 127
    assert exits(events(tmp_path))[0]["data"]["exit_code"] == 127
