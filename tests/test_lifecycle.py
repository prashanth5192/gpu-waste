import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from gpu_waste_meter._store import read_events

ROOT = Path(__file__).resolve().parents[1]


def run_script(tmp_path, body, *, env_extra=None, wait=True):
    script = tmp_path / "train.py"
    script.write_text(textwrap.dedent(body))
    env = dict(os.environ, GWM_HOME=str(tmp_path / "home"), GWM_JOB="lc", GWM_ATTEMPT="0",
               PYTHONPATH=str(ROOT), GWM_NVML="0", RANK="0", LOCAL_RANK="0", WORLD_SIZE="2")
    env.pop("GWM_DISABLE", None)
    env.update(env_extra or {})
    p = subprocess.Popen([sys.executable, str(script)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if wait:
        p.communicate(timeout=30)
    return p


def events(tmp_path, rank=0):
    return list(read_events(tmp_path / "home" / "lc" / "0" / f"rank{rank}.jsonl"))


def types(evs):
    return [e["type"] for e in evs]


def test_clean_exit(tmp_path):
    p = run_script(tmp_path, "import gpu_waste_meter\nprint('ok')\n")
    assert p.returncode == 0
    evs = events(tmp_path)
    assert types(evs) == ["attempt_start", "attempt_end"]
    assert evs[0]["source"] == "inproc" and evs[0]["rank"] == 0
    assert evs[0]["data"]["world_size"] == 2
    assert evs[-1]["data"] == {"reason": "exit", "failed": False}


def test_value_error_records_type_only(tmp_path):
    p = run_script(tmp_path, """
        import gpu_waste_meter
        raise ValueError("secret message must not be recorded")
    """)
    assert p.returncode == 1
    evs = events(tmp_path)
    assert types(evs) == ["attempt_start", "exception", "attempt_end"]
    assert evs[1]["data"]["exc_type"] == "ValueError"
    assert "secret" not in (tmp_path / "home" / "lc" / "0" / "rank0.jsonl").read_text()
    assert evs[-1]["data"]["failed"] is True


def test_fake_cuda_oom(tmp_path):
    p = run_script(tmp_path, """
        import gpu_waste_meter
        class OutOfMemoryError(RuntimeError):
            pass
        OutOfMemoryError.__module__ = "torch.cuda"
        raise OutOfMemoryError("CUDA out of memory")
    """)
    assert p.returncode == 1
    assert types(events(tmp_path)) == ["attempt_start", "exception", "cuda_oom", "attempt_end"]


def test_sigterm_default_still_kills(tmp_path):
    p = run_script(tmp_path, """
        import gpu_waste_meter, sys, time
        print("ready", flush=True)
        time.sleep(30)
    """, wait=False)
    assert p.stdout.readline().strip() == b"ready"
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=30)
    assert p.returncode == -signal.SIGTERM           # default behaviour preserved
    evs = events(tmp_path)
    assert types(evs) == ["attempt_start", "sigterm", "attempt_end"]
    assert evs[-1]["data"]["reason"] == "sigterm"


def test_sigterm_chains_existing_handler(tmp_path):
    p = run_script(tmp_path, """
        import signal, sys, time
        signal.signal(signal.SIGTERM, lambda *a: sys.exit(42))
        import gpu_waste_meter
        print("ready", flush=True)
        time.sleep(30)
    """, wait=False)
    assert p.stdout.readline().strip() == b"ready"
    p.send_signal(signal.SIGTERM)
    p.communicate(timeout=30)
    assert p.returncode == 42                         # user's handler still ran
    assert "sigterm" in types(events(tmp_path))


def test_existing_excepthook_still_runs(tmp_path):
    script = tmp_path / "t.py"
    script.write_text(textwrap.dedent("""
        import sys
        sys.excepthook = lambda t, v, tb: print("MY HOOK RAN", t.__name__, flush=True)
        import gpu_waste_meter
        raise KeyError("x")
    """))
    env = dict(os.environ, GWM_HOME=str(tmp_path / "home"), GWM_JOB="lc", GWM_ATTEMPT="0",
               PYTHONPATH=str(ROOT), GWM_NVML="0", RANK="0")
    env.pop("GWM_DISABLE", None)
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, timeout=30)
    assert b"MY HOOK RAN KeyError" in r.stdout
    assert "exception" in types(events(tmp_path))


def test_thread_exception(tmp_path):
    p = run_script(tmp_path, """
        import gpu_waste_meter, threading
        t = threading.Thread(target=lambda: 1 / 0, name="worker")
        t.start(); t.join()
    """)
    evs = [e for e in events(tmp_path) if e["type"] == "exception"]
    assert evs and evs[0]["data"] == {"exc_type": "ZeroDivisionError", "where": "thread:worker"}


def test_heartbeat(tmp_path):
    p = run_script(tmp_path, "import gpu_waste_meter, time\ntime.sleep(0.5)\n",
                   env_extra={"GWM_HEARTBEAT_S": "0.1"})
    assert types(events(tmp_path)).count("heartbeat") >= 3


def test_disable(tmp_path):
    p = run_script(tmp_path, "import gpu_waste_meter\n", env_extra={"GWM_DISABLE": "1"})
    assert p.returncode == 0
    assert not (tmp_path / "home").exists()


def test_rank_file_name(tmp_path):
    run_script(tmp_path, "import gpu_waste_meter\n", env_extra={"RANK": "5", "LOCAL_RANK": "1"})
    assert types(events(tmp_path, rank=5))[0] == "attempt_start"


def test_wrapper_and_inproc_share_attempt_dir(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("import gpu_waste_meter\n")
    env = dict(os.environ, GWM_HOME=str(tmp_path / "home"), GWM_JOB="lc",
               PYTHONPATH=str(ROOT), GWM_NVML="0")
    for v in ("GWM_DISABLE", "GWM_ATTEMPT", "RANK", "LOCAL_RANK", "TORCHELASTIC_RESTART_COUNT"):
        env.pop(v, None)
    r = subprocess.run([sys.executable, "-m", "gpu_waste_meter.cli", "run", "--",
                        sys.executable, str(script)], env=env, timeout=30)
    assert r.returncode == 0
    d = tmp_path / "home" / "lc" / "0"
    assert sorted(p.name for p in d.iterdir()) == ["rank0.jsonl", "wrapper.jsonl"]


def test_cli_process_is_not_instrumented(tmp_path):
    env = dict(os.environ, GWM_HOME=str(tmp_path / "home"), GWM_JOB="lc",
               PYTHONPATH=str(ROOT), GWM_NVML="0")
    env.pop("GWM_DISABLE", None)
    r = subprocess.run([sys.executable, "-m", "gpu_waste_meter.cli", "run", "--",
                        sys.executable, "-c", "pass"], env=env, timeout=30)
    assert r.returncode == 0
    files = [p.name for p in (tmp_path / "home").rglob("*.jsonl")]
    assert files == ["wrapper.jsonl"]
