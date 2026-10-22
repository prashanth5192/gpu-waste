import json
import os
import signal
import subprocess
import sys
import textwrap
import time

from gpu_waste_meter._store import EventWriter, read_events


def test_roundtrip(tmp_path):
    p = tmp_path / "a" / "rank0.jsonl"
    w = EventWriter(p)
    for i in range(10):
        w.write({"type": "heartbeat", "i": i})
    w.close()
    evs = list(read_events(p))
    assert [e["i"] for e in evs] == list(range(10))
    assert all("ts" in e for e in evs)


def test_reader_skips_corrupt_lines(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\nnot json\n\n{"a":2}\n{"a":3}')  # valid last line, no trailing newline
    assert [e["a"] for e in read_events(p)] == [1, 2, 3]
    p.write_text('{"a":1}\n{"a":')  # truncated trailing line
    assert [e["a"] for e in read_events(p)] == [1]


def test_survives_sigkill(tmp_path):
    p = tmp_path / "k.jsonl"
    child = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {os.getcwd()!r})
        from gpu_waste_meter._store import EventWriter
        w = EventWriter({str(p)!r})
        i = 0
        while True:
            w.write({{"type": "heartbeat", "i": i, "pad": "x" * 200}})
            i += 1
    """)
    proc = subprocess.Popen([sys.executable, "-c", child])
    deadline = time.time() + 10
    while time.time() < deadline and (not p.exists() or p.stat().st_size < 50_000):
        time.sleep(0.05)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    lines = p.read_bytes().split(b"\n")
    complete = [l for l in lines[:-1] if l]
    assert len(complete) > 100
    for l in complete:  # every complete line is valid JSON
        json.loads(l)
    evs = list(read_events(p))
    assert [e["i"] for e in evs] == list(range(len(evs)))  # no gaps, no reordering


def test_unwritable_path_never_raises(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    w = EventWriter(ro / "sub" / "x.jsonl")
    w.write({"type": "x"})  # must not raise
    w.close()
    if os.geteuid() != 0:  # root ignores permissions
        assert w.broken
