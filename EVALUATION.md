# Evaluating gpu_waste_meter v0 (T1–T5)

A hands-on checklist to verify each piece yourself. **Parts 0–6 run on a laptop
(macOS or Linux), no GPU needed.** Part 7 needs a GPU box.

Each check lists the command, what you should see, and what it proves.
Time: ~20 min for Parts 0–6.

---

## Part 0 · Setup (2 min)

```bash
git clone https://github.com/prashanth5192/gpu-waste && cd gpu-waste
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'
pip install torch            # optional: only for Part 6 (torchrun); ~200 MB
export GWM_HOME=/tmp/gwm-eval   # keep evaluation data out of ~/.gpu_waste
```

**Check:** `gpu-waste --help` lists `run` and `report`.

## Part 1 · Full test suite (1 min)

```bash
python -m pytest -q
```

**Expect:** `41 passed`. No GPU or torch needed.

| Test file | Task | What it proves |
|---|---|---|
| `test_env.py`, `test_store.py` | T1 | job/attempt/rank identity; JSONL survives `kill -9` mid-write |
| `test_run.py` | T2 | wrapper exit codes, signals, stderr tee, SIGTERM forwarding |
| `test_report.py` | T3 | report snapshot; every SPEC section present; root-cause heuristics |
| `test_nvml.py` | T4 | 120 mocked samples → 2 per-minute events; `CUDA_VISIBLE_DEVICES` |
| `test_lifecycle.py` | T5 | in-process events; chained hooks; `GWM_DISABLE` |

---

## Part 2 · T1 — crash-safe evidence store

The claim: evidence survives the process being killed at any moment.

```bash
python - <<'EOF'
import subprocess, sys, time, signal, json
p = subprocess.Popen([sys.executable, "-c",
  "from gpu_waste_meter._store import EventWriter\n"
  "w = EventWriter('/tmp/gwm-eval/crash.jsonl')\n"
  "i = 0\n"
  "while True: w.write({'i': i}); i += 1"])
time.sleep(1); p.send_signal(signal.SIGKILL); p.wait()
lines = open('/tmp/gwm-eval/crash.jsonl','rb').read().split(b'\n')
ok = bad = 0
for l in lines[:-1]:
    try: json.loads(l); ok += 1
    except ValueError: bad += 1
print(f"complete lines: {ok}, corrupt: {bad}, trailing partial: {lines[-1][:30]!r}")
EOF
```

**Expect:** tens of thousands of complete lines, `corrupt: 0`. At most one partial
trailing line (the reader skips it).

## Part 3 · T2 — `gpu-waste run`

```bash
gpu-waste run -- python3 -c "import sys; print('boom', file=sys.stderr); sys.exit(3)"
echo "exit=$?"                       # → boom, then exit=3
cat $GWM_HOME/*/*/wrapper.jsonl | python3 -m json.tool --json-lines | grep -E '"type"|exit_code|signal|stderr'
```

**Expect:** `attempt_start` then `child_exit` with `"exit_code": 3` and
`"stderr_tail": ["boom"]`. The `boom` still printed to your terminal (tee).

Signals:

```bash
gpu-waste run -- python3 -c "import os; os.kill(os.getpid(), 9)"; echo "exit=$?"   # → 137 (128+9), "signal": 9
gpu-waste run -- sleep 60 &  sleep 1; kill -TERM $!; wait $!; echo "exit=$?"        # → 143: SIGTERM forwarded to sleep
gpu-waste run -- no-such-command; echo "exit=$?"                                    # → 127, no crash
```

Note: `OOM kills` (cgroup `memory.events`) is Linux-only; on macOS it shows `—`.

## Part 4 · T3 — `gpu-waste report`

Render the checked-in fixture (3 attempts, 2 failures, includes a corrupt and a
truncated line):

```bash
GWM_HOME=tests/fixtures gpu-waste report --job fixture-job --gpus 8 -o /tmp/fixture.md
diff /tmp/fixture.md tests/fixtures/fixture-job.expected.md && echo SAME
```

**Expect:** `SAME`. Open `/tmp/fixture.md` and check by hand:

- Allocation: 8 GPUs × 4h17m30s = **34.33 GPU-hours** (window = first → last valid event)
- Attempts: exit 1 / signal 9 (OOM kills 1) / ok
- Confirmed / Attributed / Potential / Recurring: `not measured (accounting pending)` — by design for v0
- NOT MEASURED box lists XID, NIC, fabric, checkpoint, dataloader

Without `--gpus`, GPUs and GPU-hours say `not measured` (the wrapper alone can't
know the GPU count — T5 fixes this via `world_size`).

## Part 5 · T4 — NVML sampler (no GPU: degradation only)

```bash
pip install nvidia-ml-py
python scripts/nvml_overhead.py 5
```

**Expect on a laptop:** `NVML unavailable: … NVML Shared Library Not Found`,
exit 1, no traceback. That is the required behaviour: one `context` event, then
stop. The real sampling + overhead check is Part 7.

## Part 6 · T5 — in-process hooks

Single process, each failure mode:

```bash
for f in none exception oom kill sigterm; do
  GWM_JOB=t5-$f gpu-waste run -- python3 examples/flaky_train.py --steps 6 --at-step 3 --fail $f 2>/dev/null
  echo "$f → exit $?"
  ls $GWM_HOME/t5-$f/0/
done
```

**Expect:** each attempt folder has `wrapper.jsonl` **and** `rank0.jsonl`.
Inspect one:

```bash
grep -o '"type":"[a-z_]*"' $GWM_HOME/t5-oom/0/rank0.jsonl
# attempt_start, exception, cuda_oom, attempt_end
```

On a machine without an NVIDIA GPU, each `rank0.jsonl` also has one `context`
event (`"nvml unavailable"`) right after `attempt_start` — expected.

| `--fail` | rank0.jsonl ends with | why |
|---|---|---|
| none | `attempt_end` | clean exit |
| exception | `exception`, `attempt_end` | type name only: `grep -c "simulated training crash" $GWM_HOME/t5-exception/0/rank0.jsonl` → **0** |
| oom | `exception`, `cuda_oom`, `attempt_end` | detects `torch.cuda.OutOfMemoryError` by class |
| kill | `attempt_start` only | SIGKILL leaves no chance to write — that absence *is* the signal |
| sigterm | `sigterm`, `attempt_end` | process still dies with SIGTERM (exit 143) |

Safety checks:

```bash
GWM_DISABLE=1 GWM_JOB=off python3 examples/flaky_train.py --steps 2; ls $GWM_HOME/off 2>&1   # → No such file
python3 -c "
import sys, signal
signal.signal(signal.SIGTERM, lambda *a: print('MY HANDLER'))
sys.excepthook = lambda *a: print('MY HOOK')
import gpu_waste_meter, os
os.kill(os.getpid(), signal.SIGTERM)
raise ValueError" 2>/dev/null
# → MY HANDLER, MY HOOK   (your handlers still run; we chain, never replace)
```

### Multi-rank with torchrun (needs `pip install torch`; CPU/gloo is fine)

The realistic scenario: one job, three attempts, two failures.

```bash
export GWM_JOB=demo
gpu-waste run -- torchrun --nproc-per-node=2 examples/flaky_train.py --steps 8 --at-step 3 --fail kill --fail-rank 0
gpu-waste run -- torchrun --nproc-per-node=2 examples/flaky_train.py --steps 8 --at-step 3 --fail oom
gpu-waste run -- torchrun --nproc-per-node=2 examples/flaky_train.py --steps 8
gpu-waste report && cat gpu-waste.md
```

**Expect in the Attempts table:**

| # | Outcome | First error |
|---|---|---|
| 0 | `failed (exit 1) · rank 0 died without exit (SIGKILL / OOM-killer / crash?)` | rank 1's gloo error (collateral) |
| 1 | `failed (exit 1) · CUDA OOM` | `[rank…]: torch.OutOfMemoryError: CUDA out of memory…` |
| 2 | `ok` | — |

…and `GPUs: 2 (from world_size)`, `Ranks reporting: 2`. Attempt 0 is the
interesting one: the surviving rank's error is *not* blamed — the rank that
vanished is.

Preemption (SIGTERM to the wrapper → torchrun → every rank):

```bash
GWM_JOB=preempt gpu-waste run -- torchrun --nproc-per-node=2 examples/flaky_train.py --steps 100 &
sleep 5; kill -TERM $!; wait
grep -c '"type":"sigterm"' $GWM_HOME/preempt/0/rank*.jsonl     # → 1 per rank
```

---

## Part 7 · On a GPU box (Linux + NVIDIA driver)

The two checks that need real NVIDIA GPUs.

**7a · NVML overhead (T4 manual check)**

```bash
pip install -e '.[nvml]'
python scripts/nvml_overhead.py 60
```

**Expect:** `sampling N GPU(s)`, `… % of one core`, **`PASS`** (< 1%).
Then `CUDA_VISIBLE_DEVICES=0 python scripts/nvml_overhead.py 10` → `sampling 1 GPU(s)`.

**7b · Real 2-GPU run (TASKS.md "STOP — bring to Claude for review")**

Add `import gpu_waste_meter` as the first line of a real training script, then:

```bash
gpu-waste run -- torchrun --nproc-per-node=2 train.py
gpu-waste report
```

**Expect:** `nvml_sample` events in `rank0.jsonl` only (one sampler per node),
one per GPU per minute, with plausible util / memory / power. Bring back for
review: `_store.py`, `_lifecycle.py`, the `pytest` output, and this `gpu-waste.md`.

---

## Known limitations (v0, by design)

- **No loss accounting yet.** Confirmed / Attributed / Potential print
  `not measured (accounting pending)` — that is the post-review phase.
- **In-process hooks need `import gpu_waste_meter`** in the training script.
  Without it you still get wrapper-level attempts, exit codes and stderr.
- **GPU count** comes from `world_size` (T5) or `--gpus`; wrapper-only runs need `--gpus`.
- **cgroup OOM-kill counter** is Linux-only.
- **SIGKILL is invisible in-process** by nature; it is inferred from a rank that
  started but never wrote `attempt_end`.

## Where the data lives

```
$GWM_HOME (default ~/.gpu_waste)/<job>/<attempt>/
    wrapper.jsonl      gpu-waste run: attempt_start, child_exit
    rank<N>.jsonl      in-process: attempt_start, heartbeat, exception, cuda_oom,
                       sigterm, attempt_end, nvml_sample (local rank 0)
```

Every line is one JSON event (SPEC.md "Event record"). The report is derived
only from these files, so `gpu-waste report` can be re-run any time.
