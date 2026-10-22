# Task cards for local coding agent (read SPEC.md first)

Each card = one session. Write the test first. Stop when the test passes.

---
## T1 · `_env.py` + `_store.py`
**Build**
- `_env.py`: `get_identity() -> dict(job, attempt, rank, local_rank, world_size, node)` per SPEC rules.
- `_store.py`: `EventWriter(path)` with `.write(event_dict)`; appends one JSON line, `flush()` + `os.fsync()` at most every 5 s and on `close()`.
**Test** `tests/test_store.py`
- Write 100 events in a subprocess, `os.kill(pid, 9)` mid-loop, then every line in the file parses as JSON (last partial line allowed and skipped by reader).
- `get_identity()` with/without `TORCHELASTIC_RUN_ID`, `SLURM_JOB_ID`, `RANK`.
**Done when** `pytest tests/test_store.py` passes.

---
## T2 · `cli.py run` (wrapper)
**Build**
- `gpu-waste run -- <cmd...>`: spawn child, write `attempt_start` to `wrapper.jsonl`, wait, write `child_exit` with exit code / signal and last 200 lines of stderr (tee'd, still shown to user).
- Forward SIGTERM/SIGINT to child. Exit with the child's code.
- Read `/sys/fs/cgroup/memory.events` `oom_kill` before/after if readable; record delta.
**Test** `tests/test_run.py`
- `gpu-waste run -- python -c "import sys; sys.exit(3)"` → exit 3, one `child_exit` with code 3.
- child that `os.kill(os.getpid(), 9)` → `child_exit` with signal 9.
**Done when** both pass.

---
## T3 · `report.py` + `cli.py report`
**Build**
- Read all JSONL under `~/.gpu_waste/<job>/`, skip bad lines.
- Render `gpu-waste.md` with every SPEC section. Sections with no data print `not measured`.
- For now: allocation = GPUs × (last ts − first ts); attempts table from `attempt_start`/`child_exit`. Leave loss sections as `not measured (accounting pending)`.
**Test** `tests/test_report.py` using fixture JSONL in `tests/fixtures/` (3 attempts, 2 failures) → snapshot-compare the markdown.
**Done when** snapshot matches.

---
## T4 · `_nvml.py`
**Build**
- Daemon thread, 1 Hz, `pynvml` util.gpu, mem.used, power for each visible GPU (`CUDA_VISIBLE_DEVICES` aware).
- Aggregate to 1-minute mean/min/max; write one `nvml_sample` per GPU per minute.
- If `pynvml` init fails → write one `context` event "nvml unavailable", stop.
**Test** mock `pynvml`; 120 fake samples → 2 aggregated events. Overhead check script (manual): <1% CPU on your box.
**Done when** test passes and manual check OK.

---
## T5 · `_lifecycle.py` + `__init__.py` auto-start
**Build**
- On `import gpu_waste_meter`: get identity, open writer `rank<N>.jsonl`, write `attempt_start`, start heartbeat thread (every 5 s), start NVML.
- `atexit` → `attempt_end`; chained `sys.excepthook` / `threading.excepthook` → `exception` (type name only, no message); chained SIGTERM → `sigterm`.
- Detect `torch.cuda.OutOfMemoryError` → `cuda_oom`.
- `GWM_DISABLE=1` disables everything.
**Test** subprocess scripts: clean exit, raise ValueError, raise fake OOM class, SIGTERM → expected events. Verify a pre-existing excepthook still runs.
**Done when** all pass.

---
## STOP — bring to Claude for review
After T5, send Claude: `_store.py`, `_lifecycle.py`, the test output, and one real `gpu-waste.md` from a 2-GPU torchrun run on your workstation.
