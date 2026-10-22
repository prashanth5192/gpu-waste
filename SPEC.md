# gpu_waste_meter — SPEC (v0)

One-page spec for coding agents. Do not add features not listed here.

## What it is
Open-source Python package. Records where allocated GPU time is lost in PyTorch training jobs. Local only, no network, no account.

- Package: `gpu_waste_meter` · CLI: `gpu-waste` · Output: `gpu-waste.md`
- Dependencies: `pynvml` only. `torch` imported lazily; absent torch = feature "not measured".

## UX
```
pip install gpu_waste_meter
gpu-waste run -- torchrun --nproc-per-node=8 train.py   # recommended
gpu-waste report                                         # writes gpu-waste.md
```
Optional in user code: `import gpu_waste_meter as gwm; gwm.step(global_step)`

## Hard rules (every module)
1. Never raise into the user's program. Any internal error → log once, mark feature "not measured", continue.
2. Never change return values or behavior of torch functions we wrap.
3. Chain, don't replace, existing signal handlers / excepthooks.
4. Evidence is append-only JSONL, flushed every ≤5 s. Report is derived from it and can always be regenerated.
5. No network calls anywhere.

## Evidence layout
```
~/.gpu_waste/<job_id>/<attempt_id>/
    wrapper.jsonl        # from gpu-waste run (one per node)
    rank<N>.jsonl        # from in-process hooks
    fr_<rank>.*          # Flight Recorder dumps, if any
```
`job_id`: `GWM_JOB` env > `TORCHELASTIC_RUN_ID` > `SLURM_JOB_ID` > sha1(argv+cwd)[:12]
`attempt_id`: `TORCHELASTIC_RESTART_COUNT` or wrapper-assigned counter.

## Event record (one JSON object per line)
```
{"ts": float_unix, "job": str, "attempt": int, "rank": int|null, "node": str,
 "type": str, "confidence": "confirmed"|"attributed"|"potential"|"context",
 "source": "wrapper"|"inproc"|"nvml"|"fr"|"nvrx"|"clockwork", "data": {...}}
```
Types (v0): `attempt_start, attempt_end, heartbeat, exception, cuda_oom, sigterm, child_exit, nccl_timeout, collective_sample, step, nvml_sample, fr_dump`

## Accounting (per attempt, per second, one bucket per second)
Priority: `recovery_gap > confirmed_hang > stall > slowdown > normal`
- **Confirmed**: failed attempts, recovery gaps, NCCL hangs, recomputed steps (needs `step`)
- **Attributed**: collective time above baseline, stalls with PIDs alive
- **Potential**: SM util < 30% on *normal* seconds only — never summed into loss
GPU-hours = seconds × GPUs / 3600.

## Report must always include
Allocation · Confirmed · Attributed · Potential · Attempts table · Recurring patterns · Coverage (ranks reporting, % collectives observed, step() present) · NOT MEASURED box (XID, NIC, fabric, checkpoint, dataloader).

## Build order
T1 store+env → T2 wrapper → T3 report → T4 NVML → T5 lifecycle hooks → (later, needs Claude review) FR/NCCL, collective wrappers, step(), accounting.
