# gpu_waste_meter

Measure where your GPU time goes — without changing your training code.

> **Status: pre-alpha (v0).** Records attempts, failures and GPU utilization;
> loss accounting is not implemented yet. See `SPEC.md` for the design,
> `TASKS.md` for build order, and `EVALUATION.md` to verify it yourself.

```
pip install gpu_waste_meter
gpu-waste run -- torchrun --nproc-per-node=8 train.py
gpu-waste report      # writes gpu-waste.md
```

Local only. No account, no upload, no daemon.

## What gets recorded

| Layer | How to enable | Records |
|---|---|---|
| Wrapper | prefix your launch with `gpu-waste run --` | each attempt's start, exit code / signal, last 200 stderr lines, cgroup OOM kills |
| In-process | `import gpu_waste_meter` at the top of `train.py` | per-rank start / end, heartbeats, exception type, CUDA OOM, SIGTERM; per-GPU utilization, memory and power each minute (`pip install 'gpu_waste_meter[nvml]'`) |

Evidence is append-only JSONL under `~/.gpu_waste/<job>/<attempt>/` and
survives crashes, including `kill -9`. `gpu-waste report` can be re-run any time.

## Report

`gpu-waste.md` contains: Allocation (GPU-hours), Attempts (outcome and root
cause per attempt), Coverage, and an explicit NOT MEASURED box. Anything the
tool did not measure says `not measured` instead of guessing.

```
gpu-waste report [--job JOB] [-o gpu-waste.md] [--gpus N]
```

## Configuration

| Variable | Effect |
|---|---|
| `GWM_DISABLE=1` | turn off all in-process recording |
| `GWM_HOME` | evidence directory (default `~/.gpu_waste`) |
| `GWM_JOB` | override the job id (default: torchrun / SLURM id, else a hash of command + cwd) |
| `GWM_NVML=0` | skip GPU sampling |

## Try it without a GPU

```
pip install -e '.[test]'
python -m pytest -q
gpu-waste run -- python examples/flaky_train.py --fail oom
gpu-waste report && cat gpu-waste.md
```
