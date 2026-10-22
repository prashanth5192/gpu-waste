"""Toy training loop for evaluating gpu_waste_meter — no GPU needed.

Simulates common ways training attempts die, so you can see them in gpu-waste.md.

    python examples/flaky_train.py                       # clean run
    python examples/flaky_train.py --fail exception      # Python crash
    python examples/flaky_train.py --fail oom            # CUDA OOM (fake class unless torch+GPU)
    python examples/flaky_train.py --fail kill           # SIGKILL (like the kernel OOM killer)
    python examples/flaky_train.py --fail sigterm        # scheduler preemption

Under torchrun (needs `pip install torch`), ranks join a gloo process group and
all-reduce every step; `--fail-rank R` makes only rank R fail:

    torchrun --nproc-per-node=2 examples/flaky_train.py --fail exception --fail-rank 1
"""

import gpu_waste_meter  # noqa: F401  (import first: starts in-process recording)

import argparse
import os
import signal
import sys
import time


def fake_cuda_oom():
    try:
        import torch
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
    except ImportError:
        class OutOfMemoryError(RuntimeError):
            pass
        OutOfMemoryError.__module__ = "torch.cuda"
        raise OutOfMemoryError("CUDA out of memory (simulated)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step-time", type=float, default=0.1)
    ap.add_argument("--fail", choices=["none", "exception", "oom", "kill", "sigterm"],
                    default="none")
    ap.add_argument("--at-step", type=int, default=10)
    ap.add_argument("--fail-rank", type=int, default=None,
                    help="only this rank fails (default: all ranks)")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    dist = None
    if world > 1:
        import torch
        import torch.distributed as dist
        dist.init_process_group("gloo")

    fails_here = args.fail != "none" and args.fail_rank in (None, rank)
    for step in range(args.steps):
        if fails_here and step == args.at_step:
            print(f"[rank {rank}] step {step}: injecting failure '{args.fail}'",
                  file=sys.stderr, flush=True)
            if args.fail == "exception":
                raise RuntimeError("simulated training crash")
            if args.fail == "oom":
                fake_cuda_oom()
            if args.fail == "kill":
                os.kill(os.getpid(), signal.SIGKILL)
            if args.fail == "sigterm":
                os.kill(os.getpid(), signal.SIGTERM)
        if dist is not None:
            t = torch.ones(1) * (rank + 1)
            dist.all_reduce(t)
        time.sleep(args.step_time)
        print(f"[rank {rank}] step {step} loss={1.0 / (step + 1):.4f}", file=sys.stderr, flush=True)

    if dist is not None:
        dist.destroy_process_group()
    print(f"[rank {rank}] done", file=sys.stderr)


if __name__ == "__main__":
    main()
