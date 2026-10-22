"""Manual T4 check: CPU overhead of the NVML sampler on a real GPU box.

    pip install -e '.[nvml]'
    python scripts/nvml_overhead.py            # 60 s, default
    python scripts/nvml_overhead.py 120        # custom duration

Pass criterion (TASKS.md T4): sampler CPU < 1% of one core.
"""

import sys
import time

from gpu_waste_meter._nvml import NvmlSampler

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
events = []
s = NvmlSampler(lambda t, c, d: events.append((t, d)))
cpu0, wall0 = time.process_time(), time.monotonic()
if not s.start():
    print("NVML unavailable:", events)
    sys.exit(1)
print(f"sampling {len(s._gpus)} GPU(s) for {seconds:.0f} s ...")
time.sleep(seconds)
s.stop()
cpu, wall = time.process_time() - cpu0, time.monotonic() - wall0
pct = 100 * cpu / wall
n = sum(1 for t, _ in events if t == "nvml_sample")
print(f"CPU {cpu:.3f} s over {wall:.1f} s wall = {pct:.3f}% of one core; {n} nvml_sample events")
print("PASS" if pct < 1.0 else "FAIL (>= 1%)")
sys.exit(0 if pct < 1.0 else 1)
