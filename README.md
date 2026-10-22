# gpu_waste_meter

Measure where your GPU time goes — without changing your training code.

> **Status: pre-alpha (v0 in progress).** See `SPEC.md` for the design and `TASKS.md` for build order.

```
pip install gpu_waste_meter
gpu-waste run -- torchrun --nproc-per-node=8 train.py
gpu-waste report      # writes gpu-waste.md
```

Local only. No account, no upload, no daemon.
