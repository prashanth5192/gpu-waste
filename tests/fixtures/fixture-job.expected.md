# GPU waste report — `fixture-job`

## Allocation

- Window: 2026-09-21 14:13:20 UTC → 2026-09-21 18:30:50 UTC (4h 17m 30s)
- GPUs: 8 (from --gpus)
- Allocated: **34.33 GPU-hours**

## Confirmed loss

not measured (accounting pending)

## Attributed loss

not measured (accounting pending)

## Potential (not counted as loss)

not measured (accounting pending)

## Attempts

3 attempts, 2 failed.

| # | Start | Duration | Outcome | OOM kills | First error (stderr) |
|---|---|---|---|---|---|
| 0 | 2026-09-21 14:13:20 UTC | 45m 30s | failed (exit 1) | 0 | torch.distributed.DistBackendError: NCCL error: unhandled system error \| rank 3 |
| 1 | 2026-09-21 14:59:50 UTC | 30m 00s | failed (signal 9) | 1 | [step 1200] loss=2.31 |
| 2 | 2026-09-21 15:30:50 UTC | 3h 00m 00s | ok | 0 | — |

## Recurring patterns

not measured (accounting pending)

## Coverage

- Wrapper (`gpu-waste run`): yes
- Ranks reporting: none (in-process hooks not active)
- Collectives observed: not measured
- `step()` present: no

## NOT MEASURED

> These sources of waste are **not** included in any number above:
>
> - **XID / GPU hardware errors** — no NVIDIA XID collection yet
> - **NIC errors** — no NIC counter collection yet
> - **Fabric / network** — no fabric telemetry yet
> - **Checkpoint save / load time** — no checkpoint hooks yet
> - **Dataloader stalls** — no dataloader hooks yet
