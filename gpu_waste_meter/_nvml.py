"""Per-GPU NVML sampling: 1 Hz samples, one aggregated `nvml_sample` per GPU per minute.

`pynvml` (package `nvidia-ml-py`) is optional. If it is missing or NVML init
fails, one `context` event "nvml unavailable" is written and sampling stops.
Never raises into the caller.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

METRICS = ("util_gpu", "mem_used_mb", "power_w")


def _visible_ids() -> list[str] | None:
    """Parse CUDA_VISIBLE_DEVICES. None = all devices; [] = no devices."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    ids = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lstrip("-").isdigit() and int(tok) < 0:
            break                       # CUDA stops at the first invalid index
        ids.append(tok)
    return ids


class NvmlSampler:
    def __init__(self, emit: Callable[[str, str, dict], None], interval: float = 1.0,
                 window: float = 60.0, nvml=None):
        self._emit = emit
        self.interval = interval
        self.window = window
        self._nvml = nvml
        self._gpus: list[tuple[int, str, object]] = []   # (index, uuid, handle)
        self._buf: dict[int, dict] = {}                  # gpu index -> window state
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.available = False

    # ------------------------------------------------------------------ setup
    def init(self) -> bool:
        try:
            if self._nvml is None:
                import pynvml  # type: ignore
                self._nvml = pynvml
            nv = self._nvml
            nv.nvmlInit()
            count = nv.nvmlDeviceGetCount()
            wanted = _visible_ids()
            for i in range(count):
                h = nv.nvmlDeviceGetHandleByIndex(i)
                uuid = nv.nvmlDeviceGetUUID(h)
                uuid = uuid.decode() if isinstance(uuid, bytes) else str(uuid)
                if wanted is None or str(i) in wanted or any(
                        w.startswith("GPU-") and uuid.startswith(w) for w in wanted):
                    self._gpus.append((i, uuid, h))
            self.available = True
            return True
        except Exception as e:
            self._emit("context", "context",
                       {"msg": "nvml unavailable", "error": f"{type(e).__name__}: {e}"})
            return False

    # ------------------------------------------------------------------ sampling
    def _read(self, h) -> dict:
        nv, out = self._nvml, {}
        try:
            out["util_gpu"] = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            pass
        try:
            out["mem_used_mb"] = nv.nvmlDeviceGetMemoryInfo(h).used / 2**20
        except Exception:
            pass
        try:
            out["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            pass
        return out

    def sample_once(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        for idx, uuid, h in self._gpus:
            self.add_sample(idx, uuid, now, self._read(h))

    def add_sample(self, idx: int, uuid: str, now: float, values: dict) -> None:
        with self._lock:
            st = self._buf.get(idx)
            if st is not None and now - st["start"] >= self.window:
                self._flush_one(idx)
                st = None
            if st is None:
                st = self._buf[idx] = {"start": now, "uuid": uuid, "n": 0,
                                       "vals": {m: [] for m in METRICS}}
            st["n"] += 1
            for m in METRICS:
                if m in values:
                    st["vals"][m].append(values[m])

    def _flush_one(self, idx: int) -> None:
        st = self._buf.pop(idx, None)
        if not st or not st["n"]:
            return
        data = {"gpu": idx, "uuid": st["uuid"], "window_start": st["start"],
                "window_s": self.window, "n": st["n"]}
        for m, vs in st["vals"].items():
            data[m] = ({"mean": round(sum(vs) / len(vs), 2), "min": min(vs), "max": max(vs)}
                       if vs else None)
        self._emit("nvml_sample", "context", data)

    def flush(self) -> None:
        with self._lock:
            for idx in list(self._buf):
                self._flush_one(idx)

    # ------------------------------------------------------------------ thread
    def _loop(self) -> None:
        nxt = time.monotonic()
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception:
                pass
            nxt += self.interval
            self._stop.wait(max(0.0, nxt - time.monotonic()))

    def start(self) -> bool:
        if not self.init():
            return False
        if not self._gpus:
            self._emit("context", "context", {"msg": "nvml: no visible GPUs"})
            return False
        self._thread = threading.Thread(target=self._loop, name="gwm-nvml", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval + 1)
        self.flush()
        try:
            if self.available:
                self._nvml.nvmlShutdown()
        except Exception:
            pass
