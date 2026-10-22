import time
import types

from gpu_waste_meter._nvml import NvmlSampler, _visible_ids


class FakeNvml:
    """Minimal stand-in for pynvml with N GPUs."""

    def __init__(self, n=1, fail_init=False):
        self.n, self.fail_init, self.shutdown = n, fail_init, False
        self.tick = 0

    def nvmlInit(self):
        if self.fail_init:
            raise RuntimeError("NVML Shared Library Not Found")

    def nvmlShutdown(self):
        self.shutdown = True

    def nvmlDeviceGetCount(self):
        return self.n

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetUUID(self, h):
        return f"GPU-0000000{h}-aaaa".encode()

    def nvmlDeviceGetUtilizationRates(self, h):
        self.tick += 1
        return types.SimpleNamespace(gpu=self.tick % 101)

    def nvmlDeviceGetMemoryInfo(self, h):
        return types.SimpleNamespace(used=(h + 1) * 2**30)

    def nvmlDeviceGetPowerUsage(self, h):
        return 300_000  # mW


def collect():
    events = []
    return events, lambda t, c, d: events.append({"type": t, "confidence": c, "data": d})


def test_120_samples_make_2_events():
    events, emit = collect()
    s = NvmlSampler(emit, nvml=FakeNvml(1))
    assert s.init()
    for i in range(120):
        s.sample_once(now=1000.0 + i)
    s.flush()
    samples = [e for e in events if e["type"] == "nvml_sample"]
    assert len(samples) == 2
    for e in samples:
        d = e["data"]
        assert d["n"] == 60 and d["gpu"] == 0
        assert d["mem_used_mb"] == {"mean": 1024.0, "min": 1024.0, "max": 1024.0}
        assert d["power_w"]["mean"] == 300.0
        assert d["util_gpu"]["min"] <= d["util_gpu"]["mean"] <= d["util_gpu"]["max"]
    assert samples[1]["data"]["window_start"] == 1060.0


def test_one_event_per_gpu_per_minute():
    events, emit = collect()
    s = NvmlSampler(emit, nvml=FakeNvml(4))
    s.init()
    for i in range(60):
        s.sample_once(now=float(i))
    s.flush()
    assert sorted(e["data"]["gpu"] for e in events) == [0, 1, 2, 3]


def test_init_failure_writes_one_context_event():
    events, emit = collect()
    s = NvmlSampler(emit, nvml=FakeNvml(fail_init=True))
    assert s.start() is False
    assert len(events) == 1
    assert events[0]["type"] == "context" and events[0]["data"]["msg"] == "nvml unavailable"
    s.stop()  # must not raise


def test_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3")
    events, emit = collect()
    s = NvmlSampler(emit, nvml=FakeNvml(4))
    s.init()
    assert [g[0] for g in s._gpus] == [1, 3]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-00000002")
    s = NvmlSampler(emit, nvml=FakeNvml(4))
    s.init()
    assert [g[0] for g in s._gpus] == [2]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,-1,2")
    assert _visible_ids() == ["0"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert _visible_ids() == []


def test_thread_runs_and_stops():
    events, emit = collect()
    nv = FakeNvml(2)
    s = NvmlSampler(emit, interval=0.01, window=0.05, nvml=nv)
    assert s.start()
    time.sleep(0.3)
    s.stop()
    assert nv.shutdown
    assert len([e for e in events if e["type"] == "nvml_sample"]) >= 4


def test_metric_failure_is_partial_not_fatal():
    class Flaky(FakeNvml):
        def nvmlDeviceGetPowerUsage(self, h):
            raise RuntimeError("not supported")
    events, emit = collect()
    s = NvmlSampler(emit, nvml=Flaky(1))
    s.init()
    s.sample_once(now=0.0)
    s.flush()
    assert events[0]["data"]["power_w"] is None and events[0]["data"]["util_gpu"] is not None
