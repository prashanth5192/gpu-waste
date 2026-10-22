"""In-process lifecycle hooks, started on `import gpu_waste_meter`.

Writes to ~/.gpu_waste/<job>/<attempt>/rank<N>.jsonl:
    attempt_start, heartbeat (every 5 s), attempt_end (atexit),
    exception (type name only), cuda_oom, sigterm, plus nvml_sample (local rank 0).

Hard rules: never raise into the user's program; chain, don't replace, existing
excepthooks and signal handlers; `GWM_DISABLE=1` turns everything off.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading
import time

from . import _env
from ._store import EventWriter, attempt_dir

HEARTBEAT_S = float(os.environ.get("GWM_HEARTBEAT_S", "5"))

_state: dict = {"started": False}
_lock = threading.Lock()


def disabled() -> bool:
    return os.environ.get("GWM_DISABLE", "").strip().lower() in ("1", "true", "yes")


def _is_cuda_oom(exc_type) -> bool:
    """torch.cuda.OutOfMemoryError (or subclass) without importing torch."""
    try:
        for cls in getattr(exc_type, "__mro__", ()):
            if cls.__name__ == "OutOfMemoryError" and cls.__module__.startswith("torch"):
                return True
    except Exception:
        pass
    return False


def _emit(type_: str, confidence: str, data: dict | None = None) -> None:
    try:
        w = _state.get("writer")
        if w is None:
            return
        ident = _state["ident"]
        w.write({
            "ts": time.time(), "job": ident["job"], "attempt": ident["attempt"],
            "rank": ident["rank"], "node": ident["node"], "type": type_,
            "confidence": confidence, "source": "inproc" if type_ != "nvml_sample" else "nvml",
            "data": data or {},
        })
    except Exception:
        pass


def _record_exception(exc_type, where: str) -> None:
    name = getattr(exc_type, "__qualname__", None) or getattr(exc_type, "__name__", "?")
    module = getattr(exc_type, "__module__", "")
    full = name if module in ("builtins", "") else f"{module}.{name}"
    _emit("exception", "confirmed", {"exc_type": full, "where": where})
    if _is_cuda_oom(exc_type):
        _emit("cuda_oom", "confirmed", {"exc_type": full, "where": where})
    _state["failed"] = True


# --------------------------------------------------------------------------- hooks

def _install_excepthooks() -> None:
    prev_sys = sys.excepthook

    def sys_hook(exc_type, exc, tb):
        try:
            if not issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
                _record_exception(exc_type, "main")
        except Exception:
            pass
        prev_sys(exc_type, exc, tb)

    sys.excepthook = sys_hook

    prev_thr = getattr(threading, "excepthook", None)
    if prev_thr is not None:
        def thread_hook(args):
            try:
                if args.exc_type is not SystemExit:
                    _record_exception(args.exc_type, f"thread:{getattr(args.thread, 'name', '?')}")
            except Exception:
                pass
            prev_thr(args)

        threading.excepthook = thread_hook


def _install_sigterm() -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        prev = signal.getsignal(signal.SIGTERM)
    except (ValueError, OSError):
        return

    def handler(signum, frame):
        _emit("sigterm", "confirmed", {"pid": os.getpid()})
        _state["sigterm"] = True
        if callable(prev):
            prev(signum, frame)
        elif prev == signal.SIG_IGN:
            return
        else:  # SIG_DFL (or None): restore default behaviour and re-deliver
            try:
                _shutdown(reason="sigterm")
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
            except Exception:
                pass
            os.kill(os.getpid(), signal.SIGTERM)

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError):
        pass


def _heartbeat_loop(stop: threading.Event) -> None:
    n = 0
    while not stop.wait(HEARTBEAT_S):
        n += 1
        _emit("heartbeat", "context", {"n": n})


# --------------------------------------------------------------------------- public

def start() -> bool:
    """Idempotent. Returns True if instrumentation is active."""
    if disabled():
        return False
    with _lock:
        if _state["started"]:
            return True
        _state["started"] = True
        try:
            ident = _env.get_identity()
            rank = ident["rank"] if ident["rank"] is not None else 0
            _state["ident"] = ident
            _state["writer"] = EventWriter(attempt_dir(ident["job"], ident["attempt"])
                                           / f"rank{rank}.jsonl")
            _emit("attempt_start", "context", {
                "pid": os.getpid(), "local_rank": ident["local_rank"],
                "world_size": ident["world_size"], "argv": sys.argv[:20],
            })

            stop = threading.Event()
            _state["hb_stop"] = stop
            threading.Thread(target=_heartbeat_loop, args=(stop,),
                             name="gwm-heartbeat", daemon=True).start()

            # One sampler per node: local rank 0 (or a non-distributed process).
            if ident["local_rank"] in (None, 0) and os.environ.get("GWM_NVML", "1") != "0":
                from ._nvml import NvmlSampler
                sampler = NvmlSampler(_emit)
                if sampler.start():
                    _state["nvml"] = sampler

            _install_excepthooks()
            _install_sigterm()
            atexit.register(_shutdown)
        except Exception:
            pass
    return True


def _shutdown(reason: str = "exit") -> None:
    with _lock:
        if _state.get("shut"):
            return
        _state["shut"] = True
    try:
        if "hb_stop" in _state:
            _state["hb_stop"].set()
        if "nvml" in _state:
            _state["nvml"].stop()
        _emit("attempt_end", "context", {
            "reason": "sigterm" if _state.get("sigterm") else reason,
            "failed": bool(_state.get("failed")),
        })
        if _state.get("writer") is not None:
            _state["writer"].close()
    except Exception:
        pass
