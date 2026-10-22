"""gpu_waste_meter: local-first accounting of where GPU time is lost.

`import gpu_waste_meter` in a training script starts lightweight in-process
recording (see _lifecycle.py). Set GWM_DISABLE=1 to turn it off.
"""

__version__ = "0.0.1"

import os as _os
import sys as _sys

from ._env import get_identity  # noqa: F401


def _is_cli_process() -> bool:
    """Don't instrument the `gpu-waste` CLI itself (wrapper / report)."""
    argv0 = _sys.argv[0] if _sys.argv else ""
    # `python -m gpu_waste_meter.cli`: argv[0] is "-m" while the package imports.
    return argv0 == "-m" or _os.path.basename(argv0) in ("gpu-waste", "gpu-waste.exe")


def start() -> bool:
    """Start in-process recording explicitly (normally automatic on import)."""
    from . import _lifecycle
    return _lifecycle.start()


if not _is_cli_process():
    try:
        start()
    except Exception:  # never raise into the user's program
        pass
