"""Import-order guard: load ``trl`` before ``torch``.

On some GPUs (observed on Blackwell / sm_120 cards) importing ``trl`` *after* ``torch`` crashes
the process with a segfault during CUDA initialisation, while the reverse order is fine. Every
package module that imports ``torch`` therefore imports this module first::

    from . import _compat  # noqa: F401

The guard only acts when ``torch`` has not been imported yet and ``trl`` is installed. Importing
the trainer classes (not just the top-level ``trl`` package, which is lazy) is what actually pulls
in ``trl``'s torch-dependent modules in the safe order.
"""

from __future__ import annotations

import sys

if "torch" not in sys.modules:
    try:
        from trl import DPOConfig, DPOTrainer  # noqa: F401
    except ImportError:
        # trl is optional for evaluation-only installs; nothing to order then.
        pass
