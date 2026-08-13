"""Shared helper for the *_real_rtm.py driver scripts: runs a base sweep
module's main() with RTM_CHECKPOINT overridden to a real-libRadtran-trained
surrogate, saving output under a new filename while protecting the base
module's original (placeholder-physics) results file from being overwritten
in place.
"""

from __future__ import annotations

import os
from types import ModuleType


def run_protected(base: ModuleType, checkpoint_override: str, base_out_path: str, real_out_path: str) -> None:
    base.RTM_CHECKPOINT = checkpoint_override
    backup_path = base_out_path + ".protected_bak"
    if os.path.exists(base_out_path):
        os.rename(base_out_path, backup_path)
    try:
        base.main()
        os.replace(base_out_path, real_out_path)
    finally:
        if os.path.exists(backup_path):
            os.rename(backup_path, base_out_path)
    print(f"Saved output to {real_out_path}, original placeholder-physics results file restored/untouched")
