"""
sitecustomize loaded only for choice 'b' via PYTHONPATH.
It hides flash_attn from importlib.find_spec and actual imports,
without affecting other packages (e.g., bitsandbytes).
"""

import importlib.machinery as _mach
import sys

# If already patched, skip (idempotent)
if not getattr(sys, "_flash_attn_blocked", False):
    _orig_find_spec = _mach.PathFinder.find_spec

    def _blocked_find_spec(*args, **kwargs):
        """
        Works for both classmethod and instance calls:
        - PathFinder.find_spec(fullname, path=None, target=None)
        - PathFinder().find_spec(fullname, path=None, target=None)
        The first arg is self/cls, second is fullname.
        """
        fullname = args[1] if len(args) > 1 else kwargs.get("fullname")
        if isinstance(fullname, str) and (
            fullname == "flash_attn" or fullname.startswith("flash_attn.")
        ):
            return None  # makes _is_package_available return False
        return _orig_find_spec(*args, **kwargs)

    _mach.PathFinder.find_spec = _blocked_find_spec
    sys._flash_attn_blocked = True

    # Also drop any preloaded flash_attn modules if present
    for _k in list(sys.modules.keys()):
        if _k == "flash_attn" or _k.startswith("flash_attn."):
            sys.modules.pop(_k, None)
