# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Import the optimized-kernel packages and report what happened, as a THROWAWAY process.

Run as a script; prints one JSON object on stdout::

    {"xformers": {"imports": true, "runs": false, "error": "OSError: ..."}, ...}

This exists as a separate process, not a function call, for three reasons -- each of them
something that has already bitten this codebase:

* A package whose ``__init__`` raises leaves every submodule it already executed behind in
  ``sys.modules``. The next import re-runs ``__init__`` with those served from cache, so
  attributes are never rebound and the package imports "successfully" while missing pieces.
  See ``utils/torch_warmup.purge_partial_import`` and unslothai/unsloth#7580. A diagnostic
  that poisons the import cache for the warm and for every later request is worse than no
  diagnostic.
* ``import bitsandbytes`` creates a CUDA context. The backend deliberately never latches
  one -- main.py pins CUDA_DEVICE_ORDER before any torch import, and the export planner
  budgets from free VRAM read before a context exists -- so a diagnostic must not
  permanently take several hundred MB off every later VRAM reading.
* A genuinely broken native wheel can abort the interpreter rather than raise (pybind11
  answers a duplicate type registration with ``std::terminate``). In a child that is just a
  failed probe. In the server it is a dead app, on exactly the broken installs this is
  meant to describe.

Kept dependency-free (stdlib only, no studio imports) so the parent can run it with the
same interpreter and nothing else on the path.
"""

import json
import sys
from typing import Any, Dict


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def probe_xformers() -> Dict[str, Any]:
    """Does xformers import, and do its C++/CUDA extensions load?

    The extension load is what fails on an ABI mismatch: ``torch.ops.load_library`` raises
    OSError, and xformers catches it and downgrades it to a logger warning, which is why
    the package imports and reports a version while having no memory-efficient attention.

    Prefer the outcome ``xformers/_cpp_lib.py`` already recorded at its own import over
    calling ``_register_extensions()`` again: re-calling it re-runs
    ``os.add_dll_directory`` on Windows, and the private name is not guaranteed to exist in
    every layout -- treating its absence as "broken" would put a red banner on a working
    install.
    """
    entry: Dict[str, Any] = {"imports": False, "runs": None, "error": None}
    try:
        import xformers  # noqa: F401
    except BaseException as exc:
        entry["error"] = _error(exc)
        return entry
    entry["imports"] = True

    try:
        from xformers import _cpp_lib
    except BaseException as exc:
        entry["error"] = _error(exc)
        return entry

    if hasattr(_cpp_lib, "_cpp_library_load_exception"):
        failure = _cpp_lib._cpp_library_load_exception
        entry["runs"] = failure is None
        if failure is not None:
            entry["error"] = _error(failure)
        return entry

    register = getattr(_cpp_lib, "_register_extensions", None)
    if register is None:
        # An xformers layout we do not recognise. Unknown is not broken.
        return entry
    try:
        register()
        entry["runs"] = True
    except BaseException as exc:
        entry["runs"] = False
        entry["error"] = _error(exc)
    return entry


def probe_flash_attn() -> Dict[str, Any]:
    """flash-attn, forced past the lazy bit: the CUDA extension lives in the interface."""
    entry = probe_import("flash_attn")
    if not entry["imports"]:
        return entry
    try:
        import flash_attn.flash_attn_interface  # noqa: F401

        entry["runs"] = True
    except BaseException as exc:
        entry["runs"] = False
        entry["error"] = _error(exc)
    return entry


def probe_import(import_name: str) -> Dict[str, Any]:
    """Plain import. An ABI mismatch in a native wheel surfaces here as an undefined symbol."""
    entry: Dict[str, Any] = {"imports": False, "runs": None, "error": None}
    try:
        __import__(import_name)
        entry["imports"] = True
    except BaseException as exc:
        entry["error"] = _error(exc)
    return entry


PROBES = {
    "xformers": probe_xformers,
    "flash_attn": probe_flash_attn,
    "torchao": lambda: probe_import("torchao"),
    "bitsandbytes": lambda: probe_import("bitsandbytes"),
}


def main(argv) -> int:
    """Probe the names given on the command line (default: all of them)."""
    wanted = [name for name in (argv or PROBES) if name in PROBES]
    results = {}
    for name in wanted:
        try:
            results[name] = PROBES[name]()
        except BaseException as exc:
            results[name] = {"imports": False, "runs": None, "error": _error(exc)}
    # Some of these packages print to stdout on import, so the JSON goes out last behind a
    # marker the parent can seek to rather than assuming it owns the stream.
    sys.stdout.write("\n__UNSLOTH_ACCELERATOR_PROBE__" + json.dumps(results) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
