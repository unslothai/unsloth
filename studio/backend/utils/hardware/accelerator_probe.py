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
import os
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
        return _with_kernel_verdict(entry)

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
    return _with_kernel_verdict(entry)


def _with_kernel_verdict(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Downgrade a loaded-but-useless xFormers from "runs" to "does not run".

    The library loading is necessary, not sufficient: a build with no kernel for THIS GPU
    (an sm_120 card against a wheel that ships none) loads fine, and every attention call
    is then capability-rejected straight back to SDPA. That is precisely the degraded state
    this report exists to surface, and it was rendering as "Working".

    Static, not a forward pass: this child runs with no visible GPU on purpose (a probe
    must not latch a CUDA context or take VRAM from a run in progress), so the verdict
    comes from the op table plus the capability read out of nvidia-smi, which needs no
    context. Unknown at any step leaves the load-status answer alone -- an install that
    cannot be checked is not an install that is broken.
    """
    capability = _device_compute_capability()
    if capability is None:
        return entry
    try:
        from xformers.ops import fmha
    except BaseException:
        return entry
    ops = getattr(fmha, "ALL_FW_OPS", None)
    if not ops:
        return entry
    usable = False
    for op in ops:
        minimum = getattr(op, "CUDA_MINIMUM_COMPUTE_CAPABILITY", None)
        maximum = getattr(op, "CUDA_MAXIMUM_COMPUTE_CAPABILITY", None)
        if getattr(op, "OPERATOR", False) is None:
            continue  # the build did not ship this op at all
        if minimum is not None and capability < minimum:
            continue
        if maximum is not None and capability > maximum:
            continue
        usable = True
        break
    if not usable:
        entry["runs"] = False
        entry["error"] = (
            f"xformers loaded but ships no memory-efficient attention kernel for this GPU "
            f"(compute capability {capability[0]}.{capability[1]}), so attention falls "
            f"back to SDPA"
        )
    return entry


def _device_compute_capability():
    """This host's compute capability as ``(major, minor)``, or None when unknown.

    Through nvidia-smi rather than torch: reading it from torch initialises CUDA, and the
    whole point of this child is that it never holds a context. UNSLOTH_PROBE_DEVICE_CC
    overrides, which is also how the tests drive it.
    """
    override = os.environ.get("UNSLOTH_PROBE_DEVICE_CC", "").strip()
    text = override
    if not text:
        try:
            import shutil
            import subprocess

            exe = shutil.which("nvidia-smi")
            if not exe:
                return None
            result = subprocess.run(
                [exe, "--query-gpu=compute_cap", "--format=csv,noheader"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                encoding = "utf-8",
                errors = "replace",
                timeout = 10,
            )
            if result.returncode != 0:
                return None
            # First line: the probe describes one host, and a mixed box's weakest card is
            # not what a "does this install work" answer should hinge on.
            text = (result.stdout or "").strip().splitlines()[0] if result.stdout.strip() else ""
        except BaseException:
            return None
    try:
        major, _, minor = text.strip().partition(".")
        return (int(major), int(minor or 0))
    except (AttributeError, ValueError):
        return None


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


def probe_bitsandbytes() -> Dict[str, Any]:
    """bitsandbytes, past the import: are the ctypes kernel handles real?

    From 0.46 a wheel whose native library never loaded still imports cleanly and hands
    back a ``throw_on_call`` closure for every symbol, so ``imports=True`` says nothing and
    the row rendered as "Working" while the first 4-bit op was going to die mid-run. The
    repository already answers this question in ``unsloth/bnb_availability.py``, a leaf
    module that imports nothing from unsloth -- loaded here BY PATH so the verdict is the
    same one the loader gates on, without dragging ``unsloth/__init__`` (and its patching)
    into a diagnostic child.
    """
    entry = probe_import("bitsandbytes")
    if not entry["imports"]:
        return entry
    ready = _load_bnb_availability()
    if ready is None:
        # Cannot find the checker: leave runs unknown rather than inventing a verdict.
        return entry
    try:
        import bitsandbytes

        ready.check_native_kernels(bitsandbytes, _device_type())
        entry["runs"] = True
    except BaseException as exc:
        entry["runs"] = False
        entry["error"] = _error(exc)
    return entry


def _device_type() -> str:
    """The device type ``bitsandbytes_symbols`` keys on. Only "xpu" differs."""
    return os.environ.get("UNSLOTH_PROBE_DEVICE_TYPE", "cuda").strip().lower() or "cuda"


def _load_bnb_availability():
    """``unsloth.bnb_availability`` loaded standalone, or None when it cannot be found."""
    try:
        import importlib.util

        spec = importlib.util.find_spec("unsloth")
        origin = getattr(spec, "origin", None) if spec is not None else None
        if not origin:
            return None
        path = os.path.join(os.path.dirname(origin), "bnb_availability.py")
        if not os.path.isfile(path):
            return None
        leaf = importlib.util.spec_from_file_location("_unsloth_bnb_availability", path)
        if leaf is None or leaf.loader is None:
            return None
        module = importlib.util.module_from_spec(leaf)
        leaf.loader.exec_module(module)
        return module
    except BaseException:
        return None


def probe_torchao() -> Dict[str, Any]:
    """torchao, past the import: did its C++/CUDA extension actually load?

    A supported install can import torchao while its kernels are absent -- the Python
    stack pins 0.17.0 on torch 2.10+cu130 precisely because its torch-2.11 extensions are
    "cleanly skipped" rather than crashed -- and the quantization kernels are then not
    there. Reporting that as "Working" is the false all-clear this report exists to avoid.
    """
    entry = probe_import("torchao")
    if not entry["imports"]:
        return entry
    try:
        import torch

        ops = getattr(torch.ops, "torchao", None)
        # torch._ops._OpNamespace resolves lazily, so its dir() is the registered set.
        entry["runs"] = bool(ops is not None and dir(ops))
        if entry["runs"] is False:
            entry["error"] = (
                "torchao imported but registered no native operators: its C++/CUDA "
                "extension was skipped for this torch build, so the optimized "
                "quantization kernels are not available"
            )
    except BaseException as exc:
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
    "torchao": probe_torchao,
    "bitsandbytes": probe_bitsandbytes,
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
