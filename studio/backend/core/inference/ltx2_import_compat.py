# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Let the diffusers LTX-2 pipelines import on the transformers Studio pins.

diffusers 0.40.0 (and main since huggingface/diffusers#14447, "Ltx 2.5") imports
``Gemma4UnifiedForConditionalGeneration`` at module level in every ``diffusers.pipelines.ltx2.pipeline_ltx2*``
module. That class first ships in transformers 5.10, while Studio pins transformers 5.5.0, so ``LTX2Pipeline``
cannot even be imported and every LTX-2 and LTX-2.3 load and LTX-2 LoRA run fails with
``cannot import name 'Gemma4UnifiedForConditionalGeneration' from 'transformers'`` (tracked upstream as
huggingface/diffusers#14773).

The name is only used in the ``text_encoder`` annotation and docstring: the LTX-2 and LTX-2.3 checkpoints Studio
serves use a Gemma3 encoder. So when the installed transformers lacks it, ``ensure_ltx2_pipelines_importable`` binds
an inert placeholder on ``transformers`` just long enough to import the pipeline modules that name it, then removes
it again. Nothing else in the process ever sees the placeholder on ``transformers``, and a transformers that has the
real class is left completely alone. The placeholder refuses to be constructed or loaded, so a checkpoint that truly
needs the Gemma4 unified encoder fails with a message naming the transformers release it needs instead of loading
something wrong.

Retire this module once diffusers guards the import (or Studio's transformers pin reaches 5.10).
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
import sys
import threading
from typing import Any, Optional

# transformers names the diffusers LTX-2 pipelines import at module level, with the first transformers release that
# exports each one.
LTX2_OPTIONAL_TRANSFORMERS_NAMES: dict[str, str] = {
    "Gemma4UnifiedForConditionalGeneration": "5.10",
}

_LTX2_PACKAGE = "diffusers.pipelines.ltx2"
_PLACEHOLDER_MARKER = "_unsloth_ltx2_import_placeholder"

_lock = threading.Lock()
_done = False


def is_ltx2_pipeline_class(name: Optional[str]) -> bool:
    """Whether ``name`` is a diffusers class that lives in the LTX-2 pipeline package."""
    return isinstance(name, str) and name.startswith("LTX2") and name.endswith("Pipeline")


def _make_placeholder(name: str, min_version: str) -> type:
    message = (
        f"This checkpoint needs transformers' {name}, which this environment's transformers does not "
        f"provide (it needs transformers >= {min_version}). LTX-2 and LTX-2.3 use a Gemma3 text encoder "
        f"and do not need it."
    )

    def _refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise ImportError(message)

    return type(
        name,
        (),
        {
            "__doc__": f"Import-time stand-in for transformers.{name}; see {__name__}.",
            "__module__": __name__,
            _PLACEHOLDER_MARKER: True,
            "__init__": _refuse,
            "from_pretrained": classmethod(_refuse),
            "from_config": classmethod(_refuse),
        },
    )


def _transformers_has(transformers: Any, name: str) -> bool:
    # transformers' top level is lazy: this getattr imports the real model module when there is one. A release that
    # names the class but cannot import it is treated as lacking it, which is what the diffusers import would hit.
    try:
        value = getattr(transformers, name)
    except Exception:  # noqa: BLE001 -- AttributeError on an old release, RuntimeError on a broken lazy import
        return False
    return not getattr(value, _PLACEHOLDER_MARKER, False)


def _modules_naming(package: Any, names: list[str]) -> list[str]:
    """Submodules of ``package`` whose source mentions any of ``names``, in a stable order."""
    found = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key = lambda i: i.name):
        full = f"{package.__name__}.{info.name}"
        try:
            spec = importlib.util.find_spec(full)
            origin = getattr(spec, "origin", None)
            if not origin or not origin.endswith(".py"):
                continue
            with open(origin, encoding = "utf-8") as fh:
                source = fh.read()
        except Exception:  # noqa: BLE001 -- unreadable source: leave it to the normal import path
            continue
        if any(name in source for name in names):
            found.append(full)
    return found


def ensure_ltx2_pipelines_importable(logger: Any = None) -> bool:
    """Make ``diffusers.pipelines.ltx2`` importable when transformers lacks a name it only annotates with.

    Idempotent and cheap after the first success. Returns True when nothing more is needed (the class exists, the
    modules are imported, or there is no LTX-2 package to fix), False when it could not help, in which case the
    caller's own import reports the real error unchanged. Never raises.
    """
    global _done
    if _done:
        return True
    with _lock:
        if _done:
            return True
        try:
            import transformers
        except Exception:  # noqa: BLE001 -- no transformers: the caller's import reports it
            return False
        missing = [
            name
            for name in LTX2_OPTIONAL_TRANSFORMERS_NAMES
            if not _transformers_has(transformers, name)
        ]
        if not missing:
            _done = True
            return True
        try:
            package = importlib.import_module(_LTX2_PACKAGE)
        except Exception:  # noqa: BLE001 -- no diffusers, or one without LTX-2: nothing to shim
            return False
        targets = [m for m in _modules_naming(package, missing) if m not in sys.modules]
        try:
            # transformers re-executes its own __init__ (direct_transformers_import) the first time processing_utils is
            # imported, which REPLACES sys.modules["transformers"] and would drop a name bound on the old object. The
            # LTX-2 pipelines import ProcessorMixin anyway, so settle the swap first.
            importlib.import_module("transformers.processing_utils")
        except Exception:  # noqa: BLE001 -- the pipeline import below reports whatever this was
            pass

        placeholders = {
            name: _make_placeholder(name, LTX2_OPTIONAL_TRANSFORMERS_NAMES[name])
            for name in missing
        }
        touched: list[Any] = []

        def _bind() -> None:
            # Bind on whatever module object `from transformers import ...` will read right now.
            current = sys.modules.get("transformers", transformers)
            for name, placeholder in placeholders.items():
                if name not in current.__dict__:
                    setattr(current, name, placeholder)
                    if current not in touched:
                        touched.append(current)

        failed: list[tuple[str, str]] = []
        try:
            for module_name in targets:
                for attempt in (0, 1):
                    _bind()
                    try:
                        importlib.import_module(module_name)
                        break
                    except Exception as exc:  # noqa: BLE001 -- a different failure: the real import re-raises it
                        if attempt == 0 and any(name in str(exc) for name in missing):
                            continue  # transformers was swapped mid-import: rebind and retry once
                        failed.append((module_name, f"{type(exc).__name__}: {exc}"))
                        break
        finally:
            for module in touched:
                for name, placeholder in placeholders.items():
                    if module.__dict__.get(name) is placeholder:
                        delattr(module, name)

        if logger is not None:
            try:
                logger.info(
                    "ltx2_import_compat: transformers %s lacks %s; imported %d LTX-2 pipeline module(s) "
                    "with an inert stand-in",
                    getattr(transformers, "__version__", "unknown"),
                    ", ".join(missing),
                    len(targets) - len(failed),
                )
                for module_name, error in failed:
                    logger.warning(
                        "ltx2_import_compat: %s still fails to import: %s", module_name, error
                    )
            except Exception:  # noqa: BLE001, S110 -- logging must never turn this into a failure
                pass
        _done = not failed
        return _done


def _reset_for_tests() -> None:
    global _done
    with _lock:
        _done = False
