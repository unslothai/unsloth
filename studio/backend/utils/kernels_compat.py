# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keeps ``kernels`` at a version the installed transformers imports: transformers < 5.13 builds unversioned
``LayerRepository`` objects at import, which kernels >= 0.15 rejects. Stdlib only."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
import threading
from typing import Any, Optional

# flash_4_hub's floor and the installer's pin (requirements/extras-no-deps.txt).
KERNELS_FLOOR = "0.12.3"
# First kernels release that rejects a LayerRepository without a revision or version.
KERNELS_STRICT = (0, 15)
# First transformers release whose hub_kernels mapping passes version= to every LayerRepository.
TRANSFORMERS_KERNELS_STRICT_OK = (5, 13)
# kernels >= 0.14.1 requires huggingface-hub >= 1.10.0.
_KERNELS_HUB_FLOOR = (1, 10)
KERNELS_CONFLICT_MARKER = "Either a revision or a version must be specified"

_ATTENTION_INSTALL_ENV = "UNSLOTH_DIFFUSION_ATTENTION_INSTALL"
_VERIFY_TIMEOUT_S = 180
_VERIFY_CODE = "import transformers.activations"

_heal_lock = threading.Lock()
_healed = False
_hidden_reason: Optional[str] = None


def _dist_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except Exception:  # noqa: BLE001 - missing or broken metadata reads as not installed
        return None


def _release(text: Optional[str]) -> Optional[tuple[int, ...]]:
    if not text:
        return None
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)", text)
    if m is None:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def _is_prerelease(text: str) -> bool:
    return re.search(r"(dev|a|alpha|b|beta|rc|c|pre|preview)\d*$", text.strip().lower()) is not None


def transformers_accepts_strict_kernels(transformers_version: Optional[str]) -> bool:
    """Known to import next to kernels >= 0.15 (not a 5.13.0 pre-release or an unknown version)."""
    have = _release(transformers_version)
    if have is None:
        return False
    if _is_prerelease(transformers_version or "") and tuple(
        have[:3]
    ) == TRANSFORMERS_KERNELS_STRICT_OK + (0,):
        return False
    return have[:2] >= TRANSFORMERS_KERNELS_STRICT_OK


def kernels_install_spec(
    transformers_version: Optional[str] = None, hub_version: Optional[str] = None
) -> str:
    """Pip requirement for the ``--no-deps`` install. Exact below 5.13: 0.14.x needs ``kernels-data``, which
    ``--no-deps`` leaves out."""
    hub = _release(hub_version)
    if not transformers_accepts_strict_kernels(transformers_version) or (
        hub is not None and hub[:2] < _KERNELS_HUB_FLOOR
    ):
        return f"kernels=={KERNELS_FLOOR}"
    return f"kernels>={KERNELS_FLOOR}"


def current_kernels_install_spec() -> str:
    return kernels_install_spec(_dist_version("transformers"), _dist_version("huggingface_hub"))


def predicted_conflict(kernels_version: Optional[str], transformers_version: Optional[str]) -> bool:
    kv = _release(kernels_version)
    if kv is None or transformers_version is None or _release(transformers_version) is None:
        return False
    return kv[:2] >= KERNELS_STRICT and not transformers_accepts_strict_kernels(
        transformers_version
    )


def verify_transformers_import(timeout: float = _VERIFY_TIMEOUT_S) -> tuple[Optional[bool], str]:
    """Fresh-interpreter import check: True, False (with stderr tail) or None (could not tell)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _VERIFY_CODE],
            capture_output = True,
            timeout = timeout,
            env = dict(os.environ),
        )
    except subprocess.TimeoutExpired:
        return None, f"the import check timed out after {timeout:.0f}s"
    except Exception as exc:  # noqa: BLE001
        return None, f"the import check could not run ({type(exc).__name__}: {exc})"
    if proc.returncode == 0:
        return True, ""
    err = proc.stderr
    if isinstance(err, bytes):
        err = err.decode("utf-8", errors = "replace")
    lines = [ln for ln in (err or "").strip().splitlines() if ln.strip()]
    return False, (lines[-1] if lines else f"exit code {proc.returncode}")[-500:]


def hide_kernels(reason: str) -> None:
    """Make ``kernels`` unimportable here; transformers' hub_kernels tolerates the ImportError."""
    global _hidden_reason
    for name in [m for m in sys.modules if m == "kernels" or m.startswith("kernels.")]:
        sys.modules.pop(name, None)
    sys.modules["kernels"] = None
    _hidden_reason = reason
    _reset_import_caches()


def hidden_reason() -> Optional[str]:
    return _hidden_reason if sys.modules.get("kernels", object()) is None else None


def _reset_import_caches() -> None:
    importlib.invalidate_caches()
    for module_name in ("transformers.utils.import_utils", "diffusers.utils.import_utils"):
        mod = sys.modules.get(module_name)
        fn = getattr(mod, "is_kernels_available", None) if mod is not None else None
        cache_clear = getattr(fn, "cache_clear", None)
        if callable(cache_clear):
            try:
                cache_clear()
            except Exception:  # noqa: BLE001
                pass
    mod = sys.modules.get("diffusers.utils.import_utils")
    if mod is not None and sys.modules.get("kernels", object()) is None:
        for attr, value in (("_kernels_available", False), ("_kernels_version", "N/A")):
            if hasattr(mod, attr):
                setattr(mod, attr, value)


def _pip(args: list[str], timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", *args],
        capture_output = True,
        timeout = timeout,
        check = True,
    )


def restore_kernels(previous_version: Optional[str]) -> bool:
    try:
        if previous_version is None:
            _pip(["uninstall", "-y", "kernels"])
        else:
            _pip(["install", "--only-binary", ":all:", "--no-deps", f"kernels=={previous_version}"])
        importlib.invalidate_caches()
        return True
    except Exception:  # noqa: BLE001
        return False


def check_kernels_install(previous_version: Optional[str], logger: Any = None) -> bool:
    """True when transformers still imports after the install; else roll back and hide kernels."""
    if _dist_version("transformers") is None:
        return True
    ok, detail = verify_transformers_import()
    if ok is not False:
        if ok is None and logger is not None:
            logger.warning(
                "diffusion.attention: could not confirm transformers still imports after installing kernels %s: %s",
                _dist_version("kernels"),
                detail,
            )
        return True
    installed = _dist_version("kernels")
    restored = restore_kernels(previous_version)
    reason = (
        f"kernels {installed} breaks `import transformers.activations` with transformers "
        f"{_dist_version('transformers')} ({detail})"
    )
    hide_kernels(reason)
    if logger is not None:
        logger.warning(
            "diffusion.attention: %s; %s, using the default attention backend",
            reason,
            (
                (
                    "removed it again"
                    if previous_version is None
                    else f"restored kernels {previous_version}"
                )
                if restored
                else "could not roll the install back"
            ),
        )
    return False


def heal_kernels_conflict(logger: Any = None) -> Optional[str]:
    """Once per process before diffusers imports: repair or hide a kernels transformers cannot import."""
    global _healed
    if _healed:
        return None
    with _heal_lock:
        if _healed:
            return None
        try:
            return _heal(logger)
        except Exception as exc:  # noqa: BLE001 - must never break an import
            if logger is not None:
                logger.debug("kernels compatibility check failed: %r", exc)
            return None
        finally:
            # Set only once the repair is done, so a concurrent caller waits on the lock instead of importing.
            _healed = True


def _heal(logger: Any) -> Optional[str]:
    kernels_version = _dist_version("kernels")
    transformers_version = _dist_version("transformers")
    if not predicted_conflict(kernels_version, transformers_version):
        return None
    if sys.modules.get("kernels", object()) is None:
        return None
    ok, detail = verify_transformers_import()
    if ok is not False or KERNELS_CONFLICT_MARKER not in detail:
        return None
    spec = kernels_install_spec(transformers_version, _dist_version("huggingface_hub"))
    gate = os.environ.get(_ATTENTION_INSTALL_ENV, "auto").strip().lower()
    already_imported = "kernels" in sys.modules
    if gate not in ("0", "false", "no", "off"):
        try:
            _pip(["install", "--only-binary", ":all:", "--no-deps", spec])
            importlib.invalidate_caches()
            ok, detail = verify_transformers_import()
        except Exception as exc:  # noqa: BLE001
            stderr = getattr(exc, "stderr", None)
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors = "replace")
            ok, detail = False, ((stderr or "").strip()[-300:] or str(exc))
        if ok:
            if not already_imported:
                if logger is not None:
                    logger.warning(
                        "kernels %s cannot be imported by transformers %s; reinstalled %s (now %s)",
                        kernels_version,
                        transformers_version,
                        spec,
                        _dist_version("kernels"),
                    )
                _reset_import_caches()
                return f"repaired: {spec}"
    reason = (
        f"kernels {kernels_version} cannot be imported by transformers {transformers_version} "
        f"({KERNELS_CONFLICT_MARKER})"
    )
    hide_kernels(reason)
    if logger is not None:
        logger.warning(
            "%s; Hugging Face hub kernels are disabled in this process%s. To repair, run: "
            "python -m pip install --no-deps '%s'",
            reason,
            " (the repaired version takes effect after Unsloth restarts)" if ok else "",
            spec,
        )
    return f"hidden: {reason}"
