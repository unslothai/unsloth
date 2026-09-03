# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Per-notebook transformers version activation for the Unsloth Docker image.

The image bakes coherent transformers "sidecars" and prepends one to sys.path before
transformers is imported. Mirrors studio/backend/utils/transformers_version.py.

Selection has a FLOOR as well as a ceiling (see sidecar_for): vLLM is version-locked
to transformers, so a sidecar older than the baked vLLM accepts is not an older
transformers, it is an ImportError at `import unsloth`.
"""

from __future__ import annotations
import os, sys, glob, json

SIDECAR_ROOT = os.environ.get("UNSLOTH_TF_SIDECAR_ROOT", "/opt/unsloth-venv/tf-sidecars")
MARKER = os.environ.get("UNSLOTH_NB_TF_MARKER", "/tmp/unsloth_nb/requested_transformers")

# Lowest transformers the baked vLLM can import. Written by the Dockerfile's sidecar
# verification step, so it tracks the vLLM the image bakes rather than a literal.
FLOOR_FILE = os.path.join(SIDECAR_ROOT, ".vllm_min_transformers")


def _logging_enabled() -> bool:
    """Sidecar activation is silent by default; UNSLOTH_ENABLE_LOGGING=1 surfaces it."""
    return os.environ.get("UNSLOTH_ENABLE_LOGGING", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


# fallback when a notebook names a new model but pins nothing; ported from Studio
_TIER_SUBSTRINGS = {
    "5.10.2": ("gemma-4-12b", "gemma4-12b"),
    "5.5.0": ("gemma-4", "gemma4", "qwen3.6"),
    "5.3.0": (
        "ministral-3",
        "glm-4.7-flash",
        "qwen3-30b-a3b",
        "qwen3.5",
        "qwen3-next",
        "qwen3_5",
        "lfm2.5-vl",
    ),
}


def _baked():
    out = {}
    for d in sorted(glob.glob(os.path.join(SIDECAR_ROOT, "t_*"))):
        out[os.path.basename(d)[2:].replace("_", ".")] = d
    return out


def min_version():
    """Lowest transformers this image's vLLM can import; None keeps the pre-floor
    behaviour. UNSLOTH_TF_SIDECAR_MIN overrides."""
    v = os.environ.get("UNSLOTH_TF_SIDECAR_MIN", "").strip()
    if v:
        return v
    try:
        with open(FLOOR_FILE) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _eligible():
    baked = _baked()
    if not baked:
        return []
    try:
        from packaging.version import Version
    except Exception:
        return None
    floor = min_version()
    try:
        low = Version(floor) if floor else None
    except Exception:
        low = None
    rows = []
    for v, d in baked.items():
        try:
            ver = Version(v)
        except Exception:
            continue
        if low is not None and ver < low:
            continue  # vLLM cannot import it
        rows.append((ver, v, d))
    rows.sort()
    return rows


def tier_for_model(model_name: str):
    if not model_name:
        return None
    low = model_name.lower()
    # newest tiers first, so gemma-4-12b wins over gemma-4
    for ver in ("5.10.2", "5.5.0", "5.3.0"):
        if any(s in low for s in _TIER_SUBSTRINGS[ver]):
            return ver
    return None


def sidecar_for(version: str):
    """FLOOR then CEILING: a sidecar the baked vLLM cannot import is never eligible
    whatever the notebook pinned, so a request below the floor clamps UP; among the
    eligible ones take the smallest >= the request, since a model added in X needs at
    least X. Above every eligible sidecar returns None, i.e. the base venv."""
    if not version:
        return None
    rows = _eligible()
    if rows is None:  # no packaging: only an exact, still-eligible match is safe
        baked = _baked()
        d = baked.get(version)
        floor = min_version()
        return d if (d and (not floor or version == floor)) else None
    if not rows:
        return None
    for _ver, v, d in rows:
        if v == version:
            return d
    try:
        from packaging.version import Version
        want = Version(version)
    except Exception:
        return None
    for ver, _v, d in rows:
        if ver >= want:
            return d
    return None


def requested_version():
    try:
        with open(MARKER) as f:
            v = f.read().strip()
        return v or None
    except OSError:
        return None


def activate(version: str | None, *, quiet: bool = False):
    """Prepend the matching sidecar to sys.path, if transformers is not imported yet."""
    if not version:
        return None
    d = sidecar_for(version)
    if not d:
        return None
    if "transformers" in sys.modules:
        if not quiet:
            print(
                f"[unsloth-nb] transformers already imported; cannot switch to "
                f"{version} in-process (restart the kernel, or use `unsloth-run`).",
                file = sys.stderr,
            )
        return None
    if d not in sys.path:
        sys.path.insert(0, d)
    os.environ["PYTHONPATH"] = d + os.pathsep + os.environ.get("PYTHONPATH", "")
    if not quiet and _logging_enabled():
        print(f"[unsloth-nb] activated transformers sidecar for {version}: {d}")
    return d


def resolve(model_name: str | None = None):
    return requested_version() or tier_for_model(model_name or "")


def _pre_run_cell(_info = None):
    v = requested_version()
    if v and "transformers" not in sys.modules:
        activate(v)


def register_ipython():
    try:
        ip = get_ipython()  # noqa: F821 (provided by IPython)
    except NameError:
        return
    if ip is not None and not getattr(ip, "_unsloth_tf_hook", False):
        ip.events.register("pre_run_cell", _pre_run_cell)
        ip._unsloth_tf_hook = True
