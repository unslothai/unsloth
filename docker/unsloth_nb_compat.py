# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Per-notebook transformers version activation for the Unsloth Docker image.

Problem: unslothai/notebooks pin many different transformers versions in their
install cells (transformers==4.56.2 on ~115, 5.5.0/5.3.0/5.10.x on newer model
families). The baked base venv ships ONE transformers (latest 5.x). Running an
old-model notebook against it, or letting the install cell pip-install a pinned
version on top, either breaks the model or clobbers the cu128 torch/vLLM stack.

Solution (mirrors Unsloth Studio's studio/backend/utils/transformers_version.py):
keep the base venv intact and ship coherent transformers "sidecars" -- each is a
`pip install --target <dir> --no-deps transformers==X` plus the matched
huggingface_hub/tokenizers/safetensors. To use version X we just prepend its
sidecar dir to sys.path BEFORE transformers is imported; the rest of the stack
(torch, vllm, unsloth, peft, trl) comes from the base venv unchanged. Verified:
base unsloth loads + generates under both a 4.57.6 and a 5.5.0 sidecar on B200.

Two activation paths:
  * driven/headless: `unsloth-run <notebook>` sets PYTHONPATH at kernel launch.
  * manual JupyterLab: an IPython pre_run_cell hook (registered by the baked
    startup file) activates the sidecar before the first model cell, using the
    version the notebook's own install cell asked for (recorded by the pip shim).
"""

from __future__ import annotations
import os, sys, glob, json

SIDECAR_ROOT = os.environ.get("UNSLOTH_TF_SIDECAR_ROOT", "/opt/unsloth-venv/tf-sidecars")
# The pip/uv shim writes the transformers version a notebook asked for here.
MARKER = os.environ.get("UNSLOTH_NB_TF_MARKER", "/tmp/unsloth_nb/requested_transformers")


def _logging_enabled() -> bool:
    """Sidecar activation is silent by default; users found the per-cell
    `[unsloth-nb] activated transformers sidecar ...` line noisy. Set
    UNSLOTH_ENABLE_LOGGING=1 to surface it (and other [unsloth-nb] diagnostics)."""
    return os.environ.get("UNSLOTH_ENABLE_LOGGING", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
        "off",
    )


# Model-name -> minimum transformers tier (substring match on the lowered id),
# ported from Studio. Fallback when a notebook names a new model but pins nothing.
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
    """Return {version_str: dir} for every baked sidecar."""
    out = {}
    for d in sorted(glob.glob(os.path.join(SIDECAR_ROOT, "t_*"))):
        out[os.path.basename(d)[2:].replace("_", ".")] = d
    return out


def tier_for_model(model_name: str):
    """Best-effort minimum transformers version for a model id (or None)."""
    if not model_name:
        return None
    low = model_name.lower()
    # check newest tiers first so gemma-4-12b wins over gemma-4
    for ver in ("5.10.2", "5.5.0", "5.3.0"):
        if any(s in low for s in _TIER_SUBSTRINGS[ver]):
            return ver
    return None


def sidecar_for(version: str):
    """Map a requested/needed transformers version to a baked sidecar dir.

    Uses ceiling semantics: the smallest baked version >= the request, because a
    model added in version X needs *at least* X. If the request is newer than
    every baked sidecar, return None -> use the base venv (the newest 5.x)."""
    baked = _baked()
    if not baked or not version:
        return None
    if version in baked:
        return baked[version]
    try:
        from packaging.version import Version
        want = Version(version)
    except Exception:
        return None
    ge = sorted((Version(v), d) for v, d in baked.items() if Version(v) >= want)
    return ge[0][1] if ge else None


def requested_version():
    """transformers version a notebook asked for (recorded by the pip shim)."""
    try:
        with open(MARKER) as f:
            v = f.read().strip()
        return v or None
    except OSError:
        return None


def activate(version: str | None, *, quiet: bool = False):
    """Prepend the matching sidecar to sys.path if transformers isn't imported yet.

    Returns the activated dir, or None if the base venv is used / activation is
    no longer possible (transformers already imported)."""
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
    """Resolve the version to use: the notebook's pin first, else the model tier."""
    return requested_version() or tier_for_model(model_name or "")


# -- manual JupyterLab integration: activate before the first model cell --------
def _pre_run_cell(_info = None):
    v = requested_version()
    if v and "transformers" not in sys.modules:
        activate(v)


def register_ipython():
    """Register the pre_run_cell hook (called from the baked IPython startup)."""
    try:
        ip = get_ipython()  # noqa: F821 (provided by IPython)
    except NameError:
        return
    if ip is not None and not getattr(ip, "_unsloth_tf_hook", False):
        ip.events.register("pre_run_cell", _pre_run_cell)
        ip._unsloth_tf_hook = True
