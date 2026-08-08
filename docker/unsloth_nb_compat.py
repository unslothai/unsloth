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
(torch, vllm, unsloth, peft, trl) comes from the base venv unchanged.

That "rest of the stack" is the catch, and it is why selection has a FLOOR as
well as a ceiling (see sidecar_for): vLLM is version-locked to transformers, so a
sidecar older than what the baked vLLM accepts does not give the notebook an
older transformers, it gives it an ImportError at `import unsloth`. The image
therefore only ships sidecars whose vLLM import has been verified at build time,
and records the lowest of them as the floor.

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

# Lowest transformers the image's baked vLLM can import. A sidecar below this is
# not "an older transformers", it is a BROKEN image: `import unsloth` dies before
# the first model cell. Written by the Dockerfile's sidecar verification step
# (which imports vllm.transformers_utils.config under every candidate and drops
# the ones that raise), so it tracks whatever vLLM the image actually bakes
# instead of a literal that rots on the next bump. Measured on vLLM 0.26.0:
#
#   transformers 4.57.6  FAIL  "Support for Transformers v4 ... removed in vLLM v0.24.0"
#   transformers 5.3.0   FAIL  "cannot import name 'ALLOWED_LAYER_TYPES'"
#   transformers 5.5.0   OK
#   transformers 5.10.2  OK
#   transformers 5.14.1  OK (the baked one, no sidecar)
FLOOR_FILE = os.path.join(SIDECAR_ROOT, ".vllm_min_transformers")


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


def min_version():
    """Lowest transformers this image's vLLM can import, or None if unrecorded.

    UNSLOTH_TF_SIDECAR_MIN overrides, so a hand-mounted sidecar root can declare
    its own floor. Returns None when neither is set, which keeps the pre-floor
    behaviour for any environment that never ran the build-time verification."""
    v = os.environ.get("UNSLOTH_TF_SIDECAR_MIN", "").strip()
    if v:
        return v
    try:
        with open(FLOOR_FILE) as f:
            return f.read().strip() or None
    except OSError:
        return None


def _eligible():
    """Baked sidecars the floor allows, as a sorted [(Version, version_str, dir)].

    Returns None when the versions cannot be parsed (no packaging available)."""
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
            continue  # vLLM cannot import it; activating it only breaks the run
        rows.append((ver, v, d))
    rows.sort()
    return rows


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

    FLOOR then CEILING, in that order:

      * floor -- a sidecar the baked vLLM cannot import is never eligible, no
        matter what the notebook pinned. Selecting one used to break `import
        unsloth` in 254 of the 433 shipped notebooks, because the two common pin
        families (4.5x -> the 4.57.6 sidecar, 5.2/5.3 -> the 5.3.0 sidecar) both
        landed on a sidecar vLLM 0.26.0 refuses. A request below the floor is
        clamped UP to the lowest eligible sidecar: that is the closest version to
        what the notebook asked for that this image can actually run.
      * ceiling -- among the eligible sidecars pick the smallest >= the request,
        because a model added in version X needs *at least* X.

    A request newer than every eligible sidecar returns None -> use the base venv
    (the newest 5.x), which is always vLLM-compatible."""
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
