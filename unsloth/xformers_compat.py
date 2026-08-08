# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""torch <-> xformers ABI compatibility, resolved without importing xformers or torch.

Every xformers wheel is compiled against ONE torch release AND one CUDA major. Getting
either wrong does not fail the install: the wheels are ``cp39-abi3`` (or ``py39-none``
from 0.0.35), so a wheel built on CPython 3.10 drops cleanly into 3.13, and
``pip install --no-deps`` never reads the wheel's ``Requires-Dist: torch==X`` at all.
What fails is ``torch.ops.load_library("xformers/_C.pyd")`` at import time, which
``xformers/_cpp_lib.py`` catches and downgrades to a ``logging`` warning -- so the package
imports, reports a version, and silently has no memory-efficient attention. That is
unslothai's NVIDIA P0: a ``cu128``-built wheel shipped next to a ``cu130`` runtime.

Two independent signals live here, both cheap and both offline:

1. ``XFORMERS_TORCH_PINS`` / ``XFORMERS_BUILT_FOR_TORCH`` -- what each xformers release
   declares, and what it was actually compiled against. Usable BEFORE anything is
   installed, which is what picks the wheel.
2. ``xformers_build_metadata()`` -- the installed wheel's own ``xformers/cpp_lib.json``,
   which records the torch version, CUDA version and Python it was built with, e.g.
   ``{"version": {"cuda": 1208, "torch": "2.10.0+cu128", "python": "3.10.11"}}``. This is
   the authority once a wheel is on disk. xformers itself does not *compare* these fields
   -- ``_register_extensions()`` just tries the load and, on ``OSError``, quotes them back
   in ``xFormersInvalidLibException`` -- so reading the file is the only way to know
   *before* the load, and it costs no import and fires no warning.

Pure module: stdlib only, no torch and no xformers import at any point. It lives at the
top of ``unsloth/`` (next to device_type.py / import_fixes.py) rather than in
``unsloth/utils/``, because ``unsloth/utils/__init__.py`` imports attention_dispatch,
which imports ``unsloth.models._utils`` -- the very module that needs this table.

``studio/backend/utils/hardware/hardware.py`` deliberately re-implements the
``cpp_lib.json`` read instead of importing this module: the studio backend runs with
``studio/backend`` on ``sys.path`` and importing ``unsloth.xformers_compat`` would execute
``unsloth/__init__.py``, which drags in torch. Only the ~15-line file read is duplicated,
not the tables.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

__all__ = [
    "XFORMERS_TORCH_PINS",
    "XFORMERS_BUILT_FOR_TORCH",
    "TORCH_TO_XFORMERS",
    "normalize_release",
    "normalize_release_with_post",
    "cuda_major_from_torch_version",
    "format_build_cuda",
    "expected_torch_for_xformers",
    "xformers_for_torch",
    "declared_torch_pin",
    "xformers_build_metadata",
    "xformers_build_summary",
    "describe_xformers_mismatch",
]


# xformers release -> the exact torch release its wheels declare in Requires-Dist.
# Read off the published wheel METADATA (pypi.org/pypi/xformers/<version>/json ->
# info.requires_dist), not guessed; do not extend this by pattern-matching version
# numbers, read the wheel.
#
# 0.0.35 is deliberately absent: it is the first release to declare a RANGE
# (``torch>=2.10``) rather than ``torch==X``, so its declared pin no longer says what it
# was built against and pip can no longer keep the pair consistent on its own. Its build
# torch lives in XFORMERS_BUILT_FOR_TORCH below.
XFORMERS_TORCH_PINS: Dict[str, str] = {
    "0.0.28": "2.4.1",
    "0.0.28.post1": "2.4.1",
    "0.0.28.post2": "2.5.0",
    "0.0.28.post3": "2.5.1",
    "0.0.29": "2.5.1",
    "0.0.29.post1": "2.5.1",
    "0.0.29.post2": "2.6.0",
    "0.0.29.post3": "2.6.0",
    "0.0.30": "2.7.0",
    "0.0.31": "2.7.1",
    "0.0.31.post1": "2.7.1",
    "0.0.32.post1": "2.8.0",
    "0.0.32.post2": "2.8.0",
    "0.0.33": "2.9.0",
    "0.0.33.post1": "2.9.0",
    "0.0.33.post2": "2.9.1",
    "0.0.34": "2.10.0",
}

# xformers release -> the torch release its ``_C`` extension was actually COMPILED
# against, i.e. ``cpp_lib.json``'s ``version.torch`` with the ``+cuXXX`` tag dropped
# (the tag varies per wheel variant, the release does not).
#
# For everything up to 0.0.34 this equals the declared pin, which is the point: the pin
# was trustworthy. 0.0.35 is where they diverge -- it declares ``torch>=2.10`` but its
# ``_C`` is still a single build against 2.10.0, so pip will happily pair it with torch
# 2.11/2.12/2.13 and the extension will not load. There is no xformers release built for
# torch 2.11 or later, so ``xformers_for_torch`` returns None there rather than guessing.
XFORMERS_BUILT_FOR_TORCH: Dict[str, str] = {
    "0.0.33": "2.9.0",
    "0.0.33.post1": "2.9.0",
    "0.0.33.post2": "2.9.1",
    "0.0.34": "2.10.0",
    "0.0.35": "2.10.0",
}

# torch release -> the NEWEST xformers built for it. The inverse of the tables above with
# the post-releases winning, which is what an installer wants: 2.9.0 has both 0.0.33 and
# 0.0.33.post1, and the post is the one to ship. 0.0.35 never wins 2.10.0 over 0.0.34:
# both are built against 2.10.0, and 0.0.34's ``torch==2.10.0`` pin is the one pip can
# still enforce.
TORCH_TO_XFORMERS: Dict[str, str] = {
    "2.4.1": "0.0.28.post1",
    "2.5.0": "0.0.28.post2",
    "2.5.1": "0.0.29.post1",
    "2.6.0": "0.0.29.post3",
    "2.7.0": "0.0.30",
    "2.7.1": "0.0.31.post1",
    "2.8.0": "0.0.32.post2",
    "2.9.0": "0.0.33.post1",
    "2.9.1": "0.0.33.post2",
    "2.10.0": "0.0.34",
}


def normalize_release(version: Any) -> Optional[str]:
    """``'2.10.0+cu130'`` / ``'2.11.0.dev20260101'`` -> ``'2.10.0'`` / ``'2.11.0'``.

    Drops the local tag and any pre-release/dev suffix, matching the release part that
    ``Requires-Dist: torch==X`` compares against. None when unparseable.
    """
    if not version:
        return None
    release = str(version).strip().split("+", 1)[0]
    match = re.match(r"^(\d+(?:\.\d+)*)", release)
    if match is None:
        return None
    return match.group(1)


def normalize_release_with_post(version: Any) -> Optional[str]:
    """Like normalize_release but KEEPS ``.postN``. None for a pre-release.

    xformers keys on the post: 0.0.33 and 0.0.33.post2 are built for different torch
    releases (2.9.0 and 2.9.1), so dropping it would answer the wrong question.

    A ``.dev`` / ``a`` / ``b`` / ``rc`` build is reported as unknown rather than folded
    onto its release: the fourteen ``0.0.35.devNNNN`` wheels on PyPI are built against
    torch nightlies, so answering "0.0.35, therefore torch 2.10.0" for one of them is
    confidently wrong, and unknown is the honest answer.
    """
    if not version:
        return None
    release = str(version).strip().split("+", 1)[0]
    match = re.match(r"^(\d+(?:\.\d+)*(?:\.post\d+)?)$", release)
    if match is None:
        return None
    return match.group(1)


def cuda_major_from_torch_version(torch_version: Any) -> Optional[int]:
    """``'2.10.0+cu130'`` -> 13, ``'2.10.0+cu128'`` -> 12. None for rocm/cpu/tagless.

    Mirrors ``_cuda_major_from_torch_version`` in studio/install_python_stack.py.
    """
    if not torch_version:
        return None
    parts = str(torch_version).split("+", 1)
    if len(parts) < 2 or not parts[1].startswith("cu"):
        return None
    digits = re.sub(r"[^0-9].*", "", parts[1][2:])  # 'cu130' -> '130'
    if not digits:
        return None
    return int(digits) // 10  # '130' -> 13, '128' -> 12, '118' -> 11


def format_build_cuda(build_cuda: Any) -> Optional[str]:
    """cpp_lib.json's integer CUDA version -> ``'12.8'``. None when absent (ROCm/CPU).

    xformers' setup.py stores ``major * 100 + minor``: 1208 is CUDA 12.8, 1300 is 13.0.
    Its own exception message prints the raw integer, which is unreadable, so format it.
    """
    if not isinstance(build_cuda, int) or isinstance(build_cuda, bool):
        return None
    return f"{build_cuda // 100}.{build_cuda % 100}"


def expected_torch_for_xformers(xformers_version: Any) -> Optional[str]:
    """The torch release ``xformers_version`` was built for, or None if unknown to us.

    Prefers the recorded build torch over the declared pin: from 0.0.35 the pin is a
    range and no longer names a single release.
    """
    release = normalize_release_with_post(xformers_version)
    if release is None:
        return None
    built_for = XFORMERS_BUILT_FOR_TORCH.get(release)
    if built_for is not None:
        return built_for
    return XFORMERS_TORCH_PINS.get(release)


def xformers_for_torch(torch_version: Any) -> Optional[str]:
    """The newest xformers release built for ``torch_version``, or None if unknown.

    Takes a full torch version (local tag and all) so callers can pass
    ``torch.__version__`` straight in.
    """
    release = normalize_release(torch_version)
    if release is None:
        return None
    return TORCH_TO_XFORMERS.get(release)


def declared_torch_pin(xformers_version: Any = None) -> Optional[str]:
    """The installed xformers distribution's own ``Requires-Dist: torch==X`` pin.

    Prefers the resident metadata over the static table, so a release we have never seen
    still answers correctly -- but only when it describes the same version the caller
    asked about, otherwise a stale table lookup is the honest answer.

    When the resident pin is a range rather than ``==`` (0.0.35 and later) there is no
    declared pin to report, so this falls back to what the wheel was actually built
    against. Callers must phrase that as "is built for", never "declares": the range
    release deliberately does not declare a single torch.
    """
    wanted = normalize_release_with_post(xformers_version)
    try:
        from importlib.metadata import requires as _requires, version as _version
        resident = normalize_release_with_post(_version("xformers"))
        requirements = _requires("xformers") or ()
    except Exception:
        resident, requirements = None, ()
    if wanted is None or resident is None or wanted == resident:
        for requirement in requirements:
            match = re.match(r"^\s*torch\s*==\s*([0-9][0-9A-Za-z.\-+]*)", str(requirement))
            if match is not None:
                return normalize_release(match.group(1))
    return expected_torch_for_xformers(xformers_version)


def xformers_build_metadata() -> Optional[Dict[str, Any]]:
    """The installed xformers wheel's ``cpp_lib.json``, WITHOUT importing xformers.

    ``importlib.util.find_spec`` only locates the package, it does not execute
    ``xformers/__init__.py`` -- which matters because importing xformers is what emits the
    warning we are trying to explain, and because it drags in torch. Returns None when
    xformers is absent, is an editable/source checkout with no built extension, or ships
    no cpp_lib.json.
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec("xformers")
    except Exception:
        # find_spec raises (not returns None) on a half-removed dist, and ImportError
        # here must never take the caller down: this is diagnostics.
        return None
    if spec is None:
        return None
    locations = list(getattr(spec, "submodule_search_locations", None) or ())
    origin = getattr(spec, "origin", None)
    if origin:
        locations.append(os.path.dirname(origin))
    for location in locations:
        path = os.path.join(location, "cpp_lib.json")
        try:
            with open(path, "r", encoding = "utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            continue
        if isinstance(metadata, dict) and isinstance(metadata.get("version"), dict):
            return metadata
    return None


def xformers_build_summary(
    build_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Optional[str]]]:
    """``cpp_lib.json`` -> ``{"torch": ..., "cuda": "12.8", "python": ...}`` for reporting.

    None when there is no build metadata to summarise. Every value is a string or None,
    so this drops straight into a JSON API response.
    """
    if build_metadata is None:
        build_metadata = xformers_build_metadata()
    version_block = (build_metadata or {}).get("version")
    if not isinstance(version_block, dict):
        return None
    built_torch = version_block.get("torch")
    built_python = version_block.get("python")
    built_hip = version_block.get("hip")
    return {
        "torch": str(built_torch) if built_torch else None,
        "cuda": format_build_cuda(version_block.get("cuda")),
        "hip": str(built_hip) if built_hip else None,
        "python": str(built_python) if built_python else None,
    }


def _running_detail(torch_version: Any, python_version: Any = None) -> str:
    """``'torch 2.10.0+cu130'``, plus Python when the caller supplied it."""
    detail = f"torch {torch_version}"
    if python_version:
        detail += f" / Python {python_version}"
    return detail


def _built_detail(summary: Dict[str, Optional[str]]) -> str:
    """``'torch 2.10.0+cu128 / Python 3.10.11'`` from a build summary."""
    parts = []
    if summary.get("torch"):
        parts.append(f"torch {summary['torch']}")
    elif summary.get("cuda"):
        parts.append(f"CUDA {summary['cuda']}")
    if summary.get("python"):
        parts.append(f"Python {summary['python']}")
    return " / ".join(parts) if parts else "an unknown build"


def describe_xformers_mismatch(
    torch_version: Any,
    torch_cuda: Any = None,
    xformers_version: Any = None,
    build_metadata: Optional[Dict[str, Any]] = None,
    python_version: Any = None,
) -> Optional[str]:
    """One sentence naming why this xformers cannot load its kernels here, else None.

    ``torch_version`` is ``torch.__version__`` (local tag included -- it carries the CUDA
    family) and ``torch_cuda`` is ``torch.version.cuda``. Both the recorded build metadata
    and the declared pin are consulted; the build metadata wins because it describes the
    binary that is actually on disk. Returns None when the two agree, when xformers is
    absent, or when there is not enough information to be sure -- this must never cry
    wolf, the caller prints it as a warning on the default path.

    A Python-version difference alone is NOT a mismatch and never triggers this: the
    wheels are abi3/none-tagged and ``_C`` is loaded through ``torch.ops.load_library``,
    not the CPython ABI, so 3.10-built kernels run fine on 3.13. It is still reported as
    context, because xformers' own message leads with it and users chase it first.
    """
    running_release = normalize_release(torch_version)
    if running_release is None:
        return None

    if build_metadata is None:
        build_metadata = xformers_build_metadata()
    summary = xformers_build_summary(build_metadata) or {}
    built_torch = summary.get("torch")
    built_release = normalize_release(built_torch)
    running = _running_detail(torch_version, python_version)

    if built_release is not None and built_release != running_release:
        return (
            f"xformers was built for {_built_detail(summary)} but you are running "
            f"{running}; its C++/CUDA extensions cannot load, so memory-efficient "
            f"attention is unavailable"
        )

    # Same torch release, different CUDA major: the case NVIDIA hit (a cu128 wheel beside
    # a cu130 runtime). Majors only -- CUDA minor version compatibility means a cu126
    # wheel loads fine against a cu128 torch, and flagging that would be crying wolf.
    built_cuda = summary.get("cuda")
    running_cuda_major = cuda_major_from_torch_version(torch_version)
    if running_cuda_major is None and torch_cuda:
        try:
            running_cuda_major = int(str(torch_cuda).split(".", 1)[0])
        except (TypeError, ValueError):
            running_cuda_major = None
    if built_cuda is not None and running_cuda_major is not None:
        built_cuda_major = int(built_cuda.split(".", 1)[0])
        if built_cuda_major != running_cuda_major:
            return (
                f"xformers was built for {_built_detail(summary)} but you are running "
                f"{running} (CUDA {running_cuda_major}.x); its C++/CUDA extensions cannot "
                f"load, so memory-efficient attention is unavailable"
            )

    # No build metadata (source install, or a wheel that ships none): fall back to the
    # declared torch pin, which at least catches a wholesale torch-release mismatch.
    if built_release is None:
        pinned = declared_torch_pin(xformers_version)
        if pinned is not None and pinned != running_release:
            return (
                f"xformers {xformers_version or 'installed'} is built for torch {pinned} "
                f"but you are running {running}; its C++/CUDA extensions cannot load, so "
                f"memory-efficient attention is unavailable"
            )
    return None
