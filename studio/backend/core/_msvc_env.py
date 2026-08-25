# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gate torch.compile on Triton's C toolchain being usable, on Windows. See #7595.

On Windows, `import triton` succeeding does NOT mean a Triton compile will
succeed. Triton's AMD/HIP driver JIT-compiles `hip_utils.c` with `clang-cl` on
first GPU touch, and without the CRT headers that dies with:

    fatal error: 'stdlib.h' file not found

Where those headers come from is the part worth being precise about, because it
decides what this module can usefully do. `triton/runtime/build.py` calls
`triton.windows_utils.find_msvc_winsdk()` and passes the result to the compiler
as `/I` flags; that search reads `VCINSTALLDIR`, vswhere, `PATH`, the registry
and the well-known Program Files roots. It never reads `INCLUDE`. So Triton
finds its own toolchain whenever one is installed, and `INCLUDE` only matters as
the compiler's own fallback when Triton passes no `/I` at all.

That leaves nothing for us to repair: when Visual Studio is present Triton
already works, and when it is absent there is no toolchain to point it at. What
IS worth doing is refusing to enable torch.compile when the headers are
unreachable, so the user gets an actionable message instead of a hard crash on
the first compile.

`gate_torch_compile_on_windows()` is what the workers call. It asks Triton's own
discovery (the same oracle the real compile uses, so it cannot disagree with
what actually happens), falling back to `INCLUDE` for Triton builds old enough
to rely on it.
"""

from __future__ import annotations

import os
import sys
import logging

logger = logging.getLogger(__name__)


def _have_crt_headers() -> bool:
    """True if stdlib.h is reachable through the current INCLUDE.

    Only the compiler's fallback path: modern triton-windows passes explicit
    /I dirs and never consults INCLUDE.
    """
    for d in os.environ.get("INCLUDE", "").split(os.pathsep):
        if d and os.path.isfile(os.path.join(d, "stdlib.h")):
            return True
    return False


def _triton_finds_crt_headers() -> bool:
    """True if Triton's own MSVC/WinSDK search turns up the CRT headers.

    This is the search `triton/runtime/build.py` runs to build its `/I` list, so
    asking it here is the same question the real compile will ask.
    """
    try:
        from triton.windows_utils import find_msvc_winsdk
    except Exception:  # noqa: BLE001 -- absent/older/broken Triton: fall back to INCLUDE
        return False
    try:
        _, inc_dirs, _ = find_msvc_winsdk()
    except Exception:  # noqa: BLE001 -- discovery is best-effort, never fatal
        logger.debug("Triton's MSVC/WinSDK discovery raised", exc_info = True)
        return False
    return any(d and os.path.isfile(os.path.join(d, "stdlib.h")) for d in inc_dirs)


def _needs_msvc_headers() -> bool:
    """Whether the compiler Triton will actually pick needs the MSVC/SDK headers.

    Only cl.exe and clang-cl do. `get_cc()` prefers the ROCm wheel's clang-cl, which is why AMD
    hits this and nobody else does; without that wheel it falls through to the bundled TinyCC,
    which carries its own headers and is never passed a `/I` for the SDK. Gating a TinyCC box on
    MSVC would disable torch.compile on the ordinary Windows NVIDIA install, which works fine.
    """
    try:
        from triton.runtime.build import get_cc, is_clang_cl, is_msvc
    except Exception:  # noqa: BLE001 -- older/absent Triton: assume the MSVC path, as before
        return True
    try:
        cc = get_cc()
    except Exception:  # noqa: BLE001 -- no compiler at all is its own failure, not ours to judge
        logger.debug("Triton's compiler selection raised", exc_info = True)
        return True
    return bool(is_msvc(cc) or is_clang_cl(cc))


def crt_headers_reachable() -> bool:
    """Whether a Triton compile in THIS process will find the CRT headers.

    True off win32, where none of this applies.
    """
    if sys.platform != "win32":
        return True
    if not _needs_msvc_headers():
        return True
    return _triton_finds_crt_headers() or _have_crt_headers()


def gate_torch_compile_on_windows(log: logging.Logger) -> None:
    """Disable torch.compile unless Triton *and* its C toolchain are usable.

    No-op off win32. Workers call this before importing torch: Triton being
    importable is necessary but not sufficient, because its clang-cl JIT still
    needs the CRT headers (#7595).
    """
    if sys.platform != "win32":
        return
    try:
        import triton  # noqa: F401
    except ImportError:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        log.warning(
            "Triton not found on Windows — torch.compile disabled. "
            'Install for better performance: pip install "triton-windows<3.7"'
        )
        return

    if crt_headers_reachable():
        log.info("Triton available — torch.compile enabled")
        return
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    # Deliberately not promising that everything now works. This turns off the compiles we own
    # (inductor's), which is the whole of the diffusion path. It cannot turn off a kernel written
    # as @triton.jit, and Unsloth's own kernels are, so a train still needs the toolchain.
    log.warning(
        "Triton is installed but its C toolchain has no CRT headers, so its "
        "clang-cl JIT would fail on 'stdlib.h' (#7595). torch.compile disabled; "
        "directly launched Triton kernels still need the toolchain. "
        "Install Visual Studio Build Tools (C++ workload): winget "
        'install Microsoft.VisualStudio.2022.BuildTools --override "--add '
        'Microsoft.VisualStudio.Workload.VCTools --includeRecommended".'
    )
