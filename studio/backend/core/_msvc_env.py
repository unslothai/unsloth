# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gate torch.compile on Triton's C toolchain, on Windows (#7595). `import triton` succeeding does not
mean a compile will: Triton's clang-cl JIT dies on `'stdlib.h'` mid-run when the CRT headers are absent.
It passes its own `/I` dirs and never reads `INCLUDE`, so there is nothing to repair, only a state to refuse."""

from __future__ import annotations

import os
import sys
import logging

logger = logging.getLogger(__name__)


def _headers_complete(dirs) -> bool:
    """`stdlib.h` alone is a standalone SDK: its ucrt pulls in `vcruntime.h` from the VC toolset,
    so the compile still dies there (measured with the toolset dir removed). Both, or not whole."""

    def found(name: str) -> bool:
        return any(d and os.path.isfile(os.path.join(d, name)) for d in dirs)

    return found("stdlib.h") and found("vcruntime.h")


def _have_crt_headers() -> bool:
    return _headers_complete(os.environ.get("INCLUDE", "").split(os.pathsep))


def _triton_finds_crt_headers() -> bool | None:
    """None means the search could not be run, which is not the same as running and finding nothing.
    Only the latter may gate: `find_msvc_winsdk` is private, so a rename or a changed return arity
    would otherwise read as "no headers" and disable torch.compile on a machine with Visual Studio."""
    try:
        from triton.windows_utils import find_msvc_winsdk  # noqa: PLC0415
        _, inc_dirs, _ = find_msvc_winsdk()
    except Exception:  # noqa: BLE001
        logger.debug("Triton's MSVC/WinSDK discovery is unavailable", exc_info = True)
        return None
    return _headers_complete(inc_dirs)


def _triton_is_triton_windows() -> bool:
    """XPU Triton and triton-windows own the same top-level name; only the distribution says which."""
    try:
        import importlib.metadata as md  # noqa: PLC0415
        dists = md.packages_distributions().get("triton") or ()
    except Exception:  # noqa: BLE001
        logger.debug("Could not resolve which distribution owns `triton`", exc_info = True)
        return False
    return any(d.lower().replace("_", "-") == "triton-windows" for d in dists)


def _rocm_clang_cl_present() -> bool:
    import sysconfig  # noqa: PLC0415
    return os.path.isfile(
        os.path.join(
            sysconfig.get_path("platlib"), "_rocm_sdk_core", "lib", "llvm", "bin", "clang-cl.exe"
        )
    )


def _cc_needs_msvc_headers(cc: str) -> bool:
    """Triton's own predicates decide, so this cannot disagree with what it does. They are imported
    apart from `get_cc` because `is_clang_cl` only arrived in triton-windows 3.5.1.post23: importing
    all three together failed on every earlier release and threw away a usable `get_cc` with it.
    The fallback is what both predicates do, a case-insensitive basename match."""
    try:
        from triton.runtime.build import is_clang_cl, is_msvc  # noqa: PLC0415
        return bool(is_msvc(cc) or is_clang_cl(cc))
    except Exception:  # noqa: BLE001
        logger.debug("Triton's compiler predicates are unavailable", exc_info = True)
        return os.path.basename(str(cc)).lower() in ("cl", "cl.exe", "clang-cl", "clang-cl.exe")


def _triton_cc() -> str:
    """triton-windows 3.8.0.post28, what a bare `pip install triton-windows` gives you today, has no
    `get_cc` at all: it was renamed `_find_compiler(language)`, whose "c" branch is the old body.
    Without this the newest release never reaches the compiler question and answers from the wheel
    layout instead."""
    try:
        from triton.runtime.build import get_cc  # noqa: PLC0415
    except ImportError:
        from triton.runtime.build import _find_compiler  # noqa: PLC0415
        return _find_compiler("c")
    return get_cc()


def _needs_msvc_headers() -> bool:
    """Only cl and clang-cl need them; Triton otherwise picks bundled TinyCC, which carries its own.
    The fallback needs both halves: an in-place ROCm-to-XPU repair leaves the compiler on disk without
    the Triton that selects it. Unanswerable means ungated: gating wrongly breaks a working box."""
    try:
        cc = _triton_cc()
    except Exception:  # noqa: BLE001
        logger.debug("Triton's compiler selection is unavailable", exc_info = True)
        return _triton_is_triton_windows() and _rocm_clang_cl_present()
    return _cc_needs_msvc_headers(cc)


def _toolchain_summary() -> str:
    """0 include dirs means no Visual Studio; dirs missing only `vcruntime.h` means an SDK
    without the VC toolset."""
    try:
        cc = os.path.basename(_triton_cc())
    except Exception:  # noqa: BLE001
        cc = "unknown"
    try:
        from triton.windows_utils import find_msvc_winsdk  # noqa: PLC0415
        _, inc_dirs, _ = find_msvc_winsdk()
    except Exception:  # noqa: BLE001
        inc_dirs = []
    logger.debug("Triton include dirs: %s", list(inc_dirs))
    missing = [
        h
        for h in ("stdlib.h", "vcruntime.h")
        if not any(d and os.path.isfile(os.path.join(d, h)) for d in inc_dirs)
    ]
    return (
        f"compiler={cc}, Triton include dirs={len(inc_dirs)}, "
        f"missing headers={','.join(missing) or 'none'}, "
        f"INCLUDE={'set' if os.environ.get('INCLUDE') else 'unset'}"
    )


def crt_headers_reachable() -> bool:
    if sys.platform != "win32":
        return True
    if not _needs_msvc_headers():
        return True
    if _have_crt_headers():
        return True
    # INCLUDE above is a positive signal only. A machine with Visual Studio but no INCLUDE is the
    # normal case (Studio is not launched from a Developer Command Prompt), so the sole evidence
    # that may gate is Triton's own search running and coming back empty.
    return _triton_finds_crt_headers() is not False


def gate_torch_compile_on_windows(log: logging.Logger) -> None:
    if sys.platform != "win32":
        return
    try:
        import triton  # noqa: F401, PLC0415
    except ImportError:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        log.warning(
            "Triton not found on Windows — torch.compile disabled. "
            'Install for better performance: pip install "triton-windows<3.7"'
        )
        return

    try:
        reachable = crt_headers_reachable()
    except Exception:  # noqa: BLE001 -- a probe must not take down the worker it exists to protect
        logger.debug("The toolchain probe raised; leaving torch.compile alone", exc_info = True)
        reachable = True
    if reachable:
        log.info("Triton available — torch.compile enabled")
        return

    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    # This only turns off the compiles we own; Unsloth's @triton.jit kernels still need the toolchain.
    log.warning(
        "Triton is installed but its C toolchain has no CRT headers, so its "
        "clang-cl JIT would fail on 'stdlib.h' (#7595). torch.compile disabled; "
        "directly launched Triton kernels still need the toolchain. "
        "Install Visual Studio Build Tools (C++ workload): winget "
        'install Microsoft.VisualStudio.2022.BuildTools --override "--add '
        'Microsoft.VisualStudio.Workload.VCTools --includeRecommended". '
        "Probe saw: %s.",
        _toolchain_summary(),
    )
