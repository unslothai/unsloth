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


def _triton_include_dirs() -> list | None:
    """None means the search could not be run, which is not the same as running and finding nothing.
    Only the latter may gate: `find_msvc_winsdk` is private, so a rename or a changed return arity
    would otherwise read as "no headers" and disable torch.compile on a machine with Visual Studio."""
    try:
        from triton.windows_utils import find_msvc_winsdk  # noqa: PLC0415
        _, inc_dirs, _ = find_msvc_winsdk()
    except Exception:  # noqa: BLE001
        logger.debug("Triton's MSVC/WinSDK discovery is unavailable", exc_info = True)
        return None
    return list(inc_dirs)


def _triton_finds_crt_headers() -> bool | None:
    dirs = _triton_include_dirs()
    return None if dirs is None else _headers_complete(dirs)


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


def _compiles_a_trivial_translation_unit(cc: str, inc_dirs) -> bool | None:
    """Ask the compiler instead of predicting it. None means the probe could not be run.

    Header discovery is an inference and it is wrong in the expensive direction: clang-cl
    locates MSVC through its own search, so on a measured R9700 it compiled with INCLUDE,
    VCINSTALLDIR and WindowsSdkDir all cleared, injecting an -internal-isystem we can see
    no trace of. Predicting from directories alone would disable torch.compile there."""
    import subprocess  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    try:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "probe.c")
            with open(src, "w", encoding = "utf-8") as fh:
                fh.write("#include <stdlib.h>\nint main(void){return 0;}\n")
            # Syntax-only: no link, so a missing lib path cannot masquerade as a missing
            # header, and nothing is written outside the temporary directory.
            argv = [cc, "/Zs", src] + [f"/I{d}" for d in inc_dirs if d]
            done = subprocess.run(
                argv,
                cwd = tmp,
                capture_output = True,
                timeout = 90,
            )
    except Exception:  # noqa: BLE001
        logger.debug("The compiler probe could not be run", exc_info = True)
        return None
    if done.returncode != 0:
        logger.debug("Compiler probe failed: %s", (done.stderr or b"")[-400:])
    return done.returncode == 0


def crt_headers_reachable() -> bool:
    if sys.platform != "win32":
        return True
    if not _needs_msvc_headers():
        return True
    triton_dirs = _triton_include_dirs()

    # The compiler is asked FIRST, because every check below it is inference and inference is
    # wrong in both directions. It over-gates when clang-cl locates MSVC through its own search,
    # which the directory listing cannot see. It under-gates on a partial or mismatched SDK: the
    # two markers are only the entry points, and `vcruntime.h` includes `sal.h`, which lives in
    # the SDK's `shared` directory rather than beside either marker, so a dir set can carry both
    # markers and still fail to compile. Only a compile answers the question the JIT will ask.
    if triton_dirs is not None:
        try:
            verdict = _compiles_a_trivial_translation_unit(_triton_cc(), triton_dirs)
        except Exception:  # noqa: BLE001
            verdict = None
        if verdict is not None:
            return verdict

    # Fallback, for a host where the probe could not be run at all. Unknown is not evidence of
    # a broken toolchain, so these keep their fail-open shape.
    env_dirs = os.environ.get("INCLUDE", "").split(os.pathsep)
    if _headers_complete(env_dirs):
        return True
    # INCLUDE is a positive signal only: unset is the normal case (Studio is not launched from a
    # Developer Command Prompt), so only Triton's own search coming back empty may gate.
    if triton_dirs is None:
        return True
    # Judged over the UNION, because that is what the compile sees: clang-cl reads INCLUDE as
    # system include paths and Triton passes its own dirs as /I. Judging each alone rejects a
    # split toolchain (VC toolset on INCLUDE, SDK discovered by Triton) that compiles fine.
    return _headers_complete(triton_dirs + env_dirs)


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
