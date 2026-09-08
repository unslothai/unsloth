# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A mixed NVIDIA+AMD host must have SOME route to its AMD card.

Every installer stops probing AMD the moment nvidia-smi lists a GPU. That is the
right default -- CUDA is the better-supported stack and a false AMD positive would
swap a working install -- but it left a mixed host with no route at all: chat can be
pointed at Vulkan in Settings, and training reads whatever torch was installed, so
the AMD card is invisible to it. The only bypass was an index pin, which is
undocumented and names a wheel family rather than a preference (#10450).

``UNSLOTH_FORCE_ROCM_TORCH=1`` is the request, mirroring ``UNSLOTH_FORCE_VULKAN``
for the llama.cpp bundle and the ``backend == "rocm"`` re-probe in
``install_llama_prebuilt._route_to_vulkan_prebuilt``.

One torch install serves one vendor, so the request SWAPS the stack: these tests
assert it is honoured, and that it changes nothing at all unless it is set.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

_STACK = Path(__file__).resolve().parents[2] / "install_python_stack.py"


@pytest.fixture(scope = "module")
def stack():
    """install_python_stack imported by path: it is a top-level installer script, not
    a package module, so the backend's own import path does not reach it."""
    spec = importlib.util.spec_from_file_location("_unsloth_install_python_stack", _STACK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_the_request_is_recognised(stack, monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", value)
    assert stack._rocm_torch_explicitly_requested() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
def test_anything_else_is_not_a_request(stack, monkeypatch, value):
    """Including the empty string: an exported-but-empty variable is how a shell
    passes "unset", and reading it as a request would swap a CUDA host's stack."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", value)
    assert stack._rocm_torch_explicitly_requested() is False


def test_unset_is_not_a_request(stack, monkeypatch):
    monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    assert stack._rocm_torch_explicitly_requested() is False


def test_an_nvidia_host_still_hides_the_amd_card_by_default(stack, monkeypatch):
    """The control. Every assertion below is only meaningful if the default holds."""
    monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    assert stack._has_rocm_gpu() is False


def _rocminfo(monkeypatch, stack, output: str):
    """A host where rocminfo is on PATH and prints ``output``; amd-smi absent."""
    monkeypatch.setattr(
        stack.shutil, "which", lambda c: "/usr/bin/rocminfo" if c == "rocminfo" else None
    )
    monkeypatch.setattr(
        stack.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode = 0, stdout = output),
    )
    # The sysfs fallback must not answer for the shelled probes above.
    monkeypatch.setattr(stack.os.path, "isdir", lambda p: False)


def test_the_request_lets_the_amd_probe_run_on_a_mixed_host(stack, monkeypatch):
    """Fails before the fix: _has_rocm_gpu returned False on any NVIDIA host before
    consulting a single AMD probe, so nothing downstream could see the AMD card."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    _rocminfo(monkeypatch, stack, "  Name:                    gfx1201")
    assert stack._has_rocm_gpu() is True


def test_the_same_host_without_the_request_still_answers_no(stack, monkeypatch):
    """The pair to the test above, on an identical host: the ONLY difference is the
    variable, so a probe that started answering for some other reason fails here."""
    monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    _rocminfo(monkeypatch, stack, "  Name:                    gfx1201")
    assert stack._has_rocm_gpu() is False


def test_the_request_does_not_invent_a_card(stack, monkeypatch):
    """The request relaxes which vendor wins, not whether there is a card to serve.
    A host with no AMD GPU that sets it must still install nothing, or a typo turns a
    working CUDA box into a CPU one."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    _rocminfo(monkeypatch, stack, "  Name:                    gfx000")  # CPU agent only
    assert stack._has_rocm_gpu() is False


def _shell_function(name: str) -> str:
    """The text of one function from install.sh, by brace matching.

    Only the named functions are extracted, never the whole file: install.sh runs its
    installer at top level, so sourcing it in a test would attempt a real install.
    """
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    depth = 0
    for end in range(start, len(lines)):
        depth += lines[end].count("{") - lines[end].count("}")
        if depth == 0:
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated function {name}")


def _index_url(env: str, stubs: str) -> str:
    """get_torch_index_url() under a stubbed host, returning the wheel index it picks.

    This is the seam the request has to move: the family chosen here is what install.sh
    exports as UNSLOTH_TORCH_BACKEND, and _ensure_rocm_torch returns on its first line
    for a "cuda" backend. Run through bash rather than reimplemented, since a Python
    copy of the shell logic would agree with itself and prove nothing.
    """
    import subprocess

    script = "\n".join(
        [
            _shell_function("_rocm_torch_explicitly_requested"),
            stubs,
            # The selector asks whether the request has a wheel route, not whether a card
            # is present, so its helpers are lifted too. Stubbing the ANSWER here would
            # make the arch cases below assert about the stub.
            _shell_function("_amd_hardware_corroborated"),
            _shell_function("_amd_gfx_has_wheel_route"),
            _shell_function("_amd_request_has_a_wheel_route"),
            _shell_function("get_torch_index_url"),
            f"{env} get_torch_index_url",
        ]
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
    ).stdout


# A bash function definition ends at its closing brace, so these are newline
# separated: two on one line is a syntax error, which the first version of this
# harness produced and read as "the flag did nothing".
_MIXED_HOST = "\n".join(
    [
        "_has_usable_nvidia_gpu() { return 0; }",
        "_has_amd_rocm_gpu() { return 0; }",
        "_probe_amd_gfx_arch() { echo gfx1201; }",
        "_amd_arch_index_family_for_gfx() { echo gfx120X-all; }",
        "_kfd_gfx_targets() { echo gfx1201; }",
        "_infer_linux_amd_gfx_arch() { echo gfx1201; }",
        "_amd_sole_index_arch() { echo gfx1201; }",
        "_amd_gpu_present_via_pci() { return 0; }",
        "_detect_rocm_version_tag() { echo rocm7.0; }",
        "_amd_agreed_index_family() { echo gfx120X-all; }",
        "_rocm_sdk_install_hint() { echo ''; }",
        "nvidia-smi() { echo 'CUDA Version: 13.0'; }",
    ]
)


def test_the_shell_installer_selects_a_rocm_index_under_the_request(stack):
    """Fails before the fix: get_torch_index_url set _nvidia_detected=1 and returned a
    cu* family whatever the AMD probes said, so the request could not swap anything."""
    out = _index_url("UNSLOTH_FORCE_ROCM_TORCH=1", _MIXED_HOST)
    assert "rocm" in out or "gfx" in out, out


def test_the_same_host_without_the_request_still_selects_cuda(stack):
    """The control, identical but for the variable."""
    out = _index_url("UNSLOTH_FORCE_ROCM_TORCH=0", _MIXED_HOST)
    assert "rocm" not in out and "gfx" not in out, out


_NO_AMD_CARD = "\n".join(
    [
        _MIXED_HOST.replace(
            "_has_amd_rocm_gpu() { return 0; }", "_has_amd_rocm_gpu() { return 1; }"
        ),
        # Every source the route test can read, silenced together. Silencing only the
        # ROCm runtime leaves the arch probes answering, which is the host the
        # runtime-less reroute exists for rather than a machine with no AMD card.
        "_probe_amd_gfx_arch() { :; }",
        "_kfd_gfx_targets() { :; }",
        "_infer_linux_amd_gfx_arch() { :; }",
        "_amd_gpu_present_via_pci() { return 1; }",
    ]
)


def test_the_request_does_not_select_rocm_without_an_amd_card(stack):
    """A typo on a pure NVIDIA box must not turn a working CUDA machine into a ROCm or
    CPU one: the AMD route test still has to pass."""
    out = _index_url("UNSLOTH_FORCE_ROCM_TORCH=1", _NO_AMD_CARD)
    assert "rocm" not in out and "gfx" not in out, out


def test_the_cuda_repair_stands_down_under_the_request(stack, monkeypatch):
    """A standalone `studio update` leaves _TORCH_BACKEND empty, so the CUDA repair
    runs first, sees an NVIDIA GPU beside the requested HIP build, reads it as
    poisoning and reinstalls the CUDA trio. Fails before the fix."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    # The request stands down only for a card an index can actually serve, so the mixed
    # host this describes has to say so. Stubbed rather than left to the host, since
    # otherwise this asserts about whatever silicon the test runner happens to have.
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: True)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [])
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: True)
    monkeypatch.setattr(stack, "_physical_amd_gfx_archs", lambda: ["gfx1100"])
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: False)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    probed = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_probe_torch_runtime",
        lambda *a, **k: (
            probed.__setitem__("ran", True),
            (True, True, "2.11.0+rocm7.0", True, False),
        )[1],
    )
    stack._ensure_cuda_torch()
    assert probed["ran"] is False


def test_the_cuda_repair_still_runs_without_the_request(stack, monkeypatch):
    """The control: the same host, no request, and the repair must still classify."""
    monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    probed = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_probe_torch_runtime",
        lambda *a, **k: (
            probed.__setitem__("ran", True),
            (True, True, "2.11.0+rocm7.0", True, False),
        )[1],
    )
    monkeypatch.setattr(stack, "pip_install", lambda *a, **k: None)
    stack._ensure_cuda_torch()
    assert probed["ran"] is True


def test_windows_is_not_swapped_by_the_request_alone(stack, monkeypatch):
    """Windows picks the wheel in install.ps1 and republishes an expected tag that
    _ensure_expected_torch_flavor restores, so honouring the request only in Python
    would install ROCm and have it reverted. Left to the index pin until both
    PowerShell installers honour it too."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "IS_WINDOWS", True)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_explicit_rocm_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_explicit_unknown_family_torch_index_url", lambda: None)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    detected = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_detect_windows_gfx_arch",
        lambda *a, **k: (detected.__setitem__("ran", True), "gfx1151")[1],
    )
    stack._ensure_rocm_torch()
    assert detected["ran"] is False


def _nvidia_wins(env: str, stubs: str) -> bool:
    """_nvidia_gpu_wins_over_amd() under a stubbed host.

    This is the predicate the two per-arch reroutes consult. They run at TOP LEVEL and
    probe the host afresh, so get_torch_index_url clearing its own _nvidia_detected
    never reaches them; the request has to be asked here as well or a mixed host takes
    the cpu index the reroute exists to rewrite.
    """
    import subprocess

    script = "\n".join(
        [
            _shell_function("_rocm_torch_explicitly_requested"),
            stubs,
            _shell_function("_amd_hardware_corroborated"),
            _shell_function("_amd_arch_index_family_for_gfx"),
            _shell_function("_amd_gfx_has_wheel_route"),
            _shell_function("_amd_request_has_a_wheel_route"),
            _shell_function("_nvidia_gpu_wins_over_amd"),
            f"{env} _nvidia_gpu_wins_over_amd && echo NVIDIA || echo AMD",
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True)
    assert out.stdout.strip() in ("NVIDIA", "AMD"), out
    return out.stdout.strip() == "NVIDIA"


# The probes the route test reads, not its answer: stubbing _amd_request_has_a_wheel_route
# itself would make every case below assert about the stub.
_NO_AMD = "\n".join(
    [
        "_has_amd_rocm_gpu() { return 1; }",
        "_probe_amd_gfx_arch() { :; }",
        "_kfd_gfx_targets() { :; }",
        "_infer_linux_amd_gfx_arch() { :; }",
        "_amd_gpu_present_via_pci() { return 1; }",
    ]
)

_ROUTABLE_AMD = "\n".join(
    [
        # Kept so a predicate asking the OLD presence question still RUNS: an undefined
        # function would make the unroutable case below fail for the wrong reason.
        "_has_amd_rocm_gpu() { return 0; }",
        "_probe_amd_gfx_arch() { echo gfx1201; }",
        "_kfd_gfx_targets() { echo gfx1201; }",
        "_infer_linux_amd_gfx_arch() { echo gfx1201; }",
        "_amd_gpu_present_via_pci() { return 0; }",
    ]
)

_PURE_NVIDIA = "\n".join(["_has_usable_nvidia_gpu() { return 0; }", _NO_AMD])

_PURE_AMD = "\n".join(["_has_usable_nvidia_gpu() { return 1; }", _ROUTABLE_AMD])

_MIXED = "\n".join(["_has_usable_nvidia_gpu() { return 0; }", _ROUTABLE_AMD])


def test_the_reroute_predicate_yields_to_the_request_on_a_mixed_host():
    """The #10450 gap the selector fix alone left open: an unreadable ROCm version
    makes get_torch_index_url return the cpu index deliberately, for the per-arch
    reroute to rewrite. A reroute that still asked the bare NVIDIA probe declined,
    and the request finished with CPU torch beside a working NVIDIA card."""
    assert not _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _MIXED)


def test_the_same_mixed_host_without_the_request_keeps_cuda():
    """The control, differing only in the environment."""
    assert _nvidia_wins("", _MIXED)


def test_the_request_does_not_yield_a_pure_nvidia_host():
    """It relaxes which vendor wins, not whether there is a card to serve. Without
    this a request set on a box with no AMD card would reroute it to AMD wheels."""
    assert _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _PURE_NVIDIA)


def test_a_host_with_no_nvidia_card_is_unchanged_either_way():
    """The automatic path is untouched: with no NVIDIA GPU the answer never depended
    on the request, and must not start to."""
    assert not _nvidia_wins("", _PURE_AMD)
    assert not _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _PURE_AMD)


def test_the_reroutes_ask_the_request_aware_predicate():
    """The predicate is only worth testing if the reroutes actually consult it.

    Read off install.sh rather than assumed: a bare _has_usable_nvidia_gpu at either
    site would restore the gap while every test above still passed.
    """
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    body = install_sh.read_text(encoding = "utf-8")
    marker = "_amd_no_rocm_version_reroute=false"
    tail = body[body.index(marker) :]
    assert "! _nvidia_gpu_wins_over_amd" in tail
    assert "! _has_usable_nvidia_gpu" not in tail, (
        "a per-arch reroute still asks the bare NVIDIA probe, so the explicit "
        "request cannot reach the AMD wheels it selected"
    )


def test_the_cuda_repair_still_runs_when_the_request_finds_no_amd_card(stack, monkeypatch):
    """Standing down needs a card to stand down FOR, which is the shell selector's
    rule too: a request on a box with no AMD GPU selects nothing and falls through to
    CUDA. Leaving the variable set on a host whose AMD card has since been removed
    must not silence this repair, because _ensure_rocm_torch then finds no target
    either and a stale HIP build would be left on a working NVIDIA GPU with nothing
    to fix it."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: False)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    probed = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_probe_torch_runtime",
        lambda *a, **k: (
            probed.__setitem__("ran", True),
            (True, True, "2.11.0+rocm7.0", True, False),
        )[1],
    )
    monkeypatch.setattr(stack, "pip_install", lambda *a, **k: None)
    stack._ensure_cuda_torch()
    assert probed["ran"] is True


def _cuda_restore_block() -> str:
    """install.sh's request-downgrade guard, lifted by text.

    Restated here it would agree with itself; taken from the file it fails when the
    guard is removed, which is the only thing worth asserting about it.

    An absent guard returns the empty string rather than raising, so removing it makes
    these tests describe the behaviour BEFORE the fix instead of an extraction error.
    That is what keeps the two controls below controls: they must still pass with no
    guard present, and only the dead-end case must fail.
    """
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if line.startswith('if [ "$_torch_index_pinned" = false ]')
            and "_rocm_torch_explicitly_requested" in "".join(lines[i : i + 2])
        ),
        None,
    )
    if start is None:
        return ""
    end = next(i for i in range(start, len(lines)) if lines[i] == "fi")
    return "\n".join(lines[start : end + 1])


def _index_after_the_guard(env: str, resolved: str, cuda_answer: str) -> str:
    """TORCH_INDEX_URL after the guard, given what the AMD branch resolved.

    ``cuda_answer`` is what the same selector returns with the request suppressed,
    which the guard obtains by calling get_torch_index_url again; it is stubbed so the
    test states the two inputs rather than re-deriving one of them.
    """
    import subprocess

    script = "\n".join(
        [
            _shell_function("_rocm_torch_explicitly_requested"),
            # Taken from install.sh rather than restated: a classifier restated
            # here would agree with itself, and the mirror case below is exactly
            # where it must not.
            _shell_function("_torch_index_url_leaf"),
            _shell_function("_torch_index_url_is_rocm"),
            "_has_usable_nvidia_gpu() { return 0; }",
            f"get_torch_index_url() {{ echo {cuda_answer!r}; }}",
            "_torch_index_pinned=false",
            "SKIP_TORCH=false",
            f"TORCH_INDEX_URL={resolved!r}",
            f"{env} true",
            _cuda_restore_block(),
            'printf "%s\\n" "$TORCH_INDEX_URL"',
        ]
    )
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        env = {**os.environ, **dict([env.split("=", 1)] if "=" in env else [])},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


_CUDA = "https://download.pytorch.org/whl/cu130"


def test_a_dead_end_amd_route_keeps_cuda():
    """ROCm 5.x, an arch no index covers, or a non-x86_64 host all end the AMD branch
    at the cpu index. The request asked for a swap; handing back CPU torch on a machine
    with a working NVIDIA GPU is a downgrade, and is the outcome the PR description
    says this feature must not have."""
    assert (
        _index_after_the_guard(
            "UNSLOTH_FORCE_ROCM_TORCH=1",
            resolved = "https://download.pytorch.org/whl/cpu",
            cuda_answer = _CUDA,
        )
        == _CUDA
    )


def test_a_resolved_rocm_index_is_left_alone():
    """The control that makes the test above mean something: when the request DID
    reach ROCm wheels, the guard must not undo it."""
    rocm = "https://repo.amd.com/rocm/whl/gfx1151"
    assert (
        _index_after_the_guard(
            "UNSLOTH_FORCE_ROCM_TORCH=1",
            resolved = rocm,
            cuda_answer = _CUDA,
        )
        == rocm
    )


def test_a_deliberate_cpu_install_without_the_request_is_untouched():
    """No request, no guard. A CPU index chosen for any other reason is a decision
    this must not reverse."""
    cpu = "https://download.pytorch.org/whl/cpu"
    assert _index_after_the_guard("", resolved = cpu, cuda_answer = _CUDA) == cpu


def test_a_mirror_whose_base_path_says_rocm_is_not_a_rocm_index():
    """The whole-URL form this guard first used matched a mirror's BASE path, so
    https://mirror.local/rocm-cache/cpu read as ROCm wheels and the guard stood down --
    leaving CPU torch on a working NVIDIA card, which is the one outcome it exists to
    prevent. install.sh already carries a comment warning about this exact leaf, three
    lines below the block that got it wrong."""
    assert (
        _index_after_the_guard(
            "UNSLOTH_FORCE_ROCM_TORCH=1",
            resolved = "https://mirror.local/rocm-cache/cpu",
            cuda_answer = _CUDA,
        )
        == _CUDA
    )


def test_a_mirrored_rocm_index_is_still_left_alone():
    """The control for the case above, one path segment apart: the same mirror serving
    an actual ROCm leaf must keep it. Without this the fix could be "call every mirror
    CPU", which passes the test above and breaks every mirrored ROCm install."""
    rocm = "https://mirror.local/rocm-cache/rocm7.2"
    assert (
        _index_after_the_guard(
            "UNSLOTH_FORCE_ROCM_TORCH=1",
            resolved = rocm,
            cuda_answer = _CUDA,
        )
        == rocm
    )


def test_the_radeon_repo_leaf_counts_as_a_rocm_index():
    """repo.radeon.com ends in rocm-rel-X.Y, which _is_pip_rocm_family_leaf declines --
    correctly, since it answers the narrower "is this a pip ROCm FAMILY index". Reusing
    that helper here would have called a working Radeon install a dead end and replaced
    it with CUDA wheels, so the classifier is deliberately the broader one."""
    radeon = "https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4/"
    assert (
        _index_after_the_guard(
            "UNSLOTH_FORCE_ROCM_TORCH=1",
            resolved = radeon,
            cuda_answer = _CUDA,
        )
        == radeon
    )


def test_a_miscomputing_arch_host_keeps_the_cuda_repair(stack, monkeypatch):
    """A card is necessary and not sufficient. _ensure_rocm_torch bails on an arch
    measured to compute incorrectly under ROCm and keeps CPU torch, so a gfx1033-style
    host has a ROCm GPU and no ROCm route -- and standing down for it leaves the CUDA
    repair silenced while nothing installs ROCm, so _ensure_cpu_torch demotes the HIP
    build and the NVIDIA card ends up on CPU torch. The same downgrade install.sh's
    guard exists to prevent, on the other side of the installer.

    Fails on the pre-fix condition, which asked only whether a card was present."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: True)
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: True)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    probed = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_probe_torch_runtime",
        lambda *a, **k: (
            probed.__setitem__("ran", True),
            (True, True, "2.11.0+rocm7.0", True, False),
        )[1],
    )
    monkeypatch.setattr(stack, "pip_install", lambda *a, **k: None)
    stack._ensure_cuda_torch()
    assert probed["ran"] is True


def test_the_rocm_installer_bails_on_the_same_arch(stack, monkeypatch):
    """The other half of the pair, and the reason the test above is not arbitrary: the
    helper the repair now consults is the one _ensure_rocm_torch itself returns on, so
    the two sides of the installer cannot disagree about whether a route exists."""
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: True)
    monkeypatch.setattr(stack, "_explicit_rocm_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_explicit_unknown_family_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: True)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    installed = {"ran": False}
    monkeypatch.setattr(
        stack,
        "pip_install",
        lambda *a, **k: installed.__setitem__("ran", True),
    )
    stack._ensure_rocm_torch()
    assert installed["ran"] is False


def _gpu_summary_branch(resolved: str, request: str) -> str:
    """Which arm the GPU detection summary takes, given the index the resolution left.

    Only the two condition lines are lifted -- the arms themselves probe rocminfo and
    call step() -- and a summary whose conditions cannot be found yields the PRE-FIX
    pair rather than raising, so removing the fix describes the old behaviour instead
    of breaking the extraction.
    """
    import subprocess

    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    head = next(
        (
            l
            for l in lines
            if l.startswith("if _has_usable_nvidia_gpu && ! _torch_index_url_is_rocm")
        ),
        "if _nvidia_gpu_wins_over_amd; then",
    )
    tail = next(
        (l for l in lines if l.startswith("elif _torch_index_url_is_rocm")),
        'elif case "$TORCH_INDEX_URL" in */rocm*|*/gfx*) true ;; *) false ;; esac; then',
    )
    script = "\n".join(
        [
            _shell_function("_rocm_torch_explicitly_requested"),
            _shell_function("_torch_index_url_leaf"),
            _shell_function("_torch_index_url_is_rocm"),
            _shell_function("_nvidia_gpu_wins_over_amd"),
            "_has_usable_nvidia_gpu() { return 0; }",
            "_has_amd_rocm_gpu() { return 0; }",
            f"TORCH_INDEX_URL={resolved!r}",
            # export, not a VAR=VAL command prefix: a prefix applies to that one command and
            # leaves the variable unset for everything after it, so the request would never
            # be in effect and the first case below would pass without the fix.
            f"export {request}" if request else ":",
            head,
            "echo nvidia",
            tail,
            "echo amd",
            "else",
            "echo amd-cpu-fallback",
            "fi",
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_the_summary_reports_cuda_after_the_cuda_restore():
    """The mixed host whose request found no ROCm route: the guard put the CUDA index
    back, but the request is still set and the AMD card is still there, so the reroute
    predicate still answers "AMD wins". Asked of the predicate, the summary announced
    an AMD CPU fallback one line after warning that the CUDA build was being kept.

    Fails before the fix, which is the whole of the item."""
    assert _gpu_summary_branch(_CUDA, "UNSLOTH_FORCE_ROCM_TORCH=1") == "nvidia"


def test_the_summary_still_reports_amd_when_the_request_won():
    """The control: when the request DID reach ROCm wheels, the summary must report the
    card those wheels are for, which is what the predicate was introduced to fix."""
    assert (
        _gpu_summary_branch(
            "https://repo.amd.com/rocm/whl/gfx1151",
            "UNSLOTH_FORCE_ROCM_TORCH=1",
        )
        == "amd"
    )


def test_the_summary_on_a_pure_cuda_install_is_unchanged():
    """And the control for the control: no request, CUDA wheels, NVIDIA reported, which
    is what every automatic install on an NVIDIA host must keep seeing."""
    assert _gpu_summary_branch(_CUDA, "") == "nvidia"


def _route_shell(probe: str, inferred: str, pci_ok: bool) -> bool:
    """install.sh's request route test, on a stubbed host.

    The three probes are stubbed and everything else is lifted from install.sh, so the
    answer depends only on the arch reasoning under test.
    """
    import subprocess

    script = "\n".join([
        f'_probe_amd_gfx_arch() {{ printf "%s\\n" {probe!r}; }}',
        "_kfd_gfx_targets() { :; }",
        f'_infer_linux_amd_gfx_arch() {{ [ -n {inferred!r} ] && printf "%s\\n" {inferred!r}; }}',
        f"_amd_gpu_present_via_pci() {{ return {0 if pci_ok else 1}; }}",
        _shell_function("_amd_hardware_corroborated"),
        _shell_function("_amd_arch_index_family_for_gfx"),
        _shell_function("_amd_gfx_has_wheel_route"),
        _shell_function("_amd_request_has_a_wheel_route"),
        "_amd_request_has_a_wheel_route && echo yes || echo no",
    ])
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip() == "yes"


def test_an_arch_no_index_can_serve_does_not_depose_the_nvidia_card():
    """gfx1010 (RDNA 1) is in neither the generic wheel nor any per-arch index, so the
    request cannot buy this host a working ROCm stack. On presence alone it cleared
    _nvidia_detected, the AMD branch picked the generic rocm index because a ROCm version
    was readable, and the CUDA restore then accepted that as a successful route -- so a
    working CUDA install was replaced with wheels carrying no kernels for the card.

    install_python_stack.py states the rule this now follows: an arch in neither route
    "must never depose a card that can"."""
    assert _route_shell("gfx1010", "", pci_ok = True) is False


def test_a_routable_arch_still_deposes_it():
    """The control: gfx1100 is in the generic wheel, so the request still swaps. Without
    this the fix could be "never yield", which passes the test above and removes the
    feature."""
    assert _route_shell("gfx1100", "", pci_ok = True) is True


def test_a_runtime_less_but_inferable_card_is_served_like_a_pure_amd_host():
    """A Strix box with no rocminfo, amd-smi or KFD node is routed to per-arch wheels on a
    pure-AMD host, by the reroute this predicate gates. Requiring a working ROCm runtime
    answered differently for the same silicon depending only on whether an NVIDIA card sat
    beside it, and blocked both reroutes."""
    assert _route_shell("", "gfx1151", pci_ok = True) is True


def test_a_declared_arch_alone_is_not_a_card():
    """The control that keeps the case above safe. _infer_linux_amd_gfx_arch returns
    UNSLOTH_ROCM_GFX_ARCH before it looks at any hardware, so a stale or copied-in value
    names an arch on a host with no AMD GPU at all. Accepting inference without
    corroborating the silicon would force AMD wheels over a working CUDA stack there."""
    assert _route_shell("", "gfx1030", pci_ok = False) is False


def test_gfx906_alone_is_routable():
    """gfx906's only route is the rocm6.3 legacy tag."""
    assert _route_shell("gfx906", "", pci_ok = True) is True


def test_gfx906_beside_a_second_amd_arch_is_not():
    """...and that tag opens only when gfx906 is the sole arch on the machine, so on a
    mixed-AMD box it is unroutable. _MIXED_HOST_UNROUTABLE says the same on the Python
    side; this keeps the two answering alike."""
    assert _route_shell("gfx906\ngfx1010", "", pci_ok = True) is False


def _viable(stack, monkeypatch, *, corroborated: bool, archs: list, miscomputing = False):
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: corroborated)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [])
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: corroborated)
    monkeypatch.setattr(stack, "_physical_amd_gfx_archs", lambda: archs)
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: miscomputing)
    return stack._forced_rocm_route_is_viable()


def test_the_python_route_test_matches_the_shell_one(stack, monkeypatch):
    """Both halves of the installer have to answer the same question the same way, or a
    standalone `studio update` undoes what install.sh chose."""
    assert _viable(stack, monkeypatch, corroborated = True, archs = ["gfx1010"]) is False
    assert _viable(stack, monkeypatch, corroborated = True, archs = ["gfx1100"]) is True
    assert _viable(stack, monkeypatch, corroborated = False, archs = ["gfx1030"]) is False
    assert _viable(
        stack, monkeypatch, corroborated = True, archs = ["gfx1033"], miscomputing = True,
    ) is False


def test_an_unroutable_card_keeps_the_cuda_repair(stack, monkeypatch):
    """The repair had stood down for any host with a visible AMD GPU. With an unusable HIP
    build and a gfx1010 beside a working NVIDIA card, nothing then classified the stale
    build, _ensure_rocm_torch found no wheel tag, and the NVIDIA GPU was left with no
    working torch at all. Fails on the round-four condition, which excluded only the
    miscomputing arches."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: True)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [])
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: True)
    monkeypatch.setattr(stack, "_physical_amd_gfx_archs", lambda: ["gfx1010"])
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: False)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    probed = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_probe_torch_runtime",
        lambda *a, **k: (
            probed.__setitem__("ran", True),
            (True, True, "2.11.0+rocm7.0", True, False),
        )[1],
    )
    monkeypatch.setattr(stack, "pip_install", lambda *a, **k: None)
    stack._ensure_cuda_torch()
    assert probed["ran"] is True


def test_an_inferable_card_stands_the_cuda_repair_down(stack, monkeypatch):
    """The other direction, and the control for the test above: a runtime-less host whose
    arch is inferable is one _ensure_rocm_torch will serve, so repairing CUDA here only to
    have ROCm force-installed immediately after is a reinstall cycle on every update."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: False)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [])
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: True)
    monkeypatch.setattr(stack, "_physical_amd_gfx_archs", lambda: ["gfx1151"])
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: False)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    probed = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_probe_torch_runtime",
        lambda *a, **k: (
            probed.__setitem__("ran", True),
            (True, True, "2.11.0+rocm7.0", True, False),
        )[1],
    )
    stack._ensure_cuda_torch()
    assert probed["ran"] is False


def test_a_declared_arch_does_not_force_rocm_over_a_working_cuda_stack(stack, monkeypatch):
    """The request skips the NVIDIA precedence return, leaving the presence gate as the
    only thing between a stale UNSLOTH_ROCM_GFX_ARCH and per-arch AMD wheels installed
    over CUDA on a host with no AMD card. Fails before the corroboration gate."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", "gfx1030")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "_explicit_rocm_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_explicit_unknown_family_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: False)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [])
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: False)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    installed = {"ran": False}
    monkeypatch.setattr(
        stack, "pip_install", lambda *a, **k: installed.__setitem__("ran", True),
    )
    stack._ensure_rocm_torch()
    assert installed["ran"] is False


def test_a_real_card_with_a_declared_arch_is_still_served(stack, monkeypatch):
    """The control: declaring an arch is the documented routing hint for a runtime-less
    host, so it must keep working once the silicon is corroborated. A fix that simply
    stopped trusting the variable would pass the test above and break that host."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", "gfx1151")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "_explicit_rocm_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_explicit_unknown_family_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: False)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [])
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: True)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    reached = {"ran": False}
    monkeypatch.setattr(
        stack, "_detect_rocm_version",
        lambda *a, **k: (reached.__setitem__("ran", True), (7, 2))[1],
    )
    monkeypatch.setattr(stack, "pip_install", lambda *a, **k: None)
    monkeypatch.setattr(
        stack, "_probe_torch_runtime", lambda *a, **k: (True, True, "2.11.0", False, True),
    )
    stack._ensure_rocm_torch()
    assert reached["ran"] is True


# A card ROCm sees perfectly well and no index can serve. The two questions disagree here,
# which is the whole point: the old predicate yields the NVIDIA GPU, the new one does not.
_UNROUTABLE_AMD = "\n".join(
    [
        "_has_usable_nvidia_gpu() { return 0; }",
        "_has_amd_rocm_gpu() { return 0; }",
        "_probe_amd_gfx_arch() { echo gfx1010; }",
        "_kfd_gfx_targets() { echo gfx1010; }",
        "_infer_linux_amd_gfx_arch() { echo gfx1010; }",
        "_amd_gpu_present_via_pci() { return 0; }",
    ]
)


def test_the_reroute_predicate_keeps_cuda_for_an_unroutable_card():
    """The call site, not the helper: both per-arch reroutes consult this predicate, and
    on presence alone a gfx1010 cleared the way for AMD wheels that carry no kernels for
    it. Fails on the round-four predicate, which asked _has_amd_rocm_gpu."""
    assert _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _UNROUTABLE_AMD)


def test_the_reroute_predicate_still_yields_for_a_routable_one():
    """The control: same host, routable arch, and the request must still win. A fix that
    stopped yielding altogether passes the test above and removes the feature."""
    assert not _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _MIXED)
