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


# `command -v rocminfo` must not reach a real one on a ROCm build host, or these answers
# would depend on the machine running the suite. Each harness names its own inventory.
_rocminfo_stub = "rocminfo() { return 1; }"


def _fake_rocminfo(devices: "list[str]") -> str:
    """A rocminfo whose output has the shape the real one has.

    Every GPU agent names its target TWICE -- once as "Name:" and again inside "ISA Info"
    -- which is the whole defect: a flat grep returns two rows per card, and a mask ordinal
    indexed into that list reads the wrong device. A stub emitting one line per GPU would
    pass whether or not the parsing splits on agent headers.
    """
    blocks = [
        "Agent 1",
        "*******",
        "  Name:                    AMD Ryzen 9",
        "  Marketing Name:          AMD Ryzen 9",
        "  Device Type:             CPU",
    ]
    for _i, _gfx in enumerate(devices):
        blocks += [
            "*******",
            f"Agent {_i + 2}",
            "*******",
            f"  Name:                    {_gfx}",
            "  Marketing Name:          AMD Radeon Graphics",
            "  Device Type:             GPU",
            "  ISA Info:",
            "    ISA 1",
            f"      Name:                    amdgcn-amd-amdhsa--{_gfx}",
        ]
    body = "\n".join(blocks).replace("'", "")
    return "rocminfo() { cat <<'ROCMINFO_EOF'\n" + body + "\nROCMINFO_EOF\n}"


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
            _shell_function("_amd_visible_masks_select_no_gpu"),
            _shell_function("_amd_generic_tag_carries_gfx"),
            _shell_function("_amd_mask_survivors"),
            _shell_function("_amd_runtime_gfx_target"),
            _shell_function("_amd_masks_lead_with_the_first_device"),
            _shell_function("_rocminfo_gpu_records"),
            _shell_function("_amd_ordered_gfx_devices"),
            "_ensure_rocm_probe_env() { :; }",
            _rocminfo_stub,
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
            _shell_function("_amd_visible_masks_select_no_gpu"),
            _shell_function("_amd_generic_tag_carries_gfx"),
            _shell_function("_amd_mask_survivors"),
            _shell_function("_amd_runtime_gfx_target"),
            _shell_function("_amd_masks_lead_with_the_first_device"),
            _shell_function("_rocminfo_gpu_records"),
            _shell_function("_amd_ordered_gfx_devices"),
            "_ensure_rocm_probe_env() { :; }",
            _rocminfo_stub,
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

    script = "\n".join(
        [
            f'_probe_amd_gfx_arch() {{ printf "%s\\n" {probe!r}; }}',
            "_kfd_gfx_targets() { :; }",
            f'_infer_linux_amd_gfx_arch() {{ [ -n {inferred!r} ] && printf "%s\\n" {inferred!r}; }}',
            f"_amd_gpu_present_via_pci() {{ return {0 if pci_ok else 1}; }}",
            _shell_function("_amd_hardware_corroborated"),
            _shell_function("_amd_arch_index_family_for_gfx"),
            _shell_function("_amd_gfx_has_wheel_route"),
            _shell_function("_amd_visible_masks_select_no_gpu"),
            _shell_function("_amd_generic_tag_carries_gfx"),
            _shell_function("_amd_mask_survivors"),
            _shell_function("_amd_runtime_gfx_target"),
            _shell_function("_amd_masks_lead_with_the_first_device"),
            _shell_function("_rocminfo_gpu_records"),
            _shell_function("_amd_ordered_gfx_devices"),
            "_ensure_rocm_probe_env() { :; }",
            _rocminfo_stub,
            _shell_function("_amd_request_has_a_wheel_route"),
            "_detect_rocm_version_tag() { printf '%s\\n' rocm7.2; }",
            "_amd_request_has_a_wheel_route && echo yes || echo no",
        ]
    )
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        # A declared arch decides before the inventory, so one inherited from the runner's
        # environment would answer every case here instead of the probe under test.
        env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_ROCM_GFX_ARCH"},
    )
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


def _viable(
    stack,
    monkeypatch,
    *,
    corroborated: bool,
    archs: list,
    miscomputing = False,
    machine: str = "x86_64",
):
    monkeypatch.setattr(stack.platform, "machine", lambda: machine)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: corroborated)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [])
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: corroborated)
    monkeypatch.setattr(stack, "_physical_amd_gfx_archs", lambda: archs)
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: miscomputing)
    return stack._forced_rocm_route_is_viable()


def test_a_host_with_no_published_rocm_wheels_is_not_a_viable_route(stack, monkeypatch):
    """_ensure_rocm_torch() returns on its Linux-x86_64 gate without installing anything,
    because ROCm wheels are published nowhere else. Answering "viable" on an aarch64 host
    stands the CUDA repair down for a swap that then never happens, and the venv keeps
    whatever CPU or stale HIP torch it had -- on a machine with a usable NVIDIA GPU."""
    assert (
        _viable(stack, monkeypatch, corroborated = True, archs = ["gfx1100"], machine = "aarch64")
        is False
    )


def test_the_same_host_on_x86_64_is_still_viable(stack, monkeypatch):
    """The control: same card, same corroboration, an architecture the wheels exist for.
    Without it the gate above could be "never viable"."""
    assert (
        _viable(stack, monkeypatch, corroborated = True, archs = ["gfx1100"], machine = "x86_64") is True
    )


def test_the_python_route_test_matches_the_shell_one(stack, monkeypatch):
    """Both halves of the installer have to answer the same question the same way, or a
    standalone `studio update` undoes what install.sh chose."""
    assert _viable(stack, monkeypatch, corroborated = True, archs = ["gfx1010"]) is False
    assert _viable(stack, monkeypatch, corroborated = True, archs = ["gfx1100"]) is True
    assert _viable(stack, monkeypatch, corroborated = False, archs = ["gfx1030"]) is False
    assert (
        _viable(
            stack,
            monkeypatch,
            corroborated = True,
            archs = ["gfx1033"],
            miscomputing = True,
        )
        is False
    )


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
        stack,
        "pip_install",
        lambda *a, **k: installed.__setitem__("ran", True),
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
        stack,
        "_detect_rocm_version",
        lambda *a, **k: (reached.__setitem__("ran", True), (7, 2))[1],
    )
    monkeypatch.setattr(stack, "pip_install", lambda *a, **k: None)
    monkeypatch.setattr(
        stack,
        "_probe_torch_runtime",
        lambda *a, **k: (True, True, "2.11.0", False, True),
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


def _shell_request_flag(value: "str | None") -> bool:
    """install.sh's own request test, run verbatim, for one value of the variable."""
    import subprocess

    # Through the environment rather than the script text: a tab quoted into a bash
    # single-quoted string arrives as a literal backslash-t, so a harness that inlined the
    # value would test a different string than the one named.
    env = dict(os.environ)
    env.pop("UNSLOTH_FORCE_ROCM_TORCH", None)
    if value is not None:
        env["UNSLOTH_FORCE_ROCM_TORCH"] = value
    script = "\n".join(
        [
            _shell_function("_rocm_torch_explicitly_requested"),
            "_rocm_torch_explicitly_requested && echo yes || echo no",
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, env = env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip() == "yes"


@pytest.mark.parametrize("value", [" true ", "\ttrue", "1 ", " ON"])
def test_the_shell_flag_is_trimmed_like_its_python_twin(value):
    """install_python_stack.py reads this variable through .strip(); install.sh did not.

    Untrimmed, an env file or launcher passing " true " left install.sh on CUDA and
    exporting an authoritative CUDA backend, which then stopped the Python half honouring
    the identical value -- the two halves disagreeing about one string.
    """
    assert _shell_request_flag(value) is True


@pytest.mark.parametrize("value", ["  ", "", "tru e", None])
def test_trimming_does_not_widen_what_counts_as_a_request(value):
    """The control. Trimming must not turn whitespace, or an internally-spaced value,
    into a request: that would swap a CUDA host's stack on an exported-but-empty
    variable, which is how a shell passes "unset"."""
    assert _shell_request_flag(value) is False


def _route_shell_masked(
    physical: "list[str]",
    rocm_tag: str = "rocm7.2",
    devices: "list[str] | None" = None,
    **mask: str,
) -> bool:
    """The request route test on a masked host, with the mask resolution left live.

    Only _probe_amd_gfx_arch is stubbed, and it returns the WHOLE inventory in every mode --
    which is what the real one does here, because rocminfo honours only ROCR_VISIBLE_DEVICES
    and amd-smi honours neither. So this harness reproduces the condition the resolution has
    to survive rather than assuming a probe that narrows for it.

    ``devices`` is the same host as rocminfo enumerates it, one row per GPU agent, and
    defaults to ``physical``. They differ where the real flat probe would: rocminfo names
    each agent's target twice, so a two-card host greps as four rows.

    The arches are passed as a list and emitted as separate printf arguments: a "\n" inside
    a bash single-quoted string is a literal backslash-n, so a harness that joined them would
    hand the route test one unroutable token and answer no whatever the code does.
    """
    import subprocess

    emit = 'printf "%s\\n" ' + " ".join(repr(a) for a in physical) if physical else ":"
    script = "\n".join(
        [
            f"_probe_amd_gfx_arch() {{ {emit}; }}",
            "_kfd_gfx_targets() { :; }",
            "_infer_linux_amd_gfx_arch() { :; }",
            "_amd_gpu_present_via_pci() { return 0; }",
            _shell_function("_amd_hardware_corroborated"),
            _shell_function("_amd_arch_index_family_for_gfx"),
            _shell_function("_amd_gfx_has_wheel_route"),
            _shell_function("_amd_visible_masks_select_no_gpu"),
            _shell_function("_amd_generic_tag_carries_gfx"),
            _shell_function("_amd_mask_survivors"),
            _shell_function("_amd_runtime_gfx_target"),
            _shell_function("_amd_masks_lead_with_the_first_device"),
            _shell_function("_rocminfo_gpu_records"),
            _shell_function("_amd_ordered_gfx_devices"),
            "_ensure_rocm_probe_env() { :; }",
            _fake_rocminfo(physical if devices is None else devices),
            _shell_function("_amd_request_has_a_wheel_route"),
            f"_detect_rocm_version_tag() {{ printf '%s\\n' {rocm_tag!r}; }}",
            "_amd_request_has_a_wheel_route && echo yes || echo no",
        ]
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("VISIBLE_DEVICES") and k != "UNSLOTH_ROCM_GFX_ARCH"
    }
    env.update(mask)
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, env = env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip() == "yes"


def test_a_mask_that_selects_only_an_unroutable_card_keeps_cuda():
    """The probes cannot answer this: rocminfo is filtered only by ROCR_VISIBLE_DEVICES and
    amd-smi by neither, so a probe run with a HIP mask in place returns the whole machine and
    the route test answered yes on the strength of the card the mask had just hidden. The
    request then traded a working CUDA stack for wheels carrying no kernels for the card that
    will actually run. The mask is resolved against the enumeration order instead."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "1") is False


def test_a_mask_that_selects_a_routable_card_still_deposes_it():
    """The control: same host, mask pointing the other way. Without it the fix could be
    "a mask always keeps CUDA", which passes the test above and removes the feature on
    every masked host."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "0") is True


def test_the_flat_probes_duplicate_rows_do_not_shift_the_ordinals():
    """rocminfo names each agent's target twice, so the flat probe returns four rows for a
    two-card host. Indexing THAT by a mask ordinal reads the wrong device: HIP=1 lands on
    the second row, which is still card 0. The route test then approves ROCm on the
    strength of a gfx1100 the mask had hidden, and the gfx1010 that will actually run has
    no kernels in any wheel. Resolved against one row per agent instead."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1100", "gfx1010", "gfx1010"],
            devices = ["gfx1100", "gfx1010"],
            HIP_VISIBLE_DEVICES = "1",
        )
        is False
    )


def test_the_same_duplicated_host_still_yields_for_the_routable_card():
    """The control: same duplicated inventory, mask pointing at the card that does have a
    route. Without it the fix could be "duplicated rows always keep CUDA"."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1100", "gfx1010", "gfx1010"],
            devices = ["gfx1100", "gfx1010"],
            HIP_VISIBLE_DEVICES = "0",
        )
        is True
    )


def test_a_mask_past_the_first_device_fails_closed_with_no_per_device_list():
    """Only rocminfo enumerates in the order the masks index -- amd-smi orders by KFD
    discovery, which is why _amd_smi_hip_order exists -- so a host without it cannot say
    which card an ordinal names. Guessing from the flat list is what the test above shows
    to be wrong, and guessing yes replaces a working CUDA stack."""
    assert _route_shell_masked(["gfx1010", "gfx1100"], devices = [], HIP_VISIBLE_DEVICES = "1") is False


def test_the_first_device_is_still_answerable_without_a_per_device_list():
    """The control, and the reason this is not simply "no rocminfo, no route": a mask that
    leads with ordinal 0 selects the first device, and the first row of the flat inventory
    names that same device however many duplicates follow it. Denying here would strand
    every host that sets CUDA_VISIBLE_DEVICES=0 and has no rocminfo installed."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], devices = [], HIP_VISIBLE_DEVICES = "0") is True


def test_the_rocr_layer_is_resolved_the_same_way():
    """rocminfo does honour this one, but the route test must not depend on which probe
    answered: amd-smi honours neither mask, so the resolution has to be its own."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], ROCR_VISIBLE_DEVICES = "1") is False


@pytest.mark.parametrize(
    "mask",
    [
        {"HIP_VISIBLE_DEVICES": ""},
        {"HIP_VISIBLE_DEVICES": "-1"},
        {"ROCR_VISIBLE_DEVICES": "-1"},
        {"ROCR_VISIBLE_DEVICES": "-1", "HIP_VISIBLE_DEVICES": "0"},
        {"CUDA_VISIBLE_DEVICES": ""},
    ],
)
def test_a_mask_that_exposes_no_device_leaves_cuda_alone(mask):
    """A deliberate no-GPU selection, not a detection miss: there is nothing for the request
    to swap TO, so falling back to the physical inventory here installed ROCm wheels on a
    host whose runtime exposes no AMD GPU at all. The fourth case is the composition -- ROCr
    filters beneath HIP, so an empty ROCr mask hides everything however HIP reads."""
    assert _route_shell_masked(["gfx1100"], **mask) is False


def test_an_unset_mask_is_not_a_no_gpu_selection():
    """Its control, and the reason ${VAR+x} rather than ${VAR:-}: an unset mask hides
    nothing, and reading it like a set-but-empty one would keep CUDA on every host."""
    assert _route_shell_masked(["gfx1100"]) is True


@pytest.mark.parametrize("value", ["GPU-abcdef", "9"])
def test_a_mask_this_cannot_resolve_fails_closed(value):
    """ROCr accepts UUIDs, and an ordinal can name no device at all. Neither can be resolved
    against an arch list, and the harm is asymmetric: a wrong yes replaces a working CUDA
    stack with wheels carrying no kernels for the card that runs, where a wrong no leaves
    the user where they were."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = value) is False


def test_the_entries_before_an_unresolvable_one_are_still_exposed():
    """Failing closed is about what cannot be resolved, not about the whole variable: the
    runtime reads the list left to right and stops at the first entry naming no device, so
    "0,GPU-..." still exposes device 0. Treating the whole mask as unresolvable would keep
    CUDA on a host whose selected card is routable."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "0,GPU-abcdef") is True


def test_an_unroutable_card_before_an_unresolvable_entry_still_keeps_cuda():
    """And its control, so the rule above is the prefix rather than "the first entry wins
    if anything follows it"."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "1,GPU-abcdef") is False


def test_a_masked_gfx906_is_judged_against_the_physical_host():
    """gfx906's only route is the rocm6.3 legacy tag, and the reroute that grants it inspects
    the UNMASKED inventory, refusing when a second AMD arch is present. Counting the masked
    set made a selected gfx906 look like the sole arch and therefore routable, after which
    that reroute declines and the card is left on newer wheels whose BLAS kernels do not
    support it."""
    assert _route_shell_masked(["gfx906", "gfx1100"], HIP_VISIBLE_DEVICES = "0") is False


def test_a_masked_gfx906_alone_is_still_routable():
    """The control: on a host where gfx906 IS the sole arch, the legacy tag opens and a mask
    naming it must not change that."""
    assert _route_shell_masked(["gfx906"], HIP_VISIBLE_DEVICES = "0") is True


def test_an_explicit_cuda_pin_outranks_the_request(stack, monkeypatch):
    """_rocm_torch_explicitly_requested's docstring promises an index pin still outranks
    the request, and _rocm_pin is the ROCm pin, so a CUDA pin leaves it None and the
    request skipped the NVIDIA-precedence return. _ensure_cuda_torch installs the pinned
    build and this function replaced it, so the environment ended up holding the stack the
    pin ruled out."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cu128")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: False)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    reached = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_has_rocm_gpu",
        lambda: (reached.__setitem__("ran", True), True)[1],
    )
    stack._ensure_rocm_torch()
    assert reached["ran"] is False


def test_a_rocm_pin_is_still_honoured_over_an_nvidia_card(stack, monkeypatch):
    """The control: the pin that names ROCm wheels must still win, or the fix would read
    as "any pin keeps CUDA" and break the headless/CI case _rocm_pin exists for."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "rocm7.0")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    # A ROCm pin skips the whole vendor-precedence block, so reaching the version probe
    # is what "the pin still won" looks like from outside.
    reached = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_detect_rocm_version",
        lambda *a, **k: (reached.__setitem__("ran", True), None)[1],
    )
    stack._ensure_rocm_torch()
    assert reached["ran"] is True


def _viable_masked(stack, monkeypatch, *, devices: list, **mask: str) -> bool:
    """_forced_rocm_route_is_viable on a masked host, with the resolution left live.

    Only the probe is stubbed, so _runtime_gfx_target does the real ROCr/HIP composition:
    stubbing the resolved target instead would assert about the stub.
    """
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: True)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [])
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: True)
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: False)
    monkeypatch.setattr(stack, "_infer_linux_amd_gfx_arch", lambda: None)
    monkeypatch.setattr(stack, "_detect_amd_gfx_codes", lambda **k: list(devices))
    monkeypatch.setattr(stack, "_physical_amd_gfx_archs", lambda: list(devices))
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    for var, value in mask.items():
        monkeypatch.setenv(var, value)
    return stack._forced_rocm_route_is_viable()


def test_the_python_route_test_judges_the_selected_card_too(stack, monkeypatch):
    """It asked whether SOME arch on the bus had a route, which answers yes on a mixed host
    whose mask selects the unsupported one. _ensure_cuda_torch then stands down for a card
    the ROCm wheels have no kernels for, leaving neither AMD nor NVIDIA usable."""
    assert (
        _viable_masked(stack, monkeypatch, devices = ["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "1")
        is False
    )


def test_the_same_host_selecting_the_routable_card_is_still_viable(stack, monkeypatch):
    """The control: same two cards, mask pointing the other way."""
    assert (
        _viable_masked(stack, monkeypatch, devices = ["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "0")
        is True
    )


def test_a_mask_exposing_no_device_is_not_a_viable_route(stack, monkeypatch):
    """The Python half of the same rule the shell now applies: a no-GPU mask is a deliberate
    selection, so there is nothing to swap to."""
    assert (
        _viable_masked(stack, monkeypatch, devices = ["gfx1100"], HIP_VISIBLE_DEVICES = "-1") is False
    )


def test_an_unmasked_host_is_judged_exactly_as_before(stack, monkeypatch):
    """And the control for all three: with no mask set the answer is the inventory's, which
    is what this function did before and what the shell twin still does."""
    assert _viable_masked(stack, monkeypatch, devices = ["gfx1100", "gfx1010"]) is True
    assert _viable_masked(stack, monkeypatch, devices = ["gfx1010"]) is False


def _rocm_repair_reached(
    stack,
    monkeypatch,
    *,
    archs: list,
    viable: bool,
    pin: "str | None" = None,
    pin_url: "str | None" = None,
) -> bool:
    """Whether _ensure_rocm_torch gets past its vendor-precedence gates on a mixed host."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: False)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: True)
    monkeypatch.setattr(stack, "_infer_linux_amd_gfx_arch", lambda: None)
    monkeypatch.setattr(stack, "_forced_rocm_route_is_viable", lambda: viable)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    # Cleared here rather than left to the environment, then set from the argument: a
    # caller that set the pin before calling would otherwise have it deleted underneath it.
    for var in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
        monkeypatch.delenv(var, raising = False)
    if pin is not None:
        monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", pin)
    if pin_url is not None:
        monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", pin_url)
    reached = {"ran": False}
    monkeypatch.setattr(
        stack,
        "_detect_rocm_version",
        lambda *a, **k: (reached.__setitem__("ran", True), None)[1],
    )
    stack._ensure_rocm_torch()
    return reached["ran"]


def test_the_request_needs_a_viable_route_to_bypass_nvidia(stack, monkeypatch):
    """Presence was the bar, and presence is not the bar this feature states. On NVIDIA plus
    a present-but-unroutable gfx1010, _ensure_cuda_torch correctly kept the working CUDA
    build and this function then installed generic ROCm wheels over it, because the request
    alone cleared NVIDIA precedence and the AMD-presence check passed."""
    assert _rocm_repair_reached(stack, monkeypatch, archs = ["gfx1010"], viable = False) is False


def test_a_viable_route_still_lets_the_request_through(stack, monkeypatch):
    """The control: the same mixed host with a card an index can serve. Without it the fix
    reads as "the request never wins on a mixed host", which is the whole feature."""
    assert _rocm_repair_reached(stack, monkeypatch, archs = ["gfx1100"], viable = True) is True


@pytest.mark.parametrize("family", ["cpu", "cu128"])
def test_an_explicit_pin_of_another_family_outranks_the_request(stack, monkeypatch, family):
    """A /cpu pin had the multi-GB ROCm stack installed here and then undone by
    _ensure_cpu_torch, on every update. _rocm_pin only names a ROCm pin, so any other family
    leaves it None and this is the one place the check can happen."""
    assert (
        _rocm_repair_reached(stack, monkeypatch, archs = ["gfx1100"], viable = True, pin = family)
        is False
    )


def test_a_rocm_pin_does_not_outrank_the_request(stack, monkeypatch):
    """The control the CPU/CUDA pair needs: a pin naming ROCm agrees with the request, so it
    must not be swept up by the same check. Without this, "any pin outranks the request" and
    "the wrong family outranks it" are the same passing test."""
    assert (
        _rocm_repair_reached(
            stack,
            monkeypatch,
            archs = ["gfx1100"],
            viable = True,
            pin_url = "https://download.pytorch.org/whl/rocm6.4",
        )
        is True
    )


def test_an_unknown_family_pin_is_left_alone(stack, monkeypatch):
    """The boundary, and pre-existing behaviour rather than anything this adds: an unknown
    leaf was applied verbatim at install time, so _ensure_rocm_torch declines to judge it and
    returns before any of the vendor gates. Asserted so the CPU gate is not credited for it."""
    assert (
        _rocm_repair_reached(
            stack,
            monkeypatch,
            archs = ["gfx1100"],
            viable = True,
            pin_url = "https://mirror.example/simple",
        )
        is False
    )


def test_the_two_mask_layers_compose_in_the_order_the_runtime_reads_them():
    """ROCr filters the physical list and renumbers the survivors; the HIP layer then
    indexes those. Reading either alone answers on the wrong card: here ROCr leaves only
    gfx1010, so HIP ordinal 0 is that card, while a HIP-only reading calls it gfx1100 and
    lets the request replace a working CUDA stack with wheels that cannot run it.

    Fails before the fix, which took HIP_VISIBLE_DEVICES when both were set."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1010"], ROCR_VISIBLE_DEVICES = "1", HIP_VISIBLE_DEVICES = "0"
        )
        is False
    )


def test_the_same_two_layers_pointing_at_the_routable_card_still_depose_cuda():
    """The control: same host, same stacking, ROCr selecting the other card. Without it the
    fix could be "any two stacked masks keep CUDA", which passes the test above."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1010"], ROCR_VISIBLE_DEVICES = "0", HIP_VISIBLE_DEVICES = "0"
        )
        is True
    )


def test_two_cards_of_one_arch_do_not_shift_the_ordinals():
    """An ordinal names a DEVICE, not an architecture. Deduplicating before indexing turns
    [gfx1010, gfx1010, gfx1100] into a two-entry list, so ordinal 1 reads as the gfx1100
    when the runtime will hand over the second gfx1010.

    Fails before the fix, which deduped inside the resolver."""
    assert _route_shell_masked(["gfx1010", "gfx1010", "gfx1100"], HIP_VISIBLE_DEVICES = "1") is False


def test_the_same_host_selecting_past_the_pair_is_still_routable():
    """The control for the cardinality rule: ordinal 2 on that host really is the gfx1100,
    and a resolver that simply refused every repeated arch would fail this."""
    assert _route_shell_masked(["gfx1010", "gfx1010", "gfx1100"], HIP_VISIBLE_DEVICES = "2") is True


def test_the_first_listed_device_is_the_one_judged():
    """HIP remaps runtime ordinal 0 onto the head of the mask, so HIP_VISIBLE_DEVICES=1,0
    targets the second physical card. Judging the whole exposed set passes on a routable
    card the runtime is not going to select.

    Fails before the fix, which returned yes if ANY exposed arch had a route."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "1,0") is False


def test_the_same_pair_listed_the_other_way_round_is_routable():
    """The control: the identical two devices, mask order reversed, so the selected target
    is the routable one and the request still wins."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "0,1") is True


def test_a_declared_arch_decides_before_the_inventory():
    """UNSLOTH_ROCM_GFX_ARCH is what _probe_amd_gfx_arch's default mode returns and what
    get_torch_index_url installs from, so judging the physical inventory instead can pass on
    a routable sibling while the wheels are chosen for the declared card.

    Fails before the fix, which always read the physical probe."""
    assert _route_shell_masked(["gfx1100"], UNSLOTH_ROCM_GFX_ARCH = "gfx1010") is False


def test_a_declared_arch_that_is_routable_still_deposes_cuda():
    """The control, and the #7301 host: the declaration exists to serve a runtime-less card,
    so a routable one must still win even when the inventory disagrees with it."""
    assert _route_shell_masked(["gfx1010"], UNSLOTH_ROCM_GFX_ARCH = "gfx1100") is True


def test_a_generic_only_arch_is_judged_against_the_tag_this_host_resolves():
    """gfx950 has no per-arch index, so its only route is the generic wheel -- and the tag
    comes from the installed ROCm version, which a stale /opt/rocm beside a current amdgpu
    can leave older than the card. _GENERIC_WHEEL_GFX_MIN_ROCM in install_python_stack.py
    puts gfx950 at ROCm 7.0.

    Fails before the fix, which read membership of the generic list as a route outright."""
    assert _route_shell_masked(["gfx950"], rocm_tag = "rocm6.4") is False


def test_the_same_arch_on_a_tag_that_carries_it_is_routable():
    """The control: the identical host one tag later, where the wheel does carry gfx950."""
    assert _route_shell_masked(["gfx950"], rocm_tag = "rocm7.0") is True


def test_an_arch_with_its_own_index_is_not_held_to_the_generic_floor():
    """The boundary: gfx1200 predates the rocm6.0 generic wheel too, but it has a per-arch
    AMD index that carries it, and the reroute is what serves it. Holding it to the generic
    floor would keep CUDA on a host this feature is meant to move."""
    assert _route_shell_masked(["gfx1200"], rocm_tag = "rocm6.0") is True


def _corroborated_on_wsl(
    stack,
    monkeypatch,
    *,
    nvidia: bool,
    rocm_sees_a_gpu: bool = False,
) -> bool:
    """_amd_hardware_is_corroborated() on a WSL box, with the WSL evidence present."""
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: rocm_sees_a_gpu)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: [])
    monkeypatch.setattr(stack, "_is_wsl", lambda: True)
    monkeypatch.setattr(stack, "_wsl_rocm_runtime_present", lambda: True)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: nvidia)
    return stack._amd_hardware_is_corroborated()


def test_leftover_wsl_rocm_files_are_not_a_card_beside_an_nvidia_gpu(stack, monkeypatch):
    """/dev/dxg is the generic WSL GPU bridge an NVIDIA passthrough creates too, and
    librocdxg is a file an uninstalled ROCm leaves behind, so neither names a vendor.
    _has_rocm_gpu() has already answered no by this point, so on an NVIDIA-only WSL box
    the pair is the whole evidence -- and taking it would let a stale UNSLOTH_ROCM_GFX_ARCH
    force AMD wheels over a working CUDA stack, which is the one thing this predicate
    exists to prevent."""
    assert _corroborated_on_wsl(stack, monkeypatch, nvidia = True) is False


def test_the_same_wsl_box_with_no_nvidia_card_still_counts(stack, monkeypatch):
    """The control, and why this is not simply "WSL never corroborates": with no NVIDIA
    GPU there is no working stack to lose, so the leftover reading still beats declining
    and a WSL AMD host whose runtime cannot answer keeps its route."""
    assert _corroborated_on_wsl(stack, monkeypatch, nvidia = False) is True


def test_a_wsl_box_whose_runtime_names_an_agent_counts_beside_nvidia(stack, monkeypatch):
    """The other control: a real AMD adapter under WSL is still corroborated with an NVIDIA
    card present, because the ROCm runtime enumerates it rather than a leftover file
    standing in for it. That answer comes from the _has_rocm_gpu() arm above the WSL
    branch, which is where install.sh reaches for _probe_amd_gfx_arch physical."""
    assert _corroborated_on_wsl(stack, monkeypatch, nvidia = True, rocm_sees_a_gpu = True) is True


def _needs_repair_passes_the_nvidia_gate(stack, monkeypatch, *, viable: bool) -> bool:
    """Whether _amd_torch_needs_dependency_pass() gets past its NVIDIA fast path.

    Read by sentinel rather than by the return value: everything after that gate would
    have to be stubbed to reach a True, and stubbing it would be asserting about the
    stubs. The next gate raises instead, so "passed" and "returned False here" are
    distinguishable.
    """

    class _Reached(Exception):
        pass

    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "IS_LINUX", True)
    monkeypatch.setattr(stack.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "rocm")
    monkeypatch.setattr(stack, "_explicit_rocm_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_explicit_torch_index_url", lambda: None)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_rocm_torch_explicitly_requested", lambda: True)
    monkeypatch.setattr(stack, "_forced_rocm_route_is_viable", lambda: viable)

    def _sentinel():
        raise _Reached

    monkeypatch.setattr(stack, "_visible_masks_select_no_gpu", _sentinel)
    try:
        assert stack._amd_torch_needs_dependency_pass() is False
    except _Reached:
        return True
    return False


def test_an_unroutable_request_does_not_bypass_the_nvidia_fast_path(stack, monkeypatch):
    """On a mixed host whose selected AMD card has no wheel route, _ensure_cuda_torch()
    and _ensure_rocm_torch() both leave CUDA in place. Reporting "needs repair" there is
    answered by studio/setup.sh running the whole dependency pass, and the flag does not
    clear itself, so that pass reruns on every launch for a swap that can never happen."""
    assert _needs_repair_passes_the_nvidia_gate(stack, monkeypatch, viable = False) is False


def test_a_routable_request_still_bypasses_it(stack, monkeypatch):
    """The control: the request is the whole feature, so a viable one must still outrank
    the NVIDIA card here or the repair it gates can never run."""
    assert _needs_repair_passes_the_nvidia_gate(stack, monkeypatch, viable = True) is True
