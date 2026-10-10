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
import subprocess
import sys
import types
from pathlib import Path

import pytest

_STACK = Path(__file__).resolve().parents[2] / "install_python_stack.py"


@pytest.fixture(scope = "module")
def stack():
    """install_python_stack imported by path: it is a top-level installer script, not a package
    module, so the backend's own import path does not reach it."""
    spec = importlib.util.spec_from_file_location("_unsloth_install_python_stack", _STACK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StoppedAtProbe(BaseException):
    """Raised by the stubbed ROCm version probe so nothing past it runs."""


@pytest.fixture(autouse = True)
def _no_real_install(stack, monkeypatch):
    """No test in this file may reach an installation."""

    def _refuse(*args, **kwargs):
        pytest.fail(f"the test reached a real install: pip_install{args[:2]}")

    monkeypatch.setattr(stack, "pip_install", _refuse)
    monkeypatch.setattr(stack, "pip_install_try", _refuse)


def _reaches_the_version_probe(stack, monkeypatch) -> bool:
    """Whether _ensure_rocm_torch gets as far as the ROCm version probe."""
    reached = {"ran": False}

    def _probe(*args, **kwargs):
        reached["ran"] = True
        raise _StoppedAtProbe

    monkeypatch.setattr(stack, "_detect_rocm_version", _probe)
    try:
        stack._ensure_rocm_torch()
    except _StoppedAtProbe:
        pass
    return reached["ran"]


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_the_request_is_recognised(stack, monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", value)
    assert stack._rocm_torch_explicitly_requested() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
def test_anything_else_is_not_a_request(stack, monkeypatch, value):
    """Including the empty string: an exported-but-empty variable is how a shell passes "unset",
    and reading it as a request would swap a CUDA host's stack."""
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
    """Fails before the fix: _has_rocm_gpu returned False on any NVIDIA host before consulting a
    single AMD probe, so nothing downstream could see the AMD card."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    _rocminfo(monkeypatch, stack, "  Name:                    gfx1201")
    assert stack._has_rocm_gpu() is True


def test_the_same_host_without_the_request_still_answers_no(stack, monkeypatch):
    """The pair to the test above, on an identical host: the ONLY difference is the variable, so a
    probe that started answering for some other reason fails here."""
    monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    _rocminfo(monkeypatch, stack, "  Name:                    gfx1201")
    assert stack._has_rocm_gpu() is False


def test_the_request_does_not_invent_a_card(stack, monkeypatch):
    """The request relaxes which vendor wins, not whether there is a card to serve."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    _rocminfo(monkeypatch, stack, "  Name:                    gfx000")  # CPU agent only
    assert stack._has_rocm_gpu() is False


def _shell_function(name: str) -> str:
    """The text of one function from install.sh, by brace matching."""
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}() {{"))
    depth = 0
    for end in range(start, len(lines)):
        depth += lines[end].count("{") - lines[end].count("}")
        if depth == 0:
            return "\n".join(lines[start : end + 1])
    raise AssertionError(f"unterminated function {name}")


def _shell_constant(name: str) -> str:
    """One top-level `NAME=value` assignment from install.sh."""
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    return next(
        line
        for line in install_sh.read_text(encoding = "utf-8").splitlines()
        if line.startswith(f"{name}=")
    )


def _clean_env(extra: "dict | None" = None) -> dict:
    """The runner's environment minus the two things that decide these cases by themselves."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("VISIBLE_DEVICES") and k != "UNSLOTH_ROCM_GFX_ARCH"
    }
    env.update(extra or {})
    return env


def _bash(script: str, *, env: "dict | None" = None) -> str:
    """Run a lifted script under bash and return its stdout, failing with its own stderr."""
    out = subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, env = env or _clean_env()
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


# `command -v rocminfo` must not reach a real one on a ROCm build host, or these answers
# would depend on the machine running the suite. Each harness names its own inventory.
_rocminfo_stub = "rocminfo() { return 1; }"


def _wheel_route_defs(*, arch_family: bool = True, rocminfo: str = _rocminfo_stub) -> "list[str]":
    """Everything _amd_request_has_a_wheel_route consults, lifted from install.sh."""
    return [
        _shell_function("_amd_hardware_corroborated"),
        *([_shell_function("_amd_arch_index_family_for_gfx")] if arch_family else []),
        _shell_function("_amd_gfx_has_wheel_route"),
        _shell_function("_amd_visible_masks_select_no_gpu"),
        _shell_function("_amd_generic_tag_carries_gfx"),
        _shell_function("_amd_mask_survivors"),
        _shell_function("_amd_runtime_gfx_target"),
        _shell_function("_amd_gfx_is_shadowing_integrated"),
        _shell_function("_amd_prefer_discrete_gfx"),
        _shell_function("_rocminfo_gpu_records"),
        _shell_function("_amd_ordered_gfx_devices"),
        "_ensure_rocm_probe_env() { :; }",
        rocminfo,
        _shell_function("_amd_request_has_a_wheel_route"),
    ]


def _fake_rocminfo(devices: "list[str]") -> str:
    """A rocminfo whose output has the shape the real one has."""
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
    """get_torch_index_url() under a stubbed host, returning the wheel index it picks."""
    script = "\n".join(
        [
            # Every case here describes a Linux x86_64 host, so say so rather than letting
            # the RUNNER answer. get_torch_index_url returns the cpu index outright for any
            # other `uname -m`, which on the ubuntu-24.04-arm leg turned these into "the
            # request selected cpu": two of them failed for a reason that is about the
            # runner, and the cpu-index CONTROLS below passed for the same wrong reason.
            'uname() { case "${1:-}" in -m) echo x86_64 ;; *) echo Linux ;; esac; }',
            _shell_function("_rocm_torch_explicitly_requested"),
            stubs,
            # The stubs above name this host's arch family, so it is not lifted here.
            *_wheel_route_defs(arch_family = False),
            # The generic route floors its leaf through this helper; left undefined, the
            # substitution is empty and every generic pick reads as a bare ".../whl/".
            _shell_constant("_ROCM_BNB_GENERIC_FLOOR_TAG"),
            _shell_function("_rocm_bnb_compatible_generic_tag"),
            _shell_function("get_torch_index_url"),
            f"{env} get_torch_index_url",
        ]
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        env = _clean_env(),
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
    """Fails before the fix: get_torch_index_url set _nvidia_detected=1 and returned a cu* family
    whatever the AMD probes said, so the request could not swap anything."""
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
    """A typo on a pure NVIDIA box must not turn a working CUDA machine into a ROCm or CPU one: the
    AMD route test still has to pass."""
    out = _index_url("UNSLOTH_FORCE_ROCM_TORCH=1", _NO_AMD_CARD)
    assert "rocm" not in out and "gfx" not in out, out


@pytest.mark.parametrize(
    "rocm_gpu, archs",
    [(True, ["gfx1100"]), (False, ["gfx1151"])],
    ids = ["runtime-visible", "runtime-less-but-inferable"],
)
def test_the_cuda_repair_stands_down_under_the_request(stack, monkeypatch, rocm_gpu, archs):
    """A standalone `studio update` leaves _TORCH_BACKEND empty, so the CUDA repair runs
    first, sees an NVIDIA GPU beside the requested HIP build, reads it as poisoning and
    reinstalls the CUDA trio. The second row is the control: a runtime-less host whose arch
    is inferable is one _ensure_rocm_torch will serve, so repairing CUDA here only to have
    ROCm forced back is a reinstall cycle on every update."""
    _mixed_host(stack, monkeypatch, rocm_gpu = rocm_gpu, archs = archs, display = True)
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
    _ensure_expected_torch_flavor restores, so honouring the request only in Python would
    install ROCm and have it reverted."""
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
    """_nvidia_gpu_wins_over_amd() under a stubbed host."""
    script = "\n".join(
        [
            _shell_function("_rocm_torch_explicitly_requested"),
            stubs,
            *_wheel_route_defs(),
            _shell_function("_nvidia_gpu_wins_over_amd"),
            f"{env} _nvidia_gpu_wins_over_amd && echo NVIDIA || echo AMD",
        ]
    )
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, env = _clean_env())
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
    """The #10450 gap the selector fix alone left open: an unreadable ROCm version makes
    get_torch_index_url return the cpu index deliberately, for the per-arch reroute to rewrite."""
    assert not _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _MIXED)


def test_the_same_mixed_host_without_the_request_keeps_cuda():
    """The control, differing only in the environment."""
    assert _nvidia_wins("", _MIXED)


def test_the_request_does_not_yield_a_pure_nvidia_host():
    """It relaxes which vendor wins, not whether there is a card to serve."""
    assert _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _PURE_NVIDIA)


def test_a_host_with_no_nvidia_card_is_unchanged_either_way():
    """The automatic path is untouched: with no NVIDIA GPU the answer never depended on the
    request, and must not start to."""
    assert not _nvidia_wins("", _PURE_AMD)
    assert not _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _PURE_AMD)


def test_the_reroutes_ask_the_request_aware_predicate():
    """The predicate is only worth testing if the reroutes actually consult it."""
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
    """Standing down needs a card to stand down FOR, which is the shell selector's rule too: a
    request on a box with no AMD GPU selects nothing and falls through to CUDA. Leaving the
    variable set on a host whose AMD card has since been removed must not silence this repair,
    because _ensure_rocm_torch then finds no target either and a stale HIP build would be left
    on a working NVIDIA GPU with nothing to fix it."""
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
    """install.sh's request-downgrade guard, lifted by text."""
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
    """TORCH_INDEX_URL after the guard, given what the AMD branch resolved."""
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
    return _bash(script, env = _clean_env(dict([env.split("=", 1)] if "=" in env else [])))


_CUDA = "https://download.pytorch.org/whl/cu130"


def test_a_dead_end_amd_route_keeps_cuda():
    """ROCm 5.x, an arch no index covers, or a non-x86_64 host all end the AMD branch at the cpu
    index."""
    assert (
        _index_after_the_guard(
            "UNSLOTH_FORCE_ROCM_TORCH=1",
            resolved = "https://download.pytorch.org/whl/cpu",
            cuda_answer = _CUDA,
        )
        == _CUDA
    )


def test_a_resolved_rocm_index_is_left_alone():
    """The control that makes the test above mean something: when the request DID reach ROCm
    wheels, the guard must not undo it."""
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
    """No request, no guard."""
    cpu = "https://download.pytorch.org/whl/cpu"
    assert _index_after_the_guard("", resolved = cpu, cuda_answer = _CUDA) == cpu


def test_a_mirror_whose_base_path_says_rocm_is_not_a_rocm_index():
    """The whole-URL form this guard first used matched a mirror's BASE path, so
    https://mirror.local/rocm-cache/cpu read as ROCm wheels and the guard stood down -- leaving
    CPU torch on a working NVIDIA card, which is the one outcome it exists to prevent."""
    assert (
        _index_after_the_guard(
            "UNSLOTH_FORCE_ROCM_TORCH=1",
            resolved = "https://mirror.local/rocm-cache/cpu",
            cuda_answer = _CUDA,
        )
        == _CUDA
    )


def test_a_mirrored_rocm_index_is_still_left_alone():
    """The control for the case above, one path segment apart: the same mirror serving an actual
    ROCm leaf must keep it."""
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
    """repo.radeon.com ends in rocm-rel-X.Y, which _is_pip_rocm_family_leaf declines -- correctly,
    since it answers the narrower "is this a pip ROCm FAMILY index"."""
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
    """A card is necessary and not sufficient."""
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
    """The other half of the pair, and the reason the test above is not arbitrary: the helper the
    repair now consults is the one _ensure_rocm_torch itself returns on, so the two sides of the
    installer cannot disagree about whether a route exists."""
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


def _gpu_summary_branch(
    resolved: str,
    request: str,
    pinned: bool = False,
) -> str:
    """Which arm the GPU detection summary takes, given the index the resolution left."""
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    head = next(
        (l for l in lines if l.startswith("if _has_usable_nvidia_gpu && ")),
        "if _nvidia_gpu_wins_over_amd; then",
    )
    # The condition spans a continuation; take the rest of it verbatim.
    if head.rstrip().endswith("\\"):
        _at = lines.index(head)
        while lines[_at].rstrip().endswith("\\"):
            _at += 1
            head = head.rstrip().rstrip("\\") + " " + lines[_at].strip()
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
            f"_torch_index_pinned={'true' if pinned else 'false'}",
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
    return _bash(script)


def test_a_pinned_rocm_index_still_reports_the_nvidia_card():
    """A pin names a wheel family, not a card. An NVIDIA host pinned to a ROCm index has no AMD
    card to describe, so reading the resolved index alone hid the NVIDIA identity behind an
    "AMD ROCm" banner for hardware the machine does not have."""
    assert (
        _gpu_summary_branch("https://download.pytorch.org/whl/rocm7.2", "", pinned = True) == "nvidia"
    )


def test_an_unpinned_rocm_resolution_still_reports_amd():
    """The control: the exemption is the pin, not the index. The same URL reached by resolution
    still reports the card its wheels are for."""
    assert _gpu_summary_branch("https://download.pytorch.org/whl/rocm7.2", "") == "amd"


def test_the_summary_reports_cuda_after_the_cuda_restore():
    """The mixed host whose request found no ROCm route: the guard put the CUDA index back, but the
    request is still set and the AMD card is still there, so the reroute predicate still answers
    "AMD wins"."""
    assert _gpu_summary_branch(_CUDA, "UNSLOTH_FORCE_ROCM_TORCH=1") == "nvidia"


def test_the_summary_still_reports_amd_when_the_request_won():
    """The control: when the request DID reach ROCm wheels, the summary must report the card those
    wheels are for, which is what the predicate was introduced to fix."""
    assert (
        _gpu_summary_branch(
            "https://repo.amd.com/rocm/whl/gfx1151",
            "UNSLOTH_FORCE_ROCM_TORCH=1",
        )
        == "amd"
    )


def test_the_summary_on_a_pure_cuda_install_is_unchanged():
    """And the control for the control: no request, CUDA wheels, NVIDIA reported, which is what
    every automatic install on an NVIDIA host must keep seeing."""
    assert _gpu_summary_branch(_CUDA, "") == "nvidia"


def _route_shell(probe: str, inferred: str, pci_ok: bool) -> bool:
    """install.sh's request route test, on a stubbed host."""
    script = "\n".join(
        [
            f'_probe_amd_gfx_arch() {{ printf "%s\\n" {probe!r}; }}',
            "_kfd_gfx_targets() { :; }",
            f'_infer_linux_amd_gfx_arch() {{ [ -n {inferred!r} ] && printf "%s\\n" {inferred!r}; }}',
            f"_amd_gpu_present_via_pci() {{ return {0 if pci_ok else 1}; }}",
            *_wheel_route_defs(),
            "_detect_rocm_version_tag() { printf '%s\\n' rocm7.2; }",
            "_amd_request_has_a_wheel_route && echo yes || echo no",
        ]
    )
    return _bash(script, env = _clean_env()) == "yes"


@pytest.mark.parametrize(
    "probe, inferred, pci_ok, routes",
    [
        # gfx1010 (RDNA 1) is in neither the generic wheel nor any per-arch index.
        ("gfx1010", "", True, False),
        ("gfx1100", "", True, True),
        # A Strix box with no rocminfo, amd-smi or KFD node is served per-arch wheels on a
        # pure-AMD host by the reroute this predicate gates, so it is a route here too.
        ("", "gfx1151", True, True),
        ("", "gfx1030", False, False),
        # gfx906's only route is the rocm6.3 legacy tag, which opens solely when it is the
        # sole arch on the machine.
        ("gfx906", "", True, True),
        ("gfx906\ngfx1010", "", True, False),
    ],
    ids = [
        "no-index",
        "generic-wheel",
        "inferable-no-runtime",
        "declared-without-a-card",
        "gfx906-sole-arch",
        "gfx906-mixed-amd",
    ],
)
def test_the_request_needs_a_card_whose_arch_an_index_serves(probe, inferred, pci_ok, routes):
    """Presence is not the bar: an arch no index can serve must never depose a card that can,
    and each row is paired with its control."""
    assert _route_shell(probe, inferred, pci_ok = pci_ok) is routes


def _mixed_host(stack, monkeypatch, **over):
    """The host this feature exists for, stated by difference."""
    o = dict(
        request = True,
        nvidia = True,
        rocm_gpu = True,
        archs = ["gfx1100"],
        devices = None,
        masked = None,
        kfd = [],
        wsl = False,
        display = None,
        miscomputing = False,
        machine = "x86_64",
        windows = False,
        macos = False,
        backend = "",
        inferred = None,
        rocm = (6, 4),
        torch = "2.13.0+cu130",
        declared = None,
        pin = None,
        pin_url = None,
        masks = {},
    )
    o.update(over)
    if o["display"] is None:
        o["display"] = o["rocm_gpu"]
    devices = o["archs"] if o["devices"] is None else o["devices"]

    monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    if o["request"]:
        monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    for _var in (
        "UNSLOTH_ROCM_GFX_ARCH",
        "UNSLOTH_TORCH_INDEX_URL",
        "UNSLOTH_TORCH_INDEX_FAMILY",
        "UNSLOTH_ROCM_TORCH_INSTALLED",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(_var, raising = False)
    for _var, _val in (
        ("UNSLOTH_ROCM_GFX_ARCH", o["declared"]),
        ("UNSLOTH_TORCH_INDEX_FAMILY", o["pin"]),
        ("UNSLOTH_TORCH_INDEX_URL", o["pin_url"]),
    ):
        if _val is not None:
            monkeypatch.setenv(_var, _val)
    for _var, _val in o["masks"].items():
        monkeypatch.setenv(_var, _val)

    monkeypatch.setattr(stack.platform, "machine", lambda: o["machine"])
    monkeypatch.setattr(stack, "IS_WINDOWS", o["windows"])
    monkeypatch.setattr(stack, "IS_MACOS", o["macos"])
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_TORCH_BACKEND", o["backend"])
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: o["nvidia"])
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: o["rocm_gpu"])
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda *a, **k: list(o["kfd"]))
    monkeypatch.setattr(stack, "_is_wsl", lambda: o["wsl"])
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: o["display"])
    monkeypatch.setattr(stack, "_miscomputing_arch_host", lambda: o["miscomputing"])
    monkeypatch.setattr(stack, "_infer_linux_amd_gfx_arch", lambda *a, **k: o["inferred"])
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda *a, **k: o["rocm"])
    monkeypatch.setattr(
        stack,
        "_physical_amd_gfx_archs",
        lambda: list(o["archs"]) or ([o["inferred"]] if o["inferred"] else []),
    )
    # The masked list is what a probe that HONOURS the mask returns; ignore_visible_masks asks
    # for the machine before any mask, which is the distinction the ROCr layer rests on.
    monkeypatch.setattr(
        stack,
        "_detect_amd_gfx_codes",
        lambda **k: list(
            devices if o["masked"] is None or k.get("ignore_visible_masks") else o["masked"]
        ),
    )
    monkeypatch.setattr(
        stack,
        "_probe_torch_runtime",
        lambda *a, **k: (True, True, o["torch"], "rocm" in o["torch"], ""),
    )
    # The pin readers are deliberately NOT stubbed: the environment above is already
    # cleared, so they answer None on their own, and a case that sets a pin needs the real
    # reader to see it.
    return o


def _viable(
    stack,
    monkeypatch,
    *,
    corroborated: bool,
    archs: list,
    miscomputing = False,
    machine: str = "x86_64",
):
    _mixed_host(
        stack,
        monkeypatch,
        request = False,
        rocm_gpu = corroborated,
        archs = archs,
        miscomputing = miscomputing,
        machine = machine,
    )
    return stack._forced_rocm_route_is_viable()


def test_a_host_with_no_published_rocm_wheels_is_not_a_viable_route(stack, monkeypatch):
    """_ensure_rocm_torch() returns on its Linux-x86_64 gate without installing anything, because
    ROCm wheels are published nowhere else."""
    assert (
        _viable(stack, monkeypatch, corroborated = True, archs = ["gfx1100"], machine = "aarch64")
        is False
    )


def test_the_same_host_on_x86_64_is_still_viable(stack, monkeypatch):
    """The control: same card, same corroboration, an architecture the wheels exist for."""
    assert (
        _viable(stack, monkeypatch, corroborated = True, archs = ["gfx1100"], machine = "x86_64") is True
    )


def test_the_python_route_test_matches_the_shell_one(stack, monkeypatch):
    """Both halves of the installer have to answer the same question the same way, or a standalone
    `studio update` undoes what install.sh chose."""
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
    """The repair had stood down for any host with a visible AMD GPU. With an unusable HIP build
    and a gfx1010 beside a working NVIDIA card, nothing then classified the stale build,
    _ensure_rocm_torch found no wheel tag, and the NVIDIA GPU was left with no working torch at
    all."""
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


def test_a_declared_arch_does_not_force_rocm_over_a_working_cuda_stack(stack, monkeypatch):
    """The request skips the NVIDIA precedence return, leaving the presence gate as the only thing
    between a stale UNSLOTH_ROCM_GFX_ARCH and per-arch AMD wheels installed over CUDA on a host
    with no AMD card."""
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
    """The control: declaring an arch is the documented routing hint for a runtime-less host, so it
    must keep working once the silicon is corroborated."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", "gfx1151")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack.platform, "machine", lambda: "x86_64")
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
    monkeypatch.setattr(
        stack,
        "_probe_torch_runtime",
        lambda *a, **k: (True, True, "2.11.0", False, True),
    )
    # Not a no-op pip_install stub: this route installs through pip_install_try, which is
    # a real subprocess too, so stopping at the probe is what keeps the test hermetic.
    assert _reaches_the_version_probe(stack, monkeypatch) is True


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
    """The call site, not the helper: both per-arch reroutes consult this predicate, and on
    presence alone a gfx1010 cleared the way for AMD wheels that carry no kernels for it."""
    assert _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _UNROUTABLE_AMD)


def test_the_reroute_predicate_still_yields_for_a_routable_one():
    """The control: same host, routable arch, and the request must still win."""
    assert not _nvidia_wins("UNSLOTH_FORCE_ROCM_TORCH=1", _MIXED)


def _shell_request_flag(value: "str | None") -> bool:
    """install.sh's own request test, run verbatim, for one value of the variable."""
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
    return _bash(script, env = env) == "yes"


@pytest.mark.parametrize("value", [" true ", "\ttrue", "1 ", " ON"])
def test_the_shell_flag_is_trimmed_like_its_python_twin(value):
    """install_python_stack.py reads this variable through .strip(); install.sh did not."""
    assert _shell_request_flag(value) is True


@pytest.mark.parametrize("value", ["  ", "", "tru e", None])
def test_trimming_does_not_widen_what_counts_as_a_request(value):
    """The control."""
    assert _shell_request_flag(value) is False


def _route_shell_masked(
    physical: "list[str]",
    rocm_tag: str = "rocm7.2",
    devices: "list[str] | None" = None,
    kfd: "list[str] | None" = None,
    **mask: str,
) -> bool:
    """The request route test on a masked host, with the mask resolution left live."""
    return (
        _bash(
            _route_script(physical, rocm_tag, devices, kfd, tail = _ROUTE_YES_NO),
            env = _route_env(mask),
        )
        == "yes"
    )


_ROUTE_YES_NO = "_amd_request_has_a_wheel_route && echo yes || echo no"
# The published target, not the verdict: on a host whose integrated GPU enumerates first the
# interesting failure is not whether ROCm wins but which card it wins FOR.
_ROUTE_TARGET = (
    '_amd_request_has_a_wheel_route >/dev/null 2>&1; printf "%s" "${_AMD_REQUEST_TARGET_GFX:-}"'
)


def _route_env(mask: dict) -> dict:
    return _clean_env(mask)


def _route_target_masked(
    physical: "list[str]",
    rocm_tag: str = "rocm7.2",
    devices: "list[str] | None" = None,
    kfd: "list[str] | None" = None,
    **mask: str,
) -> str:
    """The gfx the request would install FOR, or "" when it declines."""
    return _bash(
        _route_script(physical, rocm_tag, devices, kfd, tail = _ROUTE_TARGET),
        env = _route_env(mask),
    )


def _route_script(
    physical: "list[str]",
    rocm_tag: str,
    devices: "list[str] | None",
    kfd: "list[str] | None",
    *,
    tail: str,
) -> str:
    emit = 'printf "%s\\n" ' + " ".join(repr(a) for a in physical) if physical else ":"
    # KFD node order is what the mask ordinals index, and once rocminfo says nothing it is
    # the only ordered source left, so a test about ordering cannot leave it stubbed empty.
    kfd_emit = 'printf "%s\\n" ' + " ".join(repr(a) for a in kfd) if kfd else ":"
    script = "\n".join(
        [
            f"_probe_amd_gfx_arch() {{ {emit}; }}",
            f"_kfd_gfx_targets() {{ {kfd_emit}; }}",
            "_infer_linux_amd_gfx_arch() { :; }",
            "_amd_gpu_present_via_pci() { return 0; }",
            *_wheel_route_defs(rocminfo = _fake_rocminfo(physical if devices is None else devices)),
            f"_detect_rocm_version_tag() {{ printf '%s\\n' {rocm_tag!r}; }}",
            tail,
        ]
    )
    return script


@pytest.mark.parametrize(
    "physical, mask, routes",
    [
        # No probe can answer this: rocminfo is filtered only by ROCR_VISIBLE_DEVICES and
        # amd-smi by neither, so a probe run under a HIP mask returns the whole machine and
        # the route test answered yes on the card the mask had just hidden.
        (["gfx1100", "gfx1010"], "1", False),
        (["gfx1100", "gfx1010"], "0", True),
        # Failing closed is about what cannot be resolved, not the whole variable: the
        # runtime reads left to right and stops at the first entry naming no device, so the
        # survivors are the PREFIX.
        (["gfx1100", "gfx1010"], "0,GPU-abcdef", True),
        (["gfx1100", "gfx1010"], "1,GPU-abcdef", False),
        # gfx906's only route is the rocm6.3 legacy tag, and the reroute granting it inspects
        # the UNMASKED inventory, refusing when a second AMD arch is present.
        (["gfx906", "gfx1100"], "0", False),
        # HIP remaps runtime ordinal 0 onto the head of the mask.
        (["gfx1100", "gfx1010"], "1,0", False),
        (["gfx1100", "gfx1010"], "0,1", True),
    ],
    ids = [
        "selected-unroutable",
        "selected-routable",
        "prefix-resolves",
        "prefix-unroutable",
        "gfx906-not-sole-arch",
        "mask-head-second-card",
        "mask-head-first-card",
    ],
)
def test_the_shell_route_judges_the_card_the_mask_selects(physical, mask, routes):
    """install.sh's half of the rule above, on the same host shapes."""
    assert _route_shell_masked(physical, HIP_VISIBLE_DEVICES = mask) is routes


def test_the_flat_probes_duplicate_rows_do_not_shift_the_ordinals():
    """rocminfo names each agent's target twice, so the flat probe returns four rows for a two-card
    host."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1100", "gfx1010", "gfx1010"],
            devices = ["gfx1100", "gfx1010"],
            HIP_VISIBLE_DEVICES = "1",
        )
        is False
    )


def test_the_same_duplicated_host_still_yields_for_the_routable_card():
    """The control: same duplicated inventory, mask pointing at the card that does have a route."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1100", "gfx1010", "gfx1010"],
            devices = ["gfx1100", "gfx1010"],
            HIP_VISIBLE_DEVICES = "0",
        )
        is True
    )


def test_a_mask_past_the_first_device_fails_closed_with_no_per_device_list():
    """Only rocminfo enumerates in the order the masks index -- amd-smi orders by KFD discovery,
    which is why _amd_smi_hip_order exists -- so a host without it cannot say which card an
    ordinal names."""
    assert _route_shell_masked(["gfx1010", "gfx1100"], devices = [], HIP_VISIBLE_DEVICES = "1") is False


def test_one_arch_is_still_answerable_without_a_per_device_list():
    """The control, and the reason this is not simply "no rocminfo, no route": with a single arch
    on the whole host every ordinal names it, so no enumeration order can change the answer and
    the flat inventory is enough."""
    assert _route_shell_masked(["gfx1100", "gfx1100"], devices = [], HIP_VISIBLE_DEVICES = "0") is True


def test_discovery_order_does_not_answer_for_runtime_device_zero():
    """With no rocminfo the flat inventory came from amd-smi, which enumerates in KFD DISCOVERY
    order -- the whole reason _amd_smi_hip_order exists -- so its first row can describe a
    different GPU from the one HIP hands torch at ordinal 0. Reading it approved the swap off a
    routable first row while the gfx1010 that actually runs has kernels in no wheel."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], devices = [], HIP_VISIBLE_DEVICES = "0") is False


def test_the_kernel_topology_supplies_the_order_amd_smi_cannot():
    """KFD node order IS the order HIP and ROCr index, so the repair is to read it rather than to
    decline -- the same swap _runtime_gfx_target makes in install_python_stack.py."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1010"],
            devices = [],
            kfd = ["gfx1010", "gfx1100"],
            HIP_VISIBLE_DEVICES = "0",
        )
        is False
    )


def test_the_same_kernel_topology_still_yields_for_the_routable_card():
    """The control: same host, same KFD order, the mask pointing at the card with a route."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1010"],
            devices = [],
            kfd = ["gfx1010", "gfx1100"],
            HIP_VISIBLE_DEVICES = "1",
        )
        is True
    )


@pytest.mark.parametrize(
    "mask, routes",
    [("7", False), ("1", True)],
    ids = ["mask-resolves-nothing", "mask-names-a-device"],
)
def test_a_declared_arch_does_not_answer_over_the_mask(mask, routes):
    """UNSLOTH_ROCM_GFX_ARCH takes an early return above the mask resolution, so a declared
    arch beside HIP_VISIBLE_DEVICES=7 approved the swap where HIP exposes no device at all.
    The control keeps the escape hatch: the same declaration over a mask naming a device this
    host has must still depose CUDA."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1010"],
            devices = ["gfx1100", "gfx1010"],
            UNSLOTH_ROCM_GFX_ARCH = "gfx1100",
            HIP_VISIBLE_DEVICES = mask,
        )
        is routes
    )


def test_the_rocr_layer_is_resolved_the_same_way():
    """rocminfo does honour this one, but the route test must not depend on which probe answered:
    amd-smi honours neither mask, so the resolution has to be its own."""
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
    """A deliberate no-GPU selection, not a detection miss: there is nothing for the request to
    swap TO, so falling back to the physical inventory here installed ROCm wheels on a host
    whose runtime exposes no AMD GPU at all."""
    assert _route_shell_masked(["gfx1100"], **mask) is False


def test_an_unset_mask_is_not_a_no_gpu_selection():
    """Its control, and the reason ${VAR+x} rather than ${VAR:-}: an unset mask hides nothing, and
    reading it like a set-but-empty one would keep CUDA on every host."""
    assert _route_shell_masked(["gfx1100"]) is True


@pytest.mark.parametrize("value", ["GPU-abcdef", "9"])
def test_a_mask_this_cannot_resolve_fails_closed(value):
    """ROCr accepts UUIDs, and an ordinal can name no device at all."""
    assert _route_shell_masked(["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = value) is False


def test_a_masked_gfx906_alone_is_still_routable():
    """The control: on a host where gfx906 IS the sole arch, the legacy tag opens and a mask naming
    it must not change that."""
    assert _route_shell_masked(["gfx906"], HIP_VISIBLE_DEVICES = "0") is True


def test_an_explicit_cuda_pin_outranks_the_request(stack, monkeypatch):
    """_rocm_torch_explicitly_requested's docstring promises an index pin still outranks the
    request, and _rocm_pin is the ROCm pin, so a CUDA pin leaves it None and the request skipped
    the NVIDIA-precedence return."""
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
    """The control: the pin that names ROCm wheels must still win, or the fix would read as "any
    pin keeps CUDA" and break the headless/CI case _rocm_pin exists for."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "rocm7.0")
    monkeypatch.setattr(stack, "_TORCH_BACKEND", "")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(stack, "NO_TORCH", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    # A ROCm pin skips the whole vendor-precedence block, so reaching the version probe
    # is what "the pin still won" looks like from outside.
    assert _reaches_the_version_probe(stack, monkeypatch) is True


def _viable_masked(
    stack,
    monkeypatch,
    *,
    devices: list,
    masked: "list | None" = None,
    inferred: "str | None" = None,
    rocm: "tuple | None" = (6, 4),
    declared: "str | None" = None,
    **mask: str,
) -> bool:
    """_forced_rocm_route_is_viable on a masked host, with the resolution left live."""
    _mixed_host(
        stack,
        monkeypatch,
        request = False,
        archs = devices,
        devices = devices,
        masked = masked,
        inferred = inferred,
        rocm = rocm,
        declared = declared,
        masks = mask,
    )
    return stack._forced_rocm_route_is_viable()


@pytest.mark.parametrize(
    "devices, mask, viable",
    [
        # Which CARD the mask selects, not whether some arch on the bus has a route.
        (["gfx1100", "gfx1010"], "1", False),
        (["gfx1100", "gfx1010"], "0", True),
        # _pick_visible_index folds an out-of-range ordinal and a UUID alike onto GPU 0,
        # which is right for arch SELECTION and wrong for replacing a working stack: HIP
        # exposes no device for either, so both fail closed. install.sh agrees on these.
        (["gfx1100", "gfx1010"], "7", False),
        (["gfx1100", "gfx1010"], "GPU-9d4f00a1", False),
        (["gfx1010", "gfx1100"], "1", True),
        # gfx1033 is refused outright (studio/ROCM_RDNA2_APU.md), keyed on the SELECTED
        # card: a gfx1033 anywhere would be the host-wide reading this replaces.
        (["gfx1033", "gfx1100"], "0", False),
        (["gfx1033", "gfx1100"], "1", True),
    ],
    ids = [
        "selected-unroutable",
        "selected-routable",
        "past-last-device",
        "uuid",
        "in-range-second",
        "selected-miscomputing",
        "healthy-sibling",
    ],
)
def test_the_python_route_judges_the_card_the_mask_selects(
    stack, monkeypatch, devices, mask, viable
):
    """Each row is paired with the same host masked the other way, so a fix that answered
    "never viable" fails here."""
    assert _viable_masked(stack, monkeypatch, devices = devices, HIP_VISIBLE_DEVICES = mask) is viable


def test_a_mask_exposing_no_device_is_not_a_viable_route(stack, monkeypatch):
    """The Python half of the same rule the shell now applies: a no-GPU mask is a deliberate
    selection, so there is nothing to swap to."""
    assert (
        _viable_masked(stack, monkeypatch, devices = ["gfx1100"], HIP_VISIBLE_DEVICES = "-1") is False
    )


def test_an_unmasked_host_is_judged_exactly_as_before(stack, monkeypatch):
    """And the control for all three: with no mask set the answer is the inventory's, which is what
    this function did before and what the shell twin still does."""
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
    _mixed_host(stack, monkeypatch, archs = archs, pin = pin, pin_url = pin_url)
    monkeypatch.setattr(stack, "_forced_rocm_route_is_viable", lambda: viable)
    return _reaches_the_version_probe(stack, monkeypatch)


def test_the_request_needs_a_viable_route_to_bypass_nvidia(stack, monkeypatch):
    """Presence was the bar, and presence is not the bar this feature states."""
    assert _rocm_repair_reached(stack, monkeypatch, archs = ["gfx1010"], viable = False) is False


def test_a_viable_route_still_lets_the_request_through(stack, monkeypatch):
    """The control: the same mixed host with a card an index can serve."""
    assert _rocm_repair_reached(stack, monkeypatch, archs = ["gfx1100"], viable = True) is True


@pytest.mark.parametrize("family", ["cpu", "cu128"])
def test_an_explicit_pin_of_another_family_outranks_the_request(stack, monkeypatch, family):
    """A /cpu pin had the multi-GB ROCm stack installed here and then undone by _ensure_cpu_torch,
    on every update."""
    assert (
        _rocm_repair_reached(stack, monkeypatch, archs = ["gfx1100"], viable = True, pin = family)
        is False
    )


@pytest.mark.parametrize(
    "pin_url, reached",
    [("https://download.pytorch.org/whl/rocm6.4", True), ("https://mirror.example/simple", False)],
    ids = ["rocm-pin-agrees", "unknown-family-pin"],
)
def test_a_pin_of_another_family_is_the_only_pin_that_outranks_the_request(
    stack, monkeypatch, pin_url, reached
):
    """A pin naming ROCm agrees with the request and must not be swept up by the CPU/CUDA
    check. An unknown leaf was applied verbatim at install time, so _ensure_rocm_torch
    declines to judge it and returns before any vendor gate -- pre-existing, not added here."""
    assert (
        _rocm_repair_reached(stack, monkeypatch, archs = ["gfx1100"], viable = True, pin_url = pin_url)
        is reached
    )


def test_the_two_mask_layers_compose_in_the_order_the_runtime_reads_them():
    """ROCr filters the physical list and renumbers the survivors; the HIP layer then indexes
    those."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1010"], ROCR_VISIBLE_DEVICES = "1", HIP_VISIBLE_DEVICES = "0"
        )
        is False
    )


def test_the_same_two_layers_pointing_at_the_routable_card_still_depose_cuda():
    """The control: same host, same stacking, ROCr selecting the other card."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1010"], ROCR_VISIBLE_DEVICES = "0", HIP_VISIBLE_DEVICES = "0"
        )
        is True
    )


def test_two_cards_of_one_arch_do_not_shift_the_ordinals():
    """An ordinal names a DEVICE, not an architecture."""
    assert _route_shell_masked(["gfx1010", "gfx1010", "gfx1100"], HIP_VISIBLE_DEVICES = "1") is False


def test_the_same_host_selecting_past_the_pair_is_still_routable():
    """The control for the cardinality rule: ordinal 2 on that host really is the gfx1100, and a
    resolver that simply refused every repeated arch would fail this."""
    assert _route_shell_masked(["gfx1010", "gfx1010", "gfx1100"], HIP_VISIBLE_DEVICES = "2") is True


def test_a_declared_arch_decides_before_the_inventory():
    """UNSLOTH_ROCM_GFX_ARCH is what _probe_amd_gfx_arch's default mode returns and what
    get_torch_index_url installs from, so judging the physical inventory instead can pass on a
    routable sibling while the wheels are chosen for the declared card."""
    assert _route_shell_masked(["gfx1100"], UNSLOTH_ROCM_GFX_ARCH = "gfx1010") is False


def test_a_declared_arch_that_is_routable_still_deposes_cuda():
    """The control, and the #7301 host: the declaration exists to serve a runtime-less card, so a
    routable one must still win even when the inventory disagrees with it."""
    assert _route_shell_masked(["gfx1010"], UNSLOTH_ROCM_GFX_ARCH = "gfx1100") is True


def test_a_generic_only_arch_is_judged_against_the_tag_this_host_resolves():
    """gfx950 has no per-arch index, so its only route is the generic wheel -- and the tag comes
    from the installed ROCm version, which a stale /opt/rocm beside a current amdgpu can leave
    older than the card."""
    assert _route_shell_masked(["gfx950"], rocm_tag = "rocm6.4") is False


def test_the_same_arch_on_a_tag_that_carries_it_is_routable():
    """The control: the identical host one tag later, where the wheel does carry gfx950."""
    assert _route_shell_masked(["gfx950"], rocm_tag = "rocm7.0") is True


def test_an_arch_with_its_own_index_is_not_held_to_the_generic_floor():
    """The boundary: gfx1200 predates the rocm6.0 generic wheel too, but it has a per-arch AMD
    index that carries it, and the reroute is what serves it."""
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
    """/dev/dxg is the generic WSL GPU bridge an NVIDIA passthrough creates too, and librocdxg is a
    file an uninstalled ROCm leaves behind, so neither names a vendor."""
    assert _corroborated_on_wsl(stack, monkeypatch, nvidia = True) is False


def test_the_same_wsl_box_with_no_nvidia_card_still_counts(stack, monkeypatch):
    """The control, and why this is not simply "WSL never corroborates": with no NVIDIA GPU there
    is no working stack to lose, so the leftover reading still beats declining and a WSL AMD
    host whose runtime cannot answer keeps its route."""
    assert _corroborated_on_wsl(stack, monkeypatch, nvidia = False) is True


def test_a_wsl_box_whose_runtime_names_an_agent_counts_beside_nvidia(stack, monkeypatch):
    """The other control: a real AMD adapter under WSL is still corroborated with an NVIDIA card
    present, because the ROCm runtime enumerates it rather than a leftover file standing in for
    it."""
    assert _corroborated_on_wsl(stack, monkeypatch, nvidia = True, rocm_sees_a_gpu = True) is True


def _needs_repair_passes_the_nvidia_gate(stack, monkeypatch, *, viable: bool) -> bool:
    """Whether _amd_torch_needs_dependency_pass() gets past its NVIDIA fast path."""

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
    """On a mixed host whose selected AMD card has no wheel route, _ensure_cuda_torch() and
    _ensure_rocm_torch() both leave CUDA in place."""
    assert _needs_repair_passes_the_nvidia_gate(stack, monkeypatch, viable = False) is False


def test_a_routable_request_still_bypasses_it(stack, monkeypatch):
    """The control: the request is the whole feature, so a viable one must still outrank the NVIDIA
    card here or the repair it gates can never run."""
    assert _needs_repair_passes_the_nvidia_gate(stack, monkeypatch, viable = True) is True


def test_an_unresolvable_mask_does_not_answer_for_the_next_host(stack, monkeypatch):
    """The flag is module state, so it is reset on entry rather than only where it is decided."""
    assert (
        _viable_masked(stack, monkeypatch, devices = ["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "7")
        is False
    )
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", "gfx1100")
    assert _viable_masked(stack, monkeypatch, devices = ["gfx1100", "gfx1010"]) is True


def test_a_rocr_ordinal_past_the_last_device_is_not_a_viable_route(stack, monkeypatch):
    """ROCr's own filter (ROCR-Runtime, core/inc/amd_filter_device.h) surfaces the tokens that are
    "Legal and NOT Terminating", and an index terminates when it "lies outside the interval [0 -
    (numGpuDevices - 1)]" -- so ROCR_VISIBLE_DEVICES=7 on a two-GPU box surfaces nothing and the
    HIP layer above it indexes an empty list."""
    assert (
        _viable_masked(stack, monkeypatch, devices = ["gfx1100", "gfx1010"], ROCR_VISIBLE_DEVICES = "7")
        is False
    )


def test_a_rocr_prefix_that_survives_is_still_a_viable_route(stack, monkeypatch):
    """The control, and the reason this is a prefix rule rather than "any bad token declines":
    ROCR_VISIBLE_DEVICES=0,7 terminates at the 7 and still surfaces device 0, which is routable."""
    assert (
        _viable_masked(
            stack, monkeypatch, devices = ["gfx1100", "gfx1010"], ROCR_VISIBLE_DEVICES = "0,7"
        )
        is True
    )


def test_a_rocm_version_no_wheel_family_serves_is_not_a_viable_route(stack, monkeypatch):
    """gfx908 is in the arch tables, so the route test said yes -- but on ROCm 5.7 no generic
    rocmX.Y tag resolves, the missing-kernel reroute does not fire for an arch the generic wheel
    does carry, and _ensure_rocm_torch prints "No PyTorch wheel for ROCm 5.7" and installs
    nothing."""
    assert _viable_masked(stack, monkeypatch, devices = ["gfx908"], rocm = (5, 7)) is False


def test_the_same_arch_on_a_version_with_a_wheel_family_is_viable(stack, monkeypatch):
    """The control: the same card on ROCm 6.4, where the tag resolves and the install proceeds."""
    assert _viable_masked(stack, monkeypatch, devices = ["gfx908"], rocm = (6, 4)) is True


def test_an_arch_the_generic_wheel_lacks_still_routes_on_an_old_version(stack, monkeypatch):
    """The second control, and the one that keeps the version test from swallowing the repair it
    exists beside: gfx1103 has no generic kernels at any tag, so _ensure_rocm_torch reroutes it
    to AMD's per-arch index whatever the version reads."""
    assert _viable_masked(stack, monkeypatch, devices = ["gfx1103"], rocm = (5, 7)) is True


def test_the_route_test_does_not_read_the_hosts_rocm_version(stack, monkeypatch):
    """The guard for the pin above, since its absence is not visible in a passing run."""
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda: None)
    assert _viable_masked(stack, monkeypatch, devices = ["gfx1100"]) is True


def test_a_declared_arch_does_not_answer_over_an_unresolvable_mask_in_python(stack, monkeypatch):
    """The Python half of the shell rule above."""
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", "gfx1100")
    assert (
        _viable_masked(stack, monkeypatch, devices = ["gfx1100", "gfx1010"], HIP_VISIBLE_DEVICES = "7")
        is False
    )


def test_a_declared_arch_over_a_mask_that_resolves_is_still_viable_in_python(stack, monkeypatch):
    """The control, and the reason the resolution is gated on a mask being set at all: the ordinary
    declared-arch host has none, and must still route without paying for a probe."""
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = ["gfx1100", "gfx1010"],
            declared = "gfx1100",
            HIP_VISIBLE_DEVICES = "1",
        )
        is True
    )


# A mixed AMD host: the arch the Deck gate refuses beside one it serves, with the kernel's
# topology naming both in the order the masks index.
_VAN_GOGH_PLUS_DGPU = "\n".join(
    [
        "_has_usable_nvidia_gpu() { return 0; }",
        "_has_amd_rocm_gpu() { return 0; }",
        "_probe_amd_gfx_arch() { printf '%s\\n' gfx1033 gfx1100; }",
        # The real map, not a constant: without it gfx1033 has no per-arch family here and
        # the route test declines for the wrong reason, so the control below would pass
        # whatever the gate does.
        '_amd_arch_index_family_for_gfx() { case "$1" in'
        " gfx1100) echo gfx110X-all ;; gfx1033) echo gfx103X-all ;; *) return 1 ;; esac; }",
        "_kfd_gfx_targets() { printf '%s\\n' gfx1033 gfx1100; }",
        "_infer_linux_amd_gfx_arch() { :; }",
        "_amd_gpu_present_via_pci() { return 0; }",
        "_detect_rocm_version_tag() { echo rocm7.0; }",
        "_amd_sole_index_arch() { :; }",
        "_amd_agreed_index_family() { :; }",
        "_rocm_sdk_install_hint() { echo ''; }",
        "nvidia-smi() { echo 'CUDA Version: 13.0'; }",
    ]
)


def test_the_bad_arch_gate_answers_for_the_selected_card_under_the_request(stack):
    """The gate is a PRESENCE test because an unresolved host cannot say which card runs."""
    out = _index_url("UNSLOTH_FORCE_ROCM_TORCH=1 HIP_VISIBLE_DEVICES=1", _VAN_GOGH_PLUS_DGPU)
    assert out.strip().endswith("/rocm7.0"), out


def test_the_same_request_selecting_the_deck_still_takes_the_cpu_index(stack):
    """The control that keeps the narrowing honest: the request cannot buy ROCm wheels for the arch
    measured to compute wrong answers."""
    out = _index_url("UNSLOTH_FORCE_ROCM_TORCH=1 HIP_VISIBLE_DEVICES=0", _VAN_GOGH_PLUS_DGPU)
    assert out.strip().endswith("/cpu"), out


def test_the_presence_rule_is_unchanged_for_a_host_that_did_not_ask(stack):
    """And the control for every other host: with no request nothing has resolved a target, so the
    gate answers on presence exactly as before (#7776, studio/ROCM_RDNA2_APU.md)."""
    out = _index_url(
        "HIP_VISIBLE_DEVICES=1",
        _VAN_GOGH_PLUS_DGPU.replace(
            "_has_usable_nvidia_gpu() { return 0; }", "_has_usable_nvidia_gpu() { return 1; }"
        ),
    )
    assert out.strip().endswith("/cpu"), out


def test_a_repeated_rocr_ordinal_does_not_invent_a_device(stack, monkeypatch):
    """ROCr terminates on an index that "maps to a device that has been previously selected" (ROCR-
    Runtime, core/inc/amd_filter_device.h), so ROCR_VISIBLE_DEVICES=0,0 surfaces ONE device and
    HIP_VISIBLE_DEVICES=1 above it then indexes nothing."""
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = ["gfx1100", "gfx1010"],
            ROCR_VISIBLE_DEVICES = "0,0",
            HIP_VISIBLE_DEVICES = "1",
        )
        is False
    )


def test_two_distinct_rocr_ordinals_still_expose_both_devices(stack, monkeypatch):
    """The control."""
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = ["gfx1100", "gfx1010"],
            ROCR_VISIBLE_DEVICES = "1,0",
            HIP_VISIBLE_DEVICES = "1",
        )
        is True
    )


def test_the_rocr_subset_is_the_prefix_of_devices_the_mask_surfaces(stack, monkeypatch):
    """The buckets behind the two cases above, since a route verdict cannot show WHICH list the HIP
    layer was handed."""
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    _devices = ["gfx1100", "gfx1010", "gfx1201"]
    for _mask, _expected in (
        ("0,0", ["gfx1100"]),
        ("1,1,2", ["gfx1010"]),
        ("0,7,1", ["gfx1100"]),
        ("2,0", ["gfx1201", "gfx1100"]),
        ("7", _devices),
    ):
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _mask)
        assert stack._rocr_visible_subset(list(_devices))[0] == _expected, _mask


def test_a_declared_arch_resolves_its_mask_against_the_unmasked_list(stack, monkeypatch):
    """With no KFD topology (WSL) the declared-arch path falls back to rocminfo, and rocminfo is
    the one probe ROCR_VISIBLE_DEVICES renumbers."""
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = ["gfx1100", "gfx1010"],
            masked = ["gfx1010"],
            declared = "gfx1100",
            ROCR_VISIBLE_DEVICES = "1",
        )
        is True
    )


def test_the_same_declared_host_with_a_mask_past_its_last_device_still_declines(stack, monkeypatch):
    """The control: the unmasked list is asked for so the ordinals can be judged, not so that every
    ordinal passes."""
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", "gfx1100")
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = ["gfx1100", "gfx1010"],
            masked = [],
            ROCR_VISIBLE_DEVICES = "7",
        )
        is False
    )


def test_the_installer_stops_the_rocr_prefix_at_a_repeat_too():
    """The shell half of the duplicate rule: _amd_mask_survivors printed the repeated device a
    second time, so the HIP layer read a two-entry list and selected a phantom."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1010"],
            devices = ["gfx1100", "gfx1010"],
            ROCR_VISIBLE_DEVICES = "0,0",
            HIP_VISIBLE_DEVICES = "1",
        )
        is False
    )


def test_the_installer_still_exposes_both_devices_for_distinct_ordinals():
    """The control, matching the Python one: ROCR_VISIBLE_DEVICES=1,0 renumbers rather than
    truncates, so HIP ordinal 1 is the routable card and the request still deposes CUDA."""
    assert (
        _route_shell_masked(
            ["gfx1100", "gfx1010"],
            devices = ["gfx1100", "gfx1010"],
            ROCR_VISIBLE_DEVICES = "1,0",
            HIP_VISIBLE_DEVICES = "1",
        )
        is True
    )


def test_the_installer_does_not_apply_the_repeat_rule_to_the_hip_layer():
    """The rule is ROCr's, cited from its own filter; clr documents no such termination for
    HIP_VISIBLE_DEVICES, so the third argument scopes it to the layer it was read from."""
    assert (
        _route_shell_masked(
            ["gfx1010", "gfx1100"],
            devices = ["gfx1010", "gfx1100"],
            HIP_VISIBLE_DEVICES = "1,1",
        )
        is True
    )


def test_an_unresolvable_mask_on_an_inferred_host_is_not_a_detection_miss(stack, monkeypatch):
    """Nothing enumerates a device here, so the product name is the only arch on offer and
    HIP_VISIBLE_DEVICES=1 indexes past it."""
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = [],
            inferred = "gfx1100",
            HIP_VISIBLE_DEVICES = "1",
        )
        is False
    )


def test_the_same_inferred_host_without_a_mask_is_still_a_route(stack, monkeypatch):
    """The control that keeps the feature: a runtime-less but inferable AMD card is deliberately
    served per-arch wheels, so the rule must be about the mask and not about the host having no
    runtime."""
    assert _viable_masked(stack, monkeypatch, devices = [], inferred = "gfx1100") is True


def test_the_same_inferred_host_selecting_its_only_card_is_still_a_route(stack, monkeypatch):
    """The other control, and the one that pins WHICH ordinals decline: 0 names the single card the
    name inferred, so it resolves and the route stands."""
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = [],
            inferred = "gfx1100",
            HIP_VISIBLE_DEVICES = "0",
        )
        is True
    )


def test_the_installer_already_declines_that_mask():
    """The installer twin of the case above, and the reason it is a Python-only fix: with one arch
    and no per-device list, _amd_request_has_a_wheel_route resolves the mask against the single
    row and takes `[ -n "$_arwr_sel" ] || return 1`, so it has always failed closed where the
    Python half fell through to its inventory fallback."""
    assert _route_shell_masked(["gfx1100"], devices = [], HIP_VISIBLE_DEVICES = "1") is False


def test_the_installer_still_routes_that_host_unmasked():
    """The control: the same one-arch host with no mask is exactly what the request exists to
    serve, so the decline above must belong to the mask and not to the host shape."""
    assert _route_shell_masked(["gfx1100"], devices = []) is True


_UUID_MASK = "0,GPU-DEADBEEFDEADBEEF"


@pytest.mark.parametrize(
    "devices, viable",
    [(["gfx1010", "gfx1100"], False), (["gfx1100", "gfx1100"], True)],
    ids = ["unlike-adapters", "one-architecture"],
)
def test_a_uuid_in_the_rocr_mask_is_a_rejection_only_where_it_is_ambiguous(
    stack, monkeypatch, devices, viable
):
    """ROCR_VISIBLE_DEVICES may MIX ordinals and UUIDs, and a UUID names a device this
    installer cannot place -- but where every ordinal gives the same arch there is nothing
    ambiguous about it, which is why the branch asks about unlike adapters."""
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = devices,
            ROCR_VISIBLE_DEVICES = _UUID_MASK,
            HIP_VISIBLE_DEVICES = "0",
        )
        is viable
    )


def test_the_named_arch_the_message_offers_resolves_the_ambiguity(stack, monkeypatch):
    """The second control: the message names UNSLOTH_ROCM_GFX_ARCH as the way through, so setting
    it must restore the route rather than leave the host declined for good."""
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = ["gfx1010", "gfx1100"],
            declared = "gfx1100",
            ROCR_VISIBLE_DEVICES = _UUID_MASK,
            HIP_VISIBLE_DEVICES = "0",
        )
        is True
    )


def test_the_ambiguity_flag_does_not_survive_the_next_host(stack, monkeypatch):
    """It is a module global read by a separate function, so a stale True would decline a later,
    unrelated host."""
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = ["gfx1010", "gfx1100"],
            ROCR_VISIBLE_DEVICES = _UUID_MASK,
            HIP_VISIBLE_DEVICES = "0",
        )
        is False
    )
    assert _viable_masked(stack, monkeypatch, devices = ["gfx1100"]) is True


def test_an_integrated_gpu_enumerated_first_does_not_strand_the_discrete_card():
    """A Ryzen APU enumerates ahead of the discrete Radeon beside it, and the wheel family is
    picked for ONE arch."""
    assert _route_shell_masked(["gfx90c", "gfx1200"]) is True
    assert _route_target_masked(["gfx90c", "gfx1200"]) == "gfx1200"


def test_a_routable_integrated_gpu_still_yields_to_the_discrete_one():
    """The same divergence with a quieter symptom, and the reason the fix cannot just be "decline
    when the first arch has no route": gfx1036 IS routable through gfx103X-all, so the shell
    said yes and installed for the Raphael iGPU while a gfx1100 sat beside it."""
    assert _route_target_masked(["gfx1036", "gfx1100"]) == "gfx1100"


def test_a_discrete_card_enumerated_first_is_left_alone():
    """The control: same two cards, opposite order."""
    assert _route_target_masked(["gfx1200", "gfx90c"]) == "gfx1200"


def test_an_integrated_gpu_alone_is_still_declined():
    """The control that keeps the fix honest."""
    assert _route_shell_masked(["gfx90c"]) is False
    assert _route_target_masked(["gfx90c"]) == ""


def test_an_unroutable_sibling_does_not_rescue_an_unroutable_integrated_pick():
    """gfx1010 is in neither the generic wheel nor any per-arch index, so deposing gfx90c for it
    buys nothing and the request still declines."""
    assert _route_shell_masked(["gfx90c", "gfx1010"]) is False


def test_gfx906_is_not_a_candidate_for_the_swap():
    """gfx906's only route is the rocm6.3 legacy tag, which opens solely when it is the sole arch
    on the host, so promoting it here would install a rocm7.x wheel whose BLAS has no gfx906
    kernels and strand BOTH cards."""
    assert _route_target_masked(["gfx1036", "gfx906"]) == "gfx1036"


def test_a_pinned_host_keeps_the_device_the_user_named():
    """A set mask is the user naming a device, and _visible_devices_pinned exempts the Python rule
    for exactly that reason."""
    assert _route_target_masked(["gfx90c", "gfx1200"], HIP_VISIBLE_DEVICES = "0") == ""
    assert _route_shell_masked(["gfx90c", "gfx1200"], HIP_VISIBLE_DEVICES = "0") is False


def _versionless_reroute_block() -> str:
    """install.sh's versionless per-arch reroute, lifted by text."""
    install_sh = Path(__file__).resolve().parents[3] / "install.sh"
    lines = install_sh.read_text(encoding = "utf-8").splitlines()
    start = lines.index("_amd_no_rocm_version_reroute=false")
    end = next(i for i in range(start, len(lines)) if lines[i] == "esac")
    return "\n".join(lines[start : end + 1])


def _reroute_family(physical: "list[str]", **mask: str) -> str:
    """The wheel family the versionless reroute leaves standing, or "" when it clears it."""
    emit = 'printf "%s\\n" ' + " ".join(repr(a) for a in physical)
    script = "\n".join(
        [
            "_torch_index_pinned=false",
            "SKIP_TORCH=false",
            'TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"',
            f"_probe_amd_gfx_arch() {{ {emit}; }}",
            f"_kfd_gfx_targets() {{ {emit}; }}",
            "_infer_linux_amd_gfx_arch() { :; }",
            "_amd_gpu_present_via_pci() { return 0; }",
            "_has_amd_rocm_gpu() { return 0; }",
            # No NVIDIA card, so _nvidia_gpu_wins_over_amd answers no whatever the request
            # did and the arms below differ only in the request and the mask.
            "_has_usable_nvidia_gpu() { return 1; }",
            _shell_function("_rocm_torch_explicitly_requested"),
            *_wheel_route_defs(rocminfo = _fake_rocminfo(physical)),
            _shell_function("_nvidia_gpu_wins_over_amd"),
            _shell_function("_amd_probe_arches"),
            _shell_function("_amd_agreed_index_family"),
            _shell_function("_amd_sole_index_arch"),
            _shell_function("_hsa_spoofed_physical_gfx"),
            "_detect_rocm_version_tag() { :; }",
            _versionless_reroute_block(),
            'printf "%s" "${_amd_probed_family:-}"',
        ]
    )
    return _bash(script, env = _route_env(mask))


# gfx1033 shares gfx103X-all with the card beside it, which is what makes the family
# disqualification reach a routable sibling in the first place.
_DECK_PLUS_RDNA2 = ["gfx1033", "gfx1030"]


def test_the_versionless_reroute_keeps_the_family_the_request_selected():
    """The reroute clears the family on PRESENCE of gfx1033, which is right only while the card
    that will run is unknown."""
    assert (
        _reroute_family(_DECK_PLUS_RDNA2, UNSLOTH_FORCE_ROCM_TORCH = "1", HIP_VISIBLE_DEVICES = "1")
        == "gfx103X-all"
    )


def test_the_same_request_selecting_the_deck_still_loses_the_family():
    """The control that keeps the narrowing honest: the request cannot buy ROCm wheels for the arch
    measured to compute wrong answers (studio/ROCM_RDNA2_APU.md)."""
    assert (
        _reroute_family(_DECK_PLUS_RDNA2, UNSLOTH_FORCE_ROCM_TORCH = "1", HIP_VISIBLE_DEVICES = "0")
        == ""
    )


def test_the_presence_rule_still_holds_for_a_host_that_did_not_ask():
    """And the control for every other host: with no request nothing resolved a target, so presence
    disqualifies the shared family as before (#7776)."""
    assert _reroute_family(_DECK_PLUS_RDNA2, HIP_VISIBLE_DEVICES = "1") == ""


@pytest.mark.parametrize(
    "arch, declared, viable",
    [
        ("gfx1100", "gfx1100:sramecc-:xnack-", True),
        ("gfx1100", "gfx1100", True),
        ("gfx1010", "gfx1010:xnack-", False),
    ],
    ids = ["suffixed-routable", "bare-routable", "suffixed-unroutable"],
)
def test_a_feature_suffix_changes_the_spelling_and_not_the_answer(
    stack, monkeypatch, arch, declared, viable
):
    """rocminfo prints gcnArchName with its feature flags, and that is the spelling users copy
    into UNSLOTH_ROCM_GFX_ARCH. _amd_arch_index_url keys on the bare arch, so the suffixed
    form answered None and declined a route the plain spelling gets on the same silicon. The
    third row is the control: normalising the SPELLING must not normalise the ANSWER."""
    assert (
        _viable_masked(
            stack,
            monkeypatch,
            devices = [arch],
            inferred = declared,
            rocm = (0, 0),
            declared = declared,
        )
        is viable
    )


# ── The feature suffix rocminfo prints, which users copy verbatim ───────────────────────


@pytest.mark.parametrize(
    "declared,family",
    [
        ("gfx1100", "gfx110X-all"),
        ("gfx1100:sramecc-:xnack-", "gfx110X-all"),
        ("gfx1151", "gfx1151"),
        ("gfx1151:xnack-", "gfx1151"),
    ],
)
def test_a_feature_suffix_routes_to_the_same_index_as_the_bare_arch(
    stack, monkeypatch, declared, family
):
    """rocminfo prints gcnArchName as gfx1100:sramecc-:xnack- and that is what people paste into
    UNSLOTH_ROCM_GFX_ARCH, but both arch tables are keyed on the bare arch."""
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    url = stack._amd_arch_index_url(declared)
    assert url is not None and family in url


def test_a_suffixed_miscomputing_arch_is_still_refused(stack, monkeypatch):
    """The control that keeps the normalisation from opening a door."""
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    assert stack._amd_arch_index_url("gfx1033") is None
    assert stack._amd_arch_index_url("gfx1033:xnack-") is None


def test_an_unknown_arch_is_still_unrouted(stack, monkeypatch):
    """The other control: normalising must not invent a route for an arch no index carries."""
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    assert stack._amd_arch_index_url("gfx1010") is None
    assert stack._amd_arch_index_url("gfx1010:xnack-") is None


# ── The Van Gogh gate, and what may exempt a host from it ──────────────────────────────


def _bad_arch_verdict(
    *,
    declared: str,
    probed: str,
    request: str = "1",
) -> str:
    """Which index get_torch_index_url's miscomputing-arch gate leaves, for this host."""
    lines = (
        (Path(__file__).resolve().parents[3] / "install.sh")
        .read_text(encoding = "utf-8")
        .splitlines()
    )
    end = next(i for i, line in enumerate(lines) if "end of the miscomputing-arch gate" in line)
    # The gate opens at its own header sentence; anchoring on the first
    # `_amd_gfx_bad_arch=false` walking back would land inside the exemption branch.
    start = next(
        i
        for i in range(end, 0, -1)
        if "Archs measured to compute INCORRECTLY under ROCm" in lines[i]
    )
    # Wrapped in a function, because the lifted block ends in `return` and bash refuses
    # that at top level -- unwrapped, execution ran on past the verdict.
    gate = "_gate() {\n" + "\n".join(lines[start:end]) + "\n}"
    script = "\n".join(
        [
            _shell_function("_rocm_torch_explicitly_requested"),
            *_wheel_route_defs(arch_family = False),
            "_amd_arch_index_family_for_gfx() { case $1 in gfx1030|gfx1033) echo gfx103X-all ;;"
            " gfx1100) echo gfx110X-all ;; *) return 1 ;; esac; }",
            "_has_usable_nvidia_gpu() { return 0; }",
            "_amd_gpu_present_via_pci() { return 0; }",
            f"_probe_amd_gfx_arch() {{ echo {probed}; }}",
            f"_kfd_gfx_targets() {{ echo {probed}; }}",
            f"_infer_linux_amd_gfx_arch() {{ echo {declared or probed}; }}",
            "_detect_rocm_version_tag() { echo rocm7.1; }",
            "_ensure_rocm_probe_env() { :; }",
            '_base="https://download.pytorch.org/whl"',
            f'_amd_gfx_probe="{probed}"',
            "_amd_request_has_a_wheel_route || true",
            gate,
            # The gate either PRINTS an index and returns, or falls through printing nothing.
            "_verdict=$(_gate || true)",
            'printf "%s\\n" "${_verdict:-kept}"',
        ]
    )
    env = _clean_env({"UNSLOTH_FORCE_ROCM_TORCH": request})
    if declared:
        env["UNSLOTH_ROCM_GFX_ARCH"] = declared
    out = subprocess.run(["bash", "-c", script], capture_output = True, text = True, env = env)
    return (out.stdout.strip().splitlines() or [""])[-1]


def test_a_declared_arch_cannot_exempt_a_physical_gfx1033():
    """A Steam Deck beside an NVIDIA card, with a stale or copied UNSLOTH_ROCM_GFX_ARCH. The
    declared gfx1030 is published as the request's target, and the gate used to read any non-
    gfx1033 target as "the selected card is fine"."""
    assert _bad_arch_verdict(declared = "gfx1030", probed = "gfx1033").endswith("/cpu")


def test_a_probed_target_still_exempts_a_routable_sibling():
    """The control, and the case the target was published for: the probe itself resolved a gfx1100,
    so the gfx1033 in the inventory is a card the runtime did not select."""
    assert _bad_arch_verdict(declared = "", probed = "gfx1100") == "kept"


def test_the_gate_still_fires_with_no_request_at_all():
    """The other control: presence remains the rule for every host that has not asked."""
    assert _bad_arch_verdict(declared = "", probed = "gfx1033", request = "0").endswith("/cpu")


def _reroute_family_for_target(
    *,
    inventory: list,
    target: str,
    source: str,
    bad_arch: str = "false",
) -> str:
    """The wheel family the versionless reroute would use, for this host."""
    lines = (
        (Path(__file__).resolve().parents[3] / "install.sh")
        .read_text(encoding = "utf-8")
        .splitlines()
    )
    # From the bad-arch verdict, not from the target block alone: the gate CLEARS the family
    # before the target may set it, and a lift that starts below that clearing tests a
    # sequence install.sh never runs.
    start = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == 'if [ "$_amd_reroute_bad_arch" = true ]; then'
    )
    # To the close of the OUTER if, by indentation: the block nests one, so the first
    # bare "fi" is the inner one and cutting there left an unterminated script.
    indent = len(lines[start]) - len(lines[start].lstrip())
    # Two sibling blocks at this indent: the gate's clearing, then the target's assignment.
    closes = [
        i
        for i in range(start + 1, len(lines))
        if lines[i].strip() == "fi" and len(lines[i]) - len(lines[i].lstrip()) == indent
    ]
    block = "\n".join(lines[start : closes[1] + 1])
    script = "\n".join(
        [
            _shell_function("_amd_arch_index_family_for_gfx"),
            _shell_function("_amd_probe_arches"),
            _shell_function("_amd_agreed_index_family"),
            _shell_function("_amd_sole_index_arch"),
            f'_amd_probe_out="{chr(10).join(inventory)}"',
            '_amd_probed_family=$(_amd_agreed_index_family "$_amd_probe_out") || _amd_probed_family=""',
            '_amd_probed_gfx_first=$(_amd_sole_index_arch "$_amd_probe_out") || _amd_probed_gfx_first=""',
            f'_amd_reroute_target="{target}"',
            f'_AMD_REQUEST_TARGET_SOURCE="{source}"',
            f"_amd_reroute_bad_arch={bad_arch}",
            block,
            'printf "%s\\n" "${_amd_probed_family:-none}"',
        ]
    )
    return _bash(script)


@pytest.mark.parametrize(
    "inventory, target, source, family",
    [
        (["gfx1033", "gfx1100"], "gfx1100", "probe", "gfx110X-all"),
        (["gfx1033", "gfx1100"], "gfx1100", "declared", "none"),
        (["gfx1100", "gfx1101"], "gfx1100", "probe", "gfx110X-all"),
    ],
    ids = ["cross-family-probe", "cross-family-declared", "single-family"],
)
def test_the_versionless_reroute_family_comes_from_the_selected_card(
    inventory, target, source, family
):
    """_amd_agreed_index_family needs EVERY physical AMD GPU to share a family, so a
    cross-family pair answered empty, the reroute never fired and the downgrade guard
    restored CUDA -- the request ignored on exactly the host that resolved a routable
    target. Probe-resolved only: a declared arch cannot name the card the runtime selected."""
    assert _reroute_family_for_target(inventory = inventory, target = target, source = source) == family


def test_a_disqualified_family_is_not_put_back_by_the_target():
    """The control that matters most here."""
    assert (
        _reroute_family_for_target(
            inventory = ["gfx1033"], target = "gfx1033", source = "probe", bad_arch = "true"
        )
        == "none"
    )


def _unreadable_version_host(stack, monkeypatch, gfx: str) -> None:
    """A mixed NVIDIA + AMD host whose ROCm version cannot be read, carrying healthy CUDA torch:
    the build the stand-down leaves in place when nothing replaces it."""
    _mixed_host(stack, monkeypatch, archs = [gfx], kfd = [gfx], rocm = None)


@pytest.mark.parametrize("gfx", ["gfx1102"])
def test_an_unreadable_version_is_judged_as_the_installer_judges_it(stack, monkeypatch, gfx):
    """The two halves must not read the same unreadable version differently."""
    _unreadable_version_host(stack, monkeypatch, gfx)
    assert stack._forced_rocm_route_is_viable() is False


# RDNA 4 takes the AMD per-arch route like Strix; those wheels bundle their own ROCm (#11935).
@pytest.mark.parametrize("gfx", ["gfx1151", "gfx1103", "gfx1200", "gfx1201"])
def test_an_unreadable_version_still_serves_the_arches_that_do_install(stack, monkeypatch, gfx):
    """The narrowing control."""
    _unreadable_version_host(stack, monkeypatch, gfx)
    assert stack._forced_rocm_route_is_viable() is True


@pytest.mark.parametrize("gfx", ["gfx1102", "gfx1200", "gfx1201"])
def test_a_readable_version_still_routes_the_same_three_arches(stack, monkeypatch, gfx):
    """The other control: the change is scoped to an unreadable version."""
    _unreadable_version_host(stack, monkeypatch, gfx)
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda: (6, 4))
    assert stack._forced_rocm_route_is_viable() is True


def _declared_on_physical(stack, monkeypatch, *, declared: str, physical: list) -> bool:
    """_forced_rocm_route_is_viable for a DECLARED arch on a known physical inventory."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", declared)
    for _mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_mask, raising = False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: True)
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: True)
    monkeypatch.setattr(stack, "_physical_amd_gfx_archs", lambda: list(physical))
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: list(physical))
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda: (6, 4))
    monkeypatch.setattr(stack, "_explicit_rocm_torch_index_url", lambda: None)
    return stack._forced_rocm_route_is_viable()


def test_a_declared_arch_cannot_exempt_a_physical_gfx1033_in_python(stack, monkeypatch):
    """A declaration is a build target the user typed, not a statement about the silicon."""
    assert (
        _declared_on_physical(
            stack, monkeypatch, declared = "gfx1030", physical = ["gfx1033", "gfx1100"]
        )
        is False
    )


def test_a_probe_resolved_target_still_exempts_the_same_host(stack, monkeypatch):
    """The control that keeps the narrowing honest, and the case install.sh spells out by name:
    with NO declaration the probe resolves the target, the discrete-GPU preference selects the
    gfx1100 over the integrated gfx1033, and that IS a statement about the silicon."""
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    for _mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_mask, raising = False)
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: True)
    monkeypatch.setattr(stack, "_is_wsl", lambda: False)
    monkeypatch.setattr(stack, "_linux_amd_display_device_present", lambda: True)
    monkeypatch.setattr(stack, "_infer_linux_amd_gfx_arch", lambda: None)
    monkeypatch.setattr(stack, "_physical_amd_gfx_archs", lambda: ["gfx1033", "gfx1100"])
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda: ["gfx1033", "gfx1100"])
    monkeypatch.setattr(
        stack, "_detect_amd_gfx_codes", lambda dedup = True, **k: ["gfx1033", "gfx1100"]
    )
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda: (6, 4))
    monkeypatch.setattr(stack, "_explicit_rocm_torch_index_url", lambda: None)
    # Proves the control is about the PROBE path: the target must be the discrete card, not
    # the integrated one the guard above rejects a declaration for.
    assert stack._runtime_gfx_target(None)[0] == "gfx1100"
    assert stack._forced_rocm_route_is_viable() is True


def test_a_declaration_still_routes_where_no_bad_arch_is_present(stack, monkeypatch):
    """The second control: the guard keys on a MISCOMPUTING arch in the machine, not on the
    declaration disagreeing with it."""
    assert (
        _declared_on_physical(
            stack, monkeypatch, declared = "gfx1030", physical = ["gfx1030", "gfx1100"]
        )
        is True
    )


def test_the_documented_strix_declaration_still_routes(stack, monkeypatch):
    """The third control, and the reason the guard cannot simply distrust declarations: the
    documented Strix Halo workaround declares gfx1100 on a physical gfx1151, which install.sh
    calls out by name."""
    assert (
        _declared_on_physical(stack, monkeypatch, declared = "gfx1100", physical = ["gfx1151"]) is True
    )


# ─────────────────────────────────────────────────────────────────────────────
# The request must not authorise a swap it withdraws the moment the swap happens.


def _viability_across_the_install(stack, monkeypatch, gfx, installed):
    """_forced_rocm_route_is_viable() on one unchanging host, asked twice: once with the CUDA wheel
    still installed, once with the ROCm wheel the request asked for in its place."""
    monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    monkeypatch.delenv("UNSLOTH_ROCM_GFX_ARCH", raising = False)
    for _mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(_mask, raising = False)
    monkeypatch.setattr(stack.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: True)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: True)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", lambda *a, **k: [gfx])
    monkeypatch.setattr(stack, "_detect_amd_gfx_codes", lambda *a, **k: [gfx])
    monkeypatch.setattr(stack, "_physical_amd_gfx_archs", lambda *a, **k: [gfx])
    monkeypatch.setattr(stack, "_infer_linux_amd_gfx_arch", lambda *a, **k: None)
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda *a, **k: None)
    for _pin in (
        "_explicit_rocm_torch_index_url",
        "_explicit_cuda_torch_index_url",
        "_explicit_cpu_torch_index_url",
        "_explicit_torch_index_url",
    ):
        monkeypatch.setattr(stack, _pin, lambda: None)
    monkeypatch.setattr(
        stack, "_probe_torch_runtime", lambda *a, **k: (True, True, installed, "", "")
    )
    return stack._forced_rocm_route_is_viable()


def test_the_route_is_still_viable_once_the_wheel_it_authorised_is_installed(stack, monkeypatch):
    """Measured on gfx950 with an unreadable ROCm version, the one shape where
    _rocm_torch_family_needs_repair is the only arm carrying the chain."""
    before = _viability_across_the_install(stack, monkeypatch, "gfx950", "2.9.0+cu128")
    after = _viability_across_the_install(stack, monkeypatch, "gfx950", "2.11.0+rocm7.1")
    assert before is True, "the request never authorised the swap, so there is nothing to keep"
    assert after is True, "the request withdrew the route its own install created"


def test_an_installed_rocm_build_on_an_unroutable_card_is_still_not_a_route(stack, monkeypatch):
    """The control on the rule above: "ROCm is already installed" must not become a blanket yes."""
    assert _viability_across_the_install(stack, monkeypatch, "gfx1010", "2.11.0+rocm7.1") is False


def test_the_cuda_repair_still_stands_down_after_the_swap(stack, monkeypatch):
    """The consequence, at the caller."""
    calls = []
    monkeypatch.setattr(stack, "pip_install", lambda *a, **k: calls.append(a[:1]))
    monkeypatch.setattr(stack, "pip_install_try", lambda *a, **k: calls.append(a[:1]))
    monkeypatch.delenv("UNSLOTH_TORCH_BACKEND", raising = False)
    monkeypatch.delenv("UNSLOTH_ROCM_TORCH_INSTALLED", raising = False)
    _viability_across_the_install(stack, monkeypatch, "gfx950", "2.11.0+rocm7.1")
    stack._ensure_cuda_torch()
    assert not calls, f"the CUDA repair overwrote the requested ROCm build: {calls}"


# ─────────────────────────────────────────────────────────────────────────────
# A feature suffix must not change which pins the install gets.


def _inferred_install_args(stack, monkeypatch, arch):
    """The pip arguments _ensure_rocm_torch uses on its inferred-arch branch for ``arch``."""
    calls = []
    monkeypatch.setattr(stack, "pip_install", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(stack, "pip_install_try", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", arch)
    monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    # ROCm wheels are x86_64 only, so _ensure_rocm_torch returns on its arch gate anywhere
    # else and this helper answers [] on the ubuntu-24.04-arm runner. The case is about the
    # arch SPELLING, not the host's architecture, so the host is stated like the rest.
    monkeypatch.setattr(stack.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(stack, "IS_WINDOWS", False)
    monkeypatch.setattr(stack, "IS_MACOS", False)
    monkeypatch.setattr(stack, "_has_usable_nvidia_gpu", lambda: False)
    monkeypatch.setattr(stack, "_has_rocm_gpu", lambda: False)
    monkeypatch.setattr(
        stack, "_probe_torch_runtime", lambda *a, **k: (True, True, "2.9.0", "", "")
    )
    monkeypatch.setattr(stack, "_detect_rocm_version", lambda *a, **k: None)
    for _pin in ("_explicit_rocm_torch_index_url", "_explicit_torch_index_url"):
        monkeypatch.setattr(stack, _pin, lambda: None)
    stack._ensure_rocm_torch()
    return [a for a in (calls[0] if calls else ()) if isinstance(a, str) and a.startswith("torch")]


@pytest.mark.parametrize(
    "bare,suffixed",
    [("gfx1151", "gfx1151:xnack-"), ("gfx1200", "gfx1200:sramecc+:xnack-")],
)
def test_a_suffixed_arch_installs_the_same_pins_as_the_bare_one(stack, monkeypatch, bare, suffixed):
    """gcnArchName is what rocminfo prints and what users copy into UNSLOTH_ROCM_GFX_ARCH.
    Stripping the suffix in _amd_arch_index_url is what opens the inferred-arch branch for that
    spelling; the package table two lines down was still keyed on the raw string, so it missed
    and installed unpinned torch/torchvision/torchaudio -- losing the ABI bound the table exists
    to hold, on a host the bare spelling pins correctly."""
    assert _inferred_install_args(stack, monkeypatch, suffixed) == _inferred_install_args(
        stack, monkeypatch, bare
    )


def test_the_suffixed_arch_reaches_that_branch_at_all(stack, monkeypatch):
    """The control: the assertion above is only worth anything if the suffixed spelling actually
    installs something."""
    assert _inferred_install_args(stack, monkeypatch, "gfx1151:xnack-")


@pytest.mark.parametrize("arch", ["gfx90a:sramecc+:xnack-", "gfx908:xnack-"])
def test_an_arch_outside_the_pin_table_is_still_bounded(stack, monkeypatch, arch):
    """The suffix strip opens this branch for archs the pin table does not name, and the fallback
    was three bare package names: those hosts reached an arch-index install with no companion
    bound at all, where every other arch-index install carries one. gfx103X / gfx110X joined the
    pin table with unslothai/unsloth#11814, so CDNA is what is left outside it."""
    assert _inferred_install_args(stack, monkeypatch, arch) == list(
        stack._ROCM_ARCH_INDEX_TORCH_PKG_SPEC
    )


def test_an_arch_inside_the_pin_table_keeps_its_own_pins(stack, monkeypatch):
    """The control: the fallback must not displace the table's tighter, per-arch bound."""
    assert _inferred_install_args(stack, monkeypatch, "gfx1151") == list(
        stack._WINDOWS_ROCM_TORCH_PKG_SPECS["gfx1151"]
    )


def _mask_probe_count(stack, monkeypatch, requested):
    """How many device probes _runtime_gfx_target runs for a DECLARED arch under a set mask."""
    probes = {"n": 0}

    def _counted(*a, **k):
        probes["n"] += 1
        return []

    monkeypatch.setenv("UNSLOTH_ROCM_GFX_ARCH", "gfx1100")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    if requested:
        monkeypatch.setenv("UNSLOTH_FORCE_ROCM_TORCH", "1")
    else:
        monkeypatch.delenv("UNSLOTH_FORCE_ROCM_TORCH", raising = False)
    monkeypatch.setattr(stack, "_kfd_gfx_targets", _counted)
    monkeypatch.setattr(stack, "_detect_amd_gfx_codes", _counted)
    target = stack._runtime_gfx_target(None)[0]
    assert target == "gfx1100", target
    return probes["n"]


def test_a_declared_arch_under_a_mask_costs_no_probe_unless_asked(stack, monkeypatch):
    """Those probes only fill the two mask-provenance flags, which only the forced route reads,
    and each one can spend its 15s timeout. Running them for every declared-arch host put that
    cost into an ordinary `studio update` that had asked for nothing."""
    assert _mask_probe_count(stack, monkeypatch, requested = False) == 0


def test_the_host_that_did_ask_still_resolves_its_mask(stack, monkeypatch):
    """The control: the flags are load-bearing for the route that reads them, so the request must
    still pay for them."""
    assert _mask_probe_count(stack, monkeypatch, requested = True) > 0


# ─────────────────────────────────────────────────────────────────────────────
# The two halves must read one mask string the same way.


def test_an_escaped_mask_is_not_decoded_into_an_ordinal(stack):
    """`awk -v vis=...` ESCAPE-PROCESSES its operand, so a literal four-character \061 arrived
    inside awk as the ordinal 1 and the shell half selected the second card."""
    script = "\n".join(
        [_shell_function("_amd_mask_survivors"), 'printf "%s" "$(_amd_mask_survivors "$1" "$2")"']
    )
    out = subprocess.run(
        ["bash", "-c", script, "_", "gfx1010\ngfx1100", "\\061"],
        capture_output = True,
        text = True,
        env = _clean_env(),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout == "", f"an escaped ordinal resolved a device: {out.stdout!r}"
    # And the twin's answer on the same string, which is what it has to agree with.
    os.environ["HIP_VISIBLE_DEVICES"] = "\\061"
    try:
        assert stack._hip_layer_mask_names_a_device(2) is False
    finally:
        os.environ.pop("HIP_VISIBLE_DEVICES", None)


def test_a_plain_ordinal_still_selects_its_device(stack):
    """The control: the rule above must reject the ESCAPE, not every mask."""
    script = "\n".join(
        [_shell_function("_amd_mask_survivors"), 'printf "%s" "$(_amd_mask_survivors "$1" "$2")"']
    )
    out = subprocess.run(
        ["bash", "-c", script, "_", "gfx1010\ngfx1100", "1"],
        capture_output = True,
        text = True,
        env = _clean_env(),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout == "gfx1100", out.stdout


# ─────────────────────────────────────────────────────────────────────────────
# The miscomputing-arch gate must not inherit a target from an earlier call.


def test_the_index_selector_does_not_inherit_a_previous_target(stack):
    """_AMD_REQUEST_TARGET_GFX is cleared by the helper that sets it, and that helper runs only
    when NVIDIA is detected AND the request is set."""
    selector = _shell_function("get_torch_index_url")
    assert '_AMD_REQUEST_TARGET_GFX=""' in selector.splitlines()[1] or any(
        line.strip() == '_AMD_REQUEST_TARGET_GFX=""' for line in selector.splitlines()[:12]
    ), "the selector must clear the published target on entry"
    assert any(
        line.strip() == '_AMD_REQUEST_TARGET_SOURCE=""' for line in selector.splitlines()[:12]
    ), "the selector must clear the target SOURCE on entry too"


@pytest.mark.parametrize(
    "miscomputing, viable",
    [(True, False), (False, True)],
    ids = ["every-arch-computes-wrong", "control-healthy-host"],
)
def test_a_host_gate_decides_when_no_probe_can_name_a_device(
    stack, monkeypatch, miscomputing, viable
):
    """_miscomputing_arch_host is the only thing declining when no target resolves -- a Steam
    Deck with no rocminfo, where the narrower target refusal never runs and the fallback would
    otherwise approve the swap off the inventory."""
    _mixed_host(
        stack, monkeypatch, request = False, archs = ["gfx1033"], devices = [], miscomputing = miscomputing
    )
    assert stack._forced_rocm_route_is_viable() is viable


@pytest.mark.parametrize(
    "request_set, sees_amd",
    [("", False), ("UNSLOTH_FORCE_ROCM_TORCH=1", True)],
    ids = ["default-nvidia-wins", "request-opens-the-probe"],
)
def test_the_shell_probe_hides_the_amd_card_until_it_is_asked(request_set, sees_amd):
    """install.sh's half of the gate _has_rocm_gpu carries in Python: the AMD probe returns
    early on a usable NVIDIA GPU unless this run asked for ROCm."""
    script = "\n".join(
        [
            _shell_function("_rocm_torch_explicitly_requested"),
            _shell_function("_ensure_rocm_probe_env"),
            _shell_function("_has_amd_rocm_gpu"),
            "_has_usable_nvidia_gpu() { return 0; }",
            'rocminfo() { echo "  Name:                    gfx1100"; }',
            f"{request_set} _has_amd_rocm_gpu && echo yes || echo no",
        ]
    )
    assert _bash(script, env = _clean_env()) == ("yes" if sees_amd else "no")
