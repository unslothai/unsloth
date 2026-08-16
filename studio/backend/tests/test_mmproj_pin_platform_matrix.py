# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""[OS] x [accelerator] matrix for the automatic vision-projector CPU pin.

The pin (``--no-mmproj-offload``) is emitted when the projector's VRAM is the
only thing keeping the model off the GPU. Nothing in that decision reads
``sys.platform`` directly, which is exactly why it needs an OS matrix: the claim
under test is that the *host-specific* code around it -- the Metal paravirtual
guard, the Windows full-offload tuning block, the Vulkan CPU-crash replay, the
ROCm arch gate -- neither suppresses the pin where it belongs nor conjures one
where it does not.

Matrix: [Linux, Windows, WSL, macOS] x [NVIDIA, AMD/ROCm, Vulkan, Metal, CPU-only].

Skipped cells, and why:
  macOS x NVIDIA / AMD-ROCm / Vulkan -- Studio ships a Metal build for darwin;
      there is no CUDA or ROCm runtime for macOS and no Vulkan prebuilt, so
      these are not reachable hosts rather than untested ones.
  Linux / Windows / WSL x Metal -- Metal is an Apple-only API.

Two host shapes carry no GPU into the planner and are the ones most at risk of a
mis-fire:

  * **Metal.** ``_get_gpu_memory`` has no branch that can enumerate an Apple GPU
    (nvidia-smi / amd-smi / torch.cuda, none of which answer on darwin), so
    ``gpus`` is empty. Apple memory is unified as well, so pinning the encoder
    frees nothing and only moves it to a slower path.
  * **CPU-only.** Nothing to pin to.

Both of those hosts emit ``--fit on``, because ``--fit`` starts on and is only
lowered to ``off`` by a planner that placed the model. ``--fit on`` is therefore
NOT evidence that the model did not fit; #8875 shipped a bug reading it that
way. ``test_fit_on_is_not_evidence_of_a_partial_placement`` pins that down.

Everything is simulated: this host has one NVIDIA GPU and no AMD, Apple or
Vulkan hardware. ``sys.platform`` is pinned inside the backend module (and in
``utils.hardware``, which reads its own copy) rather than globally, so
``os.pathsep`` and the rest of the interpreter stay native -- a test that fakes
``sys.platform = "darwin"`` while asserting on a native ``os.pathsep`` has
previously broken CI, and no assertion here reads a path-list separator.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Reuses the placement harness (and its module stubs). Import before anything
# from core.inference.
from test_llama_cpp_placement import _backend, _launch  # noqa: E402,F401

from core.inference import llama_cpp  # noqa: E402
from utils.hardware import hardware as hw  # noqa: E402

_PROJECTOR_BYTES = 900 * 1024 * 1024
_PIN = "--no-mmproj-offload"

# platform.system() / sys.platform per simulated host.
_OS_CELLS = {
    "linux": ("linux", "Linux"),
    "wsl": ("linux", "Linux"),
    "windows": ("win32", "Windows"),
    "macos": ("darwin", "Darwin"),
}
OS_KEYS = list(_OS_CELLS)
ACCEL_KEYS = ["nvidia", "rocm", "vulkan", "metal", "cpu"]

# (os, accel) pairs that are not reachable hosts. Value is the reason, echoed
# into the skip so the matrix report says why rather than just how many.
_IMPOSSIBLE = {
    ("macos", "nvidia"): "no CUDA runtime for macOS",
    ("macos", "rocm"): "no ROCm runtime for macOS",
    ("macos", "vulkan"): "Studio ships a Metal prebuilt for darwin, not Vulkan",
    ("linux", "metal"): "Metal is Apple-only",
    ("wsl", "metal"): "Metal is Apple-only",
    ("windows", "metal"): "Metal is Apple-only",
}

MATRIX = [
    pytest.param(
        os_key,
        accel,
        marks = (
            pytest.mark.skip(reason = _IMPOSSIBLE[(os_key, accel)])
            if (os_key, accel) in _IMPOSSIBLE
            else ()
        ),
        id = f"{os_key}-{accel}",
    )
    for os_key in OS_KEYS
    for accel in ACCEL_KEYS
]

# Hosts where the planner sees real devices, so the pin genuinely applies.
GPU_MATRIX = [p for p in MATRIX if p.values[1] in ("nvidia", "rocm", "vulkan")]
# Hosts with no enumerated device: the pin must never fire.
NO_GPU_MATRIX = [p for p in MATRIX if p.values[1] in ("metal", "cpu")]


def _apply_os(monkeypatch, os_key: str) -> None:
    """Pin the simulated host OS.

    Patched on the modules that read it, never on the interpreter: ``os.pathsep``,
    ``os.sep`` and ``pathlib`` must stay native or the test asserts against a
    filesystem that does not exist here.
    """
    platform_name, system_name = _OS_CELLS[os_key]
    monkeypatch.setattr(llama_cpp.sys, "platform", platform_name)
    monkeypatch.setattr(hw.sys, "platform", platform_name)
    monkeypatch.setattr(hw.platform, "system", lambda: system_name)
    if os_key == "wsl":
        # WSL reports as Linux and reaches the GPU through /dev/dxg; the ROCm
        # prebuilts there load the system ROCm libs first. Pinned so the cell is
        # a real WSL host rather than a relabelled Linux one.
        monkeypatch.setattr(llama_cpp, "_wsl_system_rocm_lib_dirs", lambda: ["/opt/rocm/lib"])
        monkeypatch.setattr(
            llama_cpp.os.path, "exists", lambda p: p == "/dev/dxg" or Path(p).exists()
        )


def _accel_backend(
    monkeypatch,
    tmp_path: Path,
    accel: str,
    *,
    memory,
    vision: bool = True,
    mmproj_bytes: int = _PROJECTOR_BYTES,
    model_bytes: int = 1024,
):
    """A backend wired for one accelerator cell.

    ``memory`` is the probe's answer for the GPU-bearing cells and is forced
    empty for Metal and CPU-only, which enumerate nothing.
    """
    if accel in ("metal", "cpu"):
        memory = []
    backend, gguf = _backend(tmp_path, vulkan = (accel == "vulkan"), memory = memory)
    backend._get_gguf_size_bytes = lambda _path: model_bytes
    # The real estimator returns several GB against the stub GGUF, which swamps
    # every other term and makes nothing fit on any card -- a pin test built on
    # it would pass whatever the policy did.
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * 1024 * 1024
    # ROCm reports through the same torch path as CUDA; the arch gate and the
    # Windows free-VRAM cap key off this flag.
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_host_torch_is_rocm", staticmethod(lambda: accel == "rocm")
    )
    # Physical Apple Silicon, not a VM: the paravirtual pin must not be what
    # answers for the Metal cells.
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: False)
    if vision:
        mmproj = tmp_path / "model-mmproj.gguf"
        mmproj.write_bytes(b"\x00" * 16)
        backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
        backend._mmproj_vram_bytes = lambda _path: mmproj_bytes
        backend._mmproj_matches_model_family = lambda *a, **k: True
    return backend, gguf


# A card the model alone nearly fills: the projector's 900 MiB is what tips it
# over, so pinning buys full residency. Sized once so every cell asks the same
# question of the planner and only the host shape varies.
_TIGHT_FREE_MIB = 6_000
_TIGHT_MODEL_BYTES = 4_500 * 1024 * 1024
_TIGHT_MEMORY = [(0, _TIGHT_FREE_MIB, _TIGHT_FREE_MIB + 2_000)]
_ROOMY_MEMORY = [(0, 40_000, 48_000)]


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------


@pytest.mark.parametrize("os_key, accel", GPU_MATRIX)
def test_pin_fires_where_it_buys_full_residency(monkeypatch, tmp_path, os_key, accel):
    """Every host with a real device: the pin fires, exactly once, and keeps vision."""
    _apply_os(monkeypatch, os_key)
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, accel, memory = _TIGHT_MEMORY, model_bytes = _TIGHT_MODEL_BYTES
    )

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert cmd.count(_PIN) == 1
    # Pinned, not disabled: the projector still loads, on the CPU.
    assert "--mmproj" in cmd
    assert backend.is_vision is True
    assert backend.vision_on_cpu is True
    assert backend.vision_disabled_by_user is False


@pytest.mark.parametrize("os_key, accel", GPU_MATRIX)
def test_no_pin_where_the_projector_already_fits(monkeypatch, tmp_path, os_key, accel):
    """Tier 1 on every GPU host: room to spare keeps the projector on the device."""
    _apply_os(monkeypatch, os_key)
    backend, gguf = _accel_backend(monkeypatch, tmp_path, accel, memory = _ROOMY_MEMORY)

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert _PIN not in cmd
    assert "--mmproj" in cmd
    assert backend.vision_on_cpu is False


@pytest.mark.parametrize("os_key, accel", NO_GPU_MATRIX)
def test_no_pin_without_an_enumerated_device(monkeypatch, tmp_path, os_key, accel):
    """Metal and CPU-only: nothing to pin to, so no flag on any OS.

    Metal is the load-bearing half. Its memory is unified, so moving the encoder
    off the device frees no VRAM and only costs encode speed; the pin would be
    pure loss. The model is deliberately sized so that a policy keying off
    "does it fit" rather than off an enumerated device WOULD pin here.
    """
    _apply_os(monkeypatch, os_key)
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, accel, memory = [], model_bytes = _TIGHT_MODEL_BYTES
    )

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert _PIN not in cmd
    # Vision is untouched: the projector loads and runs, on the CPU backend
    # llama.cpp picks for itself.
    assert "--mmproj" in cmd
    assert backend.vision_on_cpu is False


@pytest.mark.parametrize("os_key, accel", NO_GPU_MATRIX)
def test_fit_on_is_not_evidence_of_a_partial_placement(monkeypatch, tmp_path, os_key, accel):
    """``--fit on`` is the *starting* value, not a verdict.

    A CPU-only box and a Metal Mac both reach the launch with ``--fit on`` and
    neither placed a single layer, so anything that reads that flag as "the
    model did not fit" (as #8875 did) fires on hosts where the projector was
    never on a GPU to begin with. Assert both halves in one place: the flag IS
    on, and the pin is still absent.
    """
    _apply_os(monkeypatch, os_key)
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, accel, memory = [], model_bytes = _TIGHT_MODEL_BYTES
    )

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert cmd.count("--fit") == 1
    assert cmd[cmd.index("--fit") + 1] == "on"
    assert _PIN not in cmd


@pytest.mark.parametrize("os_key, accel", MATRIX)
def test_the_pin_is_never_emitted_twice(monkeypatch, tmp_path, os_key, accel):
    """Four emitters exist; no argv may carry the flag more than once."""
    _apply_os(monkeypatch, os_key)
    for model_bytes in (1024, _TIGHT_MODEL_BYTES, 60_000 * 1024 * 1024):
        backend, gguf = _accel_backend(
            monkeypatch,
            tmp_path,
            accel,
            memory = _TIGHT_MEMORY,
            model_bytes = model_bytes,
        )
        cmd = _launch(backend, gguf, is_vision = True)["cmd"]
        assert cmd.count(_PIN) <= 1, (os_key, accel, model_bytes)


@pytest.mark.parametrize("os_key, accel", MATRIX)
def test_disable_vision_wins_on_every_host(monkeypatch, tmp_path, os_key, accel):
    _apply_os(monkeypatch, os_key)
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, accel, memory = _TIGHT_MEMORY, model_bytes = _TIGHT_MODEL_BYTES
    )

    cmd = _launch(backend, gguf, is_vision = True, disable_vision = True)["cmd"]

    assert "--mmproj" not in cmd
    assert _PIN not in cmd
    assert backend.is_vision is False
    assert backend.vision_on_cpu is False
    assert backend.vision_disabled_by_user is True


# Every spelling llama-server accepts for the two user opt-outs. arg.cpp folds
# `_` to `-` before matching (`std::replace`), so the underscore forms are the
# same flags to the child and must be the same flags to Studio -- a detector
# that only string-compares the dashed form lets the automatic pin fire behind a
# user who named the placement, and Studio's flag is appended after the extras,
# so last-wins hands it the argument.
_USER_PLACEMENT_SPELLINGS = [
    "--mmproj-offload",
    "--mmproj_offload",
    "--no-mmproj-offload",
    "--no_mmproj_offload",
]
_USER_DISABLE_SPELLINGS = ["--no-mmproj", "--no_mmproj", "--no-mmproj-auto", "--no_mmproj_auto"]


@pytest.mark.parametrize("spelling", _USER_PLACEMENT_SPELLINGS + _USER_DISABLE_SPELLINGS)
@pytest.mark.parametrize("os_key, accel", GPU_MATRIX)
def test_an_explicit_user_flag_owns_the_placement(monkeypatch, tmp_path, os_key, accel, spelling):
    """A user who named the placement wins, on every host.

    ``--mmproj-offload`` / ``--no-mmproj-offload`` name where the projector runs;
    ``--no-mmproj`` / ``--no-mmproj-auto`` say not to load one at all. In both
    cases Studio must not add a second, automatic answer.

    The card is deliberately one the automatic policy WOULD pin, so a suppression
    that never engaged is visible rather than indistinguishable from agreement.
    """
    _apply_os(monkeypatch, os_key)
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, accel, memory = _TIGHT_MEMORY, model_bytes = _TIGHT_MODEL_BYTES
    )

    cmd = _launch(backend, gguf, is_vision = True, extra_args = [spelling])["cmd"]

    # Studio adds nothing on top of the user's choice, so the only projector
    # placement token in the argv is the one the user wrote.
    placement = [a for a in cmd if a.replace("_", "-") in ("--mmproj-offload", _PIN)]
    if spelling in _USER_PLACEMENT_SPELLINGS:
        assert placement == [spelling], f"{os_key}/{accel}: Studio raced the user for {spelling}"
        # The echo must agree with the argv the child actually receives.
        assert backend.vision_on_cpu is (spelling.replace("_", "-") == _PIN)
    else:
        assert placement == []
        # No projector was resolved, so none is placed and none is charged.
        assert "--mmproj" not in cmd
        assert backend.vision_on_cpu is False


# --------------------------------------------------------------------------
# Host-specific interactions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("os_key", OS_KEYS)
def test_the_projector_path_reaches_the_argv_verbatim(monkeypatch, tmp_path, os_key):
    """No separator rewriting anywhere on the projector path, on any host.

    ``_resolve_launch_mmproj_path`` is pure ``pathlib`` and nothing between it
    and the argv consults ``os.sep`` or ``os.pathsep``, so a Windows path with
    backslashes must arrive at llama-server exactly as resolved. Asserted as
    identity rather than by faking a separator: pinning ``sys.platform`` does not
    move ``os.pathsep``, and a test that assumes it does is how this suite broke
    CI before.
    """
    _apply_os(monkeypatch, os_key)
    backend, gguf = _accel_backend(monkeypatch, tmp_path, "cpu", memory = [])
    resolved = backend._resolve_launch_mmproj_path()

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert cmd[cmd.index("--mmproj") + 1] == resolved
    assert cmd.count("--mmproj") == 1


def test_windows_full_offload_tuning_engages_after_the_pin(monkeypatch, tmp_path):
    """The pin turns a partial placement into a full one, which is what the
    Windows ``--ctx-checkpoints 0`` / ``--cache-ram 0`` block keys off.

    That is the correct reading -- after the pin every layer really is resident,
    so the WDDM host-RAM checkpoint overhead #5692 exists to remove really does
    apply -- but it is a behaviour change on Windows that no other host sees, so
    pin it explicitly rather than leave it to be discovered.
    """
    _apply_os(monkeypatch, "windows")
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, "nvidia", memory = _TIGHT_MEMORY, model_bytes = _TIGHT_MODEL_BYTES
    )
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_ctx_checkpoints": True,
        "supports_cache_ram": True,
        "supports_no_mmproj_offload": True,
    }

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert cmd.count(_PIN) == 1
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("--ctx-checkpoints") + 1] == "0"
    assert cmd[cmd.index("--cache-ram") + 1] == "0"


def test_linux_full_offload_leaves_ctx_checkpoints_alone(monkeypatch, tmp_path):
    """The same pinned, fully-resident launch on Linux: the Windows-only block
    must not follow the pin onto other hosts."""
    _apply_os(monkeypatch, "linux")
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, "nvidia", memory = _TIGHT_MEMORY, model_bytes = _TIGHT_MODEL_BYTES
    )
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_ctx_checkpoints": True,
        "supports_cache_ram": True,
        "supports_no_mmproj_offload": True,
    }

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert cmd.count(_PIN) == 1
    assert "--ctx-checkpoints" not in cmd
    assert "--cache-ram" not in cmd


def test_wsl_is_a_linux_host_for_the_pin(monkeypatch, tmp_path):
    """WSL reports ``sys.platform == "linux"``. It must take the Linux path --
    the Windows tuning block must not engage -- while still pinning."""
    _apply_os(monkeypatch, "wsl")
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, "nvidia", memory = _TIGHT_MEMORY, model_bytes = _TIGHT_MODEL_BYTES
    )
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_ctx_checkpoints": True,
        "supports_cache_ram": True,
        "supports_no_mmproj_offload": True,
    }

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert llama_cpp.sys.platform == "linux"
    assert cmd.count(_PIN) == 1
    assert "--ctx-checkpoints" not in cmd


@pytest.mark.parametrize("spelling", ["--no-mmproj-offload", "--no_mmproj_offload"])
@pytest.mark.parametrize("os_key, accel", GPU_MATRIX)
def test_a_hand_pinned_projector_is_charged_no_vram_either(
    monkeypatch, tmp_path, os_key, accel, spelling
):
    """A user who pins by hand must get the same placement Auto's pin gets.

    The projector is on the CPU in both cases and takes 0 GPU bytes in both, so
    the planner must reach the same verdict: every layer resident,
    ``-ngl -1 --fit off``. Charging the bytes anyway made the explicit request
    strictly worse than the automatic one on the identical argv -- the model
    fits, and Studio declined to say so.
    """
    _apply_os(monkeypatch, os_key)
    (tmp_path / "auto").mkdir()
    (tmp_path / "hand").mkdir()
    auto, gguf_a = _accel_backend(
        monkeypatch,
        tmp_path / "auto",
        accel,
        memory = _TIGHT_MEMORY,
        model_bytes = _TIGHT_MODEL_BYTES,
    )
    hand, gguf_h = _accel_backend(
        monkeypatch,
        tmp_path / "hand",
        accel,
        memory = _TIGHT_MEMORY,
        model_bytes = _TIGHT_MODEL_BYTES,
    )

    auto_cmd = _launch(auto, gguf_a, is_vision = True)["cmd"]
    hand_cmd = _launch(hand, gguf_h, is_vision = True, extra_args = [spelling])["cmd"]

    # The premise: Auto pins this exact model on this exact card.
    assert auto_cmd.count(_PIN) == 1
    assert auto_cmd[auto_cmd.index("--fit") + 1] == "off"

    assert hand_cmd[hand_cmd.index("--fit") + 1] == "off"
    assert hand_cmd[hand_cmd.index("-ngl") + 1] == "-1"
    # And still exactly one placement token, the user's own.
    assert [a for a in hand_cmd if a.replace("_", "-") in ("--mmproj-offload", _PIN)] == [spelling]
    assert hand.vision_on_cpu is True


def test_a_virtualised_metal_device_still_outranks_a_user_gpu_request(monkeypatch, tmp_path):
    """The one case where Studio must race the user and win.

    A paravirtual Apple GPU returns corrupt output, so ``--mmproj-offload`` in
    the advanced arguments is a request Studio cannot honour: it appends its own
    pin after the extras and llama.cpp takes the last one. The
    hand-pin suppression above must not have opened a hole here.
    """
    _apply_os(monkeypatch, "macos")
    backend, gguf = _accel_backend(monkeypatch, tmp_path, "metal", memory = [])
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: True)
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_no_mmproj_offload": True,
    }

    cmd = _launch(backend, gguf, is_vision = True, extra_args = ["--mmproj-offload"])["cmd"]

    assert cmd.count(_PIN) == 1
    assert cmd.index(_PIN) > cmd.index("--mmproj-offload")
    assert backend.vision_on_cpu is True


@pytest.mark.parametrize("os_key, accel", GPU_MATRIX)
def test_a_build_without_the_flag_keeps_the_projector_on_the_gpu(
    monkeypatch, tmp_path, os_key, accel
):
    """Graceful degradation, not an unknown argument.

    ``--no-mmproj-offload`` is b5178, so a conclusive probe that does not list it
    means a build genuinely too old. Emitting it anyway would make llama-server
    exit on an unrecognised flag, which is a worse outcome than the spill the pin
    was avoiding.
    """
    _apply_os(monkeypatch, os_key)
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, accel, memory = _TIGHT_MEMORY, model_bytes = _TIGHT_MODEL_BYTES
    )
    # A probe that ANSWERED and did not list the flag.
    backend.probe_server_capabilities = lambda _binary = None: {"supports_metrics": False}

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert _PIN not in cmd
    assert "--mmproj" in cmd
    assert backend.is_vision is True
    assert backend.vision_on_cpu is False


@pytest.mark.parametrize("os_key, accel", GPU_MATRIX)
def test_an_unanswered_probe_still_pins(monkeypatch, tmp_path, os_key, accel):
    """A --help probe that failed is not evidence the flag is missing.

    One malformed inherited LLAMA_ARG_* makes the probe exit non-zero and every
    capability read as absent for the process. Declining the pin there would
    hand the user a spilled model over a flag every launchable build has.
    """
    _apply_os(monkeypatch, os_key)
    backend, gguf = _accel_backend(
        monkeypatch, tmp_path, accel, memory = _TIGHT_MEMORY, model_bytes = _TIGHT_MODEL_BYTES
    )
    backend.probe_server_capabilities = lambda _binary = None: {"mtp_probe_inconclusive": True}

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert cmd.count(_PIN) == 1
    assert backend.vision_on_cpu is True


def test_paravirtual_metal_still_pins(monkeypatch, tmp_path):
    """The pre-existing paravirtual-Metal pin is a different mechanism with a
    different reason (the device's output is corrupt, not its VRAM budget) and
    fires with no GPU enumerated at all. The automatic policy must not have
    displaced it."""
    _apply_os(monkeypatch, "macos")
    backend, gguf = _accel_backend(monkeypatch, tmp_path, "metal", memory = [])
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: True)
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_no_mmproj_offload": True,
    }

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert cmd.count(_PIN) == 1
    assert "--mmproj" in cmd
    assert backend.vision_on_cpu is True


def test_physical_metal_does_not_inherit_the_paravirtual_pin(monkeypatch, tmp_path):
    """The sibling of the test above: same host, real hardware, no pin. Without
    this the paravirtual test alone cannot show the guard discriminates."""
    _apply_os(monkeypatch, "macos")
    backend, gguf = _accel_backend(monkeypatch, tmp_path, "metal", memory = [])
    backend.probe_server_capabilities = lambda _binary = None: {
        "supports_no_mmproj_offload": True,
    }

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert _PIN not in cmd
    assert backend.vision_on_cpu is False


def test_metal_enumerates_no_gpu_through_the_probe(monkeypatch):
    """The premise the Metal cells rest on, asserted rather than assumed.

    ``_get_gpu_memory`` has three branches -- nvidia-smi, amd-smi, torch.cuda --
    and none of them can answer on darwin. If a future branch learns to
    enumerate an Apple GPU, the Metal rows above stop testing what they claim
    and this fails first.
    """
    monkeypatch.setattr(llama_cpp.sys, "platform", "darwin")
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _b = None: False)
    )

    def _no_such_tool(*a, **k):
        raise FileNotFoundError(a[0][0] if a and a[0] else "smi")

    monkeypatch.setattr(subprocess, "run", _no_such_tool)
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available = lambda: False, device_count = lambda: 0)
    fake_torch.version = types.SimpleNamespace(hip = None, cuda = None)
    fake_torch.__version__ = "2.6.0"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert llama_cpp.LlamaCppBackend._get_gpu_memory("/fake/llama-server") == []


# --------------------------------------------------------------------------
# Multi-GPU: the subset ranking the pin decision walks
# --------------------------------------------------------------------------


_MULTI_GPU_POOLS = {
    # Asymmetric on purpose: the ranking is "fewest GPUs first, largest first",
    # so a pool whose cards are equal cannot show the order was honoured.
    "2gpu": [(0, 6_000, 8_000), (1, 3_000, 8_000)],
    "3gpu": [(0, 5_000, 8_000), (1, 3_000, 8_000), (2, 2_000, 8_000)],
    "4gpu": [(0, 4_000, 8_000), (1, 3_000, 8_000), (2, 2_500, 8_000), (3, 1_500, 8_000)],
}


def _sweep(
    monkeypatch,
    tmp_path,
    accel,
    pool,
    *,
    os_key = "linux",
    step_mib = 200,
):
    """Launch across a range of model sizes on one pool, one row per size.

    Sizing the "just barely does not fit" model by hand needs the planner's exact
    arithmetic (per-card reserve, the 1.4x projector surcharge, the per-device
    pipeline overhead) and silently stops testing anything the moment one of
    those constants moves. Sweeping instead lets the assertions be invariants
    over the whole curve, and the "some size pins" check below is what stops a
    sweep that fits nowhere from passing vacuously.
    """
    pooled_mib = sum(free for _idx, free, _total in pool) or 4_000
    rows = []
    for mib in range(step_mib, pooled_mib + 3_000, step_mib):
        _apply_os(monkeypatch, os_key)
        backend, gguf = _accel_backend(
            monkeypatch, tmp_path, accel, memory = pool, model_bytes = mib * 1024 * 1024
        )
        cmd = _launch(backend, gguf, is_vision = True)["cmd"]
        rows.append((mib, cmd, backend))
    return rows


@pytest.mark.parametrize("pool_key", list(_MULTI_GPU_POOLS))
@pytest.mark.parametrize("accel", ["nvidia", "rocm", "vulkan"])
def test_multi_gpu_pin_only_ever_buys_full_residency(monkeypatch, tmp_path, pool_key, accel):
    """The policy's whole justification, asserted over the size curve.

    ``_mm_subsets`` walks 1..N cards; the pin fires when SOME subset holds the
    model without the projector and NONE holds it with. The claim that makes the
    trade worth taking is that pinning then buys FULL residency -- so every
    pinned launch must also be a ``--fit off`` / ``-ngl -1`` launch. A pin on a
    launch that still spills layers would be a pure loss: a 3.6x image encode
    bought with nothing.
    """
    pool = _MULTI_GPU_POOLS[pool_key]
    rows = _sweep(monkeypatch, tmp_path, accel, pool)
    pinned = [(mib, cmd, backend) for mib, cmd, backend in rows if _PIN in cmd]

    # Anti-vacuity: this pool must actually exercise the pin somewhere.
    assert pinned, f"{accel}/{pool_key}: no model size on this pool pins at all"

    for mib, cmd, backend in pinned:
        where = f"{accel}/{pool_key}@{mib}MiB"
        assert cmd.count(_PIN) == 1, where
        assert cmd[cmd.index("--fit") + 1] == "off", where
        assert cmd[cmd.index("-ngl") + 1] == "-1", where
        assert "--mmproj" in cmd, where
        assert backend.is_vision is True, where
        assert backend.vision_on_cpu is True, where

    # And the sweep must contain unpinned sizes on both sides, or "pins only
    # where it buys residency" is being read off a curve that always pins.
    assert any(_PIN not in cmd for _mib, cmd, _b in rows)


@pytest.mark.parametrize("pool_key", list(_MULTI_GPU_POOLS))
def test_the_pin_band_reaches_past_what_the_largest_card_alone_holds(
    monkeypatch, tmp_path, pool_key
):
    """The subset walk is load-bearing, not decoration.

    ``_mm_subsets`` offers the fit 1..N cards, fewest first. If it only ever
    offered the largest card, the pin would still fire and still buy full
    residency -- every shape assertion above would pass -- but it would fire on a
    strictly smaller band of models, and every multi-card user whose model needs
    two or more cards would silently lose the pin. So compare the band against
    the same sweep run on a pool holding only that largest card: the multi-card
    band must reach strictly higher.
    """
    pool = _MULTI_GPU_POOLS[pool_key]
    largest = max(pool, key = lambda g: g[1])

    pooled = [mib for mib, cmd, _b in _sweep(monkeypatch, tmp_path, "nvidia", pool) if _PIN in cmd]
    alone = [
        mib for mib, cmd, _b in _sweep(monkeypatch, tmp_path, "nvidia", [largest]) if _PIN in cmd
    ]

    assert pooled and alone
    assert max(pooled) > max(alone), (
        f"{pool_key}: the pin band tops out at {max(pooled)} MiB, the same as the "
        f"largest card alone ({max(alone)} MiB) -- the extra cards bought nothing, "
        "so the subset ranking is not being walked"
    )


@pytest.mark.parametrize("pool_key", list(_MULTI_GPU_POOLS))
def test_multi_gpu_pin_window_is_contiguous_and_bounded(monkeypatch, tmp_path, pool_key):
    """The pin band sits between "fits with the projector" and "fits neither way".

    Below the band the projector is affordable, above it the model is too large
    for the pool and the policy declines. A band that ran to the top of the sweep
    would mean the pin fires on models that stay mostly CPU-resident, which is
    the mis-fire the policy is written to avoid.
    """
    rows = _sweep(monkeypatch, tmp_path, "nvidia", _MULTI_GPU_POOLS[pool_key])
    flags = [(mib, _PIN in cmd) for mib, cmd, _b in rows]
    pinned_sizes = [mib for mib, p in flags if p]

    assert pinned_sizes
    # Smallest models: affordable with the projector, so no pin.
    assert flags[0][1] is False
    # Largest models: too big either way, so no pin.
    assert flags[-1][1] is False
    # One contiguous band, not a scattering (which would mean the subset ranking
    # answers non-monotonically in model size).
    band = [mib for mib, p in flags if p]
    lo, hi = min(band), max(band)
    assert all(p for mib, p in flags if lo <= mib <= hi), f"{pool_key}: pin band has holes"


@pytest.mark.parametrize("pool_key", list(_MULTI_GPU_POOLS))
def test_multi_gpu_no_pin_when_one_card_already_holds_everything(monkeypatch, tmp_path, pool_key):
    """A pool whose largest card fits model + projector outright.

    The subset walk starts at one GPU, so the ``fits on GPU`` answer must be
    yes at n=1 and no pin may fire, however many other cards are present.
    """
    _apply_os(monkeypatch, "linux")
    pool = [(idx, 40_000, 48_000) for idx, _f, _t in _MULTI_GPU_POOLS[pool_key]]
    backend, gguf = _accel_backend(monkeypatch, tmp_path, "nvidia", memory = pool)

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert _PIN not in cmd
    assert backend.vision_on_cpu is False


@pytest.mark.parametrize("pool_key", list(_MULTI_GPU_POOLS))
def test_multi_gpu_no_pin_when_the_pool_cannot_hold_it_either_way(monkeypatch, tmp_path, pool_key):
    """Too large for the whole pool with or without the projector.

    Freeing the projector's bytes on a stack that is mostly CPU-resident buys a
    couple of percent per token and costs a silent 3.6x on every image, so the
    policy deliberately declines. This is the failure mode a naive "does it fit"
    test would miss, since the pin looks harmless there.
    """
    _apply_os(monkeypatch, "linux")
    backend, gguf = _accel_backend(
        monkeypatch,
        tmp_path,
        "nvidia",
        memory = _MULTI_GPU_POOLS[pool_key],
        model_bytes = 200_000 * 1024 * 1024,
    )

    cmd = _launch(backend, gguf, is_vision = True)["cmd"]

    assert _PIN not in cmd
    assert "--mmproj" in cmd
    assert backend.vision_on_cpu is False


# --------------------------------------------------------------------------
# The four emitters
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv, holds_vram",
    [
        (["--mmproj", "/mm.gguf"], True),
        (["--mmproj", "/mm.gguf", "--mmproj-offload"], True),
        (["--mmproj", "/mm.gguf", _PIN], False),
        # Last-wins, both directions.
        (["--mmproj", "/mm.gguf", _PIN, "--mmproj-offload"], True),
        (["--mmproj", "/mm.gguf", "--mmproj-offload", _PIN], False),
        # arg.cpp folds the underscore, so this is the same pin to the child.
        (["--mmproj", "/mm.gguf", "--no_mmproj_offload"], False),
    ],
)
def test_a_pinned_projector_holds_no_vram_in_the_residency_bookkeeping(argv, holds_vram):
    """The other half of "a CPU-pinned projector is charged 0 VRAM".

    ``_cmd_has_gpu_companion`` is what tells the residency bookkeeping and the
    coexistence guard whether a launch keeps VRAM for a companion. It has to read
    the pin the same way llama.cpp does -- last occurrence wins, underscores
    folded -- or a pinned projector keeps being counted against a card it is not
    on.
    """
    assert llama_cpp.LlamaCppBackend._cmd_has_gpu_companion(argv) is holds_vram


def test_the_cpu_replay_does_not_re_add_a_pin_the_argv_already_carries():
    """Emitter 2 (the Vulkan GPU-init-crash CPU replay) meets emitter 4 (the
    automatic pin).

    The replay is rebuilt FROM the launched argv, and
    ``_strip_cpu_fallback_main_placement`` strips layer/fit/device/MoE flags but
    not the projector placement, so an argv that already carried the pin used to
    come back with two. Reachable only since the automatic pin: the other
    pre-launch emitter is the paravirtual-Metal one, and this replay refuses any
    non-Vulkan runtime.
    """
    cmd = [
        "/bin/llama-server",
        "-m",
        "/m.gguf",
        "-ngl",
        "-1",
        "--fit",
        "off",
        "--mmproj",
        "/mm.gguf",
        _PIN,
    ]

    replay = llama_cpp.LlamaCppBackend._cpu_isolated_replay(
        cmd, {}, {"supports_no_mmproj_offload": True}
    )

    assert replay is not None
    assert replay.count(_PIN) == 1
    # Still pinned, and still last, so nothing in the extras can outrank it.
    assert replay[-1] == _PIN


def test_the_cpu_replay_overrides_a_stale_gpu_placement_token():
    """Same path, opposite token: a ``--mmproj-offload`` left in the argv must
    not survive into a launch that has no devices at all."""
    cmd = ["/bin/llama-server", "-m", "/m.gguf", "--mmproj", "/mm.gguf", "--mmproj-offload"]

    replay = llama_cpp.LlamaCppBackend._cpu_isolated_replay(
        cmd, {}, {"supports_no_mmproj_offload": True}
    )

    assert replay is not None
    assert "--mmproj-offload" not in replay
    assert replay.count(_PIN) == 1


def test_a_cpu_fallback_reports_its_projector_as_cpu_resident():
    """``vision_on_cpu`` is what tells the UI images will be slow.

    The CPU replay pins every projector it carries (it refuses outright when it
    cannot), so a vision load that survives one is CPU-encoding whether or not
    the fit probe had already decided to pin -- which it will not have, on any
    load that fit comfortably before the crash.
    """
    backend = llama_cpp.LlamaCppBackend()
    backend._vision_on_cpu = False

    backend._apply_cpu_fallback_state(
        llama_cpp.GgufLoadIntent(gguf_path = "/m.gguf", model_identifier = "test"),
        is_vision = True,
        mmproj_has_audio = False,
    )

    assert backend.is_vision is True
    assert backend.vision_on_cpu is True


def test_a_text_only_cpu_fallback_claims_no_projector():
    backend = llama_cpp.LlamaCppBackend()

    backend._apply_cpu_fallback_state(
        llama_cpp.GgufLoadIntent(gguf_path = "/m.gguf", model_identifier = "test"),
        is_vision = False,
        mmproj_has_audio = False,
    )

    assert backend.vision_on_cpu is False


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("--mmproj-offload", True),
        ("--mmproj_offload", True),
        ("--no-mmproj-offload", True),
        ("--no_mmproj_offload", True),
        ("--mmproj-offload=true", True),
        # Neither of these places the projector, so neither may suppress the pin.
        ("--mmproj", False),
        ("--mmproj-url", False),
        ("--no-mmproj", False),
        ("--top-k", False),
    ],
)
def test_the_user_placement_detector_folds_underscores(spelling, expected):
    """arg.cpp normalises `_` to `-` before matching, so Studio must too.

    Every other flag reader in this module goes through ``_flag_name``; a raw
    string compare here let ``--mmproj_offload`` through, and because Studio's
    pin is appended after the extras, last-wins then silently moved the
    projector to the CPU against an explicit request to keep it on the GPU.
    """
    assert llama_cpp._extra_args_set_mmproj_offload([spelling]) is expected


@pytest.mark.parametrize("pool_key", list(_MULTI_GPU_POOLS))
def test_tensor_parallel_pin_stays_single_valued(monkeypatch, tmp_path, pool_key):
    """Tensor parallelism reserves per-device buffers the pin's arithmetic does
    not model. Whatever it decides, it must decide it once."""
    _apply_os(monkeypatch, "linux")
    pool = _MULTI_GPU_POOLS[pool_key]
    pooled_mib = sum(free for _idx, free, _total in pool)
    backend, gguf = _accel_backend(
        monkeypatch,
        tmp_path,
        "nvidia",
        memory = pool,
        model_bytes = int((pooled_mib * 0.9 - 400) * 1024 * 1024),
    )

    cmd = _launch(backend, gguf, is_vision = True, tensor_parallel = True)["cmd"]

    assert cmd.count(_PIN) <= 1
    assert (backend.vision_on_cpu is True) == (_PIN in cmd)
