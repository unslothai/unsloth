# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The extra-arguments pass-through across every platform and accelerator.

Unsloth emits a different command on each of these: CUDA, ROCm and Vulkan take
different offload flags, Metal takes none of them, and Windows spells the binary
and the paths differently. The claim this suite has to defend is the same on all of
them, and it is a claim about what does NOT change:

  with the box empty, the command is byte-identical to the one Unsloth emitted
  before this feature existed.

The matrix is the Cartesian product of the platforms Unsloth ships on and the
accelerators it detects, driven through the real ``load_model`` with the command
captured at the Popen boundary.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# The placement harness (module stubs, a fake GGUF, a fake GPU probe, the captured
# Popen) already exists; by path, because the tests dir is not a package.
_PLACEMENT_PATH = Path(__file__).resolve().parent / "test_llama_cpp_placement.py"
_spec = importlib.util.spec_from_file_location("_placement_harness_platforms", _PLACEMENT_PATH)
_placement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placement)
_backend = _placement._backend
_launch = _placement._launch

# (label, sys.platform, os.name, a WSL-shaped release string or None)
PLATFORMS = [
    ("linux", "linux", "posix", None),
    ("wsl2", "linux", "posix", "5.15.153.1-microsoft-standard-WSL2"),
    ("windows", "win32", "nt", None),
    ("macos", "darwin", "posix", None),
]

# (label, vulkan, memory) -- memory [] is the CPU-only / no-device answer the probe
# gives when there is nothing to place on.
ACCELERATORS = [
    ("nvidia-single", False, [(0, 20_000, 24_000)]),
    ("nvidia-multi", False, [(0, 20_000, 24_000), (1, 20_000, 24_000)]),
    ("amd-vulkan", True, [(0, 12_000, 16_000)]),
    ("cpu-only", False, []),
]

MATRIX = [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS]


def _apply_platform(monkeypatch, platform) -> None:
    """Move the seams the launch path actually branches on.

    Only ``sys.platform`` and the WSL markers: patching ``os.name`` as well swaps
    pathlib's flavour mid-run, so the harness's own tmp GGUF stops resolving and
    every assertion below becomes a lie about a file that was never opened. The
    authoritative Windows and macOS signal is the per-OS CI matrix on real runners;
    this is the branch coverage that can be had on one host.
    """
    _label, sys_platform, _os_name, wsl_release = platform
    import platform as _platform
    import sys as _sys

    monkeypatch.setattr(_sys, "platform", sys_platform, raising = False)
    if wsl_release:
        monkeypatch.setattr(_platform, "release", lambda: wsl_release, raising = False)
        monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    else:
        monkeypatch.delenv("WSL_DISTRO_NAME", raising = False)


def _stable(cmd: list[str]) -> list[str]:
    """The command with the two per-launch values masked (port, model path)."""
    masked = list(cmd)
    for index, token in enumerate(masked):
        if index and masked[index - 1] == "--port":
            masked[index] = "<port>"
        elif index and masked[index - 1] in {"-m", "--model"}:
            masked[index] = "<model>"
    return masked


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_an_empty_box_changes_nothing_anywhere(tmp_path, monkeypatch, platform, accelerator):
    # The acceptance bar, on every combination. None (inherit) and [] (explicitly
    # none) are both "the user did not put anything in the box".
    _apply_platform(monkeypatch, platform)
    _label, vulkan, memory = accelerator

    backend, gguf = _backend(tmp_path, vulkan = vulkan, memory = memory)
    baseline = _stable(_launch(backend, gguf)["cmd"])

    backend, gguf = _backend(tmp_path, vulkan = vulkan, memory = memory)
    assert _stable(_launch(backend, gguf, extra_args = None)["cmd"]) == baseline

    backend, gguf = _backend(tmp_path, vulkan = vulkan, memory = memory)
    assert _stable(_launch(backend, gguf, extra_args = [])["cmd"]) == baseline


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_an_extra_arg_lands_last_and_changes_nothing_before_it(
    tmp_path, monkeypatch, platform, accelerator
):
    # Appended, never interleaved: llama.cpp's last-wins parsing is the whole
    # mechanism, and a flag that landed early would lose to Unsloth's own.
    _apply_platform(monkeypatch, platform)
    _label, vulkan, memory = accelerator

    backend, gguf = _backend(tmp_path, vulkan = vulkan, memory = memory)
    baseline = _stable(_launch(backend, gguf)["cmd"])

    backend, gguf = _backend(tmp_path, vulkan = vulkan, memory = memory)
    with_extra = _stable(_launch(backend, gguf, extra_args = ["--top-k", "20"])["cmd"])

    assert with_extra == [*baseline, "--top-k", "20"]


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_placement_is_not_moved_by_an_unrelated_extra_arg(
    tmp_path, monkeypatch, platform, accelerator
):
    # A flag Unsloth's estimator knows nothing about must not disturb the flags it
    # computed: the offload decision belongs to the placement code on every one of
    # these accelerators, and --seed has no business changing it.
    _apply_platform(monkeypatch, platform)
    _label, vulkan, memory = accelerator
    placement_flags = {
        "-ngl",
        "--n-gpu-layers",
        "--gpu-layers",
        "--fit",
        "-sm",
        "--split-mode",
        "--tensor-split",
        "-ncmoe",
        "--n-cpu-moe",
    }

    backend, gguf = _backend(tmp_path, vulkan = vulkan, memory = memory)
    baseline = _launch(backend, gguf)["cmd"]

    backend, gguf = _backend(tmp_path, vulkan = vulkan, memory = memory)
    with_extra = _launch(backend, gguf, extra_args = ["--seed", "42"])["cmd"]

    def _placement(cmd):
        out = []
        for index, token in enumerate(cmd):
            if token in placement_flags:
                out.append((token, cmd[index + 1] if index + 1 < len(cmd) else None))
        return out

    assert _placement(with_extra) == _placement(baseline)


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_a_denied_flag_is_refused_identically_everywhere(
    tmp_path, monkeypatch, platform, accelerator
):
    # The denylist is a property of Unsloth, not of the host: a flag refused on
    # Linux must not be reachable by running the same build on Windows.
    from core.inference.llama_server_args import validate_extra_args
    _apply_platform(monkeypatch, platform)
    for denied in (["--agent"], ["--mcp-servers-json", "{}"], ["--log-file", "x"]):
        with pytest.raises(ValueError, match = "managed by Unsloth Studio"):
            validate_extra_args(denied)


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_a_windows_shaped_value_survives_as_one_token(tmp_path, monkeypatch, platform):
    # Backslashes and a drive letter are ordinary characters to the backend: the
    # split happens in the browser, and the API takes one argv token per entry.
    _apply_platform(monkeypatch, platform)
    windows_path = r"C:\\Users\\me\\models\\template.jinja"

    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 20_000, 24_000)])
    cmd = _launch(backend, gguf, extra_args = ["--chat-template-file", windows_path])["cmd"]

    assert cmd[cmd.index("--chat-template-file") + 1] == windows_path


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_the_denied_env_twins_are_scrubbed_on_every_platform(tmp_path, monkeypatch, platform):
    # llama.cpp reads LLAMA_ARG_* before argv on all of them, so denying the token
    # without the variable would leave the capability reachable wherever Unsloth runs.
    _apply_platform(monkeypatch, platform)
    monkeypatch.setenv("LLAMA_ARG_AGENT", "1")
    monkeypatch.setenv("LLAMA_ARG_TOOLS", "all")
    # The logging twin matters most of all: Unsloth classifies a failed start by
    # reading llama-server's output, and nothing it emits later overrides this.
    monkeypatch.setenv("LLAMA_ARG_LOG_FILE", "/tmp/llama.log")
    # --api-prefix moves /health, which every load waits on, and an inherited API key
    # makes the healthy child refuse requests Unsloth sends without one.
    monkeypatch.setenv("LLAMA_ARG_API_PREFIX", "/llama")
    monkeypatch.setenv("LLAMA_API_KEY", "sk-someone-elses")
    monkeypatch.setenv("LLAMA_ARG_API_KEY_FILE", "/etc/llama.keys")
    # Given both TLS twins llama-server listens on https, while Unsloth probes /health
    # and proxies over http: the child is healthy and every load times out.
    monkeypatch.setenv("LLAMA_ARG_SSL_KEY_FILE", "/etc/llama/key.pem")
    monkeypatch.setenv("LLAMA_ARG_SSL_CERT_FILE", "/etc/llama/cert.pem")
    monkeypatch.setenv("LLAMA_ARG_NO_WARMUP", "1")

    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 20_000, 24_000)])
    env = _launch(backend, gguf)["env"]

    assert "LLAMA_ARG_AGENT" not in env
    assert "LLAMA_ARG_TOOLS" not in env
    assert "LLAMA_ARG_LOG_FILE" not in env
    assert "LLAMA_ARG_API_PREFIX" not in env
    assert "LLAMA_API_KEY" not in env
    assert "LLAMA_ARG_API_KEY_FILE" not in env
    assert "LLAMA_ARG_SSL_KEY_FILE" not in env
    assert "LLAMA_ARG_SSL_CERT_FILE" not in env
    # Not a general purge: a variable that is not a denied flag's twin, and that
    # no other reconciliation claims, is the user's own configuration and stays.
    assert env.get("LLAMA_ARG_NO_WARMUP") == "1"


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_the_size_cap_leaves_room_for_the_rest_of_a_windows_command(monkeypatch, platform):
    # CreateProcess takes ONE string for the whole command line, capped at 32767
    # characters, and the model path, Unsloth's own flags and subprocess's quoting
    # come out of the same budget. A grammar that passed here and then failed inside
    # Popen would do so after the load had begun switching models.
    import sys as _sys

    from core.inference import llama_server_args as lsa

    _label, sys_platform, _os_name, _wsl = platform
    monkeypatch.setattr(_sys, "platform", sys_platform, raising = False)
    monkeypatch.setattr(lsa.sys, "platform", sys_platform, raising = False)

    limit = lsa.max_extra_args_bytes()
    if sys_platform == "win32":
        assert limit == lsa.MAX_EXTRA_ARGS_BYTES_WINDOWS
        assert limit < 32767 - 4096, "no room left for the command Unsloth builds"
    else:
        assert limit == lsa.MAX_EXTRA_ARGS_BYTES

    # And the validator refuses at that cap, naming it.
    with pytest.raises(ValueError, match = str(limit)):
        lsa.validate_extra_args(["--grammar", "x" * (limit + 1)])
    # Just under it is accepted on every platform.
    assert lsa.validate_extra_args(["--grammar", "x" * (limit - 32)])
