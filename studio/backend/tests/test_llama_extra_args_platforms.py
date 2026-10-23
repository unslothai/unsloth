# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Extra-args pass-through across every platform and accelerator: an empty box yields the old command."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# By path: the tests dir is not a package.
_PLACEMENT_PATH = Path(__file__).resolve().parent / "test_llama_cpp_placement.py"
_spec = importlib.util.spec_from_file_location("_placement_harness_platforms", _PLACEMENT_PATH)
_placement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placement)
_backend = _placement._backend
_launch = _placement._launch

# (label, sys.platform, os.name, WSL release string or None)
PLATFORMS = [
    ("linux", "linux", "posix", None),
    ("wsl2", "linux", "posix", "5.15.153.1-microsoft-standard-WSL2"),
    ("windows", "win32", "nt", None),
    ("macos", "darwin", "posix", None),
]

# (label, vulkan, memory); memory [] means no device to place on.
ACCELERATORS = [
    ("nvidia-single", False, [(0, 20_000, 24_000)]),
    ("nvidia-multi", False, [(0, 20_000, 24_000), (1, 20_000, 24_000)]),
    ("amd-vulkan", True, [(0, 12_000, 16_000)]),
    ("cpu-only", False, []),
]

MATRIX = [pytest.param(p, a, id = f"{p[0]}-{a[0]}") for p in PLATFORMS for a in ACCELERATORS]


def _apply_platform(monkeypatch, platform) -> None:
    """Patch only sys.platform and WSL markers; patching os.name too breaks pathlib mid-run."""
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
    """The command with the per-launch values masked (port, model path, API key file)."""
    masked = list(cmd)
    for index, token in enumerate(masked):
        if index and masked[index - 1] == "--port":
            masked[index] = "<port>"
        elif index and masked[index - 1] in {"-m", "--model"}:
            masked[index] = "<model>"
        elif index and masked[index - 1] == "--api-key-file":
            masked[index] = "<api-key-file>"
    return masked


def _launch_with_slot_dir(tmp_path, monkeypatch, platform, slot_dir: Path):
    """Capture the real launch command with llama.cpp slot persistence advertised."""
    _apply_platform(monkeypatch, platform)
    from utils.paths import storage_roots

    monkeypatch.setattr(storage_roots, "llama_slot_cache_root", lambda: slot_dir)
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [])
    backend.probe_server_capabilities = lambda _binary = None: {"supports_slot_save": True}
    return backend, _launch(backend, gguf)["cmd"]


def _without_flag_value(cmd: list[str], flag: str) -> list[str]:
    index = cmd.index(flag)
    return [*cmd[:index], *cmd[index + 2 :]]


def test_windows_unicode_slot_path_only_drops_optional_persistence(tmp_path, monkeypatch):
    """A/B: all launch arguments survive; only the broken optional pair is absent."""
    windows = PLATFORMS[2]
    # Relative path: tmp_path under a non-ASCII profile would make both arms drop the flag.
    monkeypatch.chdir(tmp_path)
    ascii_dir = Path("Egor") / "llama-slots"
    unicode_dir = Path("Егор") / "llama-slots"

    _ascii_backend, ascii_cmd = _launch_with_slot_dir(tmp_path, monkeypatch, windows, ascii_dir)
    unicode_backend, unicode_cmd = _launch_with_slot_dir(
        tmp_path, monkeypatch, windows, unicode_dir
    )

    assert ascii_cmd[ascii_cmd.index("--slot-save-path") + 1] == str(ascii_dir)
    assert _stable(unicode_cmd) == _without_flag_value(_stable(ascii_cmd), "--slot-save-path")
    assert unicode_backend._slot_save_dir is None
    assert unicode_backend._slot_save_binary is None


@pytest.mark.parametrize("platform", [PLATFORMS[0], PLATFORMS[1], PLATFORMS[3]])
def test_unicode_slot_path_is_unchanged_off_native_windows(tmp_path, monkeypatch, platform):
    slot_dir = tmp_path / "Егор" / "llama-slots"

    _backend_instance, cmd = _launch_with_slot_dir(tmp_path, monkeypatch, platform, slot_dir)

    assert cmd[cmd.index("--slot-save-path") + 1] == str(slot_dir)


@pytest.mark.parametrize("platform,accelerator", MATRIX)
def test_an_empty_box_changes_nothing_anywhere(tmp_path, monkeypatch, platform, accelerator):
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
    # Appended, never interleaved: llama.cpp is last-wins.
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
    from core.inference.llama_server_args import validate_extra_args
    _apply_platform(monkeypatch, platform)
    for denied in (["--agent"], ["--mcp-servers-json", "{}"], ["--log-file", "x"]):
        with pytest.raises(ValueError, match = "managed by Unsloth Studio"):
            validate_extra_args(denied)


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_a_windows_shaped_value_survives_as_one_token(tmp_path, monkeypatch, platform):
    _apply_platform(monkeypatch, platform)
    windows_path = r"C:\\Users\\me\\models\\template.jinja"

    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 20_000, 24_000)])
    cmd = _launch(backend, gguf, extra_args = ["--chat-template-file", windows_path])["cmd"]

    assert cmd[cmd.index("--chat-template-file") + 1] == windows_path


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_the_denied_env_twins_are_scrubbed_on_every_platform(tmp_path, monkeypatch, platform):
    _apply_platform(monkeypatch, platform)
    monkeypatch.setenv("LLAMA_ARG_AGENT", "1")
    monkeypatch.setenv("LLAMA_ARG_TOOLS", "all")
    monkeypatch.setenv("LLAMA_ARG_LOG_FILE", "/tmp/llama.log")
    # --api-prefix moves /health; an inherited API key makes the child refuse Unsloth.
    monkeypatch.setenv("LLAMA_ARG_API_PREFIX", "/llama")
    monkeypatch.setenv("LLAMA_API_KEY", "sk-someone-elses")
    monkeypatch.setenv("LLAMA_ARG_API_KEY_FILE", "/etc/llama.keys")
    # Both TLS twins make llama-server serve https while Unsloth probes over http.
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
    assert env.get("LLAMA_ARG_NO_WARMUP") == "1"


@pytest.mark.parametrize("platform", PLATFORMS, ids = [p[0] for p in PLATFORMS])
def test_the_size_cap_leaves_room_for_the_rest_of_a_windows_command(monkeypatch, platform):
    # Windows CreateProcess caps the whole command line at 32767 chars.
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

    with pytest.raises(ValueError, match = str(limit)):
        lsa.validate_extra_args(["--grammar", "x" * (limit + 1)])
    assert lsa.validate_extra_args(["--grammar", "x" * (limit - 32)])
