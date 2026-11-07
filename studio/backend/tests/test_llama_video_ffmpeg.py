# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

pytest.importorskip("torch")

import core.inference.llama_cpp as llama_mod  # noqa: E402
from core.inference import llama_video_input  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend, _video_ffmpeg_dir_flags  # noqa: E402
from core.inference.llama_video_input import video_tools  # noqa: E402

from test_llama_flag_catalog import _probe_with_help  # noqa: E402
from test_llama_video_frame_budget import _clip, _probe, _video_backend, needs_ffmpeg  # noqa: E402
from test_mmproj_placement_policy import _launch  # noqa: E402
from test_video_needs_ffmpeg import _load_with_video_props, _tools_dir as _bin  # noqa: E402

_CAPS = {"supports_video_ffmpeg_dir": True}
_EXE = ".exe" if os.name == "nt" else ""


def _tool_dir(tmp_path, *names):
    folder = tmp_path / "ffmpeg-bin"
    folder.mkdir(exist_ok = True)
    for name in names:
        exe = folder / f"{name}{_EXE}"
        exe.write_bytes(b"")
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return folder


def _tools_at(monkeypatch, ffmpeg, ffprobe):
    monkeypatch.setattr(llama_mod, "video_tools", lambda _dir = None: (ffmpeg, ffprobe))


def _no_tools(monkeypatch):
    monkeypatch.setattr(llama_mod, "video_tools", lambda _dir = None: None)


def test_a_launch_pins_the_ffmpeg_studio_found(monkeypatch):
    _tools_at(monkeypatch, "/opt/bin/ffmpeg", "/opt/bin/ffprobe")
    if sys.platform == "win32":
        assert _video_ffmpeg_dir_flags(_CAPS, env = {}) == ([], None)
        return
    assert _video_ffmpeg_dir_flags(_CAPS, env = {}) == (
        ["--video-ffmpeg-dir", "/opt/bin"],
        "/opt/bin",
    )


def test_tools_in_two_folders_are_left_to_path(monkeypatch):
    _tools_at(monkeypatch, "/a/ffmpeg", "/b/ffprobe")
    assert _video_ffmpeg_dir_flags(_CAPS, env = {}) == ([], None)


def test_no_ffmpeg_or_an_older_build_sends_no_flag(monkeypatch):
    _no_tools(monkeypatch)
    assert _video_ffmpeg_dir_flags(_CAPS, env = {}) == ([], None)
    _tools_at(monkeypatch, "/opt/bin/ffmpeg", "/opt/bin/ffprobe")
    assert _video_ffmpeg_dir_flags({}, env = {}) == ([], None)


def test_a_hand_typed_or_inherited_folder_is_kept(monkeypatch):
    _tools_at(monkeypatch, "/opt/bin/ffmpeg", "/opt/bin/ffprobe")
    assert _video_ffmpeg_dir_flags(_CAPS, env = {}, extra_args = ["--video-ffmpeg-dir", "/mine"]) == (
        [],
        "/mine",
    )
    assert _video_ffmpeg_dir_flags(_CAPS, env = {"LLAMA_ARG_VIDEO_FFMPEG_DIR": "/env"}) == (
        [],
        "/env",
    )


def test_an_older_build_ignores_the_inherited_folder_so_the_check_does_too(monkeypatch):
    _tools_at(monkeypatch, "/opt/bin/ffmpeg", "/opt/bin/ffprobe")
    assert _video_ffmpeg_dir_flags({}, env = {"LLAMA_ARG_VIDEO_FFMPEG_DIR": "/env"}) == ([], None)


def test_the_launch_command_carries_the_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FFMPEG_DIR", raising = False)
    _tools_at(monkeypatch, "/opt/bin/ffmpeg", "/opt/bin/ffprobe")
    backend, gguf = _video_backend(tmp_path, supports_video_ffmpeg_dir = True)
    cmd = _launch(backend, gguf)["cmd"]
    if sys.platform == "win32":
        assert "--video-ffmpeg-dir" not in cmd
        return
    assert cmd[cmd.index("--video-ffmpeg-dir") + 1] == "/opt/bin"
    assert backend._video_ffmpeg_dir == "/opt/bin"


def test_the_probe_reads_the_flag_from_help(monkeypatch, tmp_path):
    caps = _probe_with_help(
        monkeypatch,
        tmp_path,
        "--video-ffmpeg-dir DIR                  path to the directory containing ffmpeg and ffprobe\n",
    )
    assert caps["supports_video_ffmpeg_dir"] is True


def test_video_tools_needs_both_binaries_in_the_folder(tmp_path):
    assert video_tools(str(_tool_dir(tmp_path, "ffmpeg"))) is None
    folder = _tool_dir(tmp_path, "ffmpeg", "ffprobe")
    ffmpeg, ffprobe = video_tools(str(folder))
    assert Path(ffmpeg).parent == folder == Path(ffprobe).parent


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows PATHEXT lookup")
def test_command_wrappers_do_not_hide_native_ffmpeg(tmp_path, monkeypatch):
    shims = tmp_path / "shims"
    shims.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (shims / f"{name}.cmd").write_text("@exit /b 0\n", encoding = "utf-8")
    native = _tool_dir(tmp_path, "ffmpeg", "ffprobe")
    monkeypatch.setenv("PATH", os.pathsep.join((str(shims), str(native))))
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FFMPEG_DIR", raising = False)
    tools = video_tools()
    assert tools is not None
    assert all(Path(tool).parent == native for tool in tools)
    assert _video_ffmpeg_dir_flags(_CAPS) == ([], None)


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows application-directory search")
def test_automatic_pinning_does_not_replace_server_local_tools(tmp_path, monkeypatch):
    system = _bin(tmp_path / "system", "ffmpeg", "ffprobe")
    bundled = _bin(tmp_path / "server", "ffmpeg", "ffprobe")
    monkeypatch.setenv("PATH", str(system))
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FFMPEG_DIR", raising = False)
    flags, _ = _video_ffmpeg_dir_flags(_CAPS)
    search = llama_mod._video_tools_search(
        [str(bundled / "llama-server.exe"), *flags], {"PATH": str(system)}, str(tmp_path)
    )
    tools = video_tools(search)
    assert tools is not None
    assert all(Path(tool).parent == bundled for tool in tools)


def test_the_shrink_uses_the_same_folder(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(llama_video_input, "video_tools", lambda d = None: seen.append(d))
    llama_video_input.shrink_video_for_llama("AAAA", 1024, ffmpeg_dir = "/custom/ffmpeg")
    assert seen == ["/custom/ffmpeg"]


def test_an_older_build_ignores_the_ffmpeg_dir_env_after_load(tmp_path):
    good = _bin(tmp_path / "good", "ffmpeg", "ffprobe")
    stale = _bin(tmp_path / "stale")
    env = {"PATH": str(good), "LLAMA_ARG_VIDEO_FFMPEG_DIR": str(stale)}
    assert llama_mod._video_tools_reachable(["llama-server"], env, supports_ffmpeg_dir = False)
    assert not llama_mod._video_tools_reachable(["llama-server"], env)


def test_a_load_without_ffprobe_flags_the_missing_tools(tmp_path, monkeypatch):
    backend = _load_with_video_props(tmp_path, monkeypatch, _bin(tmp_path / "bin", "ffmpeg"))
    assert backend._has_video_input is False
    assert backend._video_needs_ffmpeg is True


def test_a_load_with_both_tools_does_not_blame_ffmpeg(tmp_path, monkeypatch):
    backend = _load_with_video_props(
        tmp_path, monkeypatch, _bin(tmp_path / "bin", "ffmpeg", "ffprobe")
    )
    assert backend._has_video_input is True
    assert backend._video_needs_ffmpeg is False


def test_the_shrink_searches_where_the_loaded_child_found_ffmpeg(tmp_path, monkeypatch):
    bin_dir = _bin(tmp_path / "bin", "ffmpeg", "ffprobe")
    backend = _load_with_video_props(tmp_path, monkeypatch, bin_dir)
    monkeypatch.setenv("PATH", str(tmp_path / "elsewhere"))
    ffmpeg, ffprobe = video_tools(backend._video_ffmpeg_dir)
    assert Path(ffmpeg).parent == bin_dir == Path(ffprobe).parent


def test_the_child_search_follows_the_older_build_rule(tmp_path):
    env = {"PATH": str(tmp_path / "path"), "LLAMA_ARG_VIDEO_FFMPEG_DIR": str(tmp_path / "env")}
    assert llama_mod._video_tools_search(["llama-server"], env) == str(tmp_path / "env")
    search = llama_mod._video_tools_search(["llama-server"], env, supports_ffmpeg_dir = False)
    assert str(tmp_path / "path") in search.split(os.pathsep)
    assert str(tmp_path / "env") not in search.split(os.pathsep)


@pytest.mark.parametrize("probe_result", ["timeout", "partial", "empty"])
def test_an_inconclusive_probe_keeps_the_inherited_directory(tmp_path, monkeypatch, probe_result):
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"probe fixture")
    binary.chmod(0o755)

    def run_probe(*args, **kwargs):
        if probe_result == "timeout":
            raise subprocess.TimeoutExpired(args[0], 10)
        return SimpleNamespace(
            stdout = "--top-k N  top-k sampling\n" if probe_result == "partial" else "",
            stderr = "",
            returncode = 1 if probe_result == "partial" else 0,
        )

    with monkeypatch.context() as probe_patch:
        probe_patch.setattr(llama_mod.subprocess, "run", run_probe)
        caps = LlamaCppBackend.probe_server_capabilities(str(binary))
    tools = _bin(tmp_path / "tools", "ffmpeg", "ffprobe")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setenv("LLAMA_ARG_VIDEO_FFMPEG_DIR", str(tools))
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    backend, gguf = _video_backend(
        model_dir, supports_video_ffmpeg_dir = caps["supports_video_ffmpeg_dir"]
    )
    backend._query_server_props = lambda: {
        "modalities": {"vision": True, "video": True},
        "default_generation_settings": {"n_ctx": 4096},
    }
    _launch(backend, gguf)
    assert backend._has_video_input is True
    assert caps["supports_video_ffmpeg_dir"] is None


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX directory ending in a space")
@pytest.mark.parametrize("explicit", [False, True])
def test_video_directory_whitespace_is_preserved(tmp_path, monkeypatch, explicit):
    tools = _bin(tmp_path / "tools ", "ffmpeg", "ffprobe")
    monkeypatch.setenv("PATH", str(tools))
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FFMPEG_DIR", raising = False)
    args = ["--video-ffmpeg-dir", str(tools)] if explicit else []
    flags, _ = _video_ffmpeg_dir_flags(_CAPS, extra_args = args)
    assert llama_mod._video_tools_reachable(["llama-server", *flags, *args])


@pytest.mark.skipif(sys.platform != "linux", reason = "Linux ELF loader environment")
@needs_ffmpeg
def test_preprocessing_uses_the_running_servers_loader_environment(tmp_path):
    from routes.inference import _shrink_clip_for_llama

    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the shared-library fixture")
    ffmpeg, ffprobe = video_tools()
    tools = tmp_path / "bin"
    tools.mkdir()
    marker = tmp_path / "marker.c"
    marker.write_text("int fixture_marker(void) { return 734; }\n", encoding = "utf-8")
    forward = tmp_path / "forward.c"
    forward.write_text(
        "#include <string.h>\n#include <unistd.h>\nextern int fixture_marker(void);\n"
        "int main(int argc, char **argv) {\n"
        "if (fixture_marker() != 734) return 7;\n"
        'const char *tool = strstr(argv[0], "ffprobe") ? '
        + json.dumps(ffprobe)
        + " : "
        + json.dumps(ffmpeg)
        + ";\nargv[0] = (char *)tool; execv(tool, argv); return 9; }\n",
        encoding = "utf-8",
    )
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(marker), "-o", str(tools / "libffmpeg_fixture.so")],
        check = True,
        capture_output = True,
    )
    subprocess.run(
        [
            compiler,
            str(forward),
            "-L" + str(tools),
            "-lffmpeg_fixture",
            "-o",
            str(tools / "ffmpeg"),
        ],
        check = True,
        capture_output = True,
    )
    shutil.copy2(tools / "ffmpeg", tools / "ffprobe")
    (tools / "llama-server").write_bytes(b"fixture")
    env = LlamaCppBackend._llama_server_env_for_binary(
        str(tools / "llama-server"), use_system_rocm = False
    )
    clip = _clip(tmp_path, "input.mp4", "1280x720")
    process = object()
    backend = SimpleNamespace(
        _process = process,
        _process_env = (process, env),
        _video_ffmpeg_dir = str(tools),
        extra_args = None,
    )
    shrunk = asyncio.run(_shrink_clip_for_llama(clip, backend))
    geometry = _probe(tmp_path, shrunk)
    assert (geometry["width"], geometry["height"]) == (640, 360)
    backend._process_env = (object(), env)
    assert asyncio.run(_shrink_clip_for_llama(clip, backend)) == clip
