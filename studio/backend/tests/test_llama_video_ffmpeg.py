# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server spawns ffprobe and ffmpeg per clip, so video is only offered when it can find them."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

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
from test_llama_video_frame_budget import _video_backend  # noqa: E402
from test_mmproj_placement_policy import _launch  # noqa: E402

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


def _props_backend(video, ffmpeg_dir = None, search_path = None):
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._has_video_input = False
    b._video_needs_ffmpeg = False
    b._video_ffmpeg_dir = ffmpeg_dir
    b._video_search_path = search_path
    b._query_server_props = lambda: {
        "modalities": {"video": video},
        "default_generation_settings": {"n_ctx": 4096},
    }
    return b


def test_a_video_model_on_a_machine_without_ffmpeg_does_not_offer_video(monkeypatch):
    _no_tools(monkeypatch)
    b = _props_backend(True)
    b._query_server_n_ctx()
    assert b._has_video_input is False
    assert b._video_needs_ffmpeg is True


def test_a_video_model_with_ffmpeg_offers_video(monkeypatch):
    _tools_at(monkeypatch, "/opt/bin/ffmpeg", "/opt/bin/ffprobe")
    b = _props_backend(True)
    b._query_server_n_ctx()
    assert b._has_video_input is True
    assert b._video_needs_ffmpeg is False


def test_a_model_without_video_never_blames_ffmpeg(monkeypatch):
    _no_tools(monkeypatch)
    b = _props_backend(False)
    b._query_server_n_ctx()
    assert b._has_video_input is False
    assert b._video_needs_ffmpeg is False


def test_the_gate_looks_where_llama_server_was_told_to(monkeypatch):
    asked = []
    monkeypatch.setattr(llama_mod, "video_tools", lambda d = None: asked.append(d))
    _props_backend(True, ffmpeg_dir = "/custom/ffmpeg")._query_server_n_ctx()
    assert asked == ["/custom/ffmpeg"]


def test_the_gate_searches_the_path_llama_server_was_given(monkeypatch, tmp_path):
    # e.g. ffmpeg.exe beside llama-server.exe on Windows, where only the child's PATH has it.
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    folder = _tool_dir(tmp_path, "ffmpeg", "ffprobe")
    b = _props_backend(True, search_path = os.pathsep.join([str(tmp_path / "none"), str(folder)]))
    b._query_server_n_ctx()
    assert b._has_video_input is True
    assert b._video_needs_ffmpeg is False


def test_a_launch_pins_the_ffmpeg_on_the_child_path(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    folder = _tool_dir(tmp_path, "ffmpeg", "ffprobe")
    assert _video_ffmpeg_dir_flags(_CAPS, env = {}, search_path = str(folder)) == (
        ["--video-ffmpeg-dir", str(folder)],
        str(folder),
    )


def test_a_launch_pins_the_ffmpeg_studio_found(monkeypatch):
    _tools_at(monkeypatch, "/opt/bin/ffmpeg", "/opt/bin/ffprobe")
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


def test_the_shrink_uses_the_same_folder(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(llama_video_input, "video_tools", lambda d = None: seen.append(d))
    llama_video_input.shrink_video_for_llama("AAAA", 1024, ffmpeg_dir = "/custom/ffmpeg")
    assert seen == ["/custom/ffmpeg"]
