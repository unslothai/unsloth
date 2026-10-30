# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video is offered only when llama-server can launch ffmpeg and ffprobe (#16065)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from test_mmproj_placement_policy import _backend, _launch  # noqa: E402

_EXE = ".exe" if sys.platform == "win32" else ""


def _tools_dir(root: Path, *names: str) -> Path:
    root.mkdir(parents = True, exist_ok = True)
    for name in names:
        tool = root / f"{name}{_EXE}"
        tool.write_text("#!/bin/sh\n")
        tool.chmod(0o755)
    return root


def test_both_tools_on_path_keep_video(tmp_path):
    bin_dir = _tools_dir(tmp_path / "bin", "ffmpeg", "ffprobe")
    assert llama_cpp._video_tools_reachable(["llama-server"], {"PATH": str(bin_dir)})


@pytest.mark.parametrize("present", [(), ("ffmpeg",), ("ffprobe",)])
def test_a_missing_tool_on_path_drops_video(tmp_path, present):
    bin_dir = _tools_dir(tmp_path / "bin", *present)
    assert not llama_cpp._video_tools_reachable(
        ["llama-server"], {"PATH": str(bin_dir)}, str(tmp_path)
    )


def test_the_ffmpeg_dir_flag_beats_path_and_env(tmp_path):
    good = _tools_dir(tmp_path / "good", "ffmpeg", "ffprobe")
    empty = _tools_dir(tmp_path / "empty")
    env = {"PATH": str(empty), "LLAMA_ARG_VIDEO_FFMPEG_DIR": str(empty)}
    assert llama_cpp._video_tools_reachable(["llama-server", "--video-ffmpeg-dir", str(good)], env)
    assert llama_cpp._video_tools_reachable(["llama-server", f"--video-ffmpeg-dir={good}"], env)
    env = {"PATH": str(good), "LLAMA_ARG_VIDEO_FFMPEG_DIR": str(good)}
    assert not llama_cpp._video_tools_reachable(
        ["llama-server", "--video-ffmpeg-dir", str(empty)], env
    )


@pytest.mark.skipif(sys.platform == "win32", reason = "no execute bit on Windows")
def test_non_executable_tools_in_the_ffmpeg_dir_drop_video(tmp_path):
    tools = _tools_dir(tmp_path / "tools", "ffmpeg", "ffprobe")
    for tool in tools.iterdir():
        tool.chmod(0o644)
    assert not llama_cpp._video_tools_reachable(
        ["llama-server", "--video-ffmpeg-dir", str(tools)], {}
    )


def test_the_ffmpeg_dir_env_var_is_read_without_the_flag(tmp_path):
    good = _tools_dir(tmp_path / "good", "ffmpeg", "ffprobe")
    empty = _tools_dir(tmp_path / "empty")
    env = {"PATH": str(empty), "LLAMA_ARG_VIDEO_FFMPEG_DIR": str(good)}
    assert llama_cpp._video_tools_reachable(["llama-server"], env)


def _load_with_video_props(tmp_path, monkeypatch, path, **load_kwargs):
    monkeypatch.setenv("PATH", str(path))
    monkeypatch.delenv("LLAMA_ARG_VIDEO_FFMPEG_DIR", raising = False)
    backend, gguf = _backend(tmp_path, memory = [(0, 12_000, 24_000)])
    backend._query_server_props = lambda: {
        "modalities": {"vision": True, "video": True},
        "default_generation_settings": {"n_ctx": 4096},
    }
    _launch(backend, gguf, **load_kwargs)
    return backend


def test_a_load_without_ffprobe_does_not_offer_video(tmp_path, monkeypatch):
    backend = _load_with_video_props(tmp_path, monkeypatch, _tools_dir(tmp_path / "bin", "ffmpeg"))
    assert backend._has_video_input is False


def test_a_load_with_both_tools_offers_video(tmp_path, monkeypatch):
    bin_dir = _tools_dir(tmp_path / "bin", "ffmpeg", "ffprobe")
    backend = _load_with_video_props(tmp_path, monkeypatch, bin_dir)
    assert backend._has_video_input is True


def test_a_load_given_an_ffmpeg_dir_offers_video(tmp_path, monkeypatch):
    good = _tools_dir(tmp_path / "ffmpeg", "ffmpeg", "ffprobe")
    backend = _load_with_video_props(
        tmp_path,
        monkeypatch,
        _tools_dir(tmp_path / "bin"),
        extra_args = ("--video-ffmpeg-dir", str(good)),
    )
    assert backend._has_video_input is True
