# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memory auto drops the streaming flags only when free VRAM covers a conservative resident estimate."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference import video_minimax_h3 as h3  # noqa: E402

GIB = 1024**3
AUTO_FLAGS = (
    "--offload-to-cpu",
    "--diffusion-fa",
    "--max-vram",
    "-1",
    "--stream-layers",
    "--backend",
    "diffusion=CUDA0,te=CUDA0,vae=CUDA0",
)
Q3_FILES = int((8.90 + 16.97 + 4.85 + 0.56) * GIB)
Q8_FILES = int((19.97 + 16.97 + 4.85 + 0.56) * GIB)


def test_auto_runs_resident_when_the_card_holds_the_bundle():
    need = h3.h3_native_resident_bytes(Q3_FILES, 960, 544, 124)
    flags, resident = h3.h3_native_render_flags(
        AUTO_FLAGS, memory_mode = "auto", free_bytes = need, need_bytes = need, env = {}
    )
    assert resident is True
    assert "--offload-to-cpu" not in flags and "--stream-layers" not in flags
    assert flags == [
        "--diffusion-fa",
        "--max-vram",
        "-1",
        "--backend",
        "diffusion=CUDA0,te=CUDA0,vae=CUDA0",
    ]


def test_auto_keeps_offload_when_the_card_is_one_byte_short():
    need = h3.h3_native_resident_bytes(Q3_FILES, 960, 544, 124)
    flags, resident = h3.h3_native_render_flags(
        AUTO_FLAGS, memory_mode = "auto", free_bytes = need - 1, need_bytes = need, env = {}
    )
    assert resident is False and flags == list(AUTO_FLAGS)


@pytest.mark.parametrize("free, need", [(None, 10 * GIB), (90 * GIB, None), (None, None)])
def test_an_unreadable_card_or_size_keeps_offload(free, need):
    flags, resident = h3.h3_native_render_flags(
        AUTO_FLAGS, memory_mode = "auto", free_bytes = free, need_bytes = need, env = {}
    )
    assert resident is False and flags == list(AUTO_FLAGS)


@pytest.mark.parametrize("mode", ["balanced", "low_vram"])
def test_an_explicit_offload_mode_is_left_as_asked(mode):
    flags, resident = h3.h3_native_render_flags(
        AUTO_FLAGS, memory_mode = mode, free_bytes = 500 * GIB, need_bytes = GIB, env = {}
    )
    assert resident is False and flags == list(AUTO_FLAGS)


def test_flags_that_do_not_offload_are_untouched():
    fast = ("--diffusion-fa",)
    flags, resident = h3.h3_native_render_flags(
        fast, memory_mode = "auto", free_bytes = 500 * GIB, need_bytes = GIB, env = {}
    )
    assert resident is False and flags == ["--diffusion-fa"]


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_kill_switch(value):
    flags, resident = h3.h3_native_render_flags(
        AUTO_FLAGS,
        memory_mode = "auto",
        free_bytes = 500 * GIB,
        need_bytes = GIB,
        env = {h3.H3_NATIVE_RESIDENT_ENV: value},
    )
    assert resident is False and flags == list(AUTO_FLAGS)


def test_estimate_covers_the_measured_resident_peaks():
    # Measured B200 resident peaks at 960x544x124: 33030 MiB (q3), 44318 MiB (q8).
    assert h3.h3_native_resident_bytes(Q3_FILES, 960, 544, 124) > 33030 * 1024**2
    assert h3.h3_native_resident_bytes(Q8_FILES, 960, 544, 124) > 44318 * 1024**2
    assert h3.h3_native_resident_bytes(Q3_FILES, 960, 544, 124) < 47 * GIB
    assert h3.h3_native_resident_bytes(Q3_FILES, 960, 544, 124) > 23 * GIB


def test_estimate_grows_with_the_clip_and_never_shrinks_below_h1():
    h1 = h3.h3_native_resident_bytes(Q3_FILES, 960, 544, 124)
    assert h3.h3_native_resident_bytes(Q3_FILES, 1280, 720, 241) > h1
    assert h3.h3_native_resident_bytes(Q3_FILES, 640, 384, 25) == h1


def _video_backend_tests():
    # Loaded by path because `tests` can resolve to the repo-root package.
    import importlib.util

    name = "_studio_test_video_backend"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("test_video_backend.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _backend_with_files(monkeypatch, tmp_path, calls, *, memory_mode, flags):
    backend = _video_backend_tests()._h3_native_backend(monkeypatch, calls)
    paths = {}
    sizes = {}
    for name, size in (("diffusion_model", 9), ("llm", 17), ("vae", 5), ("audio_vae", 1)):
        p = tmp_path / f"{name}.bin"
        p.write_bytes(b"")
        paths[name] = str(p)
        sizes[str(p)] = size * GIB
    # Not truncate(): NTFS allocates the full 32 GiB, filling a Windows runner's disk.
    real_getsize = os.path.getsize
    monkeypatch.setattr(os.path, "getsize", lambda p: sizes.get(str(p)) or real_getsize(p))
    import dataclasses

    state = backend._state
    backend._state = dataclasses.replace(
        state,
        pipe = h3.MiniMaxH3NativeRuntime(
            engine = state.pipe.engine, files = SimpleNamespace(**paths), offload_flags = tuple(flags)
        ),
        memory_mode = memory_mode,
        offload_policy = "group",
    )
    return backend


def test_generate_drops_offload_on_a_card_that_fits(monkeypatch, tmp_path):
    import core.inference.video as video

    calls: list = []
    backend = _backend_with_files(
        monkeypatch, tmp_path, calls, memory_mode = "auto", flags = AUTO_FLAGS
    )
    seen = []
    monkeypatch.setattr(
        video,
        "_h3_card_free_bytes",
        lambda device, ordinal: seen.append((device, ordinal)) or 90 * GIB,
    )
    result = backend.generate(prompt = "a fox", width = 960, height = 544)
    assert seen == [("cuda", None)]
    assert "--offload-to-cpu" not in calls[0]["offload"] and "--diffusion-fa" in calls[0]["offload"]
    assert result["offload_policy"] == "none"
    assert result["memory_mode"] == "auto"


def test_generate_keeps_offload_on_a_card_that_does_not(monkeypatch, tmp_path):
    import core.inference.video as video

    calls: list = []
    backend = _backend_with_files(
        monkeypatch, tmp_path, calls, memory_mode = "auto", flags = AUTO_FLAGS
    )
    monkeypatch.setattr(video, "_h3_card_free_bytes", lambda device, ordinal: 20 * GIB)
    result = backend.generate(prompt = "a fox", width = 960, height = 544)
    assert calls[0]["offload"] == list(AUTO_FLAGS)
    assert result["offload_policy"] == "group"


def test_generate_does_not_probe_the_card_for_balanced(monkeypatch, tmp_path):
    import core.inference.video as video

    calls: list = []
    backend = _backend_with_files(
        monkeypatch, tmp_path, calls, memory_mode = "balanced", flags = AUTO_FLAGS
    )
    monkeypatch.setattr(
        video, "_h3_card_free_bytes", lambda *_a: (_ for _ in ()).throw(AssertionError("probed"))
    )
    backend.generate(prompt = "a fox", width = 960, height = 544)
    assert calls[0]["offload"] == list(AUTO_FLAGS)
