# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The CUTLASS race cache key follows what decides a race verdict, not the whole GEMM source file.

The key hashed both module files byte for byte, so every Studio update touching diffusion_int8_gemm.py (install
routes, SwiGLU fusion, logging) re-raced every int8 shape on the next load: about 20 s per model on an L4. Now an
edit outside the race code keeps the cached verdicts and an edit to the Triton kernel, its launch, the race or its
timing re-races. Runs anywhere: the sources are temp copies, the library a temp file, the card faked."""

from __future__ import annotations

import os
import shutil

import pytest

from core.inference import diffusion_int8_cutlass as cut
from core.inference import diffusion_int8_gemm as g8
from core.inference import diffusion_triton_prefetch as prefetch

torch = pytest.importorskip("torch")


def _new_process():
    g8._sources_digest.cache_clear()


@pytest.fixture
def src(tmp_path, monkeypatch):
    """Editable copies of the three keyed modules, standing in for the installed files."""
    lib = tmp_path / "libunsloth_i8gemm.so"
    lib.write_bytes(b"\x7fELF" + os.urandom(1 << 16))
    paths = {}
    for name, module in (("gemm", g8), ("cut", cut), ("prefetch", prefetch)):
        path = tmp_path / f"{name}.py"
        shutil.copyfile(module.__file__, path)
        monkeypatch.setattr(module, "__file__", str(path))
        paths[name] = path
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA L4")
    monkeypatch.setattr(cut, "library_path", lambda: str(lib))
    for env in (
        cut.CUTLASS_ENV,
        cut.SUSTAINED_ENV,
        cut.FILL_TIE_ENV,
        g8.INT8_GEMM_TILES_ENV,
        g8.INT8_GEMM_UNLISTED_ENV,
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(g8, "_DEVICE_CFG", {})
    monkeypatch.setattr(g8, "_DEVICE_TILES", {})
    _new_process()
    yield paths
    _new_process()


def _edit(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))
    _new_process()


def _verdict_after(edit) -> bool:
    """A verdict cached under today's key, then the edit and a new process: is it read back?"""
    table = {
        g8._cut_cache_key(0): {"(3072, 3072, 'torch.float32', 'torch.bfloat16', True)": {"512": 2}}
    }
    edit()
    key = g8._cut_cache_key(0)
    assert key
    return key in table


# Studio edits that move no race result: the next load reads its verdicts back.
_UNRELATED = {
    "a new route in the GEMM module": (
        "gemm",
        "def is_installed(module: Any) -> bool:",
        "def _a_new_helper() -> int:\n    return 1\n\n\ndef is_installed(module: Any) -> bool:",
    ),
    "an install-path edit": (
        "gemm",
        "def call_count() -> int:\n",
        "def call_count() -> int:\n    _ = 0\n",
    ),
    "a comment in the Triton launch": (
        "gemm",
        "            ACT = int(act),\n",
        "            ACT = int(act),  # tanh GELU epilogue\n",
    ),
    "a docstring of the race": (
        "cut",
        '"""Best probed tile vs ``today`` (None = cuBLAS ``_int_mm``); tile None keeps today\'s path."""',
        '"""Best probed tile vs today\'s path."""',
    ),
    "a reformatted race constant": ("cut", "_MARGIN = 0.02", "_MARGIN = (0.02)"),
    "a native runtime edit": (
        "cut",
        "def native_gemm(",
        "def _unused() -> None:\n    pass\n\n\ndef native_gemm(",
    ),
}

# Edits that change what wins: the next load races again.
_RACE_CHANGING = {
    "the Triton kernel body": (
        "gemm",
        "return x.to(tl.bfloat16).to(tl.float32)",
        "return x.to(tl.float16).to(tl.float32)",
    ),
    "the Triton launch options": (
        "gemm",
        "            ACT = int(act),\n            num_warps = warps,",
        "            ACT = int(act),\n            num_warps = 4,",
    ),
    "today's path pick": ("gemm", "if cfg is not None and _aligned(a, w):", "if cfg is not None:"),
    "the race margin": ("cut", "_MARGIN = 0.02", "_MARGIN = 0.03"),
    "the fill tie band": ("cut", "_FILL_TIE = 0.03", "_FILL_TIE = 0.05"),
    "the fill tie-break": (
        "cut",
        "return max(near, key = lambda t: (last_tile_fill(m, rows[t]), -timing[t])), fastest",
        "return fastest, fastest",
    ),
    "the race points": ("cut", "_RACE_M = (512, 4096)", "_RACE_M = (256, 4096)"),
    "the sustained timing window": ("cut", "_SUSTAIN_S = 0.25", "_SUSTAIN_S = 0.5"),
    "the prebuilt kernels' launch": (
        "cut",
        "stream = torch.cuda.current_stream(a.device).cuda_stream",
        "stream = 0",
    ),
    # the probe decides which tiles the race times: one that admits more tiles must race them
    "the probe's dtype pairs": (
        "cut",
        "(x, w) for x in (torch.bfloat16, torch.float32) for w in (torch.bfloat16, torch.float32)",
        "(x, w) for x in (torch.bfloat16,) for w in (torch.bfloat16,)",
    ),
    "the probe shapes": ("cut", "    (1, 64, 64, False),\n", ""),
    "the probe's tie check": (
        "cut",
        "            # near-midpoint products: the double rounding must match\n",
        "            continue\n",
    ),
    "the probe's tie operands": ("gemm", "    k2 = 128\n", "    k2 = 256\n"),
    "the probed tile set": (
        "cut",
        "ok = tuple(t for t in range(len(tiles(fam))) if _probe_tile(index, fam, t, pairs))",
        "ok = tuple(t for t in range(1, len(tiles(fam))) if _probe_tile(index, fam, t, pairs))",
    ),
    "the prefetch fix version": (
        "prefetch",
        '"unsloth-sm8x-8bit-prefetch-skip-1"',
        '"unsloth-sm8x-8bit-prefetch-skip-2"',
    ),
}


@pytest.mark.parametrize("case", sorted(_UNRELATED))
def test_an_unrelated_edit_keeps_the_cached_verdict(src, case):
    name, old, new = _UNRELATED[case]
    assert _verdict_after(lambda: _edit(src[name], old, new))


@pytest.mark.parametrize("case", sorted(_RACE_CHANGING))
def test_a_race_changing_edit_invalidates_it(src, case):
    name, old, new = _RACE_CHANGING[case]
    assert not _verdict_after(lambda: _edit(src[name], old, new))


def test_a_new_library_build_invalidates_it(src):
    assert not _verdict_after(lambda: (open(cut.library_path(), "ab").write(b"\0"), _new_process()))


def test_todays_tile_on_this_card_is_part_of_the_key(src, monkeypatch):
    """An arch table, shape rule or unlisted-card race that moves this card's Triton tile re-races; stock stays."""
    stock = g8._cut_cache_key(0)
    g8._DEVICE_CFG[0] = (256, 128, 128, 8, 8, 3)
    monkeypatch.setattr(g8, "_prefetch_fix_key", lambda: True)
    tiled = g8._cut_cache_key(0)
    g8._DEVICE_CFG[0] = (128, 128, 64, 8, 4, 4)
    other = g8._cut_cache_key(0)
    monkeypatch.setattr(g8, "_prefetch_fix_key", lambda: False)
    unfixed = g8._cut_cache_key(0)
    assert len({stock, tiled, other, unfixed}) == 4
    g8._DEVICE_CFG[1] = (64, 64, 64, 4, 4, 2)  # another card's tile: not this card's key
    assert g8._cut_cache_key(0) == unfixed


def test_force_mode_does_not_reuse_auto_verdicts(src, monkeypatch):
    auto = g8._cut_cache_key(0)
    monkeypatch.setenv(cut.CUTLASS_ENV, "force")
    assert g8._cut_cache_key(0) != auto


def test_fill_ties_off_does_not_reuse_tie_broken_verdicts(src, monkeypatch):
    on = g8._cut_cache_key(0)
    monkeypatch.setenv(cut.FILL_TIE_ENV, "0")
    assert g8._cut_cache_key(0) != on


def test_every_keyed_name_is_still_defined():
    """A renamed race function would drop out of the key silently: it must be renamed here too."""
    for module in (g8, cut, prefetch):
        names = g8._RACE_SOURCES[module.__name__.rsplit(".", 1)[-1]]
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, (module.__name__, missing)
        assert "<missing>" not in g8._code(module.__file__, names)


def test_the_real_sources_key_is_stable_across_processes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA L4")
    monkeypatch.setattr(cut, "library_path", lambda: None)
    _new_process()
    first = g8._cut_cache_key(0)
    _new_process()
    assert first and g8._cut_cache_key(0) == first
