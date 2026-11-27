# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A partial conditioning-cache miss loads only the half it needs, on the tiny MiniMax-H3 release.

A clip-only edit must never load the (real: 48 GiB) Qwen3-VL conditioner, a caption-only edit never the VAE
encoders, and what either serves must be bit-identical to a from-scratch build.
"""

import random
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_h3_train_trim import _needs_tiny_stack  # noqa: E402

_TEXT = {"tokenizer", "processor", "text_encoder"}
_VAES = {"vae", "audio_vae"}


@pytest.fixture(scope = "module")
def release(tmp_path_factory):
    _needs_tiny_stack()
    from _minimax_h3_video_tiny import make_tiny_h3_video_release
    return make_tiny_h3_video_release(tmp_path_factory.mktemp("h3") / "release")


@pytest.fixture
def loads(monkeypatch):
    """Every component name the strict loader is asked for, per run."""
    from core.inference import diffusion_component_fetch as fetch

    seen: list[str] = []
    real = fetch.load_components_strict

    def spy(pipe, names, **kw):
        seen.extend(names)
        return real(pipe, names = names, **kw)

    monkeypatch.setattr(fetch, "load_components_strict", spy)
    return seen


def _clips(root: Path, seeds) -> list[str]:
    from _minimax_h3_video_tiny import write_clip
    root.mkdir(parents = True, exist_ok = True)
    return [str(write_clip(root / f"{i}.mp4", seed = s)) for i, s in enumerate(seeds)]


def _cfg(release, cache_dir, **kw):
    from core.training.diffusion_train_common import DiffusionLoraConfig

    base = dict(
        base_model = str(release),
        data_dir = "/tmp/clips",
        output_dir = str(Path(cache_dir).parent / "out"),
        mixed_precision = "bf16",
        resolution = 768,
        cond_cache_dir = str(cache_dir),
    )
    base.update(kw)
    return DiffusionLoraConfig(**base).normalized()


def _prepare(cfg, paths, captions):
    from core.training import diffusion_h3_trainer as h3
    return h3._prepare_h3_conditioning(
        cfg,
        paths,
        captions,
        "cpu",
        lambda *_a, **_k: None,
        lambda: False,
        width = 64,
        height = 64,
        num_frames = h3.H3_TRAIN_NUM_FRAMES,
        num_audio_latents = h3.h3_audio_latent_count(h3.H3_TRAIN_NUM_FRAMES),
    )


def _assert_same(got, want):
    embeds_a, cache_a = got
    embeds_b, cache_b = want
    assert embeds_a.keys() == embeds_b.keys()
    for cap in embeds_a:
        assert embeds_a[cap].dtype == embeds_b[cap].dtype
        assert torch.equal(embeds_a[cap], embeds_b[cap]), cap
    assert len(cache_a) == len(cache_b)
    for ea, eb in zip(cache_a, cache_b):
        assert len(ea) == len(eb) == 3
        for a, b in zip(ea, eb):
            assert a.dtype == b.dtype and torch.equal(a, b)


def _skip_without_mm_token_ids(release):
    import transformers
    proc = transformers.AutoProcessor.from_pretrained(str(release / "processor"))
    if not hasattr(proc, "create_mm_token_type_ids"):
        pytest.skip("this transformers' Qwen3VLProcessor has no create_mm_token_type_ids")


def test_a_full_miss_loads_every_conditioning_component(release, loads, tmp_path):
    _skip_without_mm_token_ids(release)
    paths = _clips(tmp_path / "clips", [0, 1])
    _prepare(_cfg(release, tmp_path / "cc"), paths, ["a cat", "the dog runs"])
    assert set(loads) == _TEXT | _VAES


def test_a_clip_only_edit_loads_no_text_conditioner(release, loads, tmp_path):
    _skip_without_mm_token_ids(release)
    from _minimax_h3_video_tiny import write_clip

    paths = _clips(tmp_path / "clips", [0, 1])
    captions = ["a cat", "the dog runs"]
    cfg = _cfg(release, tmp_path / "cc")
    _prepare(cfg, paths, captions)
    write_clip(Path(paths[1]), seed = 7)  # the edit: new bytes, same captions
    loads.clear()
    partial = _prepare(cfg, paths, captions)
    assert set(loads) == _VAES, f"a clip-only edit loaded {loads}"

    loads.clear()
    scratch = _prepare(_cfg(release, tmp_path / "scratch"), paths, captions)
    assert set(loads) == _TEXT | _VAES
    _assert_same(partial, scratch)


def test_a_caption_only_edit_loads_no_vae_encoder(release, loads, tmp_path):
    _skip_without_mm_token_ids(release)
    paths = _clips(tmp_path / "clips", [0, 1])
    cfg = _cfg(release, tmp_path / "cc")
    _prepare(cfg, paths, ["a cat", "the dog runs"])
    captions = ["a cat", "the dog runs fast"]
    loads.clear()
    partial = _prepare(cfg, paths, captions)
    assert set(loads) == _TEXT, f"a caption-only edit loaded {loads}"

    scratch = _prepare(_cfg(release, tmp_path / "scratch"), paths, captions)
    _assert_same(partial, scratch)


def test_a_warm_rerun_loads_nothing(release, loads, tmp_path):
    _skip_without_mm_token_ids(release)
    paths = _clips(tmp_path / "clips", [0])
    cfg = _cfg(release, tmp_path / "cc")
    first = _prepare(cfg, paths, ["a cat"])
    loads.clear()
    _assert_same(_prepare(cfg, paths, ["a cat"]), first)
    assert loads == []


def test_a_moved_namespace_loads_the_half_it_skipped(release, loads, tmp_path, monkeypatch):
    """A clip-only miss skips the conditioner; if the load then moves the revision, the captions miss too."""
    _skip_without_mm_token_ids(release)
    from _minimax_h3_video_tiny import write_clip
    from core.training import diffusion_train_extras as extras

    state = {"rev": "rev-1"}
    monkeypatch.setattr(extras, "source_revision", lambda *_a: state["rev"])
    paths = _clips(tmp_path / "clips", [0])
    cfg = _cfg(release, tmp_path / "cc")
    _prepare(cfg, paths, ["a cat"])
    write_clip(Path(paths[0]), seed = 5)

    from core.training import diffusion_h3_trainer as h3

    real = h3._load_conditioners

    def moving(*a, **kw):
        state["rev"] = "rev-2"
        return real(*a, **kw)

    monkeypatch.setattr(h3, "_load_conditioners", moving)
    loads.clear()
    moved = _prepare(cfg, paths, ["a cat"])
    assert set(loads) == _TEXT | _VAES
    monkeypatch.setattr(h3, "_load_conditioners", real)
    _assert_same(moved, _prepare(_cfg(release, tmp_path / "scratch"), paths, ["a cat"]))


def _first_step_loss(release, cache_dir, paths, captions, monkeypatch) -> float:
    from core.training import diffusion_h3_trainer as h3

    monkeypatch.setattr(h3, "_dataset_canvas", lambda _p, _s: (64, 64))
    cfg = _cfg(release, cache_dir, train_steps = 1, base_precision = "bf16", seed = 0)
    events: list[dict] = []
    torch.manual_seed(0)
    h3._train_h3(
        cfg,
        list(zip(paths, captions)),
        random.Random(0),
        "cpu",
        torch.float32,
        events.append,
        lambda: False,
        lambda: True,
    )
    losses = [e["loss"] for e in events if e.get("type") == "progress" and "loss" in e]
    assert losses, events
    return losses[0]


def test_the_first_step_after_a_clip_only_edit_matches_a_from_scratch_run(
    release, loads, tmp_path, monkeypatch
):
    _skip_without_mm_token_ids(release)
    from _minimax_h3_video_tiny import write_clip

    paths = _clips(tmp_path / "clips", [0])
    captions = ["a cat"]
    cache = tmp_path / "cc"
    _first_step_loss(release, cache, paths, captions, monkeypatch)
    write_clip(Path(paths[0]), seed = 3)
    loads.clear()
    partial = _first_step_loss(release, cache, paths, captions, monkeypatch)
    assert not _TEXT & set(loads), f"a clip-only edit loaded {loads}"
    scratch = _first_step_loss(release, tmp_path / "scratch", paths, captions, monkeypatch)
    assert partial == scratch
