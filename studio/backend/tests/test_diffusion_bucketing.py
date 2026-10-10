# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Aspect-ratio bucketing for the image LoRA trainers: the bucket set, assignment, the
one-bucket-per-batch sampler and its resume state, the per-bucket crop geometry and SDXL time
ids, every DiT family's packing on a non-square latent, the config default, and resume identity.

CPU-only."""

from __future__ import annotations

import dataclasses
import json
import math
import random
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import core.training.diffusion_checkpoint as dc
from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_lora_trainer import _load_image_tensor, _load_image_tensor_planned
from core.training.diffusion_train_common import (
    DiffusionLoraConfig,
    _config_from_dict,
    _plan_cache_variants,
    bucket_groups,
    plan_image_canvases,
    resolve_bucketing,
    resolve_train_steps,
    train_recipe_overrides,
)
from core.training.diffusion_train_extras import (
    BUCKET_DIVISOR,
    BucketBatchSampler,
    bucket_resolutions,
    cover_resize_dims,
    nearest_bucket,
    oriented_image_size,
)


def _img(
    path,
    w,
    h,
    color = (200, 30, 30),
    exif_orientation = None,
):
    im = Image.new("RGB", (w, h), color)
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        im.save(path, exif = exif)
    else:
        im.save(path)
    return str(path)


@pytest.mark.parametrize("res", [512, 768, 1024])
def test_bucket_set_is_bounded_same_area_and_on_grid(res):
    buckets = bucket_resolutions(res)
    assert (res, res) in buckets
    for w, h in buckets:
        if (w, h) != (res, res):
            assert w % BUCKET_DIVISOR == 0 and h % BUCKET_DIVISOR == 0
        assert w * h <= res * res
        assert w * h >= 0.8 * res * res
        assert 0.5 <= w / h <= 2.0
        assert (h, w) in buckets
    assert len(buckets) <= 13


def test_bucket_set_at_512_and_1024():
    assert bucket_resolutions(512) == [(384, 640), (448, 576), (512, 512), (576, 448), (640, 384)]
    assert {(1216, 832), (832, 1216), (1152, 896), (1344, 768)} <= set(bucket_resolutions(1024))


def test_off_grid_resolution_keeps_its_exact_square():
    assert (520, 520) in bucket_resolutions(520)


def test_assignment_follows_aspect():
    choices = bucket_resolutions(1024)
    assert nearest_bucket(3000, 3000, choices) == (1024, 1024)
    assert nearest_bucket(1000, 1010, choices) == (1024, 1024)
    w, h = nearest_bucket(1080, 1920, choices)
    assert h > w and abs(math.log((w / h) / (1080 / 1920))) < 0.1
    w, h = nearest_bucket(1920, 1080, choices)
    assert w > h
    assert nearest_bucket(8000, 1000, choices) == max(choices, key = lambda b: b[0] / b[1])


def test_cover_resize_is_the_legacy_short_side_resize_for_a_square_canvas():
    for w0, h0 in ((1920, 1080), (1080, 1920), (777, 333), (64, 64)):
        scale = 512 / min(w0, h0)
        legacy = (max(512, round(w0 * scale)), max(512, round(h0 * scale)))
        assert cover_resize_dims(w0, h0, 512, 512) == legacy
    rw, rh = cover_resize_dims(1080, 1920, 448, 576)
    assert rw >= 448 and rh >= 576 and (rw == 448 or rh == 576)


def test_oriented_size_reads_exif_rotation(tmp_path):
    assert oriented_image_size(_img(tmp_path / "a.jpg", 300, 200)) == (300, 200)
    assert oriented_image_size(_img(tmp_path / "b.jpg", 300, 200, exif_orientation = 6)) == (200, 300)


def _groups():
    return {(512, 512): [0, 1, 2], (640, 384): [3, 4], (384, 640): [5, 6, 7, 8]}


@pytest.mark.parametrize("k", [1, 2, 3])
def test_sampler_one_bucket_per_batch_and_every_image_per_cycle(k):
    groups = _groups()
    sampler = BucketBatchSampler(groups, random.Random(0))
    cycle = sum(-(-len(v) // k) for v in groups.values())
    seen = []
    for _ in range(cycle):
        shape, idxs = sampler.next_batch(k)
        assert len(idxs) == k and set(idxs) <= set(groups[shape])
        seen.extend(idxs)
    assert set(seen) == set(range(9))


def test_sampler_interleaves_buckets_and_is_seed_deterministic():
    a = BucketBatchSampler(_groups(), random.Random(7))
    b = BucketBatchSampler(_groups(), random.Random(7))
    da = [a.next_batch(1) for _ in range(30)]
    assert da == [b.next_batch(1) for _ in range(30)]
    assert len({s for s, _ in da[:9]}) == 3


def test_sampler_state_round_trips_mid_cycle():
    rng = random.Random(3)
    a = BucketBatchSampler(_groups(), rng)
    for _ in range(4):
        a.next_batch(2)
    state = json.loads(json.dumps(a.state_dict()))
    rng_state = rng.getstate()
    expected = [a.next_batch(2) for _ in range(12)]
    rng2 = random.Random()
    rng2.setstate(rng_state)
    b = BucketBatchSampler(_groups(), rng2)
    assert b.load_state_dict(state)
    assert [b.next_batch(2) for _ in range(12)] == expected


def test_sampler_refuses_a_foreign_state():
    a = BucketBatchSampler(_groups(), random.Random(0))
    a.next_batch(1)
    state = a.state_dict()
    other = BucketBatchSampler({(512, 512): [0, 1, 2, 3, 4, 5, 6, 7, 8]}, random.Random(0))
    assert not other.load_state_dict(state)
    assert not a.load_state_dict({"n": 9, "order": list(range(9)), "pos": 0})
    assert not a.load_state_dict(None)
    assert not a.load_state_dict({**state, "pos": 999})


def _cfg(tmp_path, **kw):
    kw.setdefault("instance_prompt", "a photo")
    return DiffusionLoraConfig(
        base_model = "stabilityai/stable-diffusion-xl-base-1.0",
        data_dir = str(tmp_path),
        output_dir = str(tmp_path / "out"),
        resolution = 512,
        **kw,
    )


def test_canvases_follow_each_image(tmp_path):
    paths = [
        _img(tmp_path / "p.png", 600, 900),
        _img(tmp_path / "l.png", 900, 600),
        _img(tmp_path / "s.png", 700, 700),
        _img(tmp_path / "exact.png", 448, 576),
    ]
    cfg = _cfg(tmp_path).normalized()
    assert cfg.bucketing is True
    canvases, room = plan_image_canvases(cfg, paths)
    assert canvases[0][1] > canvases[0][0] and canvases[1][0] > canvases[1][1]
    assert canvases[2] == (512, 512) and canvases[3] == (448, 576)
    assert room[3] == (0, 0)
    assert set(bucket_groups(canvases)) == set(canvases)
    off = dataclasses.replace(cfg, bucketing = False)
    assert plan_image_canvases(off, paths) == (None, None)


def test_an_epoch_covers_every_bucket_batch(tmp_path):
    # 3 buckets x 3 images, batch 2: 6 batches per pass, not ceil(9 / 2) = 5.
    paths = [
        _img(tmp_path / f"{kind}{i}.png", w, h)
        for kind, (w, h) in {"p": (600, 900), "l": (900, 600), "s": (700, 700)}.items()
        for i in range(3)
    ]
    cfg = _cfg(tmp_path, num_epochs = 1, train_batch_size = 2).normalized()
    assert len(bucket_groups(plan_image_canvases(cfg, paths)[0])) == 3
    assert resolve_train_steps(cfg, len(paths), paths) == 6
    assert (
        resolve_train_steps(dataclasses.replace(cfg, gradient_accumulation_steps = 4), 9, paths) == 2
    )
    assert resolve_train_steps(dataclasses.replace(cfg, bucketing = False), 9, paths) == 5
    assert resolve_train_steps(cfg, 9) == 5


def test_the_resume_preflight_target_matches_the_trainer(tmp_path):
    # The route refuses a checkpoint at or past its target, so it must match the trainer's count.
    from routes.training import _diffusion_resume_target_steps

    paths = [
        _img(tmp_path / f"{kind}{i}.png", w, h)
        for kind, (w, h) in {"p": (600, 900), "l": (900, 600), "s": (700, 700)}.items()
        for i in range(3)
    ]
    pairs = [(p, "a photo") for p in paths]
    cfg = _cfg(tmp_path, num_epochs = 1, train_batch_size = 2).normalized()
    assert _diffusion_resume_target_steps(cfg, pairs) == 6
    inherited = dataclasses.replace(
        cfg, bucketing = None, resume_from_checkpoint = str(tmp_path / "missing")
    )
    assert _diffusion_resume_target_steps(inherited, pairs) == 5


def test_plan_collapses_variants_without_crop_room():
    legacy = _plan_cache_variants(2, 4, False, True, 11)
    assert _plan_cache_variants(2, 4, False, True, 11, crop_room = None) == legacy
    bucketed = _plan_cache_variants(2, 4, False, True, 11, crop_room = [(0, 0), (40, 0)])
    # No slack: only the flip remains, so at most two variants; slack on one axis keeps its draws.
    assert len(bucketed[0]) <= 2 and all(v[:2] == (0.5, 0.5) for v in bucketed[0])
    assert all(v[1] == 0.5 for v in bucketed[1])
    assert [v[0] for v in bucketed[1]] == [v[0] for v in legacy[1]]


def test_sdxl_bucket_crop_and_time_ids(tmp_path):
    path = _img(tmp_path / "p.png", 600, 900)
    t, ids = _load_image_tensor(path, (448, 576), False, False, random.Random(0))
    assert t.shape == (3, 576, 448)
    rw, rh = cover_resize_dims(600, 900, 448, 576)
    orig_h, orig_w, top, left, target_h, target_w = ids
    assert (orig_h, orig_w, target_h, target_w) == (900, 600, 576, 448)
    assert 0 <= top <= rh - 576 and 0 <= left <= rw - 448
    t2, ids2 = _load_image_tensor_planned(path, (448, 576), True, 0.0, 0.0, True)
    assert t2.shape == (3, 576, 448)
    assert ids2[2] == (rh - 576) // 2 and ids2[4:] == (576, 448)


def test_square_canvas_tuple_is_byte_identical_to_the_legacy_int(tmp_path):
    path = _img(tmp_path / "l.png", 900, 600)
    for loader in (
        lambda r: _load_image_tensor(path, r, False, True, random.Random(5)),
        lambda r: _load_image_tensor_planned(path, r, False, 0.3, 0.7, True),
    ):
        a, ida = loader(512)
        b, idb = loader((512, 512))
        assert torch.equal(a, b) and ida == idb
    a = dit._load_pixel_tensor(path, 512, False, True, random.Random(5))
    b = dit._load_pixel_tensor(path, (512, 512), False, True, random.Random(5))
    assert torch.equal(a, b)
    assert dit._load_pixel_tensor_planned(path, (640, 384), False, 0.5, 0.5, False).shape == (
        3,
        384,
        640,
    )


class _EchoTransformer(torch.nn.Module):
    """Returns its packed input as the prediction, so forward(noisy) round-trips to noisy exactly
    when the family's pack / position ids / unpack agree on a non-square grid."""

    def __init__(self, config = None):
        super().__init__()
        self.config = config or SimpleNamespace(guidance_embeds = False)
        self.calls = []

    def forward(self, *args, **kw):
        self.calls.append(kw)
        if args:  # Z-Image: list I/O
            return ([x.clone() for x in args[0]],)
        return (kw["hidden_states"].clone(),)


def _sig(bsz, nd):
    return torch.full((bsz,), 0.5).view(bsz, *([1] * (nd - 1)))


@pytest.mark.parametrize("h,w", [(48, 80), (80, 48)])
def test_flux_round_trips_non_square(h, w):
    pytest.importorskip("diffusers")
    dit._FLUX_STATIC.clear()
    noisy = torch.randn(2, 16, h, w)
    embeds = (torch.randn(2, 7, 32), torch.randn(2, 8), torch.zeros(7, 3))
    tr = _EchoTransformer()
    out = dit._flux_forward(
        tr, noisy, torch.full((2,), 500.0), _sig(2, 4), embeds, None, "cpu", torch.float32
    )
    assert torch.equal(out, noisy)
    assert tr.calls[0]["img_ids"].shape == ((h // 2) * (w // 2), 3)


@pytest.mark.parametrize("h,w", [(48, 80), (80, 48)])
def test_qwen_round_trips_non_square(h, w):
    pytest.importorskip("diffusers")
    noisy = torch.randn(2, 16, 1, h, w)
    tr = _EchoTransformer()
    out = dit._qwen_forward(
        tr,
        noisy,
        torch.full((2,), 500.0),
        _sig(2, 5),
        (torch.randn(2, 5, 8), None),
        None,
        "cpu",
        torch.float32,
    )
    assert torch.equal(out, noisy)
    assert tr.calls[0]["img_shapes"] == [[(1, h // 2, w // 2)]] * 2


@pytest.mark.parametrize("h,w", [(48, 80), (80, 48)])
def test_zimage_round_trips_non_square(h, w):
    noisy = torch.randn(2, 16, h, w)
    out = dit._zimage_forward(
        _EchoTransformer(),
        noisy,
        torch.full((2,), 500.0),
        _sig(2, 4),
        ([torch.randn(5, 8)] * 2,),
        None,
        "cpu",
        torch.float32,
    )
    assert torch.equal(out, -noisy)


@pytest.mark.parametrize("h,w", [(48, 80), (80, 48)])
def test_krea2_round_trips_non_square(h, w):
    pytest.importorskip("diffusers.pipelines.krea2")
    noisy = torch.randn(2, 16, 1, h, w)
    tr = _EchoTransformer()
    embeds = (torch.randn(2, 6, 4, 8), torch.ones(2, 6, dtype = torch.int64))
    out = dit._krea2_forward(
        tr, noisy, torch.full((2,), 500.0), _sig(2, 5), embeds, None, "cpu", torch.float32
    )
    assert torch.equal(out, noisy)
    assert tr.calls[0]["position_ids"].shape[0] == 6 + (h // 2) * (w // 2)


@pytest.mark.parametrize("h,w", [(24, 40), (40, 24)])
def test_flux2_round_trips_non_square(h, w):
    pytest.importorskip("diffusers")
    dit._FLUX2_STATIC.clear()
    noisy = torch.randn(2, 128, h, w)
    embeds = (torch.randn(2, 7, 32), torch.zeros(2, 7, 4))
    out = dit._flux2_forward(
        _EchoTransformer(),
        noisy,
        torch.full((2,), 500.0),
        _sig(2, 4),
        embeds,
        None,
        "cpu",
        torch.float32,
    )
    assert torch.equal(out, noisy)


@pytest.mark.parametrize("h,w", [(12, 20), (20, 12)])
def test_ltx2_round_trips_non_square(h, w):
    pytest.importorskip("diffusers")
    try:
        dit._ltx2_pipeline_cls()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"LTX2Pipeline unavailable: {exc}")
    conf = SimpleNamespace(
        patch_size = 1,
        patch_size_t = 1,
        vae_scale_factors = (8, 32, 32),
        audio_sampling_rate = 16000,
        audio_hop_length = 160,
        audio_scale_factor = 4,
        audio_in_channels = 8,
    )

    class _Ltx(_EchoTransformer):
        def forward(self, *args, **kw):
            self.calls.append(kw)
            return kw["hidden_states"].clone(), None

    tr = _Ltx(conf)
    noisy = torch.randn(2, 16, 1, h, w)
    embeds = (torch.randn(2, 6, 32), torch.randn(2, 6, 32), torch.ones(2, 6, dtype = torch.int64))
    out = dit._ltx2_forward(
        tr, noisy, torch.full((2,), 500.0), _sig(2, 5), embeds, None, "cpu", torch.float32
    )
    assert torch.equal(out, noisy)
    assert (tr.calls[0]["height"], tr.calls[0]["width"]) == (h, w)


def test_bucketing_defaults_on_for_a_fresh_run_and_unset_for_a_resume(tmp_path):
    assert _cfg(tmp_path).normalized().bucketing is True
    assert _cfg(tmp_path, bucketing = False).normalized().bucketing is False
    assert (
        _config_from_dict(
            {
                "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
                "data_dir": "d",
                "output_dir": "o",
                "bucketing": "false",
            }
        ).bucketing
        is False
    )
    resumed = _cfg(tmp_path, resume_from_checkpoint = str(tmp_path / "nowhere")).normalized()
    assert resumed.bucketing is None
    assert resolve_bucketing(resumed).bucketing is False


def test_h3_recipe_pins_bucketing_off():
    cfg = SimpleNamespace(resolved_family = "minimax-h3")
    assert train_recipe_overrides(cfg)["bucketing"] is False


def _identity(tmp_path, bucketing):
    cfg = dataclasses.replace(_cfg(tmp_path).normalized(), bucketing = bucketing)
    return dc.identity_for_config(cfg)


def test_identity_records_bucketing_and_refuses_a_flip(tmp_path):
    on, off = _identity(tmp_path, True), _identity(tmp_path, False)
    assert on.bucketing == "on" and off.bucketing == "off"
    assert "aspect ratio buckets" in (on.mismatch_reason(off) or "")
    assert "aspect ratio buckets" in (off.mismatch_reason(on) or "")
    assert on.mismatch_reason(_identity(tmp_path, True)) is None


def test_a_manifest_without_the_field_trained_square(tmp_path):
    legacy = dc.CheckpointIdentity.from_dict(
        {**_identity(tmp_path, False).as_dict(), "bucketing": None}
    )
    assert legacy.bucketing is None
    assert legacy.mismatch_reason(_identity(tmp_path, False)) is None
    assert "aspect ratio buckets" in (legacy.mismatch_reason(_identity(tmp_path, True)) or "")
    # The route's pre-trainer identity has not resolved the field yet: cannot tell, not a mismatch.
    assert legacy.mismatch_reason(_identity(tmp_path, None)) is None


def test_resume_adopts_the_recorded_bucketing(tmp_path, monkeypatch):
    monkeypatch.setattr(dc, "resolve_resume_dir", lambda p: dc.Path(p))
    for recorded, expected in (("on", True), ("off", False), (None, False)):
        bundle = tmp_path / f"run_{recorded}" / "checkpoint-3"
        bundle.mkdir(parents = True)
        ident = {**_identity(tmp_path, False).as_dict(), "bucketing": recorded}
        monkeypatch.setattr(dc, "read_checkpoint", lambda p, ident = ident: {"identity": ident})
        assert dc.recorded_bucketing(str(bundle)) is expected
        cfg = _cfg(tmp_path, resume_from_checkpoint = str(bundle)).normalized()
        assert resolve_bucketing(cfg).bucketing is expected


def test_persistent_cache_keys_a_bucketed_latent_by_its_canvas(tmp_path):
    from core.training.diffusion_train_extras import PersistentConditioningCache

    path = _img(tmp_path / "p.png", 600, 900)
    cache = PersistentConditioningCache(str(tmp_path / "cc"), "fam", 512)
    variant = (0.5, 0.5, False)
    cache.put(cache.text_key("cap"), (torch.zeros(1, 3),))
    cache.put(cache.latent_key(path, variant, (448, 576)), (torch.ones(1, 4, 72, 56), None))
    plan = [[variant]]
    embeds, latents = dit._load_warm_conditioning(cache, [path], plan, ["cap"], "cpu", [(448, 576)])
    assert latents[0][0][0].shape == (1, 4, 72, 56)
    assert dit._load_warm_conditioning(cache, [path], plan, ["cap"], "cpu") == (None, None)
