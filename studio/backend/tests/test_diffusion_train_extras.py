# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the diffusion training extras: LoRA EMA math, the persistent
conditioning cache, aspect-ratio bucketing, and the short-run preset plumbing.

CPU-only; the full trainer integration is exercised by the live GPU smokes."""

from __future__ import annotations

import random

import pytest
import torch

from core.training.diffusion_train_extras import (
    BUCKET_DIVISOR,
    BucketBatchSampler,
    LoRAEMA,
    PersistentConditioningCache,
    assign_buckets,
    compute_bucket,
)
from core.training.diffusion_train_common import (
    DiffusionLoraConfig,
    FAMILY_TRAIN_DEFAULTS,
    train_defaults,
)


class _TinyLoRAish(torch.nn.Module):
    """Two params: one trainable (the 'LoRA'), one frozen (the 'base')."""

    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.ones(3))
        self.base = torch.nn.Parameter(torch.full((2,), 7.0), requires_grad = False)


# ── LoRA EMA ──────────────────────────────────────────────────────────────────
def test_ema_tracks_only_trainable_params():
    m = _TinyLoRAish()
    ema = LoRAEMA(m, decay = 0.9, warmup = False)
    assert len(ema) == 1
    assert set(ema.state_dict()) == {"lora_A"}


def test_ema_fixed_decay_math():
    m = _TinyLoRAish()
    ema = LoRAEMA(m, decay = 0.9, warmup = False)
    with torch.no_grad():
        m.lora_A.fill_(2.0)
    ema.update(m)
    # shadow = 0.9 * 1 + 0.1 * 2 = 1.1
    assert torch.allclose(ema.state_dict()["lora_A"], torch.full((3,), 1.1))
    ema.update(m)
    # shadow = 0.9 * 1.1 + 0.1 * 2 = 1.19
    assert torch.allclose(ema.state_dict()["lora_A"], torch.full((3,), 1.19))


def test_ema_warmup_ramp_is_responsive_early_and_capped_late():
    m = _TinyLoRAish()
    ema = LoRAEMA(m, decay = 0.99, warmup = True)
    # First update: decay = min(0.99, 1/11), so the shadow mostly adopts the new value.
    assert ema.effective_decay() == pytest.approx(1 / 11)
    with torch.no_grad():
        m.lora_A.fill_(2.0)
    ema.update(m)
    d = 1 / 11
    assert torch.allclose(ema.state_dict()["lora_A"], torch.full((3,), d * 1.0 + (1 - d) * 2.0))
    # Far into the run the ramp caps at the configured decay.
    ema.updates = 10_000
    assert ema.effective_decay() == pytest.approx(0.99)


def test_ema_copy_to_and_restore_roundtrip():
    m = _TinyLoRAish()
    ema = LoRAEMA(m, decay = 0.5, warmup = False)
    with torch.no_grad():
        m.lora_A.fill_(3.0)
    ema.update(m)  # shadow = 2.0
    backup = ema.copy_to(m)
    assert torch.allclose(m.lora_A.detach(), torch.full((3,), 2.0))
    ema.restore(m, backup)
    assert torch.allclose(m.lora_A.detach(), torch.full((3,), 3.0))
    # The frozen base param is never touched.
    assert torch.allclose(m.base.detach(), torch.full((2,), 7.0))


def test_ema_rejects_bad_decay():
    with pytest.raises(ValueError):
        LoRAEMA(_TinyLoRAish(), decay = 1.0)
    with pytest.raises(ValueError):
        LoRAEMA(_TinyLoRAish(), decay = -0.1)


# ── persistent conditioning cache ─────────────────────────────────────────────
def _make_image(
    tmp_path,
    name = "a.png",
    color = (255, 0, 0),
):
    from PIL import Image

    p = tmp_path / name
    Image.new("RGB", (8, 8), color).save(p)
    return str(p)


def test_cache_roundtrip_is_bit_identical(tmp_path):
    cache = PersistentConditioningCache(tmp_path / "cc", "qwen-image", 512)
    img = _make_image(tmp_path)
    key = cache.latent_key(img, (0.25, 0.75, True))
    # Posterior stats exactly as the trainer holds them: fp32, normalisation folded in.
    a = torch.randn(1, 16, 1, 64, 64, dtype = torch.float32)
    b = torch.randn(1, 16, 1, 64, 64, dtype = torch.float32)
    assert not cache.has(key)
    cache.put(key, (a, b))
    assert cache.has(key)
    ra, rb = cache.get(key)
    assert torch.equal(ra, a) and torch.equal(rb, b)
    assert ra.dtype == torch.float32


def test_cache_preserves_none_slots_for_deterministic_families(tmp_path):
    cache = PersistentConditioningCache(tmp_path / "cc", "flux.2-klein", 512)
    a = torch.randn(4, 4)
    key = "lat_manual_key"
    cache.put(key, (a, None))
    ra, rb = cache.get(key)
    assert torch.equal(ra, a)
    assert rb is None


def test_cache_text_entries_and_variable_tuples(tmp_path):
    cache = PersistentConditioningCache(tmp_path / "cc", "qwen-image", 512)
    key = cache.text_key("a photo of sks dog")
    pe = torch.randn(1, 13, 3584)
    mask = torch.ones(1, 13, dtype = torch.int64)
    cache.put(key, (pe, mask))
    rpe, rmask = cache.get(key)
    assert torch.equal(rpe, pe) and torch.equal(rmask, mask)
    # A different caption gets a different key.
    assert cache.text_key("another caption") != key


def test_cache_key_tracks_content_family_and_resolution(tmp_path):
    img = _make_image(tmp_path, "x.png")
    c1 = PersistentConditioningCache(tmp_path / "cc", "flux.1", 512)
    c2 = PersistentConditioningCache(tmp_path / "cc", "flux.1", 768)
    c3 = PersistentConditioningCache(tmp_path / "cc", "qwen-image", 512)
    v = (0.5, 0.5, False)
    k1 = c1.latent_key(img, v)
    assert c2.latent_key(img, v) != k1  # resolution in the key
    assert c3.latent_key(img, v) != k1  # family in the key
    assert c1.latent_key(img, (0.5, 0.5, True)) != k1  # variant in the key
    # Editing the file content invalidates the key; a pure rename does not.
    img2 = _make_image(tmp_path, "y.png", color = (0, 255, 0))
    assert c1.latent_key(img2, v) != k1
    import shutil

    renamed = tmp_path / "renamed.png"
    shutil.copy(img, renamed)
    assert c1.latent_key(str(renamed), v) == k1


def test_cache_corrupt_entry_returns_none(tmp_path):
    cache = PersistentConditioningCache(tmp_path / "cc", "flux.1", 512)
    cache.path_for("bad_key").write_bytes(b"not a safetensors file")
    assert cache.get("bad_key") is None
    assert cache.get("absent_key") is None


# ── aspect-ratio bucketing ────────────────────────────────────────────────────
def test_square_bucket_is_exactly_base_resolution():
    assert compute_bucket(1000, 1000, 512) == (512, 512)
    assert compute_bucket(64, 64, 768) == (768, 768)


def test_buckets_preserve_area_and_divisor():
    for w, h in ((1920, 1080), (1080, 1920), (800, 600), (512, 768)):
        bw, bh = compute_bucket(w, h, 512)
        assert bw % BUCKET_DIVISOR == 0 and bh % BUCKET_DIVISOR == 0
        # Same-area constraint: within ~20% of base^2 after snapping.
        assert 0.8 < (bw * bh) / (512 * 512) < 1.25
        # Orientation preserved.
        assert (bw >= bh) == (w >= h)


def test_extreme_ratios_clamp():
    bw, bh = compute_bucket(10_000, 100, 512, max_ratio = 2.0)
    assert bw / bh <= 2.0 + 1e-6


def test_assign_buckets_groups_by_shape():
    sizes = [(1000, 1000), (998, 1004), (1920, 1080), (1080, 1920)]
    buckets = assign_buckets(sizes, 512)
    assert buckets[(512, 512)] == [0, 1]
    assert sum(len(v) for v in buckets.values()) == len(sizes)


def test_bucket_batch_sampler_never_mixes_shapes_and_covers_all():
    buckets = {(512, 512): [0, 1, 2], (640, 384): [3, 4]}
    sampler = BucketBatchSampler(buckets, random.Random(0))
    seen: set[int] = set()
    for _ in range(50):
        shape, idxs = sampler.next_batch(2)
        assert len(idxs) == 2
        assert set(idxs) <= set(buckets[shape])
        seen.update(idxs)
    assert seen == {0, 1, 2, 3, 4}


def test_bucket_batch_sampler_wraps_small_bucket():
    sampler = BucketBatchSampler({(512, 512): [7]}, random.Random(1))
    shape, idxs = sampler.next_batch(3)
    assert shape == (512, 512) and idxs == [7, 7, 7]


def test_bucket_batch_sampler_is_seed_deterministic():
    buckets = {(512, 512): [0, 1, 2], (640, 384): [3, 4]}
    a = BucketBatchSampler(buckets, random.Random(42))
    b = BucketBatchSampler(buckets, random.Random(42))
    assert [a.next_batch(2) for _ in range(10)] == [b.next_batch(2) for _ in range(10)]


def test_bucket_batch_sampler_rejects_empty():
    with pytest.raises(ValueError):
        BucketBatchSampler({}, random.Random(0))


# ── preset plumbing ───────────────────────────────────────────────────────────
def test_flow_families_carry_warmup_presets():
    for family in ("flux.1", "qwen-image", "flux.2-klein", "flux.2-dev"):
        assert FAMILY_TRAIN_DEFAULTS[family]["lr_warmup_steps"] > 0
        assert train_defaults(family)["lr_warmup_steps"] > 0
    # Families without a measured warmup preset keep their previous defaults untouched.
    assert "lr_warmup_steps" not in FAMILY_TRAIN_DEFAULTS["sdxl"]


def _cfg(**kw):
    return DiffusionLoraConfig(
        base_model = "stabilityai/stable-diffusion-xl-base-1.0",
        data_dir = "d",
        output_dir = "o",
        **kw,
    )


def test_config_defaults_keep_current_behavior():
    n = _cfg().normalized()
    assert n.ema_decay == 0.0  # EMA off by default
    assert n.cond_cache_dir is None  # persistent cache off by default


def test_config_ema_decay_validation_and_coercion():
    assert _cfg(ema_decay = "0.99").normalized().ema_decay == 0.99
    with pytest.raises(ValueError):
        _cfg(ema_decay = 1.0).normalized()
    with pytest.raises(ValueError):
        _cfg(ema_decay = -0.1).normalized()
    with pytest.raises(ValueError):
        _cfg(ema_decay = "not-a-number").normalized()


def test_config_blank_cond_cache_dir_means_off():
    assert _cfg(cond_cache_dir = "   ").normalized().cond_cache_dir is None
    assert _cfg(cond_cache_dir = "/tmp/cc").normalized().cond_cache_dir == "/tmp/cc"


def test_source_revision_marks_a_dir_update_and_never_raises(tmp_path):
    # The trainer namespaces its conditioning cache on this, so an in-place checkpoint update must change the marker or a warm run trains on the old embeddings.
    from core.training.diffusion_train_extras import source_revision

    d = tmp_path / "ckpt"
    (d / "text_encoder").mkdir(parents = True)
    w = d / "text_encoder" / "model.safetensors"
    w.write_bytes(b"v1")
    first = source_revision(str(d))
    assert first == source_revision(str(d))  # stable while untouched
    w.write_bytes(b"v2-longer")
    second = source_revision(str(d))
    assert second != first
    # The VAE produces the cached latents, so an in-place VAE swap must invalidate them too.
    (d / "vae").mkdir()
    v = d / "vae" / "diffusion_pytorch_model.safetensors"
    v.write_bytes(b"vae-v1")
    third = source_revision(str(d))
    assert third != second
    v.write_bytes(b"vae-v2-longer")
    assert source_revision(str(d)) != third
    for ref in (None, "", "no/such/repo-xyz", "/does/not/exist", 7):
        assert isinstance(source_revision(ref), str)


def test_source_revision_reads_the_active_hub_cache(tmp_path, monkeypatch):
    """Studio can move its HF cache mid-session and loading follows the live setting, but
    huggingface_hub's HF_HUB_CACHE constant is a snapshot from import time. Reading only that left
    the marker unresolved (or pinned to the old root), so pulling a new revision of the same
    checkpoint stopped invalidating the conditioning cache and a warm run reused stale latents."""
    from core.training import diffusion_train_extras as extras

    old_root = tmp_path / "old" / "hub"
    new_root = tmp_path / "new" / "hub"
    for root, sha in ((old_root, "a" * 40), (new_root, "b" * 40)):
        refs = root / "models--org--ckpt" / "refs"
        refs.mkdir(parents = True)
        (refs / "main").write_text(sha, encoding = "utf-8")

    monkeypatch.setattr(extras, "_hub_cache_roots", lambda: [str(new_root), str(old_root)])
    assert extras.source_revision("org/ckpt") == f"rev-{'b' * 16}"

    monkeypatch.setattr(extras, "_hub_cache_roots", lambda: [str(old_root)])
    assert extras.source_revision("org/ckpt") == f"rev-{'a' * 16}"


def test_hub_cache_roots_puts_the_active_studio_cache_first(monkeypatch, tmp_path):
    from core.training import diffusion_train_extras as extras
    from utils import hf_cache_settings

    active = tmp_path / "studio" / "hub"
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(active))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "env" / "hub"))

    roots = extras._hub_cache_roots()
    assert roots and roots[0] == str(active)
    # The environment root is still consulted, just after the live setting.
    assert str(tmp_path / "env" / "hub") in roots


def test_hub_cache_roots_survives_without_studio_settings(monkeypatch, tmp_path):
    # The trainer subprocess may run without Studio's settings module importable, so the env and the library constant still have to work.
    import builtins

    from core.training import diffusion_train_extras as extras

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "utils.hf_cache_settings":
            raise ImportError("no studio settings here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "env" / "hub"))
    roots = extras._hub_cache_roots()
    monkeypatch.setattr(builtins, "__import__", real_import)
    assert str(tmp_path / "env" / "hub") in roots
