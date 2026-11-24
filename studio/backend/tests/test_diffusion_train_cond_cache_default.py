# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The persistent training conditioning cache is on by default (DiT, MiniMax-H3 video, MiniMax-H3-Image), has an off
switch, is bounded by a size cap with least-recently-used eviction, misses on every input that changes the encoded
tensors, and a warm hit is bit-identical to the fresh encode without loading the conditioner."""

from __future__ import annotations

import contextlib
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.training import diffusion_train_extras as extras  # noqa: E402
from core.training.diffusion_train_common import DiffusionLoraConfig  # noqa: E402
from core.training.diffusion_train_extras import PersistentConditioningCache  # noqa: E402


# ---------------------------------------------------------------- default / off switch


def test_unset_means_the_default_directory_under_the_studio_home():
    from utils.paths import outputs_root, studio_root

    chosen = extras.effective_cond_cache_dir(None)
    assert chosen is not None
    assert Path(chosen) == extras.default_cond_cache_dir()
    assert Path(chosen).name == "diffusion-train-cond"
    assert studio_root().resolve() in Path(chosen).resolve().parents
    # Never inside the outputs root, which the run listings scan.
    assert outputs_root().resolve() not in Path(chosen).resolve().parents
    assert extras.effective_cond_cache_dir("   ") == chosen


def test_an_off_value_and_the_env_switch_turn_the_cache_off(monkeypatch, tmp_path):
    for value in ("off", "OFF", "none", "0", "false", " disabled "):
        assert extras.effective_cond_cache_dir(value) is None, value
    explicit = str(tmp_path / "mine")
    assert extras.effective_cond_cache_dir(explicit) == explicit
    monkeypatch.setenv(extras.TRAIN_COND_CACHE_ENV, "0")
    assert extras.effective_cond_cache_dir(None) is None
    # The env switch is for the DEFAULT only: an explicit directory keeps working exactly as before.
    assert extras.effective_cond_cache_dir(explicit) == explicit


def test_normalized_keeps_unset_as_default_and_canonicalises_off():
    def cfg(value):
        return DiffusionLoraConfig(
            base_model = "unsloth/FLUX.1-dev",
            data_dir = "/tmp/d",
            output_dir = "/tmp/o",
            instance_prompt = "p",
            cond_cache_dir = value,
        ).normalized()

    assert cfg(None).cond_cache_dir is None
    assert cfg("  ").cond_cache_dir is None
    assert cfg("OFF").cond_cache_dir == "off"
    assert cfg("/tmp/cc").cond_cache_dir == "/tmp/cc"


# ---------------------------------------------------------------- size cap


def _put(
    cache,
    key,
    n = 1024,
):
    cache.put(key, (torch.zeros(n, dtype = torch.float32),))


def test_the_cache_stays_under_its_cap_evicting_oldest_first(tmp_path):
    one = None
    cache = PersistentConditioningCache(str(tmp_path), "fam", 64, max_bytes = 10**9)
    _put(cache, "txt_v1_probe")
    one = cache.path_for("txt_v1_probe").stat().st_size
    cache.path_for("txt_v1_probe").unlink()

    cap = one * 5
    cache = PersistentConditioningCache(str(tmp_path), "fam", 64, max_bytes = cap)
    now = time.time()
    for i in range(12):
        _put(cache, f"txt_v1_k{i}")
        # Distinct mtimes without sleeping.
        os.utime(cache.path_for(f"txt_v1_k{i}"), (now - 100 + i, now - 100 + i))
        total = sum(p.stat().st_size for p in tmp_path.glob("*.safetensors"))
        assert total <= cap, (i, total, cap)
    left = sorted(p.stem for p in tmp_path.glob("*.safetensors"))
    # The newest survive, the oldest went first.
    assert "txt_v1_k11" in left and "txt_v1_k0" not in left


def test_a_hit_refreshes_an_entry_so_it_outlives_older_ones(tmp_path):
    cache = PersistentConditioningCache(str(tmp_path), "fam", 64, max_bytes = 10**9)
    now = time.time()
    for i in range(3):
        _put(cache, f"txt_v1_k{i}")
        os.utime(cache.path_for(f"txt_v1_k{i}"), (now - 100 + i, now - 100 + i))
    assert cache.get("txt_v1_k0") is not None  # touched: now the most recently used
    size = cache.path_for("txt_v1_k0").stat().st_size
    small = PersistentConditioningCache(str(tmp_path), "fam", 64, max_bytes = size * 3)
    _put(small, "txt_v1_new")
    left = {p.stem for p in tmp_path.glob("*.safetensors")}
    assert "txt_v1_k0" in left and "txt_v1_new" in left
    assert "txt_v1_k1" not in left


def test_eviction_never_touches_files_the_cache_did_not_write(tmp_path):
    (tmp_path / "notes.safetensors").write_bytes(b"x" * 100_000)
    (tmp_path / "readme.txt").write_text("keep")
    cache = PersistentConditioningCache(str(tmp_path), "fam", 64, max_bytes = 1)
    _put(cache, "txt_v1_a")
    assert (tmp_path / "notes.safetensors").is_file() and (tmp_path / "readme.txt").is_file()


def test_a_write_that_would_fill_the_disk_is_skipped(monkeypatch, tmp_path):
    import shutil

    cache = PersistentConditioningCache(str(tmp_path), "fam", 64)
    real = shutil.disk_usage
    monkeypatch.setattr(
        shutil, "disk_usage", lambda p: real(p)._replace(free = extras._COND_CACHE_DISK_RESERVE_BYTES)
    )
    cache.put("txt_v1_a", (torch.ones(8),))
    assert not cache.has("txt_v1_a")
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(shutil, "disk_usage", real)
    cache.put("txt_v1_a", (torch.ones(8),))
    assert torch.equal(cache.get("txt_v1_a")[0], torch.ones(8))


def test_a_failed_write_leaves_no_partial_file(monkeypatch, tmp_path):
    import safetensors.torch

    def full_disk(
        tensors,
        filename,
        metadata = None,
    ):
        Path(filename).write_bytes(b"partial")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(safetensors.torch, "save_file", full_disk)
    cache = PersistentConditioningCache(str(tmp_path), "fam", 64)
    with pytest.raises(OSError):
        cache.put("txt_v1_a", (torch.ones(8),))
    assert list(tmp_path.iterdir()) == []


def test_an_off_value_passes_a_managed_accounts_path_check(monkeypatch, tmp_path):
    # The start route writes "off" for the trainer; it names no directory, so the account path check must not 403 it.
    from fastapi import HTTPException

    from auth import policy
    from core.training.account_jobs import validate_job_paths
    from utils.account_context import AccountContext, run_as

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    alice = AccountContext("alice-id", "alice")
    for value in ("off", "OFF", "none"):
        run_as(alice, validate_job_paths, {"cond_cache_dir": value})
    run_as(
        alice,
        validate_job_paths,
        {"cond_cache_dir": str(run_as(alice, extras.default_cond_cache_dir))},
    )
    with pytest.raises(HTTPException):
        run_as(alice, validate_job_paths, {"cond_cache_dir": str(tmp_path / "elsewhere")})


def test_the_cap_reads_its_env_and_zero_means_unbounded(monkeypatch):
    monkeypatch.setenv(extras.TRAIN_COND_CACHE_MAX_GB_ENV, "1.5")
    assert extras.cond_cache_max_bytes() == int(1.5 * 1024**3)
    monkeypatch.setenv(extras.TRAIN_COND_CACHE_MAX_GB_ENV, "0")
    assert extras.cond_cache_max_bytes() == 0
    monkeypatch.delenv(extras.TRAIN_COND_CACHE_MAX_GB_ENV)
    assert extras.cond_cache_max_bytes() == 20 * 1024**3


def test_a_long_local_base_path_namespace_still_fits_a_file_name(tmp_path):
    # Default-on puts local-path bases (tests, offline installs) through the cache: the key must stay < 255 bytes.
    family = "qwen-image-2.1_" + "/very/long/local/path" * 15 + "_dir-0123456789abcdef"
    cache = PersistentConditioningCache(str(tmp_path), family, 64, text_encoder = "te")
    key = cache.text_key("a cat")
    assert len(cache.path_for(key).name) < 255
    cache.put(key, (torch.ones(3),))
    assert torch.equal(cache.get(key)[0], torch.ones(3))
    other = PersistentConditioningCache(
        str(tmp_path), family.replace("0123", "9999"), 64, text_encoder = "te"
    )
    assert other.text_key("a cat") != key, "the hashed tail still separates namespaces"
    # A short namespace keeps its readable key unchanged.
    assert (
        PersistentConditioningCache(str(tmp_path), "flux.1_org-m_rev-1", 64).family
        == "flux.1-org-m-rev-1"
    )


# ---------------------------------------------------------------- DiT: default on


def _dit_until_transformer(monkeypatch, tmp_path, cond_cache_dir):
    """_train_dit through the conditioning phase; returns (pcache roots seen by the warm check, conditioner loads)."""
    from core.training import diffusion_dit_trainer as dit

    for name, mod in (
        ("diffusers", SimpleNamespace(FlowMatchEulerDiscreteScheduler = object)),
        ("diffusers.optimization", SimpleNamespace(get_scheduler = object)),
        ("diffusers.training_utils", SimpleNamespace(cast_training_params = object)),
        ("peft", SimpleNamespace(LoraConfig = object)),
        ("peft.utils", SimpleNamespace(get_peft_model_state_dict = object)),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr("loggers.config.quiet_third_party_progress_bars", lambda: None)
    monkeypatch.setattr(dit, "plan_image_canvases", lambda *_a: (None, False))
    monkeypatch.setattr(dit, "_plan_cache_variants", lambda *_a, **_k: [[(0.5, 0.5, False)]])
    monkeypatch.setattr(dit, "_resolve_base_precision", lambda *_a: "bf16")
    monkeypatch.setattr(dit, "isolated_sampling", lambda *_a: contextlib.nullcontext())
    monkeypatch.setattr(extras, "source_revision", lambda *_a: "local")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_LATENT_CACHE", raising = False)
    roots: list = []
    loads: list = []

    def warm(pcache, *_a, **_k):
        roots.append(pcache.root)
        return {"caption": torch.zeros(1)}, [[(torch.zeros(1), None)]]

    monkeypatch.setattr(dit, "_load_warm_conditioning", warm)

    class Reached(Exception):
        pass

    def transformer(*_a):
        raise Reached

    spec = SimpleNamespace(
        family = "flux.1",
        lora_targets = ("to_q",),
        decode_latents = False,
        load_conditioners = lambda *_a: loads.append(1) or (SimpleNamespace(), None),
        load_transformer = transformer,
    )
    cfg = DiffusionLoraConfig(
        base_model = "unsloth/FLUX.1-dev",
        data_dir = str(tmp_path),
        output_dir = str(tmp_path / "out"),
        cache_latents = True,
        cond_cache_dir = cond_cache_dir,
    )
    monkeypatch.setattr(dit, "_build_latent_cache", lambda *_a, **_k: [[(torch.zeros(1), None)]])
    monkeypatch.setattr(
        dit, "_encode_prompts_cached", lambda spec, pipe, texts, *_a: [torch.zeros(1)] * len(texts)
    )
    monkeypatch.setattr(dit, "_free_text_encoders", lambda *_a: None)
    with pytest.raises(Reached):
        dit._train_dit(
            cfg,
            spec,
            [(str(tmp_path / "image.png"), "caption")],
            random.Random(0),
            "cpu",
            None,
            lambda *_a, **_k: None,
            lambda: False,
            lambda: True,
            None,
        )
    return roots, loads


def test_the_dit_trainer_uses_the_default_cache_when_unset(monkeypatch, tmp_path):
    roots, loads = _dit_until_transformer(monkeypatch, tmp_path, None)
    assert roots == [extras.default_cond_cache_dir()]
    assert loads == [], "a warm default cache must skip the conditioner load"


def test_the_dit_trainer_honours_off(monkeypatch, tmp_path):
    roots, loads = _dit_until_transformer(monkeypatch, tmp_path, "off")
    assert roots == [] and loads == [1]


# ---------------------------------------------------------------- MiniMax-H3 video


def _h3_cfg(**kw):
    base = dict(
        base_model = "MiniMaxAI/MiniMax-H3",
        data_dir = "/tmp/clips",
        output_dir = "/tmp/out",
        mixed_precision = "bf16",
        resolution = 768,
    )
    base.update(kw)
    return DiffusionLoraConfig(**base).normalized()


class _H3Fakes:
    """Counts loads and encodes; random encodes differ, so only a cache hit can match."""

    def __init__(self, monkeypatch, num_audio):
        from core.training import diffusion_h3_trainer as h3

        self.h3 = h3
        self.loads = 0
        self.prompt_encodes = 0
        self.clip_encodes = 0
        gen = torch.Generator().manual_seed(1234)

        class _Placed:
            def to(self, _device):
                return self

        def load(cfg, device, **_k):
            self.loads += 1
            return SimpleNamespace(
                text_encoder = _Placed(), processor = None, vae = _Placed(), audio_vae = _Placed()
            )

        def prompt(pipe, caption, device):
            self.prompt_encodes += 1
            return torch.randn(1, 5, 16, generator = gen)

        def video(*_a):
            self.clip_encodes += 1
            return torch.randn(1, 4, 2, 8, 8, generator = gen), torch.rand(
                1, 4, 2, 8, 8, generator = gen
            )

        monkeypatch.setattr(h3, "_load_conditioners", load)
        monkeypatch.setattr(h3, "_encode_prompt", prompt)
        monkeypatch.setattr(h3, "decode_clip", lambda *_a, **_k: (None, None))
        monkeypatch.setattr(h3, "_encode_video_stats", video)
        monkeypatch.setattr(
            h3,
            "_encode_audio_latents",
            lambda *_a: torch.randn(1, h3.H3_AUDIO_LATENT_CHANNELS, num_audio, generator = gen),
        )


def _h3_clips(tmp_path, n = 2):
    d = tmp_path / "clips"
    d.mkdir(exist_ok = True)
    paths = []
    for i in range(n):
        p = d / f"{i}.mp4"
        p.write_bytes(f"clip-{i}".encode() * 50)
        paths.append(str(p))
    return paths


def _h3_prepare(cfg, paths, captions, **kw):
    from core.training import diffusion_h3_trainer as h3

    geom = dict(
        width = 64,
        height = 64,
        num_frames = h3.H3_TRAIN_NUM_FRAMES,
        num_audio_latents = h3.h3_audio_latent_count(h3.H3_TRAIN_NUM_FRAMES),
    )
    geom.update(kw)
    return h3._prepare_h3_conditioning(
        cfg, paths, captions, "cpu", lambda *_a, **_k: None, lambda: False, **geom
    )


def _num_audio():
    from core.training import diffusion_h3_trainer as h3
    return h3.h3_audio_latent_count(h3.H3_TRAIN_NUM_FRAMES)


def test_h3_no_longer_refuses_a_cond_cache_dir():
    from core.training.diffusion_train_common import h3_train_unsupported_reason
    assert h3_train_unsupported_reason(_h3_cfg(cond_cache_dir = "/tmp/cond")) is None


def test_h3_warm_rerun_is_bit_identical_and_never_loads_the_conditioner(monkeypatch, tmp_path):
    fakes = _H3Fakes(monkeypatch, _num_audio())
    paths = _h3_clips(tmp_path)
    captions = ["a dog barks", "rain on a roof"]
    cfg = _h3_cfg()  # unset: the default cache
    first = _h3_prepare(cfg, paths, captions)
    assert fakes.loads == 1 and fakes.prompt_encodes == 2 and fakes.clip_encodes == 2
    default = extras.default_cond_cache_dir()
    assert any(default.glob("h3clip_v*.safetensors")) and any(default.glob("txt_v*.safetensors"))

    second = _h3_prepare(cfg, paths, captions)
    assert fakes.loads == 1, "a warm cache must not load the 25-66 GB conditioner"
    assert fakes.prompt_encodes == 2 and fakes.clip_encodes == 2
    embeds_a, cache_a = first
    embeds_b, cache_b = second
    assert embeds_a.keys() == embeds_b.keys()
    for cap in embeds_a:
        assert embeds_a[cap].dtype == embeds_b[cap].dtype and torch.equal(
            embeds_a[cap], embeds_b[cap]
        )
    for ea, eb in zip(cache_a, cache_b):
        assert len(ea) == len(eb) == 3
        for a, b in zip(ea, eb):
            assert a.dtype == b.dtype and torch.equal(a, b)


@pytest.mark.parametrize("partial", [False, True])
def test_h3_an_over_budget_warm_cache_is_refused_before_it_is_held(monkeypatch, tmp_path, partial):
    fakes = _H3Fakes(monkeypatch, _num_audio())
    h3 = fakes.h3
    paths = _h3_clips(tmp_path, n = 4)
    captions = ["a dog barks"] * 4
    cfg = _h3_cfg()
    _h3_prepare(cfg, paths, captions)
    if partial:
        for entry in extras.default_cond_cache_dir().glob("h3clip_v*.safetensors"):
            if extras._file_content_hash(paths[0]) in entry.name:
                entry.unlink()
    held = []
    real_get = h3._cache_get

    def counting_get(pcache, key):
        hit = real_get(pcache, key)
        if hit is not None and key is not None and key.startswith("h3clip"):
            held.append(key)
        return hit

    monkeypatch.setattr(h3, "_cache_get", counting_get)
    monkeypatch.setattr(h3, "_latent_cache_over_budget", lambda *_a, **_k: True)
    with pytest.raises(ValueError, match = "latent cache"):
        _h3_prepare(cfg, paths, captions)
    assert (
        len(held) == 1
    ), "the budget check must run on the first cached clip, before the rest are loaded"


def test_h3_off_writes_nothing_and_re_encodes(monkeypatch, tmp_path):
    fakes = _H3Fakes(monkeypatch, _num_audio())
    paths = _h3_clips(tmp_path)
    cfg = _h3_cfg(cond_cache_dir = "off")
    _h3_prepare(cfg, paths, ["a"])
    _h3_prepare(cfg, paths, ["a"])
    assert fakes.loads == 2
    default = extras.default_cond_cache_dir()
    assert not default.exists() or not any(default.iterdir())


@pytest.mark.parametrize(
    "change",
    ["caption", "clip_bytes", "canvas", "num_frames", "audio_layout", "revision", "version"],
)
def test_h3_every_input_that_changes_the_tensors_misses(monkeypatch, tmp_path, change):
    from core.training import diffusion_h3_trainer as h3

    fakes = _H3Fakes(monkeypatch, _num_audio())
    paths = _h3_clips(tmp_path, n = 1)
    captions = ["a dog barks"]
    cfg = _h3_cfg(cond_cache_dir = str(tmp_path / "cc"))
    monkeypatch.setattr(extras, "source_revision", lambda *_a: "rev-1")
    _h3_prepare(cfg, paths, captions)
    _h3_prepare(cfg, paths, captions)
    assert fakes.loads == 1  # warm baseline
    kw = {}
    if change == "caption":
        captions = ["a dog barks loudly"]
    elif change == "clip_bytes":
        Path(paths[0]).write_bytes(b"edited")
    elif change == "canvas":
        kw = dict(width = 96, height = 64)
    elif change == "num_frames":
        kw = dict(num_frames = h3.H3_TRAIN_NUM_FRAMES + 17)
    elif change == "audio_layout":
        n = _num_audio() + 40
        kw = dict(num_audio_latents = n)
        monkeypatch.setattr(
            h3, "_encode_audio_latents", lambda *_a: torch.zeros(1, h3.H3_AUDIO_LATENT_CHANNELS, n)
        )
    elif change == "revision":
        monkeypatch.setattr(extras, "source_revision", lambda *_a: "rev-2")
    elif change == "version":
        monkeypatch.setattr(h3, "_H3_CLIP_CACHE_VERSION", "999")
    before = (fakes.prompt_encodes, fakes.clip_encodes)
    _h3_prepare(cfg, paths, captions, **kw)
    assert fakes.loads == 2, f"{change} must miss"
    if change == "caption":
        assert fakes.prompt_encodes == before[0] + 1 and fakes.clip_encodes == before[1]
    elif change in ("revision",):
        assert fakes.prompt_encodes == before[0] + 1 and fakes.clip_encodes == before[1] + 1
    else:
        assert fakes.clip_encodes == before[1] + 1


def test_h3_an_unresolved_first_run_is_found_by_the_resolved_second_run(monkeypatch, tmp_path):
    fakes = _H3Fakes(monkeypatch, _num_audio())
    paths = _h3_clips(tmp_path, n = 1)
    state = {"rev": "unresolved"}
    monkeypatch.setattr(extras, "source_revision", lambda *_a: state["rev"])
    load = fakes.h3._load_conditioners

    def downloading_load(cfg, device, **kw):
        state["rev"] = "rev-abc"  # the fetch writes refs/main
        return load(cfg, device, **kw)

    monkeypatch.setattr(fakes.h3, "_load_conditioners", downloading_load)
    cfg = _h3_cfg()
    _h3_prepare(cfg, paths, ["a"])
    _h3_prepare(cfg, paths, ["a"])
    assert fakes.loads == 1, "run 2 reads the resolved revision and must find what run 1 wrote"


def test_h3_train_hands_the_cached_tensors_to_phase_3(monkeypatch, tmp_path):
    """_train_h3 itself goes through the cache: the second run reaches the transformer with no conditioner load."""
    from core.training import diffusion_h3_trainer as h3

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_diffusion_h3_trainer import _stub_training_stack

    _stub_training_stack(monkeypatch)
    fakes = _H3Fakes(monkeypatch, _num_audio())
    monkeypatch.setattr(h3, "_dataset_canvas", lambda _p, _s: (64, 64))

    def _no_transformer(*_a, **_k):
        raise RuntimeError("reached phase 3")

    monkeypatch.setattr(h3, "_load_transformer", _no_transformer)
    paths = _h3_clips(tmp_path)
    cfg = _h3_cfg(output_dir = str(tmp_path / "out"))
    for _ in range(2):
        with pytest.raises(RuntimeError, match = "reached phase 3"):
            h3._train_h3(
                cfg,
                [(p, "a caption") for p in paths],
                random.Random(0),
                "cpu",
                torch.float32,
                lambda *_a, **_k: None,
                lambda: False,
                lambda: True,
            )
    assert fakes.loads == 1


# ---------------------------------------------------------------- MiniMax-H3-Image


def test_h3_image_no_longer_refuses_a_cond_cache_dir():
    from core.training.diffusion_train_common import h3_train_unsupported_reason
    cfg = DiffusionLoraConfig(
        base_model = "MiniMaxAI/MiniMax-H3-Image",
        data_dir = "/tmp/d",
        output_dir = "/tmp/o",
        instance_prompt = "p",
        cond_cache_dir = "/tmp/cond",
    ).normalized()
    assert h3_train_unsupported_reason(cfg) is None


def test_h3_image_keys_miss_on_encoder_scheme_revision_resolution_and_bytes(monkeypatch, tmp_path):
    from core.training import diffusion_h3_image_trainer as h3i_t

    root = tmp_path / "release"
    root.mkdir()
    pic = tmp_path / "a.png"
    pic.write_bytes(b"picture")
    cfg = SimpleNamespace(
        cond_cache_dir = str(tmp_path / "cc"),
        base_model = "MiniMaxAI/MiniMax-H3-Image",
        fetch_base_model = None,
        resolution = 64,
    )
    revs = {"MiniMaxAI/MiniMax-H3-Image": "rev-base"}
    monkeypatch.setattr(extras, "source_revision", lambda ref: revs.get(ref, "rev-te"))

    def keys():
        c = h3i_t._open_h3_image_cache(cfg, root, lambda *_a, **_k: None)
        return c.text_key("a cat"), h3i_t._picture_key(c, str(pic), cfg.resolution)

    text0, lat0 = keys()
    assert keys() == (text0, lat0)
    monkeypatch.setenv(h3i_t.TEXT_ENCODER_ENV, "bf16")
    text1, lat1 = keys()
    assert (
        text1 != text0 and lat1 == lat0
    ), "the int8 and bf16 conditioners encode different embeddings"
    monkeypatch.delenv(h3i_t.TEXT_ENCODER_ENV)
    monkeypatch.setattr(extras, "source_revision", lambda ref: revs.get(ref, "rev-te-2"))
    assert keys()[0] != text0, "a new conditioner revision must miss"
    monkeypatch.setattr(extras, "source_revision", lambda ref: "rev-base-2")
    t, lat = keys()
    assert t != text0 and lat != lat0, "a new base revision must miss"
    monkeypatch.setattr(extras, "source_revision", lambda ref: revs.get(ref, "rev-te"))
    cfg.resolution = 96
    assert keys()[1] != lat0
    cfg.resolution = 64
    pic.write_bytes(b"edited")
    assert keys()[1] != lat0
    pic.write_bytes(b"picture")
    monkeypatch.setattr(h3i_t, "_H3_IMAGE_LATENT_CACHE_VERSION", "999")
    assert keys()[1] != lat0


def _h3_image_stack():
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    pytest.importorskip("peft")


@pytest.fixture(scope = "module")
def h3_image_release(tmp_path_factory):
    _h3_image_stack()
    import _minimax_h3_image_tiny as tiny

    root = tmp_path_factory.mktemp("h3img_cc_release")
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    return root, ref, te


def _h3_image_run(release, data, out, monkeypatch, loads, **extra):
    from core.training import diffusion_h3_image_trainer as trainer

    root, ref, te = release

    def load(cfg, r):
        loads.append(1)
        return te, ref["tokenizer"], ref["processor"]

    monkeypatch.setattr(trainer, "load_text_encoder", load)
    monkeypatch.setattr(trainer, "release_text_encoder", lambda m: None)
    monkeypatch.setattr(trainer, "resolve_train_device", lambda: "cpu")
    monkeypatch.setattr("core.inference.diffusion_lora.loras_dir", lambda: out.parent / "catalog")
    monkeypatch.setenv(trainer.TRAINING_ADAPTER_ENV, "off")
    events = []
    cfg = DiffusionLoraConfig(
        base_model = str(root),
        data_dir = str(data),
        output_dir = str(out),
        model_family = "minimax-h3-image",
        resolution = 64,
        train_steps = 4,
        learning_rate = 3e-3,
        lora_rank = 4,
        lr_scheduler = "constant",
        seed = 0,
        **extra,
    )
    trainer.run_h3_image_lora_training(cfg, on_event = events.append)
    return events


def test_h3_image_warm_rerun_skips_the_conditioner_and_trains_identically(
    h3_image_release, tmp_path, monkeypatch
):
    import numpy as np
    from PIL import Image

    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (96, 64, 3), dtype = np.uint8)).save(data / "a.png")
    (data / "a.txt").write_text("a cat")
    (tmp_path / "catalog").mkdir()
    loads: list = []
    # Previews stay on: their prompt is cached with the captions, and the VAE still loads for the decode.
    extra = dict(sample_every = 2, sample_prompts = ["a red cube"])
    cold = _h3_image_run(h3_image_release, data, tmp_path / "o1", monkeypatch, loads, **extra)
    assert len(loads) == 1
    assert any(extras.default_cond_cache_dir().glob("h3img_v*.safetensors"))
    warm = _h3_image_run(h3_image_release, data, tmp_path / "o2", monkeypatch, loads, **extra)
    assert len(loads) == 1, "a warm cache must not load the conditioner"
    loss = lambda ev: [e["loss"] for e in ev if e["type"] == "progress"]  # noqa: E731
    assert loss(cold) and loss(cold) == loss(warm), "the warm run must train bit-identically"
    samples = lambda ev: [e for e in ev if e["type"] == "sample"]  # noqa: E731
    assert len(samples(warm)) == len(samples(cold)) > 0, "previews must still render on a warm hit"
    from safetensors.torch import load_file

    a = load_file(next((tmp_path / "o1").rglob("pytorch_lora_weights.safetensors")))
    b = load_file(next((tmp_path / "o2").rglob("pytorch_lora_weights.safetensors")))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_h3_image_bf16_fallback_captions_never_serve_an_int8_run(
    h3_image_release, tmp_path, monkeypatch
):
    import numpy as np
    from PIL import Image

    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (96, 64, 3), dtype = np.uint8)).save(data / "a.png")
    (data / "a.txt").write_text("a cat")
    (tmp_path / "catalog").mkdir()
    te = h3_image_release[2]
    loads: list = []
    try:
        # int8 requested (the default), but the hosted file was unusable and bf16 encoded the captions.
        te._unsloth_te_scheme = "bf16"
        _h3_image_run(h3_image_release, data, tmp_path / "o1", monkeypatch, loads)
        assert len(loads) == 1
        te._unsloth_te_scheme = "int8"
        _h3_image_run(h3_image_release, data, tmp_path / "o2", monkeypatch, loads)
        assert len(loads) == 2, "bf16 fallback embeddings must not be served to an int8 run"
        _h3_image_run(h3_image_release, data, tmp_path / "o3", monkeypatch, loads)
        assert len(loads) == 2, "the int8 run's own entries must serve the next int8 run"
    finally:
        te.__dict__.pop("_unsloth_te_scheme", None)


def test_eviction_never_touches_a_user_file_named_like_a_cache_key(tmp_path):
    from safetensors.torch import save_file

    # A user's own tensors in an explicitly chosen directory, named with a cache kind prefix.
    for name in (
        "txt_vintage",
        "txt_v1_my_embeddings",
        "lat_v2_style",
        "h3clip_v1_x",
        "h3img_v1_y",
    ):
        save_file(
            {"w": torch.ones(4096)},
            str(tmp_path / f"{name}.safetensors"),
            metadata = {"format": "pt"},
        )
    save_file(
        {"t0": torch.ones(4096)},
        str(tmp_path / "txt_v1_lookalike.safetensors"),
        metadata = {"version": "1"},
    )
    user = sorted(p.name for p in tmp_path.iterdir())
    cache = PersistentConditioningCache(str(tmp_path), "fam", 64, max_bytes = 1)
    _put(cache, "txt_v1_a")
    _put(cache, "txt_v1_b")
    left = sorted(p.name for p in tmp_path.iterdir())
    assert all(name in left for name in user), (user, left)
    # The cache still evicts its own entries under the cap.
    assert "txt_v1_a.safetensors" not in left


def test_entries_written_before_the_owner_marker_are_still_evicted(tmp_path):
    from safetensors.torch import save_file

    # The exact metadata every entry carried before the owner marker was added.
    legacy = {"version": "1", "family": "fam", "none_slots": "[1]", "count": "2"}
    save_file({"t0": torch.zeros(1024)}, str(tmp_path / "txt_v1_old.safetensors"), metadata = legacy)
    old = time.time() - 1000
    os.utime(tmp_path / "txt_v1_old.safetensors", (old, old))
    cache = PersistentConditioningCache(str(tmp_path), "fam", 64, max_bytes = 6000)
    _put(cache, "txt_v1_new")
    _put(cache, "txt_v1_newer")
    left = {p.stem for p in tmp_path.glob("*.safetensors")}
    assert "txt_v1_old" not in left and "txt_v1_newer" in left
