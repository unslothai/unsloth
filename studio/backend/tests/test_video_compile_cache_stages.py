# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stage-keyed compile cache for multi-denoiser (video) loads: ``diffusion_compile_cache.begin_stages``.

The Mega-cache API is replaced by in-memory fakes; the stage fingerprint runs against real nn.Modules on CPU."""

from __future__ import annotations

import os
import types

import pytest
import torch

from core.inference import diffusion_compile_cache as cc


class _Block(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)


class _DiT(torch.nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(
        self,
        dim: int = 8,
        layers: int = 2,
        armed: bool = True,
        dynamic = True,
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Block(dim) for _ in range(layers))
        self.config = {"dim": dim, "num_layers": layers}
        if armed:
            self._unsloth_regional_compile_kwargs = {"fullgraph": True, "dynamic": dynamic}


_KW = dict(dtype = "torch.bfloat16", quant = None, attention_backend = "_native_cudnn")


@pytest.fixture
def fake_megacache(monkeypatch):
    state = {"loaded_with": None, "saves": 0}

    def fake_save():
        state["saves"] += 1
        return (b"STAGE-ARTIFACTS", None)

    def fake_load(data: bytes):
        state["loaded_with"] = data
        return object() if data == b"STAGE-ARTIFACTS" else None

    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", fake_save, raising = False)
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", fake_load, raising = False)
    return state


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SAVE, raising = False)
    monkeypatch.setenv(cc._ENV_SYNC, "1")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path / "bundles"))
    return tmp_path


def _key(stages, **kw):
    merged = {**_KW, **kw}
    env = cc.environment_fingerprint()
    model = {
        "kind": "stages",
        "stages": [cc.stage_fingerprint(s, **merged) for s in stages],
    }
    return cc.cache_key(env, model)


# ------------------------------------------------------------------------------------------ stage discovery
def test_compiled_stages_finds_both_experts_once_and_skips_unarmed():
    high, low, encoder = _DiT(), _DiT(), _DiT(armed = False)
    pipe = types.SimpleNamespace(
        transformer = high,
        transformer_2 = low,
        components = {"transformer": high, "transformer_2": low, "text_encoder": encoder},
    )
    view = types.SimpleNamespace(transformer = low)
    assert cc.compiled_stages(pipe, view, None) == [high, low]


def test_compiled_stages_empty_when_nothing_is_armed():
    pipe = types.SimpleNamespace(transformer = _DiT(armed = False), components = {})
    assert cc.compiled_stages(pipe) == []


# ------------------------------------------------------------------------------------------ the key
def test_stage_key_ignores_the_family_name():
    """Two families sharing one class + config + compile settings trace one graph, so they share one key."""
    fp = cc.stage_fingerprint(_DiT(), **_KW)
    assert "family" not in fp
    assert _key([_DiT()]) == _key([_DiT()])


@pytest.mark.parametrize(
    "other",
    [
        lambda: [_DiT(dim = 16)],  # architecture config
        lambda: [_DiT(layers = 3)],
        lambda: [_DiT(dynamic = None)],  # the compile kwargs the regional compile really used
        lambda: [_DiT(), _DiT()],  # a second expert stage
    ],
)
def test_stage_key_tracks_the_graph(other):
    assert _key(other()) != _key([_DiT()])


@pytest.mark.parametrize(
    "field,value",
    [("dtype", "torch.float16"), ("quant", "int8"), ("attention_backend", "flash")],
)
def test_stage_key_tracks_dtype_quant_and_attention(field, value):
    assert _key([_DiT()], **{field: value}) != _key([_DiT()])


def test_driver_update_starts_a_new_bundle(cache_dir, fake_megacache, monkeypatch):
    monkeypatch.setattr(cc, "_driver_version", lambda: "570.1")
    a = cc.begin_stages([_DiT()], **_KW)
    monkeypatch.setattr(cc, "_driver_version", lambda: "580.2")
    b = cc.begin_stages([_DiT()], **_KW)
    assert a.key != b.key


# ------------------------------------------------------------------------------------------ lifecycle
def test_begin_stages_never_moves_the_process_inductor_dir(cache_dir, fake_megacache, monkeypatch):
    """The reason video declined the image cache: a second backend may be compiling under the process dir."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/shared/inductor")
    ctx = cc.begin_stages([_DiT()], **_KW)
    assert ctx is not None and ctx.prev_inductor_dir_set is False
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/shared/inductor"
    cc.save(ctx)
    cc.restore(ctx)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/shared/inductor"


def test_first_load_saves_and_the_next_process_hits(cache_dir, fake_megacache):
    ctx = cc.begin_stages([_DiT(), _DiT()], **_KW)
    assert ctx is not None and ctx.hit is False
    assert cc.save_async(ctx) is True  # SYNC env: written inline
    assert ctx.saved and ctx.manifest_path.exists()
    cc.restore(ctx)

    again = cc.begin_stages([_DiT(), _DiT()], **_KW)
    assert again.hit is True and again.key == ctx.key
    assert fake_megacache["loaded_with"] == b"STAGE-ARTIFACTS"
    # A hit starts clean: nothing new compiled, nothing rewritten.
    saves = fake_megacache["saves"]
    assert cc.save_async(again) is False
    assert fake_megacache["saves"] == saves
    # A render that compiled new graphs (a new resolution under a static compile) re-dirties it.
    cc.mark_recompiled(again)
    assert cc.save_async(again) is True
    assert fake_megacache["saves"] == saves + 1


def test_a_different_stage_set_misses(cache_dir, fake_megacache):
    ctx = cc.begin_stages([_DiT()], **_KW)
    cc.save(ctx)
    other = cc.begin_stages([_DiT(dim = 32)], **_KW)
    assert other.hit is False and other.key != ctx.key


def test_disabled_or_nothing_compiled_returns_none(cache_dir, fake_megacache, monkeypatch):
    assert cc.begin_stages([], **_KW) is None
    assert cc.begin_stages([None], **_KW) is None
    monkeypatch.setenv(cc._ENV_MODE, "0")
    assert cc.begin_stages([_DiT()], **_KW) is None


def test_no_megacache_api_returns_none(cache_dir, monkeypatch):
    monkeypatch.delattr(torch.compiler, "save_cache_artifacts", raising = False)
    assert cc.begin_stages([_DiT()], **_KW) is None


def test_a_corrupt_bundle_is_a_miss_and_the_next_save_repairs_it(cache_dir, fake_megacache):
    ctx = cc.begin_stages([_DiT()], **_KW)
    cc.save(ctx)
    ctx.bundle.write_bytes(b"garbage")
    again = cc.begin_stages([_DiT()], **_KW)
    assert again.hit is False
    assert cc.save(again) is True
    third = cc.begin_stages([_DiT()], **_KW)
    assert third.hit is True


# ------------------------------------------------------------------------------------------ video wiring
def test_video_backend_opens_one_context_for_every_stage(cache_dir, fake_megacache):
    from core.inference import video as V

    backend = V.VideoBackend.__new__(V.VideoBackend)
    high, low = _DiT(), _DiT()
    pipe = types.SimpleNamespace(transformer = high, transformer_2 = low, components = {})
    ctx, reason = backend._begin_stage_compile_cache(
        token = 7,
        roots = (pipe,),
        speed_optims = ("compiled",),
        dtype = "torch.bfloat16",
        quant = None,
        attention_backend = None,
    )
    assert reason is None and ctx is not None
    assert len(ctx.model_fp["stages"]) == 2
    assert backend._precommit_compile_caches == {7: ctx}
    # A load that dies before its commit hands the context back.
    backend._release_precommit_compile_cache(7)
    assert backend._precommit_compile_caches == {}


def test_video_backend_records_why_it_did_not_engage(cache_dir, fake_megacache):
    from core.inference import video as V

    backend = V.VideoBackend.__new__(V.VideoBackend)
    pipe = types.SimpleNamespace(transformer = _DiT(), components = {})
    ctx, reason = backend._begin_stage_compile_cache(
        token = None, roots = (pipe,), speed_optims = (), dtype = None, quant = None, attention_backend = None
    )
    assert ctx is None and "nothing compiled" in reason


def test_overlapping_loads_each_release_their_own_context(cache_dir, fake_megacache):
    """A superseded load that dies before its commit must release ITS context even after a newer load opened one."""
    from core.inference import video as V

    backend = V.VideoBackend.__new__(V.VideoBackend)

    def begin(token, dim):
        pipe = types.SimpleNamespace(transformer = _DiT(dim = dim), components = {})
        ctx, _ = backend._begin_stage_compile_cache(
            token = token,
            roots = (pipe,),
            speed_optims = ("compiled",),
            dtype = None,
            quant = None,
            attention_backend = None,
        )
        return ctx

    old, new = begin(1, 8), begin(2, 16)
    assert str(old.dir) in cc._live_dirs and str(new.dir) in cc._live_dirs
    backend._release_precommit_compile_cache(1)  # the stale load fails
    assert str(old.dir) not in cc._live_dirs
    assert str(new.dir) in cc._live_dirs
    backend._commit_precommit_compile_cache(
        2
    )  # the new load commits: its state owns the context now
    assert backend._precommit_compile_caches == {}
    assert str(new.dir) in cc._live_dirs
    cc.restore(new)


@pytest.mark.parametrize(
    "name,value",
    [
        ("UNSLOTH_DIFFUSION_WAN_FUSED_ADALN", "0"),
        ("UNSLOTH_DIFFUSION_WAN_FUSED_ROPE", "0"),
        ("UNSLOTH_DIFFUSION_PATCH_WAN_FUSED", "0"),
    ],
)
def test_wan_fusion_switches_key_bundles_apart(monkeypatch, name, value):
    """The fused Wan block is a class-level forward swap the module tree does not show, so its switches key it."""
    for env in (
        "UNSLOTH_DIFFUSION_WAN_FUSED_ADALN",
        "UNSLOTH_DIFFUSION_WAN_FUSED_ROPE",
        "UNSLOTH_DIFFUSION_PATCH_WAN_FUSED",
    ):
        monkeypatch.delenv(env, raising = False)
    stock = _key([_DiT()])
    monkeypatch.setenv(name, "1")  # the default value keeps every existing key
    assert _key([_DiT()]) == stock
    monkeypatch.setenv(name, value)
    assert _key([_DiT()]) != stock


def test_no_bundle_save_from_a_render_the_background_compile_forced_eager():
    """Such a render compiled nothing of this load: saving would publish other loads' artifacts and mark it saved."""
    import inspect

    from core.inference import video as V

    src = inspect.getsource(V.VideoBackend.generate)
    save = src.index("compile_cache.save_async(compile_ctx")
    guard = src.rindex("if compile_ctx is not None", 0, save)
    assert "not bg_eager" in src[guard:save]
