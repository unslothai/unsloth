# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tiny Wan video decoder: checkpoint routing by latent width, the TAEHV key layout, decode shapes, the opt-in
draft decode (off by default, swaps and restores vae.decode) and the live previews it feeds."""

from __future__ import annotations

import threading
import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_preview as DP
from core.inference import video_tiny_vae as TV


def _random_checkpoint(tmp_path, channels, patch, name):
    from safetensors.torch import save_file

    model = TV._build(channels, patch)
    gen = torch.Generator().manual_seed(0)
    state = {
        k: torch.randn(v.shape, generator = gen).mul_(0.05).half()
        for k, v in model.state_dict().items()
    }
    # the released files carry the encoder too; the loader must drop it
    state["encoder.0.weight"] = torch.zeros(64, 3 * patch * patch, 3, 3, dtype = torch.float16)
    path = tmp_path / name
    save_file(state, str(path))
    return str(path)


def test_routing_by_family_and_latent_width():
    assert TV.checkpoint_for("wan2.2-ti2v-5b", 48) == ("taew2_2.safetensors", 2)
    for fam in ("wan2.1-t2v-1.3b", "wan2.1-t2v-14b", "wan2.2-t2v-a14b"):
        assert TV.checkpoint_for(fam, 16) == ("taew2_1.safetensors", 1)
        assert TV.family_latent_channels(fam) == 16
    assert TV.family_latent_channels("wan2.2-ti2v-5b") == 48
    # another family's latents would decode to noise, so no tiny decoder there
    for fam, ch in (("hunyuanvideo-1.5", 32), ("ltx-2", 128), ("cosmos3-nano", 16), (None, 16)):
        assert TV.checkpoint_for(fam, ch) is None
    assert TV.checkpoint_for("wan2.2-ti2v-5b", 32) is None


def test_decoder_matches_the_released_taehv_key_layout():
    for channels, patch in ((16, 1), (48, 2)):
        sd = {k: tuple(v.shape) for k, v in TV._build(channels, patch).state_dict().items()}
        # 64 decoder tensors in taew2_1 / taew2_2 / lighttaew2_2_v2 (read from the released headers)
        assert len(sd) == 64
        assert sd["decoder.1.weight"] == (256, channels, 3, 3)
        assert sd["decoder.7.conv.weight"] == (256, 256, 1, 1)
        assert sd["decoder.13.conv.weight"] == (256, 128, 1, 1)
        assert sd["decoder.19.conv.weight"] == (128, 64, 1, 1)
        assert sd["decoder.22.weight"] == (3 * patch * patch, 64, 3, 3)
        assert sd["decoder.3.conv.0.weight"] == (256, 512, 3, 3)


@pytest.mark.parametrize("channels,patch", [(16, 1), (48, 2)])
def test_decode_shapes_and_range(tmp_path, channels, patch):
    path = _random_checkpoint(tmp_path, channels, patch, f"t{channels}.safetensors")
    model = TV.load_from_file(path, channels, patch, device = "cpu", dtype = torch.float32)
    spatial = 8 * patch
    out = model.decode(torch.randn(1, channels, 3, 4, 5))
    # 4 frames per latent with the first 3 trimmed (the Wan VAE's causal 4k+1 frames)
    assert out.shape == (1, 3, 9, 4 * spatial, 5 * spatial)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    one = model.decode(torch.randn(1, channels, 1, 4, 5))
    assert one.shape == (1, 3, 1, 4 * spatial, 5 * spatial)


def test_decode_is_causal(tmp_path):
    path = _random_checkpoint(tmp_path, 16, 1, "c.safetensors")
    model = TV.load_from_file(path, 16, 1, device = "cpu", dtype = torch.float32)
    lat = torch.randn(1, 16, 3, 4, 4)
    full = model.decode(lat)
    head = model.decode(lat[:, :, :1])
    # the first frame depends only on the first latent: the preview decode of frame 0 is the clip's frame 0
    torch.testing.assert_close(full[:, :, :1], head, atol = 1e-3, rtol = 1e-3)


def test_draft_decode_is_off_by_default(monkeypatch):
    monkeypatch.delenv(TV.DRAFT_DECODE_ENV, raising = False)
    monkeypatch.delenv(TV.TINY_VAE_ENV, raising = False)
    assert TV.draft_decode_wanted(None) is False
    assert TV.draft_decode_wanted(True) is True
    monkeypatch.setenv(TV.DRAFT_DECODE_ENV, "1")
    assert TV.draft_decode_wanted(None) is True
    assert TV.draft_decode_wanted(False) is False
    monkeypatch.setenv(TV.TINY_VAE_ENV, "0")
    assert TV.draft_decode_wanted(True) is False


def _fake_vae(z_dim):
    cfg = types.SimpleNamespace(z_dim = z_dim, latents_mean = [0.5] * z_dim, latents_std = [2.0] * z_dim)

    class _Vae:
        config = cfg

        def decode(
            self,
            z,
            return_dict = True,
        ):
            return ("full", z)

    return _Vae()


def test_draft_decode_swaps_and_restores_vae_decode(tmp_path):
    path = _random_checkpoint(tmp_path, 16, 1, "d.safetensors")
    model = TV.load_from_file(path, 16, 1, device = "cpu", dtype = torch.float32)
    vae = _fake_vae(16)
    norm = torch.randn(1, 16, 2, 3, 3)
    with TV.TinyVaeDraftDecode(vae, model) as ctx:
        assert ctx.engaged
        # the pipeline hands over un-normalised latents: latents * std + mean
        (video,) = vae.decode(norm * 2.0 + 0.5, return_dict = False)
    assert ctx.calls == 1
    assert video.shape == (1, 3, 5, 24, 24)
    torch.testing.assert_close(video, model.decode(norm) * 2 - 1, atol = 2e-3, rtol = 2e-3)
    assert vae.decode(norm, return_dict = False)[0] == "full"
    assert "decode" not in vae.__dict__


def test_draft_decode_declines_a_mismatched_vae(tmp_path):
    path = _random_checkpoint(tmp_path, 16, 1, "m.safetensors")
    model = TV.load_from_file(path, 16, 1, device = "cpu", dtype = torch.float32)
    vae = _fake_vae(48)
    with TV.TinyVaeDraftDecode(vae, model) as ctx:
        assert not ctx.engaged
        assert vae.decode(1, return_dict = False)[0] == "full"


def test_get_decoder_without_cached_weights_falls_back(monkeypatch):
    started = []
    monkeypatch.setattr(TV, "_cached_path", lambda filename: None)
    monkeypatch.setattr(TV, "prefetch", lambda family, **kw: started.append(family))
    monkeypatch.setattr(TV, "_DECODERS", {})
    assert TV.get_decoder("wan2.1-t2v-1.3b", 16, "cpu") is None
    assert started == ["wan2.1-t2v-1.3b"]
    assert TV.get_decoder("ltx-2", 128, "cpu") is None


def test_get_decoder_loads_once_and_survives_a_bad_file(tmp_path, monkeypatch):
    good = _random_checkpoint(tmp_path, 16, 1, "taew2_1.safetensors")
    bad = tmp_path / "taew2_2.safetensors"
    bad.write_bytes(b"not a safetensors file")
    monkeypatch.setattr(TV, "_DECODERS", {})
    monkeypatch.setattr(TV, "_FAILED", set())
    monkeypatch.setattr(TV, "_cached_path", lambda f: str(tmp_path / f))
    first = TV.get_decoder("wan2.1-t2v-1.3b", 16, "cpu")
    assert first is not None and TV.get_decoder("wan2.2-t2v-a14b", 16, "cpu") is first
    assert TV.get_decoder("wan2.2-ti2v-5b", 48, "cpu") is None
    del good


def test_prefetch_runs_in_the_background_once(monkeypatch):
    gate = threading.Event()
    calls = []

    def _fetch(filename):
        calls.append(filename)
        gate.wait(5)
        with TV._LOCK:
            TV._FETCHING.pop(filename, None)

    monkeypatch.setattr(TV, "_cached_path", lambda f: None)
    monkeypatch.setattr(TV, "_fetch", _fetch)
    monkeypatch.setattr(TV, "_FETCHING", {})
    monkeypatch.setattr(TV, "_FAILED", set())
    TV.prefetch("wan2.2-ti2v-5b")
    TV.prefetch("wan2.2-ti2v-5b")
    TV.prefetch("wan2.2-ti2v-5b", local_files_only = True)
    gate.set()
    for t in list(TV._FETCHING.values()):
        t.join(5)
    assert calls == ["taew2_2.safetensors"]


def test_preview_without_tiny_weights_keeps_the_linear_map_or_none(monkeypatch):
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    monkeypatch.setattr(DP, "_tiny_decoder", lambda family, device: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    pub = lambda url, seq: None
    assert (
        DP.LatentPreviewer.create(
            family = "wan2.1-t2v-1.3b",
            requested = None,
            height = 64,
            width = 64,
            device = "cuda",
            publish = pub,
        )
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_cuda_tiny_preview_publishes_without_syncing(tmp_path, monkeypatch):
    path = _random_checkpoint(tmp_path, 16, 1, "g.safetensors")
    model = TV.load_from_file(path, 16, 1, device = "cuda", dtype = torch.float16)
    monkeypatch.delenv(DP.PREVIEW_ENV, raising = False)
    monkeypatch.setattr(DP, "_tiny_decoder", lambda family, device: model)
    done = threading.Event()
    got = []

    def publish(url, seq):
        got.append(url)
        done.set()

    prev = DP.LatentPreviewer.create(
        family = "wan2.1-t2v-1.3b", requested = None, height = 240, width = 416, device = "cuda",
        publish = publish, min_interval_s = 0.0,
    )  # fmt: skip
    assert prev is not None and prev.tiny is model
    lat = torch.randn(1, 16, 5, 30, 52, device = "cuda", dtype = torch.bfloat16)
    before = lat.clone()

    class _Sched:
        sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], device = "cuda")
        _step_index = 0

    sched = _Sched()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for i in range(4):
            sched._step_index = i + 1
            prev.on_step(lat, sched)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert not prev.failed and prev.tiny_decodes == 3
    assert done.wait(10.0)
    prev.finish()
    assert torch.equal(lat, before)
    assert got[-1].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("chunk", [1, 2, 3, 0])
def test_chunked_decode_matches_one_pass(tmp_path, chunk):
    path = _random_checkpoint(tmp_path, 48, 2, "k.safetensors")
    model = TV.load_from_file(path, 48, 2, device = "cpu", dtype = torch.float32)
    lat = torch.randn(1, 48, 5, 3, 4)
    one = model.decode(lat, chunk_frames = 5)
    got = model.decode(lat, chunk_frames = chunk)
    assert got.shape == one.shape == (1, 3, 17, 48, 64)
    torch.testing.assert_close(got, one, atol = 1e-2, rtol = 1e-2)


def test_draft_decode_runs_inside_the_decode_phase_wrappers(tmp_path):
    """video.py enters the draft decoder before progress_ctx(): the decode-phase wrapper (which reports the decode and
    lets video_stream_residency return lent denoiser groups) wraps the tiny decoder instead of being replaced by it."""
    import ast
    from pathlib import Path

    from core.inference.media_decode_phase import decode_phase

    path = _random_checkpoint(tmp_path, 16, 1, "d.safetensors")
    model = TV.load_from_file(path, 16, 1, device = "cpu", dtype = torch.float32)
    vae = _fake_vae(16)
    pipe = types.SimpleNamespace(vae = vae)
    fired = []
    with TV.TinyVaeDraftDecode(vae, model) as ctx, decode_phase(pipe, lambda: fired.append(1)):
        vae.decode(torch.randn(1, 16, 2, 3, 3), return_dict = False)
    assert ctx.calls == 1 and fired == [1]
    assert "decode" not in vae.__dict__

    # The render's ``with`` statement keeps that order.
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "video.py").read_text(
        encoding = "utf-8"
    )
    orders = [
        [ast.unparse(item.context_expr) for item in node.items]
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.With)
        and any(ast.unparse(i.context_expr) == "draft_ctx" for i in node.items)
    ]
    assert orders, "the render no longer enters draft_ctx"
    for names in orders:
        assert names.index("draft_ctx") < names.index("progress_ctx()"), names


def test_tiny_checkpoint_lookup_and_download_use_studios_live_hf_cache(tmp_path, monkeypatch):
    """Both the cache lookup and the background download pass Studio's active hub cache root (Settings can move it),
    as every other video loader does through hub_cache_dir()."""
    import huggingface_hub

    from utils import hf_cache_settings

    root = tmp_path / "studio_hub"
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: root)
    seen = []

    def fake_lookup(
        repo_id,
        filename,
        cache_dir = None,
        **_k,
    ):
        seen.append(("lookup", repo_id, filename, cache_dir))
        return None

    def fake_download(
        repo_id,
        filename,
        cache_dir = None,
        **_k,
    ):
        seen.append(("download", repo_id, filename, cache_dir))
        return str(root / filename)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", fake_lookup)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    assert TV._cached_path("taew2_1.safetensors") is None
    TV._fetch("taew2_1.safetensors")
    assert seen == [
        ("lookup", TV.TINY_VAE_REPO, "taew2_1.safetensors", str(root)),
        ("download", TV.TINY_VAE_REPO, "taew2_1.safetensors", str(root)),
    ]
