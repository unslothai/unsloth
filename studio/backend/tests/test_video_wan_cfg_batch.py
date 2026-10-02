# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``video_wan_cfg_batch.py``: Wan's conditional + unconditional denoiser calls run as one batch-2 call on
fp16 loads only, every call outside that exact pair falls back to the stock order, an OOM reruns both rows alone, and a
real (tiny) WanPipeline produces the same video with one denoiser call per step instead of two."""

from __future__ import annotations

import ast
import inspect
import types

import pytest

from core.inference import video_wan_cfg_batch as cb

torch = pytest.importorskip("torch")

_CUDA = torch.cuda.is_available() and not getattr(torch.version, "hip", None)
needs_cuda = pytest.mark.skipif(not _CUDA, reason = "needs an NVIDIA GPU")


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(cb.WAN_CFG_BATCH_ENV, raising = False)
    for k in cb._COUNTS:
        cb._COUNTS[k] = 0


class _Inner:
    """A denoiser forward that records the batch of every call and returns 10 * latent + embedding mean per row."""

    def __init__(self, oom_at_batch = None):
        self.calls = []
        self.oom_at_batch = oom_at_batch

    def __call__(self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image = None,
                 return_dict = True, attention_kwargs = None):
        b = hidden_states.shape[0]
        self.calls.append(b)
        if self.oom_at_batch is not None and b == self.oom_at_batch:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        out = hidden_states * 10 + encoder_hidden_states.mean(dim = (1, 2)).view(b, *([1] * (hidden_states.dim() - 1)))
        return (out,)


class _Pipe:
    def __init__(self, cfg = True):
        self._current_timestep = 5
        self._interrupt = False
        self.do_classifier_free_guidance = cfg


def _pipe(cfg = True):
    return _Pipe(cfg)


def _step(batcher, x, t, cond, uncond):
    """WanPipeline's loop body: cond call, uncond call, guidance (read only after both)."""
    noise_pred = batcher(hidden_states = x, timestep = t, encoder_hidden_states = cond, attention_kwargs = None,
                         return_dict = False)[0]
    noise_uncond = batcher(hidden_states = x, timestep = t, encoder_hidden_states = uncond, attention_kwargs = None,
                           return_dict = False)[0]
    return noise_uncond + 5.0 * (noise_pred - noise_uncond)


def _io():
    x = torch.randn(1, 4, 2, 3, 3)
    t = torch.full((1, 6), 7.0)
    cond = torch.randn(1, 5, 8)
    uncond = torch.randn(1, 5, 8)
    return x, t, cond, uncond


def test_pair_runs_as_one_batch2_call_with_the_stock_result(monkeypatch):
    monkeypatch.setattr(cb._CfgBatcher, "_armed", lambda self, call: True)
    inner = _Inner()
    x, t, cond, uncond = _io()
    want = _step(inner, x, t, cond, uncond)
    assert inner.calls == [1, 1]
    inner.calls.clear()
    batcher = cb._CfgBatcher(_pipe(), None, inner)
    got = _step(batcher, x, t, cond, uncond)
    assert inner.calls == [2]
    assert torch.allclose(got, want)
    assert cb.counts()["batched"] == 1


def test_an_unpaired_next_call_computes_the_pending_one_first(monkeypatch):
    armed = {"on": True}
    monkeypatch.setattr(cb._CfgBatcher, "_armed", lambda self, call: armed["on"])
    inner = _Inner()
    batcher = cb._CfgBatcher(_pipe(), None, inner)
    x, t, cond, uncond = _io()
    pending = batcher(hidden_states = x, timestep = t, encoder_hidden_states = cond, return_dict = False)[0]
    armed["on"] = False
    # a different latent object: not the unconditional half of this step
    other = batcher(hidden_states = x.clone(), timestep = t, encoder_hidden_states = uncond, return_dict = False)[0]
    assert inner.calls == [1, 1]
    assert torch.allclose(pending, inner(x, t, cond)[0])
    assert torch.allclose(other, inner(x, t, uncond)[0])
    assert cb.counts() == {"batched": 0, "single": 1, "fallback": 0}


def test_flush_fills_a_pending_call(monkeypatch):
    monkeypatch.setattr(cb._CfgBatcher, "_armed", lambda self, call: True)
    inner = _Inner()
    batcher = cb._CfgBatcher(_pipe(), None, inner)
    x, t, cond, _ = _io()
    pending = batcher(hidden_states = x, timestep = t, encoder_hidden_states = cond, return_dict = False)[0]
    batcher.flush()
    assert torch.allclose(pending, inner(x, t, cond)[0])


def test_oom_in_the_batch_reruns_both_rows_alone(monkeypatch):
    monkeypatch.setattr(cb._CfgBatcher, "_armed", lambda self, call: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    inner = _Inner(oom_at_batch = 2)
    x, t, cond, uncond = _io()
    batcher = cb._CfgBatcher(_pipe(), None, inner)
    got = _step(batcher, x, t, cond, uncond)
    want = _step(_Inner(), x, t, cond, uncond)
    assert inner.calls == [2, 1, 1]
    assert torch.allclose(got, want)
    assert cb.counts()["fallback"] == 1


def test_a_non_oom_error_in_the_batch_propagates(monkeypatch):
    monkeypatch.setattr(cb._CfgBatcher, "_armed", lambda self, call: True)

    def boom(**kwargs):
        raise ValueError("bad shapes")

    batcher = cb._CfgBatcher(_pipe(), None, boom)
    x, t, cond, uncond = _io()
    with pytest.raises(ValueError):
        _step(batcher, x, t, cond, uncond)


def test_mismatched_embedding_shape_does_not_batch(monkeypatch):
    monkeypatch.setattr(cb._CfgBatcher, "_armed", lambda self, call: True)
    inner = _Inner()
    batcher = cb._CfgBatcher(_pipe(), None, inner)
    x, t, cond, _ = _io()
    uncond = torch.randn(1, 7, 8)
    got = _step(batcher, x, t, cond, uncond)
    assert 2 not in inner.calls
    assert torch.allclose(got, _step(_Inner(), x, t, cond, uncond))


@pytest.mark.parametrize(
    "pipe_state",
    [
        dict(_current_timestep = None),  # outside the denoising loop (warmups, decode)
        dict(_interrupt = True),
        dict(do_classifier_free_guidance = False),  # no unconditional call will follow
    ],
)
def test_not_armed_outside_a_cfg_denoising_step(pipe_state):
    pipe = _pipe()
    for k, v in pipe_state.items():
        setattr(pipe, k, v)
    batcher = cb._CfgBatcher(pipe, None, _Inner())
    x, t, cond, _ = _io()
    call = dict(hidden_states = x, timestep = t, encoder_hidden_states = cond, return_dict = False)
    assert batcher._armed(call) is False


def test_kill_switch_and_dtype_scope(monkeypatch):
    assert cb.wanted(torch.float16, "cuda") is (not getattr(torch.version, "hip", None))
    assert cb.wanted(torch.bfloat16, "cuda") is False
    assert cb.wanted(torch.float32, "cuda") is False
    assert cb.wanted(torch.float16, "cpu") is False
    monkeypatch.setenv(cb.WAN_CFG_BATCH_ENV, "0")
    assert cb.wanted(torch.float16, "cuda") is False


def test_installed_wan_pipeline_matches_the_loop_this_relies_on():
    wan = pytest.importorskip("diffusers.pipelines.wan.pipeline_wan")
    assert cb._pipeline_supported(wan.WanPipeline)


def test_only_a_wan_pipeline_without_a_step_cache_is_wrapped():
    class WanPipeline:
        def __call__(self):
            return None

    class WanTransformer3DModel(torch.nn.Module):
        def forward(self, x):
            return x

    pipe = WanPipeline()
    pipe.transformer = WanTransformer3DModel()
    stock = pipe.transformer.forward
    # this stand-in has no CFG loop, so the source check refuses it
    assert cb.install_for_pipe(pipe, torch.float16, "cuda") is False
    assert pipe.transformer.forward == stock
    cb._SUPPORTED[WanPipeline] = True
    try:
        assert cb.install_for_pipe(pipe, torch.float16, "cuda", cache_engaged = True) is False
        assert cb.install_for_pipe(pipe, torch.bfloat16, "cuda") is False
        assert cb.install_for_pipe(pipe, torch.float16, "cuda") is (not getattr(torch.version, "hip", None))
        if not getattr(torch.version, "hip", None):
            assert isinstance(pipe.transformer.forward, cb._CfgBatcher)
            assert cb.install_for_pipe(pipe, torch.float16, "cuda") is True  # idempotent
            assert pipe.transformer.forward._inner == stock
        cb.uninstall(pipe)
        assert pipe.transformer.forward == stock
    finally:
        cb._SUPPORTED.pop(WanPipeline, None)


def test_loader_wraps_after_the_offload_hooks_and_off_on_speed_off():
    from core.inference import video

    cls = next(c for c in vars(video).values() if inspect.isclass(c) and hasattr(c, "load_pipeline"))
    src = inspect.getsource(cls.load_pipeline)
    at = src.index("install_wan_cfg_batch(")
    assert src.index("apply_memory_plan(") < at
    gate = src.rindex("if effective_speed != SPEED_OFF:", 0, at)
    assert at - gate < 300
    assert "cache_engaged = bool(cache_engaged) or bool(cache_may_toggle)" in src
    assert '"wan_cfg_batch"' in src


def _tiny_wan_pipe(expand: bool):
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    vae = diffusers.AutoencoderKLWan(
        base_dim = 3, z_dim = 16, dim_mult = [1, 1, 1, 1], num_res_blocks = 1, temperal_downsample = [False, True, True]
    )
    transformer = diffusers.WanTransformer3DModel(
        patch_size = (1, 2, 2), num_attention_heads = 2, attention_head_dim = 12, in_channels = 16, out_channels = 16,
        text_dim = 32, freq_dim = 256, ffn_dim = 32, num_layers = 2, cross_attn_norm = True,
        qk_norm = "rms_norm_across_heads", rope_max_seq_len = 32,
    )
    # no text encoder: the renders below pass fixed prompt / negative embeddings, so nothing is downloaded
    pipe = diffusers.WanPipeline(
        tokenizer = None, text_encoder = None, vae = vae, transformer = transformer,
        scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift = 7.0), expand_timesteps = expand,
    )
    pipe.to("cuda", torch.float16)
    pipe.vae.to(torch.float32)
    pipe.set_progress_bar_config(disable = True)
    return pipe


def _render(pipe, steps = 4):
    g = torch.Generator("cpu").manual_seed(7)
    cond = torch.randn(1, 16, 32, generator = g).to("cuda", torch.float16)
    uncond = torch.randn(1, 16, 32, generator = g).to("cuda", torch.float16)
    return pipe(
        prompt_embeds = cond, negative_prompt_embeds = uncond, height = 32, width = 32, num_frames = 9,
        num_inference_steps = steps, guidance_scale = 5.0, generator = torch.Generator("cpu").manual_seed(1),
        output_type = "np",
    ).frames[0]


@needs_cuda
@pytest.mark.parametrize("expand", [False, True], ids = ["per_sample_t", "per_token_t"])
def test_real_tiny_wan_pipeline_one_batched_call_per_step(expand):
    import numpy as np

    pipe = _tiny_wan_pipe(expand)
    want = _render(pipe)
    calls = []
    inner = pipe.transformer.forward

    def counting(*a, **k):
        calls.append(k["hidden_states"].shape[0])
        return inner(*a, **k)

    pipe.transformer.forward = counting
    assert cb.install_for_pipe(pipe, torch.float16, "cuda") is True
    got = _render(pipe)
    assert calls == [2] * 4
    assert np.abs(got - want).max() < 2e-2
    cb.uninstall(pipe)
    calls.clear()
    again = _render(pipe)
    assert calls == [1] * 8 and np.array_equal(again, want)


@needs_cuda
def test_real_tiny_wan_pipeline_under_group_offload_hooks():
    """The loader's order: offload hooks attach, the batcher wraps them, and only then the first render runs.
    diffusers' lazy prefetch hook removes itself after that first forward and restores the forward it captured, which
    used to drop the batcher after step 1 (one batched call per render on a real T4)."""
    import numpy as np
    from diffusers.hooks import apply_group_offloading

    ref = _tiny_wan_pipe(True)
    want = _render(ref)
    del ref
    pipe = _tiny_wan_pipe(True)
    apply_group_offloading(
        pipe.transformer, onload_device = torch.device("cuda"), offload_device = torch.device("cpu"),
        offload_type = "block_level", num_blocks_per_group = 1, use_stream = True,
    )
    assert cb.install_for_pipe(pipe, torch.float16, "cuda") is True
    got = _render(pipe)
    assert cb.counts() == {"batched": 4, "single": 0, "fallback": 0}
    assert np.abs(got - want).max() < 2e-2
    again = _render(pipe)
    assert cb.counts()["batched"] == 8
    assert np.array_equal(got, again)


def test_a_hook_registered_after_the_batcher_is_not_wrapped_again(monkeypatch):
    monkeypatch.setattr(cb._CfgBatcher, "_armed", lambda self, call: False)

    class Dit(torch.nn.Module):
        def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict = True):
            return (hidden_states,)

    dit = Dit()
    batcher = cb._CfgBatcher(_pipe(), dit, dit.forward)
    dit.forward = batcher
    import functools

    def hooked(*a, **k):
        return batcher(*a, **k)

    dit.forward = functools.update_wrapper(hooked, batcher)  # a later hook wraps the batcher
    x, t, cond, _ = _io()
    dit(hidden_states = x, timestep = t, encoder_hidden_states = cond, return_dict = False)
    assert dit.forward is hooked  # left alone: it already reaches the batcher
    # the hook goes away and restores a forward without the batcher: the batcher puts itself back on top
    dit.forward = Dit.forward.__get__(dit)
    batcher(hidden_states = x, timestep = t, encoder_hidden_states = cond, return_dict = False)
    assert dit.forward is batcher


def test_module_is_torch_free_at_import():
    src = inspect.getsource(cb)
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("torch" in ast.unparse(n) for n in top)
