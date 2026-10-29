# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_zimage_cfg_split.py``: a guided Z-Image call after guidance-off ones compiles nothing new."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_zimage_cfg_split as zs

torch = pytest.importorskip("torch")
zmod = pytest.importorskip("diffusers.models.transformers.transformer_z_image")

_CONFIG = dict(
    all_patch_size = (2,),
    all_f_patch_size = (1,),
    in_channels = 16,
    dim = 32,
    n_layers = 2,
    n_refiner_layers = 1,
    n_heads = 2,
    n_kv_heads = 2,
    norm_eps = 1e-5,
    qk_norm = True,
    cap_feat_dim = 16,
    rope_theta = 256.0,
    t_scale = 1000.0,
    axes_dims = [8, 4, 4],
    axes_lens = [512, 64, 64],
)


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(zs.ZIMAGE_CFG_SPLIT_ENV, raising = False)
    yield
    zs.uninstall_zimage_cfg_split()


def _model():
    torch.manual_seed(0)
    return zmod.ZImageTransformer2DModel(**_CONFIG).eval()


def _latent(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(16, 1, 8, 8, generator = g)


def _caption(n, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    return torch.randn(n, 16, generator = g)


def _guided_args(latent, t, cond_len, neg_len, seed):
    # What ZImagePipeline passes on a guided step: the latent repeated, the prompt then the negative embeddings.
    return (
        [latent, latent.clone()],
        t.repeat(2),
        [_caption(cond_len, seed), _caption(neg_len, seed + 1)],
    )


def _compile_blocks(model):
    """Studio's regional compile: every repeated block's forward through one dynamic dynamo code object."""
    for block in (*model.noise_refiner, *model.context_refiner, *model.layers):
        block.forward = torch.compile(block.forward, backend = "eager", dynamic = True)


def _graphs():
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


def _run(model, *args):
    with torch.no_grad():
        return model(*args, return_dict = False)[0]


def test_guided_call_after_guidance_off_compiles_nothing_new():
    torch._dynamo.reset()
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "none") is None
    _compile_blocks(model)
    t = torch.tensor([0.4])
    with zs.render(pipe, 1):
        # Guidance-off renders at two prompt lengths (two 32-token caption buckets).
        _run(model, [_latent(0)], t, [_caption(9, 0)])
        _run(model, [_latent(1)], t, [_caption(45, 1)])
        warm = _graphs()
        assert warm > 0
        # CFG turned on: captions in different buckets (a mask when batched), then in the same bucket (no mask).
        _run(model, *_guided_args(_latent(2), t, 45, 9, 2))
        _run(model, *_guided_args(_latent(3), t, 7, 9, 3))
        _run(model, *_guided_args(_latent(4), t, 70, 9, 4))
    assert _graphs() == warm
    assert zs.zimage_cfg_split_status(pipe) == {"armed": True, "split": 3, "batched": 0}


def test_batched_guided_call_recompiles_without_the_split(monkeypatch):
    """The defect the split removes: the doubled batch (and its attention mask) are new graphs."""
    torch._dynamo.reset()
    monkeypatch.setenv(zs.ZIMAGE_CFG_SPLIT_ENV, "0")
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "none") is not None
    _compile_blocks(model)
    t = torch.tensor([0.4])
    with zs.render(pipe, 1):
        _run(model, [_latent(0)], t, [_caption(9, 0)])
        _run(model, [_latent(1)], t, [_caption(45, 1)])
        warm = _graphs()
        _run(model, *_guided_args(_latent(2), t, 45, 9, 2))
    assert _graphs() > warm


def test_split_output_is_each_branch_alone_and_matches_the_batched_call():
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    t = torch.tensor([0.4])
    x, tt, cap = _guided_args(_latent(5), t, 45, 9, 5)
    batched = _run(model, x, tt, cap)
    alone = _run(model, x[:1], tt[:1], cap[:1]) + _run(model, x[1:], tt[1:], cap[1:])
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    with zs.render(pipe, 1):
        split = _run(model, x, tt, cap)
        # return_dict=True (img2img / inpaint) joins the output dataclass the same way
        with torch.no_grad():
            split_dc = model(x, tt, cap).sample
    assert len(split) == 2
    for s, a, b, d in zip(split, alone, batched, split_dc):
        assert torch.equal(s, a)
        assert torch.equal(d, a)
        torch.testing.assert_close(s, b, atol = 1e-5, rtol = 1e-4)


def test_guidance_off_and_unarmed_calls_are_untouched():
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    t = torch.tensor([0.4, 0.4])
    x, cap = [_latent(6), _latent(7)], [_caption(9, 6), _caption(12, 7)]
    with zs.render(pipe, 2):
        # Two images per call with guidance off: one batched call, as before.
        _run(model, x, t, cap)
    # Outside a render, nothing splits.
    _run(model, *_guided_args(_latent(8), torch.tensor([0.4]), 45, 9, 8))
    assert zs.zimage_cfg_split_status(pipe)["split"] == 0


@pytest.mark.parametrize("policy", ["streaming", "sequential"])
def test_streamed_denoiser_is_not_armed(policy):
    pipe = types.SimpleNamespace(transformer = _model())
    assert "streamed" in zs.install_zimage_cfg_split(pipe, offload_policy = policy)
    assert zs.zimage_cfg_split_status(pipe)["armed"] is False


def test_group_offload_with_every_group_resident_splits():
    """12 GiB Z-Image-Turbo int8 / fp8 = group offload, all 37 groups resident; declining it cost 3 recompiles."""
    torch._dynamo.reset()
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "group") is None
    assert zs.zimage_cfg_split_status(pipe)["armed"] is True
    model._unsloth_stream_state = {
        "streamed": 0
    }  # what _keep_groups_resident records when every group is pinned
    _compile_blocks(model)
    t = torch.tensor([0.4])
    with zs.render(pipe, 1):
        _run(model, [_latent(0)], t, [_caption(9, 0)])
        _run(model, [_latent(1)], t, [_caption(45, 1)])
        warm = _graphs()
        _run(model, *_guided_args(_latent(2), t, 45, 9, 2))
        _run(model, *_guided_args(_latent(3), t, 7, 9, 3))
    assert _graphs() == warm
    assert zs.zimage_cfg_split_status(pipe)["split"] == 2


@pytest.mark.parametrize("state", [None, {"streamed": 3}, {"streamed": "?"}])
def test_group_offload_while_groups_stream_keeps_the_batched_call(state, monkeypatch):
    """Streaming groups (partial, released, or hooked with no record) keep the batched call."""
    monkeypatch.setattr(zs, "_has_offload_groups", lambda module: True)
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "group") is None
    if state is not None:
        model._unsloth_stream_state = state
    t = torch.tensor([0.4])
    with zs.render(pipe, 1):
        out = _run(model, *_guided_args(_latent(2), t, 45, 9, 2))
    stats = zs.zimage_cfg_split_status(pipe)
    assert stats["split"] == 0 and stats["batched"] == 1
    zs.uninstall_zimage_cfg_split(pipe)
    stock = _run(model, *_guided_args(_latent(2), t, 45, 9, 2))  # the unpatched, batched call
    assert all(torch.equal(a, b) for a, b in zip(out, stock))


def test_step_cache_and_step_skip_keep_the_batched_call():
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    args = _guided_args(_latent(9), torch.tensor([0.4]), 45, 9, 9)
    model._unsloth_step_cache = object()
    with zs.render(pipe, 1):
        _run(model, *args)
    del model._unsloth_step_cache

    class _Skip:
        _unsloth_outer_forward = True

        def __call__(self, *a, **k):
            return type(model).forward(model, *a, **k)

    model.__dict__["forward"] = _Skip()
    with zs.render(pipe, 1):
        _run(model, *args)
    assert zs.zimage_cfg_split_status(pipe) == {"armed": True, "split": 0, "batched": 2}


def test_not_zimage_and_uninstall_restores_the_class():
    assert (
        zs.install_zimage_cfg_split(
            types.SimpleNamespace(transformer = torch.nn.Linear(2, 2)), offload_policy = "none"
        )
        == "not a Z-Image denoiser"
    )
    cls = zmod.ZImageTransformer2DModel
    before = cls.__dict__.get("__call__")
    pipe = types.SimpleNamespace(transformer = _model())
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    assert getattr(cls.__dict__.get("__call__"), "__unsloth_zimage_cfg_split__", False)
    zs.uninstall_zimage_cfg_split(pipe)
    assert cls.__dict__.get("__call__") is before
    assert zs.zimage_cfg_split_status(pipe) == {"armed": False}


def _denoise(
    pipe,
    model,
    guided,
    steps,
    seed,
    skip = None,
    cond_len = 45,
    neg_len = 9,
):
    """A Z-Image denoise loop as ZImagePipeline runs it: one transformer call per step (2 samples when guided), the
    step-end callback after it. Returns the per-step noise predictions."""
    from core.inference import diffusion_step_skip as ss

    if skip is not None:
        ss.reset_static_step_skip(pipe, steps, step_signal = True)
    latent = _latent(seed)
    cond, neg = _caption(cond_len, seed), _caption(neg_len, seed + 1)
    preds = []
    with zs.render(pipe, 1):
        for i in range(steps):
            t = torch.tensor([1.0 - i / steps])
            if guided:
                out = _run(model, [latent, latent.clone()], t.repeat(2), [cond, neg])
                pred = out[0] + 2.0 * (out[0] - out[1])
            else:
                pred = _run(model, [latent], t, [cond])[0]
            preds.append(pred)
            latent = latent - pred / steps
            if skip is not None:
                ss.mark_step_end(pipe)
    return preds


def _static_pipe(compile_blocks):
    from core.inference import diffusion_step_skip as ss

    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    if compile_blocks:
        _compile_blocks(model)
    settings = {"mode": "taylor1", "head": 0.2, "tail": 0.1, "every": 2, "min_steps": 12}
    assert ss.install_static_step_skip(pipe, settings = settings) == "static"
    return pipe, model, ss._find(pipe)


def test_static_step_skip_splits_and_turning_cfg_on_compiles_nothing_new():
    torch._dynamo.reset()
    pipe, model, skip = _static_pipe(compile_blocks = True)
    _denoise(pipe, model, False, 12, 0, skip)
    warm = _graphs()
    assert warm > 0
    _denoise(pipe, model, True, 12, 1, skip)
    assert _graphs() == warm
    assert zs.zimage_cfg_split_status(pipe) == {"armed": True, "split": 12, "batched": 0}
    assert skip.stats["skipped"] > 0


def test_static_step_skip_split_matches_the_batched_schedule(monkeypatch):
    """Same steps skipped, same predictions: the split only keys cond as ``#0`` and uncond as ``#1``."""
    pipe, model, skip = _static_pipe(compile_blocks = False)
    split = _denoise(pipe, model, True, 12, 2, skip)
    split_stats = dict(skip.stats)
    assert zs.zimage_cfg_split_status(pipe)["split"] == 12
    zs.uninstall_zimage_cfg_split(pipe)
    from core.inference import diffusion_step_skip as ss

    ss.uninstall_static_step_skip(pipe)
    monkeypatch.setenv(zs.ZIMAGE_CFG_SPLIT_ENV, "0")
    pipe_b, model_b, skip_b = _static_pipe(compile_blocks = False)
    batched = _denoise(pipe_b, model_b, True, 12, 2, skip_b)
    assert split_stats["skipped"] == 2 * skip_b.stats["skipped"] > 0
    assert split_stats["computed"] == 2 * skip_b.stats["computed"]
    for s, b in zip(split, batched):
        torch.testing.assert_close(s, b, atol = 1e-4, rtol = 1e-4)


def test_static_step_skip_without_a_step_signal_keeps_the_batched_call():
    from core.inference import diffusion_step_skip as ss

    pipe, model, skip = _static_pipe(compile_blocks = False)
    ss.reset_static_step_skip(pipe, 12, step_signal = False)
    with zs.render(pipe, 1):
        _run(model, *_guided_args(_latent(3), torch.tensor([0.4]), 45, 9, 3))
    assert zs.zimage_cfg_split_status(pipe) == {"armed": True, "split": 0, "batched": 1}
    from core.inference.diffusion_multigpu import CfgParallelRouter

    model.__dict__["forward"] = CfgParallelRouter(model, primary = "cpu", secondary = "cpu")
    with zs.render(pipe, 1):
        _run(model, *_guided_args(_latent(3), torch.tensor([0.4]), 45, 9, 3))
    assert zs.zimage_cfg_split_status(pipe) == {"armed": True, "split": 0, "batched": 2}
    ss.reset_static_step_skip(pipe, 12, step_signal = True)
    with zs.render(pipe, 1):
        _run(model, *_guided_args(_latent(3), torch.tensor([0.4]), 45, 9, 3))
    assert zs.zimage_cfg_split_status(pipe) == {"armed": True, "split": 1, "batched": 2}


def test_cfg_parallel_router_splits_and_keeps_both_branches_on_the_primary():
    from core.inference.diffusion_multigpu import CfgParallelRouter

    torch._dynamo.reset()
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    t = torch.tensor([0.4])
    x, tt, cap = _guided_args(_latent(4), t, 45, 9, 4)
    with zs.render(pipe, 1):
        plain = _run(model, x, tt, cap)
    _compile_blocks(model)
    router = CfgParallelRouter(model, primary = "cpu", secondary = "cpu")
    model.__dict__["forward"] = router
    model._unsloth_cfg_router = router
    with zs.render(pipe, 1):
        _run(model, [_latent(0)], t, [_caption(9, 0)])
        _run(model, [_latent(1)], t, [_caption(45, 1)])
        warm = _graphs()
        routed = _run(model, x, tt, cap)
        _denoise(pipe, model, True, 3, 5)
    assert _graphs() == warm
    assert router.replica is None and router.stats["replica_calls"] == 0
    for a, b in zip(routed, plain):
        assert torch.equal(a, b)
    assert zs.zimage_cfg_split_status(pipe)["batched"] == 0


def test_compiled_only_keeps_an_eager_denoiser_batched_and_splits_once_compiled():
    """Eager tier: the split has no graph to protect and two serial calls are slower, so the guided call stays batched;
    a compile engaged after the load (deferred / auto compile) splits from the next render on."""
    torch._dynamo.reset()
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True) is None
    args = _guided_args(_latent(10), torch.tensor([0.4]), 9, 12, 10)
    with zs.render(pipe, 1):
        eager = _run(model, *args)
    assert zs.zimage_cfg_split_status(pipe) == {"armed": True, "split": 0, "batched": 0}
    with torch.no_grad():
        batched = model(*args, return_dict = False)[0]
    for a, b in zip(eager, batched):
        assert torch.equal(a, b)
    _compile_blocks(model)
    with zs.render(pipe, 1):
        _run(model, *args)
    assert zs.zimage_cfg_split_status(pipe)["split"] == 1


@pytest.mark.parametrize("engage", ["regional_mark", "compiled_block", "cuda_graphs"])
def test_compiled_only_splits_every_engaged_form(engage):
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True)
    if engage == "regional_mark":
        model._unsloth_regional_compile_kwargs = {"dynamic": True}
    elif engage == "compiled_block":
        model.layers[0].compile(backend = "eager", dynamic = True)
    else:
        pipe._unsloth_cuda_graphs = (object(),)
    with zs.render(pipe, 1):
        _run(model, *_guided_args(_latent(11), torch.tensor([0.4]), 45, 9, 11))
    assert zs.zimage_cfg_split_status(pipe)["split"] == 1


def test_studio_load_arms_the_split_compiled_only():
    import ast
    import pathlib

    source = pathlib.Path(zs.__file__).with_name("diffusion.py").read_text(encoding = "utf-8")
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "install_zimage_cfg_split"
    ]
    assert calls
    for call in calls:
        flags = {k.arg: k.value for k in call.keywords}
        assert (
            isinstance(flags.get("compiled_only"), ast.Constant)
            and flags["compiled_only"].value is True
        )


def test_compiled_only_failed_compile_runs_batched():
    """A compile that fell back to eager at its first forward leaves the regional-compile marks on the transformer; the
    load runs eager, so the guided call stays batched."""
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True)
    model._unsloth_regional_compile_kwargs = {"dynamic": True}
    model._unsloth_compile_guard = types.SimpleNamespace(error = "BackendCompilerFailed: boom")
    with zs.render(pipe, 1):
        _run(model, *_guided_args(_latent(12), torch.tensor([0.4]), 9, 12, 12))
    assert zs.zimage_cfg_split_status(pipe)["split"] == 0
    model._unsloth_compile_guard.error = None
    with zs.render(pipe, 1):
        _run(model, *_guided_args(_latent(12), torch.tensor([0.4]), 9, 12, 12))
    assert zs.zimage_cfg_split_status(pipe)["split"] == 1


def test_compiled_only_sees_graphs_on_a_shared_workflow_transformer():
    """img2img / inpaint pipelines built by from_pipe share the transformer (and its graph wrapper) but not the root
    pipe's ``_unsloth_cuda_graphs``: the live wrapper on the denoiser keeps the split."""
    from core.inference import diffusion_cuda_graph as cg

    model = _model()
    root = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(root, offload_policy = "none", compiled_only = True)
    wrapper = cg.GraphedForward(model)
    wrapper.enabled = True  # armed but never installed: the test runs eager on CPU
    workflow = types.SimpleNamespace(transformer = model)
    try:
        with zs.render(workflow, 1):
            _run(model, *_guided_args(_latent(13), torch.tensor([0.4]), 9, 12, 13))
        assert zs.zimage_cfg_split_status(root)["split"] == 1
        wrapper.enabled = False
        with zs.render(workflow, 1):
            _run(model, *_guided_args(_latent(13), torch.tensor([0.4]), 9, 12, 13))
        assert zs.zimage_cfg_split_status(root)["split"] == 1
    finally:
        cg._LIVE_WRAPPERS.discard(wrapper)


def test_compiled_only_eager_oom_retries_split_for_the_render(monkeypatch):
    """An eager guided call that runs out of memory batched retries as the two half-batch calls, and keeps splitting
    for the rest of that render; the next render tries batched again."""
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True)
    args = _guided_args(_latent(14), torch.tensor([0.4]), 9, 12, 14)
    want = _run(model, *args)
    real = type(model).forward
    batches = []

    def forward(self, x, t, cap, **kw):
        batches.append(len(x))
        if len(x) == 2:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(self, x, t, cap, **kw)

    monkeypatch.setattr(type(model), "forward", forward)
    with zs.render(pipe, 1):
        got = _run(model, *args)
        _run(model, *args)
    assert batches == [2, 1, 1, 1, 1]
    for a, b in zip(got, want):
        assert torch.allclose(a, b, atol = 1e-5)
    assert zs.zimage_cfg_split_status(pipe)["oom_split"] == 1
    batches.clear()
    with zs.render(pipe, 1):
        _run(model, *args)
    assert batches[0] == 2
    monkeypatch.setattr(
        type(model), "forward", lambda self, *a, **k: (_ for _ in ()).throw(ValueError("x"))
    )
    with pytest.raises(ValueError), zs.render(pipe, 1):
        _run(model, *args)


def test_group_offload_resident_transformer_tier_splits():
    """stream_transformer=False: only the text encoders stream, the transformer has no groups and no residency record."""
    torch._dynamo.reset()
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "group") is None
    assert zs.guided_runs_at_guidance_off_batch(pipe) is True
    _compile_blocks(model)
    t = torch.tensor([0.4])
    with zs.render(pipe, 1):
        _run(model, [_latent(0)], t, [_caption(9, 0)])
        _run(model, [_latent(1)], t, [_caption(45, 1)])
        warm = _graphs()
        _run(model, *_guided_args(_latent(2), t, 45, 9, 2))
    assert _graphs() == warm
    assert zs.zimage_cfg_split_status(pipe)["split"] == 1


def test_request_guard_reprices_doubled_when_the_split_would_be_declined():
    """Releasing groups or an AUTO FBCache engaging after the guard declines the split: the doubled call is priced."""
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none")

    def price(extra_split, extra_cfg):
        def _price(cfg):
            return extra_split if cfg is False else extra_cfg

        return _price

    for workflow in ("txt2img", "img2img", "inpaint", "upscale"):
        assert zs.split_aware_request_extra(pipe, True, workflow, price(0, 900)) == (False, 0)
    assert zs.split_aware_request_extra(pipe, None, "upscale", price(-5, 900)) == (False, -5)
    # Fits only after a release: the release can demote denoiser groups, so the batched call is priced.
    assert zs.split_aware_request_extra(pipe, True, "txt2img", price(100, 900)) == (True, 900)
    assert zs.split_aware_request_extra(pipe, None, "txt2img", price(100, 900)) == (None, 900)
    assert zs.split_aware_request_extra(pipe, True, "txt2img", price(0, 900), step_cache = True) == (
        True,
        900,
    )
    assert zs.split_aware_request_extra(pipe, True, "controlnet", price(0, 900)) == (True, 900)
    assert zs.split_aware_request_extra(pipe, False, "txt2img", price(0, 900)) == (False, 0)
    model._unsloth_step_cache = object()
    assert zs.split_aware_request_extra(pipe, True, "txt2img", price(0, 900)) == (True, 900)


def test_studio_guard_passes_the_auto_step_cache_prediction():
    import ast
    import pathlib

    source = pathlib.Path(zs.__file__).with_name("diffusion.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "split_aware_request_extra"
    ]
    assert len(calls) == 1
    assert {k.arg for k in calls[0].keywords} == {"step_cache"}
    assert "guided_runs_at_guidance_off_batch(" not in source
    predict = source.index("auto_step_cache_on = auto_step_cache_mode(cache_steps)")
    assert predict < source.index("guard_cfg, extra_mib = split_aware_request_extra(")
    assert predict < source.index("toggled = maybe_toggle_step_cache(")


def test_request_guard_sizes_a_split_guided_render_as_guidance_off(monkeypatch):
    """Pricing a split render doubled released 5 GiB of resident groups, declining the split; it needs no extra memory."""
    monkeypatch.setattr(zs, "_has_offload_groups", lambda module: True)
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.guided_runs_at_guidance_off_batch(pipe) is False  # not armed
    zs.install_zimage_cfg_split(pipe, offload_policy = "group")
    assert (
        zs.guided_runs_at_guidance_off_batch(pipe) is False
    )  # hooked, no residency record: groups stream
    model._unsloth_stream_state = {"streamed": 0}
    assert zs.guided_runs_at_guidance_off_batch(pipe) is True
    model._unsloth_stream_state = {"streamed": 2}
    assert zs.guided_runs_at_guidance_off_batch(pipe) is False
    model._unsloth_stream_state = {"streamed": 0}
    model._unsloth_step_cache = object()
    assert zs.guided_runs_at_guidance_off_batch(pipe) is False  # FBCache keeps the batched call
    del model._unsloth_step_cache
    resident = types.SimpleNamespace(transformer = _model())
    zs.install_zimage_cfg_split(resident, offload_policy = "none")
    assert zs.guided_runs_at_guidance_off_batch(resident) is True


def test_request_guard_prices_an_eager_compiled_only_load_as_doubled():
    """compiled_only: eager keeps the doubled price, compiled gets the guidance-off one."""
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "group", compiled_only = True) is None
    model._unsloth_stream_state = {"streamed": 0}
    assert zs.guided_runs_at_guidance_off_batch(pipe) is False
    model._unsloth_regional_compile_kwargs = {"dynamic": True}
    assert zs.guided_runs_at_guidance_off_batch(pipe) is True
    model._unsloth_stream_state = {"streamed": 1}
    assert zs.guided_runs_at_guidance_off_batch(pipe) is False


def test_eager_oom_split_keeps_static_skip_branch_keys(monkeypatch):
    """A batched eager call that runs out of memory consumed a static-skip ordinal; the two halves still key cond as
    ``#0`` and uncond as ``#1``, so a later skipped step reuses each branch's own prediction."""
    from core.inference import diffusion_step_skip as ss

    class ZImageTransformer2DModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fail_batched = False
            self.batches = []

        def forward(
            self,
            x,
            t,
            cap,
            return_dict = False,
        ):
            self.batches.append(len(x))
            if self.fail_batched and len(x) == 2:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return ([torch.full_like(item, float(c.item())) for item, c in zip(x, cap)],)

    monkeypatch.setattr(zs, "_empty_cache", lambda: None)
    model = ZImageTransformer2DModel().eval()
    pipe = types.SimpleNamespace(transformer = model)
    ss.install_static_step_skip(pipe)
    try:
        zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True)
        ss.reset_static_step_skip(pipe, 12, step_signal = True)
        skip = model._unsloth_static_skip
        outputs = []
        with zs.render(pipe, 1), torch.no_grad():
            for step in range(4):
                model.fail_batched = step == 2
                out = model(
                    [torch.zeros(1), torch.zeros(1)],
                    torch.tensor([1.0 - step / 12] * 2),
                    [torch.tensor([10.0]), torch.tensor([-10.0])],
                    return_dict = False,
                )[0]
                outputs.append([x.item() for x in out])
                ss.mark_step_end(pipe)
        assert not skip.plan[3] and skip.stats["skipped"] > 0
        assert outputs == [[10.0, -10.0]] * 4
        assert zs.zimage_cfg_split_status(pipe)["oom_split"] == 1
    finally:
        zs.uninstall_zimage_cfg_split(pipe)
        ss.uninstall_static_step_skip(pipe)


def test_compiled_only_eager_with_a_disabled_router_runs_batched():
    """A CFG-parallel router that gave up (replica build or run failed) pairs nothing: the eager call stays batched."""
    from core.inference.diffusion_multigpu import CfgParallelRouter

    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True)
    router = CfgParallelRouter(model, primary = "cpu", secondary = "cpu")
    router.disabled_reason = "replica failed"
    model.__dict__["forward"] = router
    model.__dict__["_unsloth_cfg_router"] = router
    try:
        with zs.render(pipe, 1):
            _run(model, *_guided_args(_latent(14), torch.tensor([0.4]), 9, 12, 14))
        assert zs.zimage_cfg_split_status(pipe)["split"] == 0
    finally:
        model.__dict__.pop("forward", None)
        model.__dict__.pop("_unsloth_cfg_router", None)


def test_compiled_only_ignores_a_poisoned_graph_handle():
    """A whole-step capture that failed for good stays in ``_unsloth_cuda_graphs`` but replays nothing."""
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True)
    handle = types.SimpleNamespace(enabled = True, poisoned = True, fallback = None, fallback_handle = None)
    pipe._unsloth_cuda_graphs = (handle,)
    args = _guided_args(_latent(15), torch.tensor([0.4]), 9, 12, 15)
    with zs.render(pipe, 1):
        _run(model, *args)
    assert zs.zimage_cfg_split_status(pipe)["split"] == 0
    handle.fallback_handle = types.SimpleNamespace(enabled = True, poisoned = False)
    with zs.render(pipe, 1):
        _run(model, *args)
    assert zs.zimage_cfg_split_status(pipe)["split"] == 1


def test_compiled_only_honours_the_background_compile_eager_gate():
    """Eager-forced with nothing recording (compile already running): the guided call runs batched."""
    from core.inference import diffusion_bg_compile as bg

    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True)
    pipe._unsloth_cuda_graphs = (object(),)
    args = _guided_args(_latent(16), torch.tensor([0.4]), 9, 12, 16)
    with bg.force_eager(), zs.render(pipe, 1):
        _run(model, *args)
    assert zs.zimage_cfg_split_status(pipe)["split"] == 0
    rec = bg.BackgroundCompile(model)
    assert rec.install()
    rec.state = "compiling"
    with bg.force_eager():
        assert zs.guided_runs_at_guidance_off_batch(pipe) is False
        with zs.render(pipe, 1):
            _run(model, *args)
    assert zs.zimage_cfg_split_status(pipe)["split"] == 0 and rec.samples == []
    with zs.render(pipe, 1):
        _run(model, *args)
    assert zs.zimage_cfg_split_status(pipe)["split"] == 1


def test_background_compile_recording_render_splits_and_records_the_compiled_shape():
    """The recording render splits, so the compiled render after the background replay compiles nothing new."""
    from core.inference import diffusion_bg_compile as bg
    from core.inference.diffusion_cuda_graph import _rebuild

    torch._dynamo.reset()
    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    assert zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True) is None
    rec = bg.BackgroundCompile(model)
    assert rec.install()
    args = _guided_args(_latent(17), torch.tensor([0.4]), 9, 12, 17)
    with torch.no_grad():
        batched = model(*args, return_dict = False)[0]
        alone = (
            model(*[a[:1] for a in args], return_dict = False)[0]
            + model(*[a[1:] for a in args], return_dict = False)[0]
        )
    with bg.force_eager():
        priced_off = zs.guided_runs_at_guidance_off_batch(pipe)
        with zs.render(pipe, 1):
            recorded = [_run(model, *args) for _ in range(3)]
    shapes = sorted(tuple(c[0].shape) for _, c, *_ in rec.samples)
    stats = dict(zs.zimage_cfg_split_status(pipe))
    # What kick() runs: regional compile, then the recorded inputs.
    _compile_blocks(model)
    with torch.no_grad():
        for spec, clones, *_ in rec.samples:
            a, k = _rebuild(spec, clones)
            model(*a, **k)
    warm = _graphs()
    assert warm > 0
    with zs.render(pipe, 1):
        _run(model, *args)
    assert _graphs() - warm == 0
    assert zs.zimage_cfg_split_status(pipe)["split"] == 4
    assert stats == {"armed": True, "split": 3, "batched": 0}
    assert shapes == [(16, 1, 8, 8)] * 2
    assert priced_off is True
    for out in recorded:
        for s, a, b in zip(out, alone, batched):
            assert torch.equal(s, a)
            torch.testing.assert_close(s, b, atol = 1e-5, rtol = 1e-4)
    assert bg.recording(model) is False


def test_eager_render_behind_an_active_router_is_priced_and_run_split():
    """Eager compiled_only behind an active router: priced guidance-off and keeps splitting if the router gives up."""
    from core.inference import diffusion_bg_compile as bg
    from core.inference.diffusion_multigpu import CfgParallelRouter

    model = _model()
    pipe = types.SimpleNamespace(transformer = model)
    zs.install_zimage_cfg_split(pipe, offload_policy = "none", compiled_only = True)
    assert zs.guided_runs_at_guidance_off_batch(pipe) is False
    router = CfgParallelRouter(model, primary = "cpu", secondary = "cpu")
    model.__dict__["forward"] = router
    model.__dict__["_unsloth_cfg_router"] = router
    args = _guided_args(_latent(17), torch.tensor([0.4]), 9, 12, 17)
    try:
        assert zs.guided_runs_at_guidance_off_batch(pipe) is True
        with bg.force_eager():
            assert zs.guided_runs_at_guidance_off_batch(pipe) is False
        with zs.render(pipe, 1):
            _run(model, *args)
            router.disabled_reason = "replica failed"
            _run(model, *args)
        assert zs.zimage_cfg_split_status(pipe)["split"] == 2
        assert zs.guided_runs_at_guidance_off_batch(pipe) is False
    finally:
        model.__dict__.pop("forward", None)
        model.__dict__.pop("_unsloth_cfg_router", None)


def test_studio_guard_disengages_a_pending_auto_cache_before_pricing():
    """A cache the AUTO toggle will turn off is disengaged before the guard, so the split is not priced as declined."""
    source = _diffusion_source()
    predict = source.index("auto_step_cache_on = auto_step_cache_mode(cache_steps)")
    guard = source.index("guard_cfg, extra_mib = split_aware_request_extra(")
    early = source.index("maybe_toggle_step_cache(", predict)
    assert predict < early < guard
    assert "if not auto_step_cache_on:" in source[predict:early]


def _diffusion_source():
    import pathlib
    return pathlib.Path(zs.__file__).with_name("diffusion.py").read_text(encoding = "utf-8")
