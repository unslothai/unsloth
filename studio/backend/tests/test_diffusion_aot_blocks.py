# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted dynamo-included block graphs (``diffusion_aot_blocks``): a restart serves repeated blocks without tracing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_aot_blocks as aot  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]


class Block(torch.nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d)
        self.fc1 = torch.nn.Linear(d, 4 * d)
        self.fc2 = torch.nn.Linear(4 * d, d)

    def forward(self, hidden_states, temb = None):
        x = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(hidden_states)), approximate = "tanh"))
        if temb is not None:
            x = x * (1 + temb[:, None])
        return hidden_states + x


class Model(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(self, n: int = 3):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block() for _ in range(n))

    def forward(self, x, temb):
        for b in self.blocks:
            x = b(x, temb = temb)
        return x


def _fake_pipe(model):
    return types.SimpleNamespace(transformer = model)


def test_install_is_inert_until_bound_and_respects_kill_switch(monkeypatch):
    monkeypatch.setattr(aot, "supported", lambda: True)
    model = Model()
    calls = []
    for b in model.blocks:
        b._compiled_call_impl = lambda *a, _b = b, **k: calls.append(_b) or _b._call_impl(*a, **k)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_AOT_BLOCKS", "0")
    assert aot.install(model, {"fullgraph": True}) is None
    monkeypatch.delenv("UNSLOTH_DIFFUSION_AOT_BLOCKS")
    assert aot.install(model, {"fullgraph": False}) is None  # graph breaks: aot_compile cannot serve it
    reg = aot.install(model, {"fullgraph": True})
    assert reg is not None and aot.registries(_fake_pipe(model)) == [reg]
    x, t = torch.randn(1, 4, 64), torch.randn(1, 64)
    with torch.no_grad():
        model(x, t)
    assert len(calls) == 3 and reg.stats["misses"] == 0 and reg.stats["hits"] == 0  # unbound: plain passthrough
    assert aot.bind(_fake_pipe(model), None) == 0 and reg.failed is not None


def test_subkey_tracks_sources_env_and_compile_kwargs(monkeypatch):
    base = aot.subkey({"fullgraph": True, "dynamic": None})
    assert aot.subkey({"fullgraph": True, "dynamic": None}) == base
    assert aot.subkey({"fullgraph": True, "dynamic": True}) != base
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SOME_PATCH", "0")
    assert aot.subkey({"fullgraph": True, "dynamic": None}) != base
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SOME_PATCH")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", "/somewhere/else")  # where the home lives changes no graph
    assert aot.subkey({"fullgraph": True, "dynamic": None}) == base
    monkeypatch.setattr(aot, "_SOURCE_DIGEST", "different-sources")
    assert aot.subkey({"fullgraph": True, "dynamic": None}) != base


def _entry(name, guard_type, value = None, derived = (), is_global = False):
    return types.SimpleNamespace(name = name, guard_type = guard_type, derived_guard_types = derived,
                                 is_global = is_global, has_value = value is not None, value = value)


def test_guard_filter_drops_only_what_cannot_be_serialised():
    entries = [
        _entry("L['self']._modules['fc1'].in_features", "EQUALS_MATCH", 64),
        _entry("G['torch']", "ID_MATCH", is_global = True),
        _entry("type(L['self']._modules['attn'].processor).__call__", "CLOSURE_MATCH"),
        # a constant match on a code object derives an identity guard: the serialiser refuses it too
        _entry("type(L['self']._modules['attn'].processor).__call__.__code__", "CONSTANT_MATCH", derived = ("ID_MATCH",)),
        _entry("L['hidden_states']", "TENSOR_MATCH", torch.zeros(2)),
    ]
    assert aot._guard_filter(entries) == [True, False, False, False, True]


def test_guard_filter_drops_guards_through_a_weight_the_pickler_cannot_copy():
    quant = pytest.importorskip("torchao.quantization")
    lin = torch.nn.Sequential(torch.nn.Linear(64, 64)).to(torch.bfloat16)
    quant.quantize_(lin, quant.Int8DynamicActivationInt8WeightConfig())
    weight = lin[0].weight
    if aot._meta_picklable(weight):
        pytest.skip("this torchao's int8 weight copies to meta: nothing to drop")
    owner = "L['self']._modules['to_q']"
    entries = [
        _entry(owner, "TYPE_MATCH", lin[0], derived = ("TYPE_MATCH",)),
        _entry(owner + "._parameters['weight']", "TENSOR_MATCH", weight),
        _entry(owner + "._parameters['weight'].original_weight_tensor.tensor_impl.int_data", "TENSOR_MATCH",
               torch.zeros(1, dtype = torch.int8)),
        _entry("L['self']._modules['to_k']", "TYPE_MATCH", torch.nn.Linear(2, 2), derived = ("TYPE_MATCH",)),
        _entry("L['self']._modules['to_q2']", "TYPE_MATCH", torch.nn.Linear(2, 2), derived = ("TYPE_MATCH",)),
    ]
    assert aot._guard_filter(entries) == [False, False, False, True, True]
    assert aot._meta_picklable(torch.zeros(3)) and aot._meta_picklable(None)


def test_fingerprint_tracks_code_classes_and_weights_the_dropped_guards_covered():
    model = Model()
    block = model.blocks[0]
    base = aot._code_fingerprint(block)
    assert aot._code_fingerprint(model.blocks[1]) == base  # same class, same code, same weight metadata
    orig = Block.forward
    try:
        Block.forward = lambda self, hidden_states, temb = None: orig(self, hidden_states, temb)  # a class patch
        assert aot._code_fingerprint(block) != base
    finally:
        Block.forward = orig
    assert aot._code_fingerprint(block) == base
    block.fc1.to(torch.float16)
    assert aot._code_fingerprint(block) != base
    block.fc1.to(torch.float32)
    block.fc2 = torch.nn.Sequential(block.fc2)  # a wrapped submodule
    assert aot._code_fingerprint(block) != base


def test_hook_check_sees_hooks_added_after_the_first_call():
    reg = aot.Registry({"fullgraph": True})
    block = Block()
    assert reg._hook_free(block)
    handle = block.fc2.register_forward_pre_hook(lambda m, a: None)
    assert not reg._hook_free(block)
    handle.remove()
    assert reg._hook_free(block)
    handle = torch.nn.modules.module.register_module_forward_hook(lambda m, i, o: None)
    try:
        assert not reg._hook_free(block)
    finally:
        handle.remove()


def _bound_forward(self, x):
    return type(self).forward(self, x)


def test_instance_bound_methods_carry_their_bound_name_only_while_pickling():
    import functools

    block = Block()
    block.fc1.forward = types.MethodType(_bound_forward, block.fc1)
    block.fc2.extra_repr = types.MethodType(functools.partial(lambda self: "x"), block.fc2)
    partial = block.fc2.extra_repr.__func__
    with aot._picklable_instance_forwards(block):
        # what dynamo's guard pickler looks up: getattr(instance, function.__name__)
        assert getattr(block.fc1, block.fc1.forward.__func__.__name__) == block.fc1.forward
        assert getattr(block.fc2, partial.__name__) == block.fc2.extra_repr
    assert _bound_forward.__name__ == "_bound_forward" and not hasattr(partial, "__name__")
    fp = aot._code_fingerprint(block)
    block.fc1.forward = types.MethodType(lambda self, x: x, block.fc1)
    assert aot._code_fingerprint(block) != fp  # the instance forward's code is fingerprinted


def test_fingerprint_is_the_same_in_every_process(tmp_path):
    """A frozenset constant iterates in per-process string-hash order: the fingerprint must not depend on it."""
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])
        from core.inference import diffusion_aot_blocks as aot
        def f(x):
            return x in {"alpha", "beta", "gamma", "delta", "epsilon"}
        import hashlib
        h = hashlib.sha256()
        aot._code_digest(f, h)
        print(h.hexdigest())
        """
    )
    outs = {
        subprocess.run([sys.executable, "-c", code, str(BACKEND)], env = dict(os.environ, PYTHONHASHSEED = str(seed)),
                       capture_output = True, text = True, timeout = 300).stdout.strip()
        for seed in (1, 2, 3, 4)
    }
    assert len(outs) == 1 and "" not in outs, outs


class _Processor:
    def __call__(self, x):
        return x


def test_fingerprint_tracks_the_attention_processor_class_code():
    block = Block()
    block.processor = _Processor()
    base = aot._code_fingerprint(block)
    orig = _Processor.__call__
    try:
        _Processor.__call__ = lambda self, x: x * 1
        assert aot._code_fingerprint(block) != base
    finally:
        _Processor.__call__ = orig
    block.processor = type("Other", (_Processor,), {})()
    assert aot._code_fingerprint(block) != base


def test_a_class_that_does_not_serialise_is_refused_once_and_on_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(aot, "supported", lambda: True)
    calls = []

    class _Refusing:
        def aot_compile(self, inputs):
            calls.append(1)
            raise RuntimeError("PackageError: ID_MATCH guard cannot be serialized.")

    monkeypatch.setattr(aot, "_compiler", lambda fn, kwargs: _Refusing())
    monkeypatch.setattr(aot, "_capturing", lambda: False)
    key = tmp_path / "key"

    def run():
        model = Model()
        for b in model.blocks:
            b._compiled_call_impl = b._call_impl
        reg = aot.install(model, {"fullgraph": True, "dynamic": None})
        aot.bind(_fake_pipe(model), types.SimpleNamespace(dir = str(key)))
        with torch.no_grad():
            model(torch.randn(1, 4, 64), torch.randn(1, 64))
            model(torch.randn(1, 8, 64), torch.randn(1, 64))  # a new signature: no second attempt either
        return reg

    reg = run()
    assert len(calls) == 1 and reg.refused == {"Block"}
    man = json.loads(next(key.rglob("manifest.json")).read_text())
    assert man["refused"] == ["Block"]
    reg = run()  # a restart reads the refusal: no second aot_compile
    assert len(calls) == 1 and reg.refused == {"Block"} and reg.stats["misses"] == 6


_SCRIPT = textwrap.dedent(
    r"""
    import json, os, sys, types
    sys.path.insert(0, os.environ["BACKEND"])
    import torch
    from torch._dynamo.utils import counters
    from core.inference import diffusion_aot_blocks as aot
    from tests.test_diffusion_aot_blocks import Model

    torch.manual_seed(0)
    if os.environ.get("FAMILY") == "flux-int8":
        # The real thing: diffusers' FLUX blocks (attention processors) with Studio's torchao int8 weights.
        from diffusers import FluxTransformer2DModel
        from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

        model = FluxTransformer2DModel(patch_size = 1, in_channels = 64, num_layers = 2, num_single_layers = 2,
                                       attention_head_dim = 64, num_attention_heads = 2, joint_attention_dim = 128,
                                       pooled_projection_dim = 128, axes_dims_rope = (16, 24, 24)).cuda().to(torch.bfloat16)
        # Studio's int8 layers (not the adaLN projections of the [1, D] timestep embedding: torch._int_mm needs M > 16).
        quantize_(model, Int8DynamicActivationInt8WeightConfig(),
                  filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear) and "blocks" in fqn and ".norm" not in fqn)
        # Studio's int8 GEMM binds a module-level function as each int8 Linear's instance ``forward``.
        def _linear_forward(self, x):
            return type(self).forward(self, x)

        for fqn, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and "blocks" in fqn and ".norm" not in fqn:
                m.forward = types.MethodType(_linear_forward, m)
        ids = torch.zeros(256, 3, device = "cuda")
        ids[:, 1], ids[:, 2] = torch.arange(256) // 16, torch.arange(256) % 16
        inputs = dict(hidden_states = torch.randn(1, 256, 64, device = "cuda", dtype = torch.bfloat16),
                      encoder_hidden_states = torch.randn(1, 32, 128, device = "cuda", dtype = torch.bfloat16),
                      pooled_projections = torch.randn(1, 128, device = "cuda", dtype = torch.bfloat16),
                      timestep = torch.tensor([0.5], device = "cuda"), img_ids = ids,
                      txt_ids = torch.zeros(32, 3, device = "cuda"), return_dict = False)
        run = lambda: model(**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in inputs.items()})[0]
        hook_target = model.transformer_blocks[0].ff
    else:
        model = Model().cuda().to(torch.bfloat16)
        x = torch.randn(1, 256, 64, device = "cuda", dtype = torch.bfloat16)
        t = torch.randn(1, 64, device = "cuda", dtype = torch.bfloat16)
        run = lambda: model(x.clone(), t.clone())  # a pipeline's latents are inference tensors, like block outputs
        hook_target = model.blocks[0].fc1
    names = set(model._repeated_blocks)
    for b in model.modules():
        if type(b).__name__ in names:
            b.compile(fullgraph = True, dynamic = None)
    if os.environ.get("HOOK") == "1":
        # A hook added after the artifact was saved: a serialised artifact would skip it, so the block must not use it.
        hook_target.register_forward_hook(lambda m, i, o: o * 2)
    reg = aot.install(model, {"fullgraph": True, "dynamic": None})
    if os.environ.get("FAMILY") == "flux-int8":
        from core.inference import diffusion_block_restride

        diffusion_block_restride.install(model, None)  # as Studio does: single block 0 shares blocks 1..N's graph
    pipe = types.SimpleNamespace(transformer = model)
    aot.bind(pipe, types.SimpleNamespace(dir = os.environ["KEYDIR"]))
    with torch.inference_mode():
        out = run()
    torch.save(out.cpu(), os.environ["OUT"])
    print("RESULT", json.dumps({"stats": reg.describe() if reg is not None else None, "classes": sorted(names),
                                "frames": counters["stats"]["unique_graphs"]}))
    """
)


def _close_to_the_normal_path(tmp_path: Path) -> None:
    """The kill-switch run compiles the same graph on the normal path in another process. Its kernels are tuned in
    that process, so the bits can differ (as two normal-path processes can on some cards); the values may not."""
    a, b = torch.load(tmp_path / "first.pt").float(), torch.load(tmp_path / "off.pt").float()
    assert torch.isfinite(a).all()
    assert ((a - b).norm() / b.norm()).item() < 1e-2


def _run(tmp_path: Path, tag: str, extra_env: dict | None = None) -> dict:
    env = dict(os.environ)
    env.update(
        BACKEND = str(BACKEND),
        KEYDIR = str(tmp_path / "key"),
        OUT = str(tmp_path / f"{tag}.pt"),
        TORCHINDUCTOR_CACHE_DIR = str(tmp_path / "inductor"),
        TRITON_CACHE_DIR = str(tmp_path / "triton"),
        PYTHONPATH = str(BACKEND),
    )
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, "-c", _SCRIPT], cwd = str(BACKEND), env = env, capture_output = True,
                       text = True, timeout = 900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert line, r.stdout[-3000:] + r.stderr[-3000:]
    return json.loads(line[-1][len("RESULT "):])


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
def test_restart_serves_blocks_from_the_artifact_without_tracing_and_bit_identical(tmp_path):
    if not aot.supported():
        pytest.skip("this torch has no aot_compile")
    first = _run(tmp_path, "first")
    # The first block compiles through aot_compile and is persisted at once; the other two reuse its graph.
    assert first["stats"]["compiled"] == 1 and first["stats"]["saved"] == 1, first
    assert first["stats"]["hits"] == 2 and first["stats"]["misses"] == 0, first
    man = json.loads(next((tmp_path / "key").rglob("manifest.json")).read_text())
    assert len(man["entries"]) == 1 and man["entries"][0]["cls"] == "Block"
    restart = _run(tmp_path, "restart")
    assert restart["stats"]["loaded"] == 1 and restart["stats"]["hits"] == 3 and restart["stats"]["misses"] == 0
    assert restart["frames"] == 0, "dynamo traced a block the artifact covers"
    assert torch.equal(torch.load(tmp_path / "first.pt"), torch.load(tmp_path / "restart.pt"))
    # A hook the artifact never saw: that block takes the normal path (and honours the hook), the others still hit.
    hooked = _run(tmp_path, "hooked", {"HOOK": "1"})
    assert hooked["stats"]["hits"] == 2 and hooked["stats"]["misses"] == 1, hooked
    assert not torch.equal(torch.load(tmp_path / "first.pt"), torch.load(tmp_path / "hooked.pt"))
    # Kill switch: the normal torch.compile path.
    off = _run(tmp_path, "off", {"UNSLOTH_DIFFUSION_AOT_BLOCKS": "0"})
    assert off["frames"] >= 1
    _close_to_the_normal_path(tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA only")
def test_real_flux_blocks_with_torchao_int8_weights_persist_and_restart_bit_identical(tmp_path):
    """diffusers' FLUX blocks (attention processor identity guards) with int8 torchao weights (a guard pickler that
    cannot copy them) and instance-bound Linear forwards: both block classes serialise, and a restart serves every
    block without tracing, bit-identical to the first start."""
    if not aot.supported():
        pytest.skip("this torch has no aot_compile")
    pytest.importorskip("torchao")
    pytest.importorskip("diffusers")
    flux = {"FAMILY": "flux-int8"}
    first = _run(tmp_path, "first", flux)
    assert first["stats"]["compiled"] == 2 and first["stats"]["saved"] == 2, first
    assert first["stats"]["hits"] == 2 and first["stats"]["misses"] == 0, first
    assert not first["stats"].get("refused"), first
    restart = _run(tmp_path, "restart", flux)
    assert restart["stats"]["loaded"] == 2 and restart["stats"]["misses"] == 0, restart
    assert restart["stats"]["hits"] == 4 and restart["frames"] == 0, restart
    assert torch.equal(torch.load(tmp_path / "first.pt"), torch.load(tmp_path / "restart.pt"))
    off = _run(tmp_path, "off", dict(flux, UNSLOTH_DIFFUSION_AOT_BLOCKS = "0"))
    _close_to_the_normal_path(tmp_path)
    hooked = _run(tmp_path, "hooked", dict(flux, HOOK = "1"))
    assert hooked["stats"]["misses"] >= 1 and not torch.equal(torch.load(tmp_path / "first.pt"),
                                                              torch.load(tmp_path / "hooked.pt"))
