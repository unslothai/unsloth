# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CP over the fast Qwen3 attention and unsloth_zoo's compiled attention: two gloo CPU ranks vs one process.
GPU-only Triton RoPE / RMSNorm are swapped for torch references; the attention path is the real one."""

import importlib
import inspect
import sys
import textwrap
import types

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
generic = importlib.import_module("tests.test_cp_ulysses_generic_gloo")

SEQ = 16
HIDDEN = 32


def _qwen3_config(kv_heads):
    config = generic._config("qwen3")
    config.hidden_size = HIDDEN
    config.intermediate_size = 64
    config.head_dim = 8
    config.num_attention_heads = 4
    config.num_key_value_heads = kv_heads
    return config


def _fast_class(cls, forward):
    return type(f"Fast{cls.__name__}", (cls,), {"forward": forward})


# unsloth_zoo's compiler template: the class forward only calls a module-level `<Class>_forward`.
def _compiled_like(cls):
    modeling = sys.modules[cls.__module__]
    source = textwrap.dedent(inspect.getsource(cls.forward))
    source = source.replace("def forward(", f"def {cls.__name__}_forward(", 1)
    lines = [line for line in source.splitlines() if not line.lstrip().startswith("@")]
    namespace = dict(vars(modeling))
    namespace["__name__"] = "unsloth_compiled_module_fake"
    exec("\n".join(lines), namespace)
    exec(
        textwrap.dedent(
            f"""
            def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values = None, **kwargs):
                return {cls.__name__}_forward(self, hidden_states = hidden_states,
                    position_embeddings = position_embeddings, attention_mask = attention_mask,
                    past_key_values = past_key_values, **kwargs)
            """
        ),
        namespace,
    )
    return type(cls.__name__, (cls,), {"forward": namespace["forward"]})


def _compile_like_model(model):
    for module in cp._attention_modules(model):
        module.__class__ = _compiled_like(type(module).__mro__[0])
    return model


class _Rotary:
    def __init__(self, head_dim, theta):
        self.head_dim, self.theta, self.asked = head_dim, theta, []

    def extend_rope_embedding(self, x, seq_len):
        self.asked.append(int(seq_len))

    def get_cached(
        self,
        seq_len,
        device_index = None,
    ):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        freqs = torch.outer(torch.arange(seq_len).float(), inv)
        emb = torch.cat([freqs, freqs], dim = -1)
        return emb.cos(), emb.sin()


def _rope(Q, K, cos, sin, position_ids):
    cos, sin = cos[position_ids].unsqueeze(1), sin[position_ids].unsqueeze(1)

    def rot(x):
        a, b = x.chunk(2, dim = -1)
        return torch.cat([-b, a], dim = -1)

    return Q * cos + rot(Q) * sin, K * cos + rot(K) * sin


def _install_cpu_kernels():
    from unsloth.kernels import qk_norm_rope
    from unsloth.models import qwen3

    qk_norm_rope.QK_NORM_ROPE_ENABLED = False
    qwen3.fast_rms_layernorm = lambda norm, X, **_: norm(X)
    qwen3.fast_rope_embedding = _rope
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False


def _fast_qwen3_attention(kv_heads):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from unsloth.models.qwen3 import Qwen3Attention_fast_forward

    config = _qwen3_config(kv_heads)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    attn = Qwen3Attention(config, 0).float().train()
    attn.__class__ = _fast_class(Qwen3Attention, Qwen3Attention_fast_forward)
    attn.rotary_emb = _Rotary(config.head_dim, getattr(config, "rope_theta", 10000.0))
    return attn


def _wrap(attn):
    root = torch.nn.Module()
    root.config = attn.config
    root.layers = torch.nn.ModuleList([torch.nn.Module()])
    root.layers[0].self_attn = attn
    return root


@pytest.mark.parametrize("kv_heads", [2, 1])
def test_unsloth_fast_qwen3_attention_passes_the_cp_check(kv_heads):
    model = _wrap(_fast_qwen3_attention(kv_heads))
    assert cp._supports_context_parallel(model, "ulysses")
    assert cp._supports_context_parallel(model, "ring")
    cp._check_cp_model(model, 2, "ulysses")


def test_compiled_fastmodel_attention_is_hf_registry_attention():
    model = _compile_like_model(generic._model("qwen3", "sdpa"))
    assert all(uh.is_hf_registry_attention(m) for m in cp._attention_modules(model))
    assert cp._generic_hf_attention(model)
    cp._check_cp_model(model, 2, "ulysses")


def test_a_fast_forward_without_the_cp_declaration_stays_refused():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from unsloth.models.qwen3 import Qwen3Attention_fast_forward

    def undeclared(self, *args, **kwargs):
        return Qwen3Attention_fast_forward(self, *args, **kwargs)

    attn = _fast_qwen3_attention(2)
    attn.__class__ = _fast_class(Qwen3Attention, undeclared)
    model = _wrap(attn)
    assert not cp._supports_context_parallel(model, "ulysses")
    with pytest.raises(NotImplementedError, match = "context parallelism supports"):
        cp._check_cp_model(model, 2, "ulysses")


def _fast_parity_worker(rank, world, kv_heads, packed, out_dir):
    _install_cpu_kernels()
    attn = _fast_qwen3_attention(kv_heads)
    model = _wrap(attn)
    cp._check_cp_model(model, world, "ulysses")
    bsz = 1 if packed else 2
    x = torch.randn(bsz, SEQ, HIDDEN, generator = torch.Generator().manual_seed(1))
    weight = torch.randn(bsz, SEQ, HIDDEN, generator = torch.Generator().manual_seed(2))
    lengths = [6, 7, 3] if packed else None
    positions = torch.arange(SEQ).expand(bsz, -1)

    pieces = [(0, SEQ)] if lengths is None else []
    start = 0
    for n in lengths or []:
        pieces.append((start, start + n))
        start += n
    full = torch.zeros(bsz, SEQ, HIDDEN)
    x_ref = x.clone().requires_grad_(True)
    for a, b in pieces:
        out, _, _ = attn(
            hidden_states = x_ref[:, a:b], position_ids = torch.arange(b - a).expand(bsz, -1)
        )
        full[:, a:b] = out.detach()
        (out * weight[:, a:b]).sum().backward()
    reference = {n: p.grad.clone() for n, p in attn.named_parameters()}
    x_grad = x_ref.grad.clone()
    attn.zero_grad(set_to_none = True)

    calls = []
    original = ad.ulysses_attention

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    ad.ulysses_attention = counted
    manager = generic._manager(world, rank)
    manager.attach_attention_hooks(model)
    inputs = {"input_ids": torch.zeros(bsz, SEQ, dtype = torch.long)}
    if packed:
        inputs["packed_seq_lengths"] = torch.tensor(lengths, dtype = torch.int32)
        positions = cp._document_positions(lengths, SEQ, x.device)
    inputs["position_ids"] = positions.clone()
    local = SEQ // world
    sl = slice(rank * local, (rank + 1) * local)
    x_cp = x[:, sl].clone().requires_grad_(True)
    attn.rotary_emb.asked.clear()
    with manager.apply(inputs):
        kwargs = {}
        if packed:
            kwargs["packed_seq_lengths"] = inputs["packed_seq_lengths"]
        out, _, _ = attn(hidden_states = x_cp, position_ids = inputs["position_ids"], **kwargs)
        (out * weight[:, sl]).sum().backward()
    ad.ulysses_attention = original
    assert calls == [1], calls
    # The rotary table must cover the global positions, not only this shard's length.
    assert attn.rotary_emb.asked and min(attn.rotary_emb.asked) >= SEQ, attn.rotary_emb.asked
    torch.testing.assert_close(out.detach(), full[:, sl], rtol = 1e-5, atol = 1e-6)
    gathered = [torch.zeros_like(x_cp.grad) for _ in range(world)]
    dist.all_gather(gathered, x_cp.grad)
    torch.testing.assert_close(torch.cat(gathered, dim = 1), x_grad, rtol = 1e-4, atol = 1e-6)
    worst = 0.0
    for name, param in attn.named_parameters():
        grad = param.grad.clone()
        dist.all_reduce(grad)
        torch.testing.assert_close(grad, reference[name], rtol = 1e-4, atol = 1e-6, msg = name)
        worst = max(worst, float((grad - reference[name]).abs().max()))
    if rank == 0 and out_dir:
        import os
        with open(
            os.path.join(out_dir, f"fast_qwen3_kv{kv_heads}_{'packed' if packed else 'rows'}.txt"),
            "w",
        ) as f:
            f.write(f"ulysses calls {len(calls)} max grad diff {worst:.3e}\n")


@pytest.mark.parametrize(
    "kv_heads, packed",
    [
        (2, False),
        (1, False),  # 1 kv head replicated over cp 2
        (2, True),
    ],
)
def test_unsloth_fast_qwen3_attention_matches_one_process(kv_heads, packed):
    _qwen3_config(kv_heads)  # skip before forking when the repo is unavailable
    generic._spawn(_fast_parity_worker, 2, kv_heads, packed, generic._OUT)


def _compiled_parity_worker(rank, world, packed, out_dir):
    original = generic._model
    generic._model = lambda kind, implementation: _compile_like_model(
        original(kind, implementation)
    )
    try:
        generic._parity_worker(rank, world, "qwen3", "sdpa", packed, out_dir)
    finally:
        generic._model = original


@pytest.mark.parametrize("packed", [False, True])
def test_compiled_fastmodel_attention_matches_one_process(packed):
    generic._config("qwen3")
    generic._spawn(_compiled_parity_worker, 2, packed, "")


_RL_REFUSAL = "DPO / GRPO context parallelism supports Unsloth's Llama-style attention only"


def test_rl_keeps_context_parallel_llama_only():
    """Qwen3 is SFTTrainer only: DPO / GRPO need an RL parity run on the fast Qwen3 attention first."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from unsloth.models.llama import LlamaAttention_fast_forward

    qwen3 = _wrap(_fast_qwen3_attention(2))
    cp._check_cp_model(qwen3, 2, "ulysses")
    with pytest.raises(NotImplementedError, match = _RL_REFUSAL):
        cp._check_cp_model(qwen3, 2, "ulysses", allow_hybrids = False)
    llama = _fast_qwen3_attention(2)
    llama.__class__ = _fast_class(Qwen3Attention, LlamaAttention_fast_forward)
    cp._check_cp_model(_wrap(llama), 2, "ulysses", allow_hybrids = False)


_REAL_COMPILED = r"""
import os, sys
os.environ["UNSLOTH_COMPILE_LOCATION"] = sys.argv[1]
import unsloth
from unsloth_zoo.compiler import unsloth_compile_transformers
unsloth_compile_transformers(model_type = "qwen3", import_from_cache = False)
import transformers.models.qwen3.modeling_qwen3 as modeling
from unsloth.utils import ulysses_hybrid as uh
forward = modeling.Qwen3Attention.forward
caller = uh._registry_caller(forward)
# The shape the one-step delegation follow relies on: the class forward only calls the module-level copy.
assert forward.__module__ == "unsloth_compiled_module_qwen3", forward.__module__
assert forward.__code__.co_names == ("Qwen3Attention_forward",), forward.__code__.co_names
assert caller is not None and caller.__name__ == "Qwen3Attention_forward", caller
assert caller.__module__ == "unsloth_compiled_module_qwen3", caller.__module__
from tests import test_cp_ulysses_generic_gloo as g
for implementation, packed in (("sdpa", False), ("sdpa", True), ("eager", False)):
    g._spawn(g._parity_worker, 2, "qwen3", implementation, packed, "")
print("REAL_COMPILED_OK")
"""


def test_real_zoo_compiled_qwen3_attention_matches_one_process(tmp_path):
    """Real unsloth_zoo compiler output (not a copy): pins the class forward shape and runs Ulysses parity on it."""
    import os
    import subprocess

    pytest.importorskip("unsloth_zoo.compiler")
    generic._config("qwen3")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Fresh interpreter: no test-only CPU fallback switches from tests/conftest.py.
    env = {k: v for k, v in os.environ.items() if k != "UNSLOTH_DISABLE_CPU_FALLBACK"}
    env.update(
        CUDA_VISIBLE_DEVICES = "",
        PYTHONPATH = os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])),
    )
    result = subprocess.run(
        [sys.executable, "-c", _REAL_COMPILED, str(tmp_path / "compiled")],
        cwd = root,
        env = env,
        capture_output = True,
        text = True,
        timeout = 1200,
    )
    assert result.returncode == 0 and "REAL_COMPILED_OK" in result.stdout, (
        result.stdout[-3000:] + result.stderr[-3000:]
    )
