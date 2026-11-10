# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""CP over Unsloth's fast attention (Qwen3, Mistral, Granite, Cohere) and unsloth_zoo's compiled attention: two gloo
CPU ranks vs one process.
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


def _family_config(family, kv_heads):
    from transformers import CohereConfig, GraniteConfig, MistralConfig

    common = dict(
        hidden_size = HIDDEN,
        intermediate_size = 64,
        head_dim = 8,
        num_attention_heads = 4,
        num_key_value_heads = kv_heads,
        num_hidden_layers = 1,
        vocab_size = 96,
        attention_dropout = 0.0,
    )
    if family == "mistral":
        return MistralConfig(sliding_window = None, **common)
    if family == "granite":
        # A non-default attention multiplier: the scale must survive the exchange.
        return GraniteConfig(attention_multiplier = 0.2, **common)
    return CohereConfig(use_qk_norm = False, **common)


def _fast_family_attention(family, kv_heads):
    """The HF attention module running Unsloth's fast forward for that family, on CPU-safe kernels."""
    if family == "qwen3":
        return _fast_qwen3_attention(kv_heads)
    module = importlib.import_module(f"unsloth.models.{family}")
    modeling = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    cls = getattr(modeling, f"{family.capitalize()}Attention")
    forward = getattr(module, f"{family.capitalize()}Attention_fast_forward")
    config = _family_config(family, kv_heads)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    attn = cls(config, 0).float().train()
    attn.__class__ = _fast_class(cls, forward)
    attn.rotary_emb = _Rotary(8, 10000.0)
    return attn


def _install_family_kernels(family):
    _install_cpu_kernels()
    if family != "qwen3":
        importlib.import_module(f"unsloth.models.{family}").fast_rope_embedding = _rope


# Granite takes the model-level table (asserts position_embeddings); Mistral / Cohere build theirs per layer.
_TABLE_ROWS = 64


def _attn_kwargs(family, attn):
    if family == "granite":
        return {"position_embeddings": attn.rotary_emb.get_cached(_TABLE_ROWS)}
    return {}


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


_MORE_FAMILIES = ["mistral", "granite", "cohere"]


@pytest.mark.parametrize("family", _MORE_FAMILIES)
def test_more_unsloth_fast_attention_passes_the_cp_check(family):
    model = _wrap(_fast_family_attention(family, 2))
    assert cp._uses_unsloth_attention(model)
    assert cp._supports_context_parallel(model, "ulysses")
    assert cp._supports_context_parallel(model, "ring")
    cp._check_cp_model(model, 2, "ulysses")


@pytest.mark.parametrize("family", _MORE_FAMILIES)
def test_more_fast_attention_keeps_window_refusal(family):
    attn = _fast_family_attention(family, 2)
    attn.config.sliding_window = 8
    with pytest.raises(NotImplementedError, match = "sliding window"):
        cp._check_cp_model(_wrap(attn), 2, "ulysses")


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


def _fast_parity_worker(
    rank,
    world,
    kv_heads,
    packed,
    out_dir,
    family = "qwen3",
):
    _install_family_kernels(family)
    attn = _fast_family_attention(family, kv_heads)
    extra = _attn_kwargs(family, attn)
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
            hidden_states = x_ref[:, a:b],
            position_ids = torch.arange(b - a).expand(bsz, -1),
            **extra,
        )
        full[:, a:b] = out.detach()
        (out * weight[:, a:b]).sum().backward()
    reference = {n: p.grad.clone() for n, p in attn.named_parameters()}
    x_grad = x_ref.grad.clone()
    attn.zero_grad(set_to_none = True)

    calls, windows = [], []
    original = ad.ulysses_attention

    def counted(*args, **kwargs):
        calls.append(1)
        windows.append(args[1].flash_dense_kwargs.get("window_size"))
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
        kwargs = dict(extra)
        if packed:
            kwargs["packed_seq_lengths"] = inputs["packed_seq_lengths"]
        out, _, _ = attn(hidden_states = x_cp, position_ids = inputs["position_ids"], **kwargs)
        (out * weight[:, sl]).sum().backward()
    ad.ulysses_attention = original
    assert calls == [1], calls
    if family == "granite":
        # Flash's dense window must span the gathered sequence (a shard-sized one cuts attention off).
        assert windows[0] is not None and min(windows[0]) >= SEQ - 1, windows
    else:
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
            os.path.join(
                out_dir, f"fast_{family}_kv{kv_heads}_{'packed' if packed else 'rows'}.txt"
            ),
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


@pytest.mark.parametrize(
    "kv_heads, packed",
    [
        (2, False),
        (1, False),  # 1 kv head replicated over cp 2
        (2, True),
    ],
)
@pytest.mark.parametrize("family", _MORE_FAMILIES)
def test_more_unsloth_fast_attention_matches_one_process(family, kv_heads, packed):
    generic._spawn(_fast_parity_worker, 2, kv_heads, packed, generic._OUT, family)


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


_RL_REFUSAL = (
    "DPO / GRPO context parallelism supports Unsloth's Llama-style and Qwen3 attention only"
)


def test_rl_context_parallel_declarations():
    """DPO / GRPO: Llama-style and Qwen3 fast attention (RL parity below); Mistral / Granite / Cohere SFTTrainer only."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from unsloth.models.llama import LlamaAttention_fast_forward

    cp._check_cp_model(_wrap(_fast_qwen3_attention(2)), 2, "ulysses", allow_hybrids = False)
    llama = _fast_qwen3_attention(2)
    llama.__class__ = _fast_class(Qwen3Attention, LlamaAttention_fast_forward)
    cp._check_cp_model(_wrap(llama), 2, "ulysses", allow_hybrids = False)
    for family in _MORE_FAMILIES:
        model = _wrap(_fast_family_attention(family, 2))
        cp._check_cp_model(model, 2, "ulysses")
        with pytest.raises(NotImplementedError, match = _RL_REFUSAL):
            cp._check_cp_model(model, 2, "ulysses", allow_hybrids = False)


class _FastQwen3RL(torch.nn.Module):
    """Two pre-norm blocks on the real fast Qwen3 attention under an embed_tokens + layers decoder (the RL hooks'
    target), scored by TRL's DPO / GRPO objectives from tests/test_context_parallel_rl.py."""

    def __init__(self, seed, dtype):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(96, HIDDEN)
        self.model.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.self_attn = _fast_qwen3_attention(2)
            layer.mlp = torch.nn.Linear(HIDDEN, HIDDEN, bias = False)
            self.model.layers.append(layer)
        self.model.config = self.model.layers[0].self_attn.config
        self.model.forward = self._decode
        self.lm_head = torch.nn.Linear(HIDDEN, 96, bias = False)
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                p.normal_(0, 0.3)
        self.to(dtype)

    def _decode(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        **kwargs,
    ):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if cp.get_cp_manager() is not None:
            assert attention_mask is None  # the RL pre-hook drops the per-shard mask
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        x = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            a, _, _ = layer.self_attn(
                hidden_states = F.rms_norm(x, (HIDDEN,)), position_ids = position_ids
            )
            x = x + a
            x = x + layer.mlp(F.silu(F.rms_norm(x, (HIDDEN,))))
        return BaseModelOutputWithPast(last_hidden_state = F.rms_norm(x, (HIDDEN,)))

    def token_logps(self, ids, mask):
        h = self.model(input_ids = ids, attention_mask = mask).last_hidden_state
        return (
            torch.log_softmax(self.lm_head(h)[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        )


def _fast_qwen3_rl_worker(rank, world, out_dir):
    import contextlib

    rl = importlib.import_module("tests.test_context_parallel_rl")
    _install_cpu_kernels()
    g = torch.Generator().manual_seed(3)
    rand = lambda n: torch.randint(1, 96, (n,), generator = g).tolist()
    prompts = [rand(7), rand(12)]
    chosen = [p + rand(n) for p, n in zip(prompts, (9, 5))]
    rejected = [p + rand(n) for p, n in zip(prompts, (4, 14))]
    ids, mask = rl._right_pad(chosen + rejected)  # 26 tokens, the padded tail crosses the shard
    completion = mask.clone()
    for i, n in enumerate([7, 12, 7, 12]):
        completion[i, :n] = 0
    dpo = (ids, mask, completion[:, 1:])
    prompt = rand(9)
    g_ids, g_mask = rl._right_pad([prompt + rand(n) for n in (11, 3, 17, 8)])
    g_completion = g_mask.clone()
    g_completion[:, :9] = 0
    grpo = (g_ids, g_mask, g_completion[:, 1:], torch.tensor([0.3, 1.0, 0.1, 0.65]))
    lines = []
    for dtype, tol in ((torch.float64, 1e-10), (torch.float32, 1e-5)):
        for objective, batch in ((rl._dpo_sigmoid, dpo), (rl._grpo_dapo, grpo)):
            results = []
            for use_cp in (False, True):
                policy = _FastQwen3RL(0, dtype)
                ref = _FastQwen3RL(1, dtype).requires_grad_(False)
                scope, calls = contextlib.nullcontext(), []
                if use_cp:
                    cp._check_cp_model(policy, world, "ulysses", allow_hybrids = False)
                    manager = generic._manager(world, rank)
                    cp.attach_rl_hooks(manager, policy)
                    cp.attach_rl_hooks(manager, ref)
                    scope = cp._manager_scope(manager)
                    original = ad.ulysses_attention

                    def counted(*args, **kwargs):
                        calls.append(1)
                        return original(*args, **kwargs)

                    ad.ulysses_attention = counted
                try:
                    with scope:
                        loss = objective(policy, ref, batch, world if use_cp else 1)
                        loss.backward()
                finally:
                    if use_cp:
                        ad.ulysses_attention = original
                grads = torch.cat([p.grad.flatten() for p in policy.parameters()])
                if use_cp:  # DDP's average over the CP group
                    # 2 layers x (policy + reference) forwards: the fast attention exchanged every time.
                    assert len(calls) == 4, calls
                    dist.all_reduce(grads)
                    grads /= world
                results.append((loss.detach(), grads))
            (loss_ref, grad_ref), (loss_cp, grad_cp) = results
            name = f"{objective.__name__} {dtype}"
            loss_rel = abs(float(loss_cp - loss_ref)) / abs(float(loss_ref))
            grad_rel = float((grad_cp - grad_ref).norm() / grad_ref.norm())
            assert loss_rel <= tol, f"{name}: loss rel {loss_rel:.3e}"
            assert grad_rel <= tol, f"{name}: grad rel {grad_rel:.3e}"
            lines.append(f"{name} loss_rel {loss_rel:.3e} grad_rel {grad_rel:.3e}")
    if rank == 0 and out_dir:
        import os
        with open(os.path.join(out_dir, "fast_qwen3_rl.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")


def test_fast_qwen3_attention_dpo_grpo_match_one_process():
    """rl = True on Qwen3: the RL decoder hooks over the real fast Qwen3 attention equal one process."""
    _qwen3_config(2)
    generic._spawn(_fast_qwen3_rl_worker, 2, generic._OUT)


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
