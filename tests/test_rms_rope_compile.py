# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""fast_rms_layernorm / fast_rope_embedding trace under torch.compile with eager's bytes."""

import hashlib
import math
import os

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level = True)

pytest.importorskip("triton")

import unsloth  # noqa: F401  (patches first)
import torch._dynamo.utils as dynamo_utils
from unsloth.kernels import fast_lora
from unsloth.kernels.rms_layernorm import Fast_RMS_Layernorm, fast_rms_layernorm
from unsloth.kernels.rope_embedding import (
    Fast_RoPE_Embedding,
    Fast_RoPE_Embedding_QK,
    fast_rope_embedding,
)

# Without triton_op the implementation may keep a graph break; results must still be right.
TRACEABLE = hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton")
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.fixture(autouse = True)
def _fresh_dynamo():
    torch._dynamo.reset()
    dynamo_utils.counters.clear()
    yield
    torch._dynamo.reset()


def _compile(fn):
    return torch.compile(fn, fullgraph = TRACEABLE)


def _graph_breaks():
    return sum(dynamo_utils.counters["graph_break"].values())


def _bytes_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    assert a.shape == b.shape and a.dtype == b.dtype
    iv = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return torch.equal(a.contiguous().view(iv), b.contiguous().view(iv))


class _Norm(torch.nn.Module):
    def __init__(self, dim, dtype, gemma):
        super().__init__()
        g = torch.Generator(device = "cuda").manual_seed(1)
        w = torch.randn(dim, device = "cuda", generator = g, dtype = torch.float32) * 0.1
        self.weight = torch.nn.Parameter((w if gemma else w + 1).to(dtype))
        self.variance_epsilon = 1e-6


def _rms_case(fn, shape, dtype, gemma, non_contiguous):
    g = torch.Generator(device = "cuda").manual_seed(0)
    norm = _Norm(shape[-1], dtype, gemma)
    if non_contiguous:
        base = torch.randn(*shape[:-1], 2 * shape[-1], device = "cuda", generator = g, dtype = dtype)
        X = base[..., ::2].detach().requires_grad_(True)
    else:
        X = torch.randn(shape, device = "cuda", generator = g, dtype = dtype).requires_grad_(True)
    dY = torch.randn(shape, device = "cuda", generator = g, dtype = dtype)
    before = X.detach().clone()
    Y = fn(norm, X, gemma)
    Y.backward(dY)
    assert _bytes_equal(X.detach(), before), "the input was mutated"
    return Y.detach(), X.grad, norm.weight.grad


@pytest.mark.parametrize("dtype", DTYPES, ids = str)
@pytest.mark.parametrize("gemma", [False, True], ids = ["llama", "gemma"])
@pytest.mark.parametrize(
    "shape,non_contiguous",
    [((64, 256), False), ((2, 33, 384), False), ((2, 33, 384), True)],
    ids = ["2d", "3d", "3d_strided"],
)
def test_rms_layernorm_compiled_matches_eager(shape, non_contiguous, gemma, dtype):
    """Compiled forward and backward are byte-identical to eager, with no graph break."""
    eager = _rms_case(fast_rms_layernorm, shape, dtype, gemma, non_contiguous)
    compiled = _rms_case(
        _compile(lambda n, x, gm: fast_rms_layernorm(n, x, gm)), shape, dtype, gemma, non_contiguous
    )
    for e, c, name in zip(eager, compiled, ("Y", "dX", "dW")):
        assert _bytes_equal(e, c), name
    if TRACEABLE:
        assert _graph_breaks() == 0


@pytest.mark.parametrize("gemma", [False, True], ids = ["llama", "gemma"])
def test_rms_layernorm_eager_is_the_autograd_function(gemma):
    """Outside torch.compile the public function is still Fast_RMS_Layernorm, bytes for bytes."""
    got = _rms_case(fast_rms_layernorm, (2, 17, 256), torch.bfloat16, gemma, False)
    ref = _rms_case(
        lambda n, x, gm: Fast_RMS_Layernorm.apply(x, n.weight, n.variance_epsilon, gm),
        (2, 17, 256),
        torch.bfloat16,
        gemma,
        False,
    )
    for e, c, name in zip(ref, got, ("Y", "dX", "dW")):
        assert _bytes_equal(e, c), name


def _cos_sin(rope_size, head_dim, dtype):
    # As LlamaRotaryEmbedding builds it: fp32 outer product, cat, cast to Q's dtype.
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype = torch.int64).float() / head_dim))
    freqs = torch.outer(torch.arange(rope_size).float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim = -1).cuda()
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rope_case(
    fn,
    bsz,
    seq,
    n_heads,
    n_kv,
    head_dim,
    dtype,
    with_indices,
    pure = False,
):
    g = torch.Generator(device = "cuda").manual_seed(0)
    q_lin = torch.randn(bsz, seq, n_heads * head_dim, device = "cuda", generator = g, dtype = dtype)
    k_lin = torch.randn(bsz, seq, n_kv * head_dim, device = "cuda", generator = g, dtype = dtype)
    q_lin.requires_grad_(True)
    k_lin.requires_grad_(True)
    Q = q_lin.view(bsz, seq, n_heads, head_dim).transpose(1, 2)
    K = k_lin.view(bsz, seq, n_kv, head_dim).transpose(1, 2)
    cos, sin = _cos_sin(max(seq, 64) + 16, head_dim, dtype)
    indices = None
    if with_indices:
        # TRL-style packed position ids: every packed sequence restarts at 0, int32.
        lens = [seq // 3, seq // 3, seq - 2 * (seq // 3)]
        row = torch.cat([torch.arange(n, dtype = torch.int32) for n in lens])
        indices = row.repeat(bsz, 1).cuda()
    dQ = torch.randn(bsz, n_heads, seq, head_dim, device = "cuda", generator = g, dtype = dtype)
    dK = torch.randn(bsz, n_kv, seq, head_dim, device = "cuda", generator = g, dtype = dtype)
    q_before, k_before = q_lin.detach().clone(), k_lin.detach().clone()
    Q_out, K_out = fn(Q, K, cos, sin, indices)
    torch.autograd.backward((Q_out, K_out), (dQ, dK))
    if pure:
        assert _bytes_equal(q_lin.detach(), q_before), "the q projection was mutated"
        assert _bytes_equal(k_lin.detach(), k_before), "the k projection was mutated"
    elif with_indices:
        if n_heads > 1:
            assert _bytes_equal(q_lin.detach(), q_before), "the q projection was mutated"
        if n_kv > 1:
            assert _bytes_equal(k_lin.detach(), k_before), "the k projection was mutated"
    return Q_out.detach(), K_out.detach(), q_lin.grad, k_lin.grad


ROPE_SHAPES = [(2, 37, 8, 8, 64), (1, 64, 32, 8, 128), (3, 20, 4, 1, 64)]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids = str)
@pytest.mark.parametrize("with_indices", [False, True], ids = ["positions", "rope_indices"])
@pytest.mark.parametrize(
    "shape", ROPE_SHAPES, ids = lambda s: f"b{s[0]}_s{s[1]}_h{s[2]}_kv{s[3]}_d{s[4]}"
)
def test_rope_compiled_matches_eager(shape, with_indices, dtype):
    """Compiled forward and backward (GQA included) are byte-identical to eager, no graph break."""
    eager = _rope_case(fast_rope_embedding, *shape, dtype, with_indices)
    compiled = _rope_case(
        _compile(lambda q, k, c, s, i: fast_rope_embedding(q, k, c, s, i)),
        *shape,
        dtype,
        with_indices,
        pure = TRACEABLE,
    )
    for e, c, name in zip(eager, compiled, ("Q", "K", "dQ", "dK")):
        assert _bytes_equal(e, c), name
    if TRACEABLE:
        assert _graph_breaks() == 0


@pytest.mark.parametrize("with_indices", [False, True], ids = ["positions", "rope_indices"])
def test_rope_eager_is_the_autograd_function(with_indices):
    """Outside torch.compile the public function is still the Fast_RoPE autograd Functions."""

    def direct(Q, K, cos, sin, indices):
        if indices is not None:
            return Fast_RoPE_Embedding_QK.apply(Q, K, cos, sin, indices)
        Q_out = Fast_RoPE_Embedding.apply(Q.transpose(1, 2).contiguous(), cos, sin).transpose(1, 2)
        K_out = Fast_RoPE_Embedding.apply(K.transpose(1, 2).contiguous(), cos, sin).transpose(1, 2)
        return Q_out, K_out

    ref = _rope_case(direct, 2, 37, 8, 4, 64, torch.bfloat16, with_indices)
    got = _rope_case(fast_rope_embedding, 2, 37, 8, 4, 64, torch.bfloat16, with_indices)
    for e, c, name in zip(ref, got, ("Q", "K", "dQ", "dK")):
        assert _bytes_equal(e, c), name


def test_rope_matches_the_rotation_formula():
    """Control for the byte tests: the kernels compute Q * cos + rotate_half(Q) * sin."""
    Q_out, K_out, _, _ = _rope_case(fast_rope_embedding, 2, 16, 4, 2, 64, torch.float32, False)
    g = torch.Generator(device = "cuda").manual_seed(0)
    q = torch.randn(2, 16, 4 * 64, device = "cuda", generator = g).view(2, 16, 4, 64).transpose(1, 2)
    cos, sin = _cos_sin(80, 64, torch.float32)
    rot = torch.cat((-q[..., 32:], q[..., :32]), dim = -1)
    torch.testing.assert_close(Q_out, q * cos[:16] + rot * sin[:16], rtol = 1e-5, atol = 1e-5)


def test_rope_dynamic_shapes():
    """dynamic=True makes the launch grid SymInts (divmod rejected them)."""
    if not TRACEABLE:
        pytest.skip("this torch has no torch.library.triton_op")
    cos, sin = _cos_sin(64, 64, torch.bfloat16)

    def f(q_lin, k_lin, bsz, seq, n_heads, n_kv):
        Q = q_lin.view(bsz, seq, n_heads, 64).transpose(1, 2)
        K = k_lin.view(bsz, seq, n_kv, 64).transpose(1, 2)
        return fast_rope_embedding(Q, K, cos, sin, None)

    compiled = torch.compile(f, fullgraph = True, dynamic = True)
    for seq in (16, 24, 40):
        g = torch.Generator(device = "cuda").manual_seed(seq)
        q_lin = torch.randn(2, seq, 8 * 64, device = "cuda", generator = g, dtype = torch.bfloat16)
        k_lin = torch.randn(2, seq, 2 * 64, device = "cuda", generator = g, dtype = torch.bfloat16)
        for e, c in zip(
            f(q_lin.clone(), k_lin.clone(), 2, seq, 8, 2), compiled(q_lin, k_lin, 2, seq, 8, 2)
        ):
            assert _bytes_equal(e, c), seq


@pytest.mark.parametrize("compiled", [False, True], ids = ["eager", "compiled"])
@pytest.mark.parametrize("with_indices", [False, True], ids = ["positions", "rope_indices"])
def test_rope_expanded_gradient(with_indices, compiled):
    """Q.sum() gives a stride-0 gradient; rotating it in place gave wrong dQ / dK."""
    g = torch.Generator(device = "cuda").manual_seed(0)
    q_lin = torch.randn(2, 16, 4 * 64, device = "cuda", generator = g).requires_grad_(True)
    k_lin = torch.randn(2, 16, 2 * 64, device = "cuda", generator = g).requires_grad_(True)
    cos, sin = _cos_sin(80, 64, torch.float32)
    indices = torch.arange(16, dtype = torch.int32).repeat(2, 1).cuda() if with_indices else None
    fn = _compile(fast_rope_embedding) if compiled else fast_rope_embedding
    Q_out, K_out = fn(
        q_lin.view(2, 16, 4, 64).transpose(1, 2),
        k_lin.view(2, 16, 2, 64).transpose(1, 2),
        cos,
        sin,
        indices,
    )
    (Q_out.sum() + K_out.sum()).backward()
    # d/dx of sum(x * cos + rotate_half(x) * sin): cos + rotate_half^T(ones) * sin, per position.
    ones = torch.ones(16, 64, device = "cuda")
    rot_t = torch.cat((ones[..., 32:], -ones[..., :32]), dim = -1)
    per_pos = cos[:16] + rot_t * sin[:16]
    torch.testing.assert_close(
        q_lin.grad.view(2, 16, 4, 64),
        per_pos[None, :, None].expand(2, 16, 4, 64),
        rtol = 1e-5,
        atol = 1e-5,
    )
    torch.testing.assert_close(
        k_lin.grad.view(2, 16, 2, 64),
        per_pos[None, :, None].expand(2, 16, 2, 64),
        rtol = 1e-5,
        atol = 1e-5,
    )


def test_tiny_llama_decoder_layers_compile_fullgraph():
    """Decoder layers compile inside Unsloth's gradient checkpointing and match eager."""
    if not TRACEABLE:
        pytest.skip("this torch has no torch.library.triton_op")
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        max_seq_length = 64,
        load_in_4bit = False,
        dtype = torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )
    model.train()
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, vocab, (2, 48), generator = g).cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    eager_loss = model(input_ids = ids, labels = ids).loss
    eager_loss.backward()
    eager_grads = [p.grad.clone() for p in params]
    model.zero_grad(set_to_none = True)

    # Before torch 2.11 fast_lora's Functions stay opaque: the only graph breaks allowed.
    trace_lora = fast_lora.TRACE_LORA_FUNCTIONS
    layers = model.base_model.model.model.layers
    for layer in layers:
        layer.forward = torch.compile(layer.forward, fullgraph = trace_lora)
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    reasons = [" ".join(str(k).split()) for k in dynamo_utils.counters["graph_break"]]
    assert all("_apply" in r for r in reasons), reasons
    assert math.isfinite(loss.item())
    assert abs(loss.item() - eager_loss.item()) <= 1e-3 * abs(eager_loss.item())
    grads = [p.grad for p in params]
    assert grads and all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    scale = max(g.abs().max().item() for g in eager_grads)
    worst = max((a.float() - b.float()).abs().max().item() for a, b in zip(grads, eager_grads))
    assert worst <= 1e-2 * scale, (worst, scale)


def test_tiny_llama_causal_lm_compiles_fullgraph():
    """The whole causal LM compiles with no requires-grad hook or use_return_dict graph break,
    and its LoRA gradients match eager (torch 2.10 still breaks in the fused loss)."""
    if not TRACEABLE:
        pytest.skip("this torch has no torch.library.triton_op")
    from unsloth import FastLanguageModel
    from unsloth_zoo.utils import Version
    import unsloth_zoo.fused_losses.cross_entropy_loss as fused_ce

    model, _ = FastLanguageModel.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        max_seq_length = 64,
        load_in_4bit = False,
        dtype = torch.bfloat16,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = False,
        random_state = 3407,
    )
    model.train()
    embeddings = model.get_input_embeddings()
    assert embeddings._forward_hooks, "enable_input_require_grads registered no hook"
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, model.config.vocab_size, (2, 48), generator = g).cuda()
    params = [p for p in model.parameters() if p.requires_grad]

    eager_loss = model(input_ids = ids, labels = ids).loss
    eager_loss.backward()
    eager_grads = [p.grad.detach().float().clone() for p in params]
    model.zero_grad(set_to_none = True)

    loss_opaque = getattr(fused_ce, "_FUSED_LOSS_OPAQUE", False)
    fullgraph = (
        fast_lora.TRACE_LORA_FUNCTIONS
        and Version(torch.__version__) >= Version("2.11.0")
        and not loss_opaque
    )
    causal_lm = model.base_model.model
    causal_lm.forward = torch.compile(causal_lm.forward, fullgraph = fullgraph)
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    reasons = [" ".join(str(k).split()) for k in dynamo_utils.counters["graph_break"]]
    assert not any("requires_grad_()" in r or "logging.Logger" in r for r in reasons), reasons
    if loss_opaque and fast_lora.TRACE_LORA_FUNCTIONS:
        assert all("_fused_loss_opaque" in r for r in reasons), reasons
    assert abs(loss.item() - eager_loss.item()) <= 1e-3 * abs(eager_loss.item())
    total = sum(float(e.norm()) for e in eager_grads)
    assert total > 0
    for p, e in zip(params, eager_grads):
        assert p.grad is not None
        err = float((p.grad.float() - e).norm())
        assert err <= 0.05 * float(e.norm()) + 1e-6, (err, float(e.norm()))


def test_input_require_grads_hook_is_exact():
    """Compiled, the hook returns x + -0.0 (exact) needing grad; eager flips requires_grad."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-LlamaForCausalLM"
    ).cuda()
    model.requires_grad_(False)
    model.enable_input_require_grads()
    emb = model.get_input_embeddings()
    with torch.no_grad():
        emb.weight[0].fill_(-0.0)
    ids = torch.tensor([[0, 1, 2]], device = "cuda")
    eager = emb(ids)
    assert eager.requires_grad
    compiled = torch.compile(lambda i: emb(i), fullgraph = TRACEABLE)(ids)
    assert compiled.requires_grad
    assert _bytes_equal(compiled.detach(), eager.detach())
    assert torch.signbit(compiled[0, 0]).all()


def test_compile_cache_key_covers_the_kernel_source():
    """A warm FX graph cache served an edited kernel as the old one; key it on the files."""
    from unsloth.kernels import rms_layernorm, rope_embedding

    if not hasattr(getattr(torch.compiler, "config", None), "cache_key_tag"):
        pytest.skip("torch has no compile cache key tag")
    if not rms_layernorm._TRACEABLE:
        pytest.skip("no triton_op, nothing is cached by op name")
    tags = torch.compiler.config.cache_key_tag.split(",")
    for mod in (rms_layernorm, rope_embedding):
        with open(mod.__file__, "rb") as file:
            digest = hashlib.sha256(file.read()).hexdigest()[:16]
        assert f"unsloth/{os.path.basename(mod.__file__)}:{digest}" in tags
