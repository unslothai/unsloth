# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL >= 1.15's fused lm_head served by unsloth_zoo's fused linear kernel, and KTO's no-grad KL pass scored from
hidden states."""

from __future__ import annotations

import inspect
import os
import sys
import textwrap
import types

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl as unsloth_rl
import unsloth.models.rl_replacements as rl

fused_linear = pytest.importorskip("unsloth_zoo.fused_losses.fused_linear")

N, H, V = 11, 8, 29


class _Kernel:
    """Stands in for TRL's ChunkedLogProbFunction and records every call it serves."""

    _calls = 0
    _last = ()

    @classmethod
    def apply(cls, *args):
        cls._calls += 1
        cls._last = args
        return ("trl",) * 5


def _reference(hidden, weight, bias, targets, temperature, softcap, logit_scale):
    """TRL's outputs in float64: scale, softcap, then temperature; mean_logits after the temperature."""
    z = hidden.double() @ weight.double().t() + bias.double()
    z = z * logit_scale
    if softcap is not None:
        z = softcap * torch.tanh(z / softcap)
    z = z / temperature
    log_p = z.log_softmax(-1)
    logps = log_p.gather(-1, targets[:, None]).squeeze(-1)
    entropy = -(log_p.exp() * log_p).sum(-1)
    lssp = (2 * log_p).logsumexp(-1)
    mean = z.mean(-1)
    top1 = z.argmax(-1) == targets
    return logps, entropy, lssp, mean, top1


def _inputs(seed = 0):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(N, H, generator = gen, requires_grad = True)
    weight = torch.randn(V, H, generator = gen)
    bias = torch.randn(V, generator = gen)
    targets = torch.randint(0, V, (N,), generator = gen)
    return hidden, weight, bias, targets


def test_adapter_matches_trl_semantics():
    hidden, weight, bias, targets = _inputs()
    adapter = unsloth_rl._UnslothFusedLinearHead(_Kernel)
    before = _Kernel._calls
    calls = fused_linear.CALLS["logprobs"]
    got = adapter.apply(
        hidden,
        weight,
        bias,
        targets,
        0.7,
        4096,
        5.0,
        1.5,
        ("log_probs", "entropy", "log_sum_sq_probs", "mean_logits", "is_top1"),
    )
    assert _Kernel._calls == before and fused_linear.CALLS["logprobs"] == calls + 1
    want = _reference(hidden.detach(), weight, bias, targets, 0.7, 5.0, 1.5)
    for g, w in zip(got[:4], want[:4]):
        torch.testing.assert_close(g.detach().double(), w, atol = 1e-5, rtol = 1e-5)
    assert torch.equal(got[4], want[4])
    (got[0].sum() + got[1].sum()).backward()
    assert hidden.grad is not None and hidden.grad.isfinite().all()


def _lazy_zoo():
    return int(str(getattr(fused_linear, "_LP_OP", "")).rsplit("_v", 1)[-1]) >= 4


def test_adapter_logged_entropy_keeps_the_forward_gradient():
    """TRL 1.15 DPO asks for the entropy and detaches it (a metric): a frozen head must then take the forward
    gradient, with no recompute backward, as TRL's kernel skips its entropy work when no gradient arrives."""
    if not _lazy_zoo():
        pytest.skip("unsloth_zoo without the lazy entropy gradient")
    hidden, weight, bias, targets = _inputs(4)
    bwd = fused_linear.CALLS["logprobs_bwd"]
    got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
        hidden,
        weight,
        bias,
        targets,
        0.7,
        4096,
        None,
        1.0,
        ("log_probs", "entropy", "mean_logits", "is_top1"),
    )
    (logged := got[1].detach())  # noqa: B018
    got[0].sum().backward()
    assert (
        fused_linear.CALLS["logprobs_bwd"] == bwd
        and hidden.grad.isfinite().all()
        and logged.isfinite().all()
    )


def test_adapter_entropy_gradient_when_used():
    """TRL's GRPO entropy bonus backpropagates the entropy: it still gets its gradient."""
    hidden, weight, bias, targets = _inputs(5)
    got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
        hidden, weight, bias, targets, 0.7, 4096, None, 1.0, ("log_probs", "entropy")
    )
    (got[0].sum() + 0.3 * got[1].sum()).backward()
    hd = hidden.detach().double().requires_grad_(True)
    z = (hd @ weight.double().t() + bias.double()) / 0.7
    logp = torch.log_softmax(z, -1)
    ent = -(logp.exp() * logp).sum(-1)
    (logp.gather(-1, targets[:, None]).sum() + 0.3 * ent.sum()).backward()
    torch.testing.assert_close(hidden.grad.double(), hd.grad, atol = 1e-4, rtol = 1e-4)


def test_adapter_returns_none_for_unrequested_outputs():
    hidden, weight, bias, targets = _inputs(1)
    got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
        hidden, weight, None, targets, 1.0, 4096, None, 1.0, ()
    )
    assert len(got) == 5 and got[0] is not None and all(x is None for x in got[1:])


@pytest.mark.parametrize("case", ["zero_scale", "fp32_hidden_half_head", "unknown_output"])
def test_adapter_hands_unreproduced_calls_to_trl(case):
    hidden, weight, bias, targets = _inputs(2)
    scale, outputs = 1.0, ("log_probs",)
    if case == "zero_scale":
        scale = 0.0
    elif case == "fp32_hidden_half_head":
        weight = weight.bfloat16()
    else:
        outputs = ("log_probs", "something_new")
    before = _Kernel._calls
    got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
        hidden, weight, bias, targets, 1.0, 4096, None, scale, outputs
    )
    assert _Kernel._calls == before + 1 and got[0] == "trl"


def test_adapter_hands_support_ids_calls_to_trl():
    """TRL > 1.16 passes a 10th argument, ``support_ids`` (score over a per-row kept set): the zoo kernel scores the
    full vocabulary, so those calls go to TRL's kernel with every argument; calls without it stay on the zoo kernel."""
    hidden, weight, bias, targets = _inputs(3)
    support = torch.zeros(hidden.shape[0], 4, dtype = torch.long)
    before = _Kernel._calls
    got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
        hidden, weight, bias, targets, 1.0, 4096, None, 1.0, ("log_probs",), support
    )
    assert _Kernel._calls == before + 1 and got[0] == "trl" and _Kernel._last[-1] is support
    got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
        hidden, weight, bias, targets, 1.0, 4096, None, 1.0, ("log_probs",), None
    )
    assert _Kernel._calls == before + 1 and got[0] is not None and got[0] != "trl"


def test_patch_is_idempotent(monkeypatch):
    utils = types.ModuleType("trl.trainer.utils")
    utils._ChunkedLogProbFunction = _Kernel
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils)
    trainer = sys.modules.get("trl.trainer")
    if trainer is not None:
        monkeypatch.setattr(trainer, "utils", utils, raising = False)
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    first = utils._ChunkedLogProbFunction
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    assert (
        isinstance(first, unsloth_rl._UnslothFusedLinearHead)
        and utils._ChunkedLogProbFunction is first
    )
    assert first.function is _Kernel


@pytest.mark.parametrize("kernel", ["modern", "no_bias"])
def test_device_and_fused_head_wrappers_see_each_other(kernel, monkeypatch):
    """A split model's device wrapper and the fused head adapter, in either order and patched again: no nesting, and
    a TRL 1.1 - 1.13 kernel (no ``bias``) never gets the adapter."""
    base = _Kernel if kernel == "modern" else _NoBiasKernel
    utils = types.ModuleType("trl.trainer.utils")
    utils._ChunkedLogProbFunction = unsloth_rl._ApplyOnInputDevice(base)
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils)
    trainer = sys.modules.get("trl.trainer")
    if trainer is not None:
        monkeypatch.setattr(trainer, "utils", utils, raising = False)
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    first = utils._ChunkedLogProbFunction
    if kernel == "no_bias":
        assert isinstance(first, unsloth_rl._ApplyOnInputDevice)
    else:
        assert isinstance(first, unsloth_rl._UnslothFusedLinearHead)
        assert getattr(first, "_unsloth_device_patched", False)
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    assert utils._ChunkedLogProbFunction is first
    wrapped = unsloth_rl._ApplyOnInputDevice(unsloth_rl._UnslothFusedLinearHead(base))
    assert getattr(wrapped, "_unsloth_fused_linear", False)


def test_stock_logps_hook_finds_fused_pass_rewrites():
    """A rewritten method that only calls the fused-pass helpers (KTO _compute_kl_logps, DPO ref precompute) is still
    the owner, so a subclass overriding get_batch_logps below it is detected."""
    from unsloth.models import rl_replacements as rr

    class Rewritten:
        def _compute_kl_logps(self, model, batch):
            return _unsloth_fused_pass_logps(self, model, batch)  # noqa: F821  (only its name is read)

    class Sub(Rewritten):
        def get_batch_logps(self, *args):
            return None

    assert rr._unsloth_stock_logps_hook(Rewritten(), "_compute_kl_logps", ("get_batch_logps",))
    assert not rr._unsloth_stock_logps_hook(Sub(), "_compute_kl_logps", ("get_batch_logps",))


def test_adapter_honours_the_kill_switch_after_install(monkeypatch):
    """UNSLOTH_FUSED_LINEAR=0 set after the adapter was installed (a notebook re-run) still serves TRL's kernel."""
    hidden, weight, bias, targets = _inputs(4)
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    before = _Kernel._calls
    got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
        hidden, weight, bias, targets, 1.0, 4096, None, 1.0, ("log_probs",)
    )
    assert _Kernel._calls == before + 1 and got[0] == "trl"


def test_patch_respects_the_kill_switch(monkeypatch):
    utils = types.ModuleType("trl.trainer.utils")
    utils._ChunkedLogProbFunction = _Kernel
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils)
    trainer = sys.modules.get("trl.trainer")
    if trainer is not None:
        monkeypatch.setattr(trainer, "utils", utils, raising = False)
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    assert utils._ChunkedLogProbFunction is _Kernel


class _NoBiasKernel(_Kernel):
    """TRL 1.1 - 1.13's ``_ChunkedLogProbFunction``: no ``bias`` (and no ``outputs``) argument."""

    @staticmethod
    def forward(
        ctx,
        last_hidden,
        weight,
        targets,
        temperature,
        chunk_size,
        final_logit_softcapping = None,
        logit_scale = 1.0,
    ):
        raise AssertionError("not reached")


def test_patch_leaves_kernels_without_bias(monkeypatch):
    """The adapter's positional layout has ``bias`` third: wrapping an older kernel would shift every argument."""
    utils = types.ModuleType("trl.trainer.utils")
    utils._ChunkedLogProbFunction = _NoBiasKernel
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils)
    trainer = sys.modules.get("trl.trainer")
    if trainer is not None:
        monkeypatch.setattr(trainer, "utils", utils, raising = False)
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    assert utils._ChunkedLogProbFunction is _NoBiasKernel


class _HeadLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import LlamaConfig

        self.config = LlamaConfig(
            vocab_size = V, hidden_size = H, num_hidden_layers = 1, num_attention_heads = 2
        )
        self.lm_head = torch.nn.Linear(H, V, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def _selector_trainer():
    return types.SimpleNamespace(accelerator = types.SimpleNamespace(unwrap_model = lambda m: m))


def _selected(selector, trainer, model):
    if selector == "orpo_cpo":
        return rl._unsloth_pref_chunked_head(trainer, model) is not None
    return rl._unsloth_dpo_chunked_head(trainer, model, {}) is not None


@pytest.mark.parametrize("selector", ["orpo_cpo", "dpo_kto"])
def test_preference_selectors_respect_the_kill_switch(selector, monkeypatch):
    model, trainer = _HeadLM(), _selector_trainer()
    assert _selected(selector, trainer, model)
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    assert not _selected(selector, trainer, model)


@pytest.mark.parametrize("selector", ["orpo_cpo", "dpo_kto"])
@pytest.mark.parametrize(
    "hook",
    ["forward_hook", "forward_pre_hook", "patched_forward", "backward_hook", "backward_pre_hook"],
)
def test_preference_selectors_reject_hooked_heads(selector, hook):
    """The chunked / fused kernels read the head's weight directly, skipping whatever its forward would add."""
    model, trainer = _HeadLM(), _selector_trainer()
    assert _selected(selector, trainer, model)
    head = model.lm_head
    if hook == "forward_hook":
        head.register_forward_hook(lambda module, args, output: output * 0.5)
    elif hook == "forward_pre_hook":
        head.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
    elif hook == "backward_hook":
        # Reading the weight directly would skip it: the head's gradients would silently change.
        head.register_full_backward_hook(
            lambda module, grad_in, grad_out: tuple(None if g is None else g * 0.5 for g in grad_in)
        )
    elif hook == "backward_pre_hook":
        head.register_full_backward_pre_hook(
            lambda module, grad_out: tuple(g * 0.5 for g in grad_out)
        )
    else:
        head.forward = lambda x: torch.nn.functional.linear(x, head.weight) * 0.5
    assert not _selected(selector, trainer, model)


def _kto_kl_source():
    """TRL's own _compute_kl_logps, read from its file: importing unsloth rebinds KTOTrainer to the generated class."""
    import ast
    import importlib.util

    pytest.importorskip("trl")
    for name in ("trl.trainer.kto_trainer", "trl.experimental.kto.kto_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_kl_logps":
                return textwrap.dedent(ast.get_source_segment(text, node, padded = True))
    pytest.skip(reason = "this TRL computes KTO's KL term elsewhere")


def test_kto_kl_pass_is_rewritten_once():
    source = _kto_kl_source()
    from packaging.version import Version
    import trl

    if Version(trl.__version__) >= Version("1.6"):
        pytest.skip(reason = "the KL rewrite targets TRL < 1.6, whose KL pass calls get_batch_logps")
    new = rl.kto_trainer_chunked_kl_logps("_compute_kl_logps", source)
    assert new != source and "_unsloth_kto_hidden_logps" in new
    compile(new, "<kto_kl>", "exec")
    assert rl.kto_trainer_chunked_kl_logps("_compute_kl_logps", new) == new
    assert rl.kto_trainer_chunked_kl_logps("forward", source) == source
    assert rl.kto_trainer_chunked_kl_logps in rl.RL_FUNCTIONS["kto_trainer"]


def test_kto_scores_completion_tokens_only():
    """The KTO helper sends only completion positions to the head and keeps the per-position vocab sums exact."""
    gen = torch.Generator().manual_seed(5)
    hidden = torch.randn(2, 9, H, generator = gen)
    head = torch.nn.Linear(H, V, bias = False)
    labels = torch.randint(0, V, (2, 9), generator = gen)
    labels[:, :4] = -100
    before = fused_linear.CALLS["logprobs"]
    logps, sums = rl._unsloth_kto_hidden_logps(
        types.SimpleNamespace(), (None, head, (0.0, 0.0, 0.0)), hidden, labels, -100
    )
    assert fused_linear.CALLS["logprobs"] == before + 1
    z = hidden.double() @ head.weight.double().t()
    log_p = z[:, :-1].log_softmax(-1)
    shifted = labels[:, 1:]
    mask = shifted != -100
    want = (log_p.gather(-1, shifted.clamp(min = 0)[..., None]).squeeze(-1) * mask).sum(-1)
    torch.testing.assert_close(logps.detach().double(), want, atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(sums.squeeze(-1).double(), z.sum(-1), atol = 1e-3, rtol = 1e-5)


def _buffer_bytes(monkeypatch):
    """Records the bytes of every logits buffer the zoo kernel allocates."""
    sizes = []
    init = fused_linear._Buffers.__init__

    def spy(self, *args, **kwargs):
        init(self, *args, **kwargs)
        sizes.append(self.buf.numel() * self.buf.element_size())

    monkeypatch.setattr(fused_linear._Buffers, "__init__", spy)
    return sizes


def _bounded_inputs(trainable_head, seed = 3):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(64, 8, generator = gen, requires_grad = True)
    weight = torch.randn(1000, 8, generator = gen, requires_grad = trainable_head)
    targets = torch.randint(0, 1000, (64,), generator = gen)
    return hidden, weight, targets


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("trainable_head", [False, True])
def test_adapter_bounds_its_logits_buffer_by_chunk_size(trainable_head, grad, monkeypatch):
    # TRL's tile is [min(N, TOKEN_CHUNK_SIZE), chunk_size] in the compute dtype: chunk_size counts vocab columns.
    # Recompute tiles shrunk to 8 KiB (TRL's 32768-column tile covers the zoo's 256 MiB ones at N >= 4096), so only
    # the [rows, V] chunked forward needs the row cap.
    monkeypatch.setattr(fused_linear, "_BWD_TILE_BYTES", 8 * 1024)
    hidden, weight, targets = _bounded_inputs(trainable_head)
    n, v, chunk_size = hidden.shape[0], weight.shape[0], 100
    trl_bytes = n * chunk_size * hidden.element_size()
    sizes = _buffer_bytes(monkeypatch)
    kwargs = []
    zoo = fused_linear.fused_linear_logprobs
    monkeypatch.setattr(
        fused_linear,
        "fused_linear_logprobs",
        lambda *a, **k: kwargs.append(k) or zoo(*a, **k),
    )
    adapter = unsloth_rl._UnslothFusedLinearHead(_Kernel)
    before = _Kernel._calls
    with torch.set_grad_enabled(grad):
        got = adapter.apply(hidden, weight, None, targets, 1.0, chunk_size, None, 1.0, ("entropy",))
        if grad:
            (got[0].sum() + got[1].sum()).backward()
    assert _Kernel._calls == before
    assert sizes and max(sizes) <= trl_bytes, (sizes, trl_bytes)
    assert kwargs[-1]["max_rows_per_chunk"] == trl_bytes // (v * 4)
    want = _reference(hidden.detach(), weight.detach(), torch.zeros(v), targets, 1.0, None, 1.0)
    torch.testing.assert_close(got[0].detach().double(), want[0], atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(got[1].detach().double(), want[1], atol = 1e-5, rtol = 1e-5)

    # TRL's default (32768 columns) is a bigger tile than the zoo kernel's own buffer here: no cap.
    sizes.clear()
    with torch.set_grad_enabled(grad):
        adapter.apply(hidden, weight, None, targets, 1.0, 32768, None, 1.0, ("entropy",))
    assert kwargs[-1]["max_rows_per_chunk"] is None and max(sizes) == n * v * 4


@pytest.mark.parametrize("grad", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 100])
def test_adapter_hands_tiles_below_its_own_to_trl(chunk_size, grad):
    # Recompute tiles cannot shrink below min(N x V, 256 MiB) without starving the GEMM: TRL serves the shape, for
    # the no-grad pass too.
    hidden, weight, targets = _bounded_inputs(False, seed = 4)
    before = _Kernel._calls
    with torch.set_grad_enabled(grad):
        got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
            hidden, weight, None, targets, 1.0, chunk_size, None, 1.0, ("log_probs",)
        )
    assert _Kernel._calls == before + 1 and got[0] == "trl"


def test_adapter_keeps_its_own_budget_at_trls_default_chunk_size():
    # At TRL's default the zoo kernel's whole-call peak is already below TRL's (frozen / trainable heads), so only an
    # explicitly smaller chunk_size bounds it: 8k tokens x 152k vocab would otherwise cap the no-grad pass at 441 rows.
    adapter = unsloth_rl._UnslothFusedLinearHead(_Kernel)
    hidden = torch.zeros(8192, 8, dtype = torch.bfloat16)
    weight = torch.zeros(151936, 8, dtype = torch.bfloat16)
    with torch.no_grad():
        assert adapter._row_cap(hidden, weight, None, adapter._default_chunk, ("log_probs",)) == 0
        assert (
            adapter._row_cap(hidden, weight, None, adapter._default_chunk // 2, ("log_probs",)) != 0
        )
