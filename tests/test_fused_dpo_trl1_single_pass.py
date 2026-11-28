# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_FUSED_PREFERENCE_LOSS=1 on TRL >= 1.0: the entropy and token accuracy metrics come out of the same
fused head pass as the policy log-probs, not a second projection of every completion row through the head
(about a fifth of a Qwen3-0.6B DPO step on an L4). Loss, gradients and metrics still match TRL (float32, CPU)."""

from __future__ import annotations

import inspect

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("trl")

import tests.test_fused_dpo_trl1 as trl1  # noqa: E402  (skips the module on TRL < 1.0)
from tests.test_fused_preference_loss import _model, dpo_module  # noqa: E402, F401


def _stats_kernel():
    try:
        from unsloth_zoo.fused_losses.fused_linear import fused_linear_logprobs
    except Exception as e:  # noqa: BLE001
        pytest.skip(reason = f"unsloth_zoo fused_linear does not import here: {type(e).__name__}")
    params = inspect.signature(fused_linear_logprobs).parameters
    if not {"return_entropy", "return_is_top1", "entropy_requires_grad"} <= set(params):
        pytest.skip(reason = "unsloth_zoo fused_linear_logprobs has no metric outputs")
    return fused_linear_logprobs


def _float_model(seed, softcap, bias):
    return _model(seed, softcap, bias).float()


def _no_second_projection(*args, **kwargs):
    raise AssertionError(
        "the metrics projected the policy hidden states through the head a second time"
    )


@trl1.needs_hook
@pytest.mark.parametrize(
    "case",
    [
        dict(),
        dict(use_weighting = True),
        dict(softcap = 3.0),
        dict(bias = True),
        dict(loss_types = ["sigmoid", "sft"], loss_weights = [0.7, 0.3]),
    ],
    ids = ["default", "weighting", "softcap", "bias", "sft"],
)
def test_metrics_share_the_logps_pass_and_match_trl(dpo_module, monkeypatch, case):
    _stats_kernel()
    case = dict(case)
    if not trl1._loss_type_known(case):
        pytest.skip("loss type not in this TRL")
    softcap = case.pop("softcap", 0.0)
    bias = case.pop("bias", False)
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, softcap))
    batch = trl1._batch(seed = 7)
    results = []
    for fused in (False, True):
        model = _float_model(0, softcap, bias)
        ref_model = _float_model(1, softcap, bias)
        trainer = trl1._trainer(model, ref_model, **case)
        if fused:
            monkeypatch.setattr(dpo_module, "_entropy_and_argmax", _no_second_projection)
        results.append(trl1._run(trainer, model, dict(batch), fused, dpo_module))
    (loss_b, metrics_b, grads_b), (loss_h, metrics_h, grads_h) = results
    torch.testing.assert_close(loss_h, loss_b, rtol = 1e-5, atol = 1e-5)
    assert set(metrics_b) == set(metrics_h)
    for key in metrics_b:
        assert metrics_h[key] == pytest.approx(metrics_b[key], rel = 1e-4, abs = 1e-4), key
    assert set(grads_b) == set(grads_h)
    for name in grads_b:
        torch.testing.assert_close(grads_h[name], grads_b[name], rtol = 1e-4, atol = 1e-5, msg = name)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("transforms", [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 2.0, 5.0)])
@pytest.mark.parametrize("bias", [False, True])
def test_single_pass_stats_equal_the_two_pass_stats(dpo_module, dtype, transforms, bias):
    _stats_kernel()
    gen = torch.Generator().manual_seed(3)
    rows, width, hidden_size, vocab = 3, 9, 16, 61
    hidden = torch.randn(rows, width, hidden_size, generator = gen).to(dtype)
    weight = (4 * torch.randn(vocab, hidden_size, generator = gen)).to(dtype)
    head_bias = torch.randn(vocab, generator = gen).to(dtype) if bias else None
    labels = torch.randint(0, vocab, (rows, width), generator = gen)
    mask = torch.rand(rows, width, generator = gen) > 0.3
    # Plant argmax hits, so the accuracy check is not vacuous.
    with torch.no_grad():
        logits = torch.nn.functional.linear(hidden, weight, head_bias)
        labels[:, ::2] = logits.argmax(-1)[:, ::2]
    head = (weight, head_bias)
    for want_lssp in (False, True):
        got = dpo_module._token_logps_and_stats(
            hidden, True, head, transforms, labels, mask, want_lssp
        )
        assert got is not None
        logps, mean_logits, lssp, entropy, is_top1 = got
        ref_logps, ref_mean, ref_lssp = dpo_module._token_logps(
            hidden, True, head, transforms, labels, mask, want_lssp
        )
        ref_entropy, predictions = dpo_module._entropy_and_argmax(
            hidden, True, head, transforms, mask
        )
        torch.testing.assert_close(logps, ref_logps)
        torch.testing.assert_close(mean_logits, ref_mean)
        if want_lssp:
            torch.testing.assert_close(lssp, ref_lssp)
        else:
            assert lssp is None
        if dtype == torch.bfloat16 and bias:
            # The two-pass metric adds the bias to bf16-rounded logits, the kernel to float32 ones: hold both
            # to the float32 math instead.
            exact = torch.nn.functional.linear(hidden.float(), weight.float(), head_bias.float())
            if transforms[0]:
                exact = exact * transforms[0]
            if transforms[1]:
                exact = exact / transforms[1]
            if transforms[2]:
                exact = transforms[2] * torch.tanh(exact / transforms[2])
            exact = -(exact.softmax(-1) * exact.log_softmax(-1)).sum(-1)
            new_err = (entropy[mask] - exact[mask]).abs().max()
            old_err = (ref_entropy[mask].float() - exact[mask]).abs().max()
            assert new_err <= max(old_err, 1e-3), (new_err, old_err)
        else:
            torch.testing.assert_close(
                entropy[mask], ref_entropy[mask].to(entropy.dtype), rtol = 2e-4, atol = 2e-4
            )
        want_top1 = (predictions == labels) & mask
        assert int(want_top1.sum()) > 0
        assert torch.equal(is_top1 & mask, want_top1)


def test_float64_head_keeps_the_two_pass_path(dpo_module):
    _stats_kernel()
    weight = torch.randn(11, 4, dtype = torch.float64)
    hidden = torch.randn(1, 3, 4, dtype = torch.float64)
    labels = torch.zeros(1, 3, dtype = torch.long)
    mask = torch.ones(1, 3, dtype = torch.bool)
    assert (
        dpo_module._token_logps_and_stats(
            hidden, True, (weight, None), (0.0, 0.0, 0.0), labels, mask
        )
        is None
    )
