# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A packed row holding more than a few thousand documents used to abort a training run with
`CUDA error: an illegal memory access was encountered`, which poisons the CUDA context so every
later op in the process fails too.

Root cause: flash-attn 2's varlen BACKWARD allocates
``dq_accum = zeros(total_q + 128 * n_seqs, n_heads, round_up(head_dim, 32))`` and indexes it
with int32, so the kernel faults once that element count reaches 2**31. xFormers dispatches a
BlockDiagonal* bias to the same flash-2 op, which is why the crash showed up on the xformers
path. Forward-only never allocates the buffer and never faults.

The parametrised bounds below are measurements from a B200 (bf16, one flattened row,
forward + backward), bisected on document count.
"""

import pytest
import torch

import unsloth  # noqa: F401
from unsloth.utils import attention_dispatch as ad


# (n_heads, head_dim, doc_len, last document count observed to run clean on a B200)
_MEASURED = [
    (16, 128, 1, 8129),
    (16, 96, 1, 10838),
    (16, 64, 1, 16257),
    (16, 128, 4, 7944),
]


@pytest.mark.parametrize("n_heads, head_dim, doc_len, last_ok", _MEASURED)
def test_guard_matches_the_measured_crash_threshold(n_heads, head_dim, doc_len, last_ok):
    """The predicted limit must sit within one document of the observed one, and must never
    sit ABOVE it -- a guard that trips late is a guard that does not exist."""

    def trips(n_docs):
        return ad._varlen_backward_overflows_int32(n_docs, n_docs * doc_len, n_heads, head_dim)

    # Inside the safe region: never give up the fast kernel for nothing.
    assert not trips(last_ok // 2)
    assert not trips(last_ok - 2)
    assert trips(last_ok + 1)
    assert trips(last_ok * 2)


def test_head_dim_is_rounded_up_to_a_multiple_of_32():
    # head_dim 96 rounds to 96, not 128, which is why that case lands at 10838 not 8129.
    assert ad._varlen_backward_dq_accum_elements(1, 0, 1, 96) == 128 * 96
    assert ad._varlen_backward_dq_accum_elements(1, 0, 1, 100) == 128 * 128
    assert ad._varlen_backward_dq_accum_elements(1, 0, 1, 64) == 128 * 64


def test_empty_partition_never_trips():
    assert not ad._varlen_backward_overflows_int32(0, 0, 16, 128)


def _context(
    n_docs,
    total_q,
    requires_grad,
    n_heads = 16,
    head_dim = 128,
):
    lengths = torch.zeros(n_docs, dtype = torch.int32)
    return ad.AttentionContext(
        bsz = 1,
        q_len = total_q,
        kv_seq_len = total_q,
        n_heads = n_heads,
        head_dim = head_dim,
        requires_grad = requires_grad,
        seq_info = (lengths, None, 1),
        attention_mask = None,
        causal_mask = None,
    )


def _run(
    monkeypatch,
    backend,
    n_docs,
    requires_grad,
    guard_disabled = False,
    softcap = None,
):
    """Drive run_attention with every real kernel stubbed, and report which branch it took."""
    taken = {}

    def _fake_xformers(*args, **kwargs):
        taken["backend"] = ad.XFORMERS
        return torch.zeros((1, 4, 16, 128))

    def _fake_flash_varlen(*args, **kwargs):
        taken["backend"] = ad.FLASH_VARLEN
        return torch.zeros((4, 16, 128))

    def _fake_sdpa(*args, **kwargs):
        taken["backend"] = ad.SDPA
        return torch.zeros((1, 16, 4, 128))

    monkeypatch.setattr(ad, "xformers_attention", _fake_xformers, raising = False)
    monkeypatch.setattr(ad, "flash_attn_varlen_func", _fake_flash_varlen, raising = False)
    monkeypatch.setattr(ad, "scaled_dot_product_attention", _fake_sdpa, raising = False)
    monkeypatch.setattr(ad, "build_xformers_block_causal_mask", lambda *a, **k: object())
    monkeypatch.setattr(ad, "build_sdpa_packed_attention_mask", lambda *a, **k: None)
    monkeypatch.setattr(ad, "_VARLEN_INT32_GUARD_DISABLED", guard_disabled)
    monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", True)
    ad._VARLEN_INT32_WARNED[0] = False

    q = torch.zeros((1, 16, 4, 128), requires_grad = requires_grad)
    kwargs = {"softcap": softcap} if softcap is not None else None
    config = ad.AttentionConfig(
        backend = backend,
        n_kv_heads = 16,
        n_groups = 1,
        flash_varlen_kwargs = kwargs,
        flash_dense_kwargs = kwargs,
    )
    ad.run_attention(config = config, context = _context(n_docs, 4, requires_grad), Q = q, K = q, V = q)
    return taken.get("backend")


@pytest.mark.parametrize("backend", [ad.XFORMERS, ad.FLASH_VARLEN])
def test_oversized_partition_falls_back_to_sdpa(monkeypatch, backend):
    assert _run(monkeypatch, backend, n_docs = 20000, requires_grad = True) == ad.SDPA


# 8129 documents at 16 heads / head_dim 128 is the last count that ran;
# 20000 is well past it.
@pytest.mark.parametrize("backend", [ad.XFORMERS, ad.FLASH_VARLEN])
def test_softcapped_model_raises_instead_of_silently_dropping_the_softcap(monkeypatch, backend):
    """Gemma 2 hands `attn_logit_softcapping` to the fast kernels through
    `flash_varlen_kwargs` alone (unsloth/models/gemma2.py), and the SDPA branch has no
    softcap at all. Downgrading a softcapped model would keep the run alive on wrong logits
    and wrong gradients, which is worse than the fault the guard prevents, so it must stop."""
    with pytest.raises(RuntimeError) as excinfo:
        _run(monkeypatch, backend, n_docs = 20000, requires_grad = True, softcap = 50.0)
    message = str(excinfo.value)
    assert "softcap=50.0" in message
    assert "Pack fewer documents per row" in message


@pytest.mark.parametrize("backend", [ad.XFORMERS, ad.FLASH_VARLEN])
def test_softcap_of_none_or_zero_still_falls_back(monkeypatch, backend):
    """Only a real softcap blocks the fallback; every other model keeps the rescue."""
    assert _run(monkeypatch, backend, n_docs = 20000, requires_grad = True, softcap = 0.0) == ad.SDPA


@pytest.mark.parametrize("backend", [ad.XFORMERS, ad.FLASH_VARLEN])
def test_softcapped_model_under_the_bound_is_untouched(monkeypatch, backend):
    """A softcapped model that does not overflow must keep its fast kernel, not raise."""
    assert _run(monkeypatch, backend, n_docs = 64, requires_grad = True, softcap = 50.0) == backend


@pytest.mark.parametrize("backend", [ad.XFORMERS, ad.FLASH_VARLEN])
def test_normal_partition_keeps_the_fast_backend(monkeypatch, backend):
    assert _run(monkeypatch, backend, n_docs = 64, requires_grad = True) == backend


@pytest.mark.parametrize("backend", [ad.XFORMERS, ad.FLASH_VARLEN])
def test_forward_only_is_never_downgraded(monkeypatch, backend):
    """Inference allocates no dq_accum and never faulted (20000 documents ran clean), so the
    guard must not cost generation anything."""
    assert _run(monkeypatch, backend, n_docs = 20000, requires_grad = False) == backend


@pytest.mark.parametrize("backend", [ad.XFORMERS, ad.FLASH_VARLEN])
def test_gradient_checkpointing_picks_the_same_backend_in_both_passes(monkeypatch, backend):
    """torch.utils.checkpoint runs its first forward under no_grad and recomputes with grad on.
    If the guard keyed only on the tensors, the two passes would take different backends and
    the backward would see activations the reported loss never came from."""

    def go(grad_enabled):
        with torch.set_grad_enabled(grad_enabled):
            taken = {}

            def _fake_xformers(*a, **k):
                taken["b"] = ad.XFORMERS
                return torch.zeros((1, 4, 16, 128))

            def _fake_flash_varlen(*a, **k):
                taken["b"] = ad.FLASH_VARLEN
                return torch.zeros((4, 16, 128))

            def _fake_sdpa(*a, **k):
                taken["b"] = ad.SDPA
                return torch.zeros((1, 16, 4, 128))

            monkeypatch.setattr(ad, "xformers_attention", _fake_xformers, raising = False)
            monkeypatch.setattr(ad, "flash_attn_varlen_func", _fake_flash_varlen, raising = False)
            monkeypatch.setattr(ad, "scaled_dot_product_attention", _fake_sdpa, raising = False)
            monkeypatch.setattr(ad, "build_xformers_block_causal_mask", lambda *a, **k: object())
            monkeypatch.setattr(ad, "build_sdpa_packed_attention_mask", lambda *a, **k: None)
            monkeypatch.setattr(ad, "_VARLEN_INT32_GUARD_DISABLED", False)
            monkeypatch.setattr(ad, "HAS_FLASH_ATTENTION", True)
            ad._VARLEN_INT32_WARNED[0] = False
            q = torch.zeros((1, 16, 4, 128))
            # The checkpointed hidden state keeps requires_grad = True in both passes.
            ctx = _context(20000, 4, requires_grad = True)
            ad.run_attention(
                config = ad.AttentionConfig(backend = backend, n_kv_heads = 16, n_groups = 1),
                context = ctx,
                Q = q,
                K = q,
                V = q,
            )
            return taken.get("b")

    assert go(False) == go(True) == ad.SDPA


def test_guard_can_be_disabled_by_env(monkeypatch):
    assert (
        _run(monkeypatch, ad.XFORMERS, n_docs = 20000, requires_grad = True, guard_disabled = True)
        == ad.XFORMERS
    )


def test_guard_warns_once_naming_the_cost(capsys):
    ad._VARLEN_INT32_WARNED[0] = False
    ad._warn_varlen_int32_overflow_once(ad.XFORMERS, 20000, 20000, 2**32)
    first = capsys.readouterr().out
    assert "illegal memory access" in first
    assert "SDPA" in first
    assert "20000 documents" in first
    # Once per process: it fires per layer per step.
    ad._warn_varlen_int32_overflow_once(ad.XFORMERS, 20000, 20000, 2**32)
    assert capsys.readouterr().out == ""
