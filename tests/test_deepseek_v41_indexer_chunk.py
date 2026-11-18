# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""DeepSeek-V4.1 indexer: query-chunked scoring matches the one-shot score and bounds the transient memory (CPU)."""

import os
import sys
import weakref

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_deepseek_v41_native import _tiny_model  # noqa: E402
from unsloth.models.deepseek_v41 import modeling as dsv41  # noqa: E402

CHUNK_ENV = "UNSLOTH_DSV41_INDEXER_Q_CHUNK"
SEQ = 40


def _reference_indexer(self, x, qr, latent, cos, sin, ccos, csin, offset, shared):
    """The unchunked indexer as it was before query chunking: one [b, s, heads, t] fp32 score, then top-k."""
    bsz, seqlen, _ = x.shape
    ratio = self.compress_ratio
    if self.owns_k and latent is not None:
        k = self.k_norm(self.k_proj(latent.to(self.k_proj.weight.dtype)))
        shared["index_k"] = dsv41.apply_rotary(k, ccos, csin)
    index_k = shared["index_k"].to(x.device)
    n_comp = index_k.shape[1]
    q = self.q_b_proj(qr).unflatten(-1, (self.n_heads, self.head_dim))
    q = dsv41.apply_rotary(q, cos, sin)
    weights = self.weights_proj(x).float() * (self.softmax_scale * self.n_heads**-0.5)
    score = torch.einsum("bshd,btd->bsht", q.float(), index_k.float())
    score = (score.relu() * weights.unsqueeze(-1)).sum(dim = 2)
    compress_lens = (torch.arange(1, seqlen + 1, device = x.device) // ratio).unsqueeze(-1)
    score = score.masked_fill(torch.arange(n_comp, device = x.device) >= compress_lens, float("-inf"))
    if self.is_candidate_source:
        shared["candidates"] = dsv41.select_candidate_blocks(
            score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size
        )
    elif self.uses_candidates and shared.get("candidates") is not None:
        score = score.masked_fill(~shared["candidates"].to(x.device), float("-inf"))
    topk = min(self.index_topk, n_comp)
    idxs = score.topk(topk, dim = -1, sorted = False).indices.sort(dim = -1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).to(torch.int32)


def _indexers(model):
    return [m for m in model.modules() if isinstance(m, dsv41.DeepseekV41Indexer)]


def _record_calls(model):
    """Capture each indexer call's inputs, output and the shared dict after it."""
    calls, handles = [], []
    for module in _indexers(model):

        def hook(mod, args, kwargs, out):
            shared = kwargs["shared"]
            calls.append(
                (
                    mod,
                    args,
                    out.clone(),
                    {k: v.clone() for k, v in shared.items() if torch.is_tensor(v)},
                )
            )

        handles.append(module.register_forward_hook(hook, with_kwargs = True))
    return calls, handles


def _run(
    monkeypatch,
    chunk,
    seed = 0,
):
    if chunk is None:
        monkeypatch.delenv(CHUNK_ENV, raising = False)
        monkeypatch.setattr(dsv41, "_INDEXER_SCORE_BUDGET", 1 << 40)
    else:
        monkeypatch.setenv(CHUNK_ENV, str(chunk))
    chunks = []
    ranges = dsv41._indexer_q_ranges
    monkeypatch.setattr(
        dsv41, "_indexer_q_ranges", lambda *a: chunks.append(len(ranges(*a))) or ranges(*a)
    )
    model = _tiny_model(seed).train()
    torch.manual_seed(1)
    ids = torch.randint(0, 512, (2, SEQ))
    calls, handles = _record_calls(model)
    out = model(input_ids = ids, labels = ids)
    out.loss.backward()
    for h in handles:
        h.remove()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    monkeypatch.setattr(dsv41, "_indexer_q_ranges", ranges)
    assert chunks and set(chunks) == ({1} if chunk is None else {-(-SEQ // chunk)}), chunks
    return out.loss.detach(), grads, calls


@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_chunked_indices_loss_and_grads_match_one_shot(monkeypatch, chunk):
    loss_ref, grads_ref, calls_ref = _run(monkeypatch, None)
    loss, grads, calls = _run(monkeypatch, chunk)
    assert len(calls) == len(calls_ref) > 0
    assert any(c[0].is_candidate_source for c in calls) and any(c[0].uses_candidates for c in calls)
    for (_, _, idx, shared), (_, _, idx_ref, shared_ref) in zip(calls, calls_ref):
        assert torch.equal(idx, idx_ref)
        assert shared.keys() == shared_ref.keys()
        for k in shared:
            assert torch.equal(shared[k], shared_ref[k]), k
    assert torch.equal(loss, loss_ref)
    assert grads.keys() == grads_ref.keys()
    for n in grads:
        assert torch.equal(grads[n], grads_ref[n]), n


@pytest.mark.parametrize("chunk", [1, 5])
@pytest.mark.parametrize("ties", [False, True])
def test_chunked_indexer_matches_unchunked_reference(monkeypatch, chunk, ties):
    """Same indices and candidate mask as the pre-chunking code on recorded layer inputs, including exact ties."""
    monkeypatch.setenv(CHUNK_ENV, str(chunk))
    model = _tiny_model().eval()
    torch.manual_seed(2)
    ids = torch.randint(0, 512, (2, SEQ))
    calls, handles = _record_calls(model)
    with torch.no_grad():
        model(input_ids = ids)
    for h in handles:
        h.remove()
    shared_new, shared_ref = {}, {}
    for module, args, _, _ in calls:
        args = list(args)
        if ties and module.owns_k:
            with (
                torch.no_grad()
            ):  # every query scores all keys equally: top-k is decided by tie-breaking only
                module.k_proj.weight.zero_()
                module.k_norm.weight.fill_(1.0)
        new = module(*args, shared_new)
        ref = _reference_indexer(module, *args, shared_ref)
        assert torch.equal(new, ref)
        for k in ("index_k", "candidates"):
            assert (k in shared_new) == (k in shared_ref)
            if k in shared_ref:
                assert torch.equal(shared_new[k], shared_ref[k]), k


class _LiveBytes(TorchDispatchMode):
    """Tracks bytes of storages created inside the mode (freed when their last tensor dies) and the peak."""

    def __init__(self):
        super().__init__()
        self.live = self.peak = self.largest = 0
        self._refs = {}

    def _drop(self, key, nbytes):
        self._refs[key] -= 1
        if self._refs[key] == 0:
            del self._refs[key]
            self.live -= nbytes

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else [out]:
            if not torch.is_tensor(t):
                continue
            storage = t.untyped_storage()
            key, nbytes = storage.data_ptr(), storage.nbytes()
            if key not in self._refs:
                self._refs[key] = 0
                self.live += nbytes
                self.peak = max(self.peak, self.live)
                self.largest = max(self.largest, nbytes)
            self._refs[key] += 1
            weakref.finalize(t, self._drop, key, nbytes)
        return out


def _indexer_peak(
    monkeypatch,
    chunk,
    heads = 8,
    head_dim = 16,
    seq = 256,
):
    if chunk is None:
        monkeypatch.delenv(CHUNK_ENV, raising = False)
        monkeypatch.setattr(dsv41, "_INDEXER_SCORE_BUDGET", 1 << 40, raising = False)
    else:
        monkeypatch.setenv(CHUNK_ENV, str(chunk))
    model = _tiny_model(index_n_heads = heads, index_head_dim = head_dim).eval()
    torch.manual_seed(3)
    ids = torch.randint(0, 512, (1, seq))
    calls, handles = _record_calls(model)
    with torch.no_grad():
        model(input_ids = ids)
    for h in handles:
        h.remove()
    module, args, _, _ = next(c for c in calls if c[0].owns_k and c[0].compress_ratio == 2)
    args = list(args)
    shared = {}
    module(
        *args, shared
    )  # fills index_k; the measured call reuses it (the non-owning layers' path)
    args[2] = None
    n_comp = shared["index_k"].shape[1]
    tracker = _LiveBytes()
    with torch.no_grad(), tracker:
        module(*args, shared)
    full_score = seq * heads * n_comp * 4
    return tracker, full_score


def test_chunked_indexer_bounds_score_memory(monkeypatch):
    """Fails on the one-shot score: its [b, s, heads, t] fp32 tensor is the largest transient."""
    one_shot, full_score = _indexer_peak(monkeypatch, None)
    chunked, _ = _indexer_peak(monkeypatch, 16)
    assert one_shot.largest >= full_score
    assert chunked.largest <= full_score // 8
    assert chunked.peak < one_shot.peak // 4, (chunked.peak, one_shot.peak)


def test_default_budget_chunks_long_sequences():
    per_row = 32 * 4096  # 32 heads x 4096 compressed keys, b = 1
    ranges = dsv41._indexer_q_ranges(8192, per_row)
    rows = ranges[0][1] - ranges[0][0]
    assert rows * per_row <= dsv41._INDEXER_SCORE_BUDGET
    assert ranges[0][0] == 0 and ranges[-1][1] == 8192
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    assert dsv41._indexer_q_ranges(0, per_row) == [(0, 0)]
