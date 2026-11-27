# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GRPO old / ref logps score the completion EOS when pad == eos, like the zoo loss mask.

With ``tokenizer.pad_token = tokenizer.eos_token`` the EOS that ends a completion has the pad id.
The zoo loss keeps that slot (``grpo_pad_is_eos_rows``) only when the old / ref logps cover it,
which ``_get_per_token_logps_and_entropies`` promises through ``_unsloth_grpo_logps_cover_eos``.
The packed path used to drop every pad-id token and leave 0 there. These tests run the real
scoring tail (left packing, packed and padded forwards) on CPU, then the real zoo loss, and
compare the per-token ratio and KL at the EOS slot with an unpacked per-row reference.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
zoo = pytest.importorskip("unsloth_zoo.rl_replacements")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _grpo_dispatch_source import load_dispatch_helpers  # noqa: E402

_SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
_HAS_HELPER = hasattr(zoo, "grpo_pad_is_eos_rows")
# The scorer only uses a helper that reads TRL's completion_mask (zoo 902b034 does not).
_HAS_MASK_HELPER = (
    _HAS_HELPER and "completion_mask" in inspect.signature(zoo.grpo_pad_is_eos_rows).parameters
)

VOCAB, HIDDEN = 24, 6
EOS = 2
DISTINCT_PAD = 0
KEEP = 4


@pytest.fixture(autouse = True)
def _restore_hidden_states_env(monkeypatch):
    # The scoring tail and the zoo loss set UNSLOTH_RETURN_HIDDEN_STATES for their forwards;
    # monkeypatch puts the caller's value back so later tests do not inherit it.
    monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", "sentinel")
    monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES")


class _CausalModel(torch.nn.Module):
    """Hidden state = tanh(running mean of the embeddings it attends to), causal per sequence.

    Honours attention_mask (padded forward) and packed_seq_lengths (packed forward), so a slot
    that is dropped or misplaced changes every later logprob in its row."""

    def __init__(self, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(VOCAB, HIDDEN)
        self.head = torch.nn.Linear(HIDDEN, VOCAB, bias = False)
        self.device = torch.device("cpu")
        self.packed_calls = 0

    def get_output_embeddings(self):
        return self.head

    def _hidden(self, ids, keep):
        e = self.emb(ids) * keep.unsqueeze(-1)
        count = keep.cumsum(dim = 1).clamp(min = 1).unsqueeze(-1)
        return torch.tanh(e.cumsum(dim = 1) / count)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        packed_seq_lengths = None,
        logits_to_keep = None,
        **kwargs,
    ):
        if packed_seq_lengths is not None:
            self.packed_calls += 1
            parts = torch.split(input_ids[0], [int(n) for n in packed_seq_lengths])
            h = torch.cat(
                [
                    self._hidden(p[None], torch.ones_like(p[None], dtype = torch.float32))[0]
                    for p in parts
                ]
            )[None]
        else:
            keep = (
                torch.ones_like(input_ids, dtype = torch.float32)
                if attention_mask is None
                else attention_mask.float()
            )
            h = self._hidden(input_ids, keep)
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        return SimpleNamespace(logits = h)


def _scoring_tail_source():
    """The real scoring tail of _get_per_token_logps_and_entropies, from ``all_logprobs_list = []``
    to its return, found by AST so the shipped code is what runs."""
    text = _SOURCE.read_text(encoding = "utf-8")
    tree = ast.parse(text)
    factory = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "grpo_trainer__get_per_token_logps_and_entropies"
    )
    inner = next(
        n
        for n in ast.walk(factory)
        if isinstance(n, ast.FunctionDef) and n.name == "_get_per_token_logps_and_entropies"
    )
    for node in ast.walk(inner):
        for field in ("body", "orelse"):
            seq = getattr(node, field, None)
            if not isinstance(seq, list):
                continue
            for i, stmt in enumerate(seq):
                if (
                    isinstance(stmt, ast.Assign)
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "all_logprobs_list"
                ):
                    lines = text.splitlines(keepends = True)[stmt.lineno - 1 : seq[-1].end_lineno]
                    body = textwrap.dedent("".join(lines))
                    return "def _tail(input_ids, attention_mask):\n" + textwrap.indent(body, "    ")
    raise AssertionError("scoring tail not found")


def _four_arg_completion_mask(
    completion_input_ids, left_pad_tokens_per_prompt, max_left_pad, pad_token_id
):
    """The released zoo signature: no eos_rows keyword, so passing it would raise."""
    n = completion_input_ids.shape[1]
    idx = torch.arange(n).unsqueeze(0)
    shift = idx >= (max_left_pad - left_pad_tokens_per_prompt).unsqueeze(1)
    return shift & (completion_input_ids != pad_token_id)


def _zoo_helpers(old_zoo):
    names = (
        "calculate_pad_tokens_in_prompt",
        "left_pack_padding",
        "create_completion_attention_mask",
        "chunked_selective_log_softmax",
        "chunked_hidden_states_selective_log_softmax",
    )
    out = {n: getattr(zoo, n) for n in names}
    if old_zoo:
        out["create_completion_attention_mask"] = _four_arg_completion_mask
    return out


def _trainer(pad):
    return SimpleNamespace(
        processing_class = SimpleNamespace(pad_token_id = pad, eos_token_id = EOS),
        args = SimpleNamespace(unsloth_grpo_mini_batch = 1, unsloth_logit_chunk_multiplier = 1),
        accelerator = SimpleNamespace(unwrap_model = lambda m, keep_fp32_wrapper = False: m, scaler = None),
        temperature = 1.0,
        _autocast_dtype = torch.float32,
        _autocast_enabled = False,
        use_vllm = False,
        num_generations = 2,
    )


def _score(
    trainer,
    model,
    ids,
    att,
    packing,
    old_zoo = False,
    monkeypatch = None,
    stale_cache = False,
):
    """Run the real scoring tail; returns the [B, KEEP + max_left_pad] logprobs.

    stale_cache: the generated trainer's copied-in mask helper predates eos_rows while the
    installed zoo has the EOS helper (a cache kept across a zoo upgrade)."""
    if old_zoo:
        monkeypatch.delattr(zoo, "grpo_pad_is_eos_rows", raising = False)
        monkeypatch.delattr(zoo, "grpo_real_token_mask", raising = False)
    ns = {
        "os": os,
        "torch": torch,
        **load_dispatch_helpers(),
        **_zoo_helpers(old_zoo or stale_cache),
        "self": trainer,
        "model": model,
        "unwrapped_model": model,
        "lm_head": model.head.weight,
        "logits_to_keep": KEEP,
        "pixel_values": None,
        "token_type_ids": None,
        "mm_token_type_ids": None,
        "B": 1,
        "multiplier": 1,
        "total_rows": ids.shape[0],
        "seq_len": ids.shape[1],
        "_grpo_vision_chunks": None,
        "vision_inputs": {},
        "_unsloth_head_hooked": False,
        "_unsloth_get_model_config": lambda m: None,
        "detect_logit_transforms": None,
        "_unsloth_get_final_logit_softcapping": lambda m: 0.0,
        "_unsloth_resolve_logit_scales": lambda c: (0.0, 0.0),
        # A PrefixGrouper capability probe: must not even be asked for a pad == eos batch.
        "_pg_enabled_fn": lambda m: trainer.__dict__.setdefault("pg_asked", True) and False,
        "UNSLOTH_GRPO_PREFIX_GROUPER_ON": True,
        "UNSLOTH_GRPO_PREFIX_GROUPER_VLLM_ON": True,
        "UNSLOTH_GRPO_SEQ_PACKING_ON": packing,
        "UNSLOTH_ZOO_HAS_MASKED_COL_GUARD": True,
        "UNSLOTH_ENABLE_LOGGING": False,
        "DEVICE_TYPE_TORCH": "cpu",
        "device_synchronize": lambda *a, **k: None,
    }
    exec(_scoring_tail_source(), ns)
    model.packed_calls = 0
    logps, _entropies, _aux = ns["_tail"](ids.clone(), att.clone())
    if packing:
        assert model.packed_calls == 1, "the packed forward did not run"
    return logps


def _trl_mask(cids):
    is_eos = cids == EOS
    idx = torch.full((cids.shape[0],), cids.shape[1], dtype = torch.long)
    idx[is_eos.any(1)] = is_eos.int().argmax(1)[is_eos.any(1)]
    return (torch.arange(cids.shape[1]).expand_as(cids) <= idx[:, None]).int()


def _batch(pad):
    prompts = [[pad, pad, 5, 6], [7, 8, 9, 10], [pad, 11, 12, 13], [pad, pad, pad, 14]]
    comps = [[15, 16, EOS, pad], [17, 18, 19, 20], [EOS, pad, pad, pad], [21, EOS, pad, pad]]
    ids = torch.tensor([p + c for p, c in zip(prompts, comps)])
    cm = _trl_mask(ids[:, -KEEP:])
    att = torch.cat([(ids[:, :-KEEP] != pad).int(), cm], 1)
    return ids, att, cm


def _reference(model, ids, att):
    """Unpacked per-row logprob of every kept completion token, [B, KEEP] in TRL's layout."""
    out = torch.zeros(ids.shape[0], KEEP)
    with torch.no_grad():
        for r in range(ids.shape[0]):
            cols = att[r].nonzero().squeeze(1)
            real = ids[r, cols][None]
            h = model(input_ids = real).logits
            lp = torch.log_softmax(h @ model.head.weight.t(), -1)[0]
            for j in range(1, real.shape[1]):
                c = int(cols[j]) - (ids.shape[1] - KEEP)
                if c >= 0:
                    out[r, c] = lp[j - 1, real[0, j]]
    return out


def _to_trl_layout(packed, mask):
    """Packed [B, W] row values at their mask slots -> TRL's right-padded [B, KEEP]."""
    out = torch.zeros(packed.shape[0], KEEP)
    for r in range(packed.shape[0]):
        vals = packed[r][mask[r].bool()]
        out[r, : vals.numel()] = vals
    return out


def _zoo_loss_inputs(trainer, policy, ids, att, cm, old, ref):
    """Run the real zoo loss; capture what reaches UnslothEfficientGRPO."""
    seen = {}

    class _Capture:
        @staticmethod
        def apply(new_logps, old_logps, ref_logps, sampling, lm_head, input_ids, mask, *rest):
            seen.update(
                new = new_logps.detach().clone(), old = old_logps, ref = ref_logps, mask = mask.clone()
            )
            z = torch.zeros(())
            return z, z, z, z, z, z

    real = zoo.UnslothEfficientGRPO
    zoo.UnslothEfficientGRPO = _Capture
    try:
        trainer.model = policy
        trainer.beta = 0.04
        zoo.grpo_accumulated_loss(
            trainer,
            ids,
            att,
            KEEP,
            cm,
            torch.zeros(ids.shape[0]),
            old,
            ref,
            loss_type = "dapo",
            num_items_in_batch = int(cm.sum()),
        )
    finally:
        zoo.UnslothEfficientGRPO = real
    return seen


needs_helper = pytest.mark.skipif(
    not _HAS_HELPER, reason = "unsloth_zoo without grpo_pad_is_eos_rows"
)
needs_mask_helper = pytest.mark.skipif(
    not _HAS_MASK_HELPER,
    reason = "unsloth_zoo grpo_pad_is_eos_rows without completion_mask",
)


@needs_mask_helper
@pytest.mark.parametrize("packing", [True, False])
def test_pad_is_eos_old_and_ref_logps_score_the_eos_slot(packing):
    ids, att, cm = _batch(EOS)
    trainer = _trainer(EOS)
    old_model, ref_model = _CausalModel(1), _CausalModel(2)
    old = _score(trainer, old_model, ids, att, packing)
    ref = _score(trainer, ref_model, ids, att, packing)
    # The scorer's own output holds the EOS logprob at the zoo's EOS slot (not 0).
    lp = zoo.calculate_pad_tokens_in_prompt(ids, KEEP, EOS)
    w = KEEP + int(lp.max())
    packed_ids = zoo.left_pack_padding(ids, EOS)
    with_eos = zoo.create_completion_attention_mask(
        packed_ids[:, -w:],
        lp,
        int(lp.max()),
        EOS,
        eos_rows = zoo.grpo_pad_is_eos_rows(trainer, ids[:, -KEEP:], completion_mask = cm),
    )
    eos_cols = with_eos & ~zoo.create_completion_attention_mask(
        packed_ids[:, -w:], lp, int(lp.max()), EOS
    )
    assert int(eos_cols.sum()) == 3
    assert (old[eos_cols] != 0).all() and (ref[eos_cols] != 0).all(), "EOS slot left unscored"
    assert not getattr(trainer, "pg_asked", False), "PrefixGrouper engaged on a pad == eos batch"
    assert trainer._unsloth_grpo_logps_cover_eos is True

    seen = _zoo_loss_inputs(trainer, _CausalModel(3), ids, att, cm, old, ref)
    mask = seen["mask"].bool()
    # The zoo mask now keeps each EOS, exactly TRL's completion_mask.
    assert mask.sum(1).tolist() == cm.sum(1).tolist()

    ref_old, ref_ref = _reference(old_model, ids, att), _reference(ref_model, ids, att)
    ref_new = _reference(_CausalModel(3), ids, att)
    got_old, got_ref = _to_trl_layout(seen["old"], mask), _to_trl_layout(seen["ref"], mask)
    got_new = _to_trl_layout(seen["new"], mask)
    m = cm.bool()
    for got, want in ((got_old, ref_old), (got_ref, ref_ref), (got_new, ref_new)):
        assert torch.allclose(got[m], want[m], atol = 1e-5), (got * m, want * m)

    eos_slot = (ids[:, -KEEP:] == EOS) & m
    assert eos_slot.sum() == 3
    assert (got_old[eos_slot] != 0).all() and (got_ref[eos_slot] != 0).all()
    # Per-token ratio and k3 KL at the EOS slot equal the unpacked reference.
    ratio, ratio_ref = torch.exp(got_new - got_old), torch.exp(ref_new - ref_old)
    kl = torch.exp(got_ref - got_new) - (got_ref - got_new) - 1
    kl_ref = torch.exp(ref_ref - ref_new) - (ref_ref - ref_new) - 1
    assert torch.allclose(ratio[eos_slot], ratio_ref[eos_slot], atol = 1e-5)
    assert torch.allclose(kl[eos_slot], kl_ref[eos_slot], atol = 1e-6)
    assert (kl_ref[eos_slot] > 0).all()


@needs_helper
@pytest.mark.parametrize("packing", [True, False])
def test_pad_is_not_eos_is_bitwise_unchanged(packing, monkeypatch):
    ids, att, _cm = _batch(DISTINCT_PAD)
    head = _score(_trainer(DISTINCT_PAD), _CausalModel(1), ids, att, packing)
    # The same tail with the eos path unavailable runs exactly the pre-change code. A fresh,
    # identical model, so both arms run the first-use per-row verifier.
    base = _score(
        _trainer(DISTINCT_PAD),
        _CausalModel(1),
        ids,
        att,
        packing,
        old_zoo = True,
        monkeypatch = monkeypatch,
    )
    assert torch.equal(head, base)


@pytest.mark.parametrize("packing", [True, False])
@pytest.mark.parametrize("pad", [EOS, DISTINCT_PAD])
def test_zoo_without_the_helper_keeps_todays_behaviour(pad, packing, monkeypatch):
    ids, att, _cm = _batch(pad)
    trainer = _trainer(pad)
    model = _CausalModel(1)
    got = _score(trainer, model, ids, att, packing, old_zoo = True, monkeypatch = monkeypatch)
    assert not getattr(trainer, "_unsloth_grpo_logps_cover_eos", False)
    # Today's rule: the id mask, with pad-id slots left out of the packed forward (0 there).
    lp = zoo.calculate_pad_tokens_in_prompt(ids, KEEP, pad)
    w = KEEP + int(lp.max())
    packed_ids = zoo.left_pack_padding(ids, pad)
    keep = _zoo_helpers(True)["create_completion_attention_mask"](
        packed_ids[:, -w:], lp, int(lp.max()), pad
    )
    assert got.shape == (ids.shape[0], w)
    if packing:
        assert (got[~keep] == 0).all()
    if pad == EOS:
        assert int(keep.sum()) == int(_cm.sum()) - 3


def _stop_string_batch():
    """_batch(EOS) with row 1 stopped by a stop string: no EOS, then pad-id (== EOS) padding."""
    ids, att, cm = _batch(EOS)
    ids[1, -KEEP:] = torch.tensor([17, 18, EOS, EOS])
    cm[1] = torch.tensor([1, 1, 0, 0])
    att[1, -KEEP:] = cm[1]
    return ids, att, cm


@needs_helper
@pytest.mark.parametrize("packing", [True, False])
def test_right_padding_without_eos_stays_out_of_the_loss(packing):
    ids, att, cm = _stop_string_batch()
    trainer = _trainer(EOS)
    old = _score(trainer, _CausalModel(1), ids, att, packing)
    ref = _score(trainer, _CausalModel(2), ids, att, packing)
    seen = _zoo_loss_inputs(trainer, _CausalModel(3), ids, att, cm, old, ref)
    kept = seen["mask"].sum(dim = 1).tolist()
    # The stop-string row keeps its 2 sampled tokens, never the unsampled padding slot.
    assert kept[1] == 2
    assert all(k <= n for k, n in zip(kept, cm.sum(dim = 1).tolist())), (kept, cm.sum(dim = 1))
    if _HAS_MASK_HELPER:
        assert kept == cm.sum(dim = 1).tolist()


@needs_helper
@pytest.mark.parametrize("packing", [True, False])
def test_helper_without_completion_mask_keeps_todays_behaviour(packing, monkeypatch):
    # zoo 902b034's helper reads token ids only, so it would mark the stop-string row's padding.
    two_arg = zoo.grpo_pad_is_eos_rows

    def grpo_pad_is_eos_rows(trainer, completion_ids):
        return two_arg(trainer, completion_ids)

    ids, att, _cm = _stop_string_batch()
    base = _score(
        _trainer(EOS), _CausalModel(1), ids, att, packing, old_zoo = True, monkeypatch = monkeypatch
    )
    monkeypatch.setattr(zoo, "grpo_pad_is_eos_rows", grpo_pad_is_eos_rows, raising = False)
    trainer = _trainer(EOS)
    got = _score(trainer, _CausalModel(1), ids, att, packing)
    assert not getattr(trainer, "_unsloth_grpo_logps_cover_eos", False)
    assert torch.equal(got, base)


@needs_mask_helper
@pytest.mark.parametrize("packing", [True, False])
def test_prefix_grouper_is_asked_when_no_row_ends_on_the_pad_id(packing):
    # pad 2 is an EOS id, but every completion stopped on the other EOS id 3 (Qwen-style
    # pad <|endoftext|>, stop <|im_end|>): there is no EOS slot to keep, so nothing changes.
    other = 3
    ids, att, _cm = _batch(DISTINCT_PAD)
    comps = torch.tensor(
        [[15, 16, other, EOS], [17, 18, 19, 20], [other, EOS, EOS, EOS], [21, other, EOS, EOS]]
    )
    prompts = ids[:, :-KEEP].masked_fill(ids[:, :-KEEP] == DISTINCT_PAD, EOS)
    ids = torch.cat([prompts, comps], 1)
    cm = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]])
    att = torch.cat([(prompts != EOS).int(), cm], 1)
    trainer = _trainer(EOS)
    trainer.eos_token_ids = [other, EOS]
    got = _score(trainer, _CausalModel(1), ids, att, packing)
    assert trainer.pg_asked is True, "PrefixGrouper skipped on a batch with no pad-id EOS"
    lp = zoo.calculate_pad_tokens_in_prompt(ids, KEEP, EOS)
    assert got.shape == (ids.shape[0], KEEP + int(lp.max()))


@needs_mask_helper
@pytest.mark.parametrize("packing", [True, False])
def test_stale_cached_mask_helper_uses_the_installed_zoo(packing):
    ids, att, _cm = _batch(EOS)
    want = _score(_trainer(EOS), _CausalModel(1), ids, att, packing)
    got = _score(_trainer(EOS), _CausalModel(1), ids, att, packing, stale_cache = True)
    assert torch.equal(got, want)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


needs_real_mask = pytest.mark.skipif(
    not hasattr(zoo, "grpo_real_token_mask"), reason = "unsloth_zoo without grpo_real_token_mask"
)


def _multi_turn_batch():
    """pad == eos == the turn end, real inside the prompt and inside tool-calling completions."""
    prompts = [[EOS, EOS, 5, EOS], [7, EOS, 9, 10]]
    prompt_mask = [[0, 0, 1, 1], [1, 1, 1, 1]]
    comps = [[15, EOS, 3, EOS, 16, EOS, EOS, EOS], [17, 18, EOS, 4, EOS, 19, EOS, EOS]]
    cm = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]])
    ids = torch.tensor([p + c for p, c in zip(prompts, comps)])
    att = torch.cat([torch.tensor(prompt_mask), cm], 1)
    return ids, att, cm


@needs_real_mask
@pytest.mark.parametrize("packing", [True, False])
def test_pad_is_eos_real_tokens_match_the_zoo_layout(packing, monkeypatch):
    # Old / ref logps and the zoo loss must lay rows out alike when pad-id tokens are real
    # (prompt turn ends, EOS inside a multi-turn completion), so every ratio and KL term
    # compares the same token.
    ids, att, cm = _multi_turn_batch()
    monkeypatch.setattr(sys.modules[__name__], "KEEP", cm.shape[1])
    trainer = _trainer(EOS)
    old_model, ref_model = _CausalModel(1), _CausalModel(2)
    old = _score(trainer, old_model, ids, att, packing)
    ref = _score(trainer, ref_model, ids, att, packing)
    assert trainer._unsloth_grpo_logps_prompt_from_mask is True
    assert not getattr(trainer, "pg_asked", False), "PrefixGrouper engaged on real pad-id tokens"

    seen = _zoo_loss_inputs(trainer, _CausalModel(3), ids, att, cm, old, ref)
    mask = seen["mask"].bool()
    assert mask.sum(1).tolist() == cm.sum(1).tolist()
    m = cm.bool()
    for got, model in (
        (seen["old"], old_model),
        (seen["ref"], ref_model),
        (seen["new"], _CausalModel(3)),
    ):
        want = _reference(model, ids, att)
        got = _to_trl_layout(got, mask)
        assert torch.allclose(got[m], want[m], atol = 1e-5), (got * m, want * m)
