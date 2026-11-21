# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A first run reads the base revision before its download; the next run must still find what it wrote."""

from __future__ import annotations

import types

import torch

from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_train_extras as extras
from core.training.diffusion_train_extras import PersistentConditioningCache, source_revision

REPO = "org/model"


def _opener(cache_dir):
    def reopen():
        return PersistentConditioningCache(
            str(cache_dir), f"flux.1_{REPO}_{source_revision(REPO)}", 512
        )

    return reopen


def test_a_cold_first_run_writes_where_the_second_run_reads(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(extras, "_hub_cache_roots", lambda: [str(hub)])
    reopen = _opener(tmp_path / "cond")
    spec = types.SimpleNamespace(
        encode_prompts = lambda pipe, caps, device: [(torch.ones(2),)] * len(caps)
    )

    first = reopen()
    assert "unresolved" in first.family
    refs = hub / "models--org--model" / "refs"
    refs.mkdir(parents = True)
    (refs / "main").write_text("0123456789abcdef0123")
    rekeyed, _ = dit._encode_prompts_after_load(spec, None, ["a cat"], "cpu", first, reopen)

    second = reopen()
    assert "unresolved" not in second.family
    assert (
        rekeyed.family == second.family
    ), "latents written after the rekey land in run 2's namespace"
    hit = second.get(second.text_key("a cat"))
    assert hit is not None and torch.equal(hit[0], torch.ones(2))


def test_a_fetch_that_moves_the_revision_never_serves_the_old_revisions_embeddings(
    monkeypatch, tmp_path
):
    hub = tmp_path / "hub"
    refs = hub / "models--org--model" / "refs"
    refs.mkdir(parents = True)
    (refs / "main").write_text("a" * 40)
    monkeypatch.setattr(extras, "_hub_cache_roots", lambda: [str(hub)])
    reopen = _opener(tmp_path / "cond")
    old = reopen()
    old.put(old.text_key("a cat"), (torch.zeros(2),))
    calls = []

    def encode(pipe, caps, device):
        calls.append(list(caps))
        return [(torch.ones(2),)] * len(caps)

    spec = types.SimpleNamespace(encode_prompts = encode)
    (refs / "main").write_text("b" * 40)  # the conditioner load fetched a new revision
    fresh, encoded = dit._encode_prompts_after_load(spec, None, ["a cat"], "cpu", old, reopen)
    assert calls == [["a cat"]], "the new weights encode the caption"
    assert torch.equal(encoded[0][0], torch.ones(2))
    assert fresh.family != old.family
    assert torch.equal(fresh.get(fresh.text_key("a cat"))[0], torch.ones(2))
    assert torch.equal(
        old.get(old.text_key("a cat"))[0], torch.zeros(2)
    ), "the old namespace is untouched"


def test_an_unchanged_namespace_keeps_the_same_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(extras, "_hub_cache_roots", lambda: [str(tmp_path / "hub")])
    reopen = _opener(tmp_path / "cond")
    cache = reopen()
    assert dit._rekey_conditioning_cache(cache, reopen) is cache
    assert dit._rekey_conditioning_cache(None, reopen) is None


def test_the_encoder_key_rekeys_once_the_hosted_encoder_is_on_disk(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(extras, "_hub_cache_roots", lambda: [str(hub)])

    def reopen():
        return PersistentConditioningCache(
            str(tmp_path / "cond"),
            "ltx-2.5_pack_rev-0000",
            512,
            text_encoder = f"te-int8-hosted-{source_revision('unsloth/LTX-2.5-FP8')}",
        )

    first = reopen()
    refs = hub / "models--unsloth--LTX-2.5-FP8" / "refs"
    refs.mkdir(parents = True)
    (refs / "main").write_text("fedcba9876543210")
    spec = types.SimpleNamespace(
        encode_prompts = lambda pipe, caps, device: [(torch.ones(2),)] * len(caps)
    )
    rekeyed, _ = dit._encode_prompts_after_load(spec, None, ["a cat"], "cpu", first, reopen)
    second = reopen()
    assert rekeyed.text_encoder == second.text_encoder != first.text_encoder
    assert second.get(second.text_key("a cat")) is not None
