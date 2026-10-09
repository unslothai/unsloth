# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
from types import SimpleNamespace

import pytest

from core.inference import mlx_speculative as spec

_DFLASH = {"architectures": ["DFlashDraftModel"], "dflash_config": {}, "vocab_size": 10}
_EAGLE3 = {"architectures": ["Eagle3Speculator"], "speculators_model_type": "eagle3"}
_ASSISTANT = {"architectures": ["Gemma4AssistantForCausalLM"], "model_type": "gemma4_assistant"}


_TARGET = "org/Qwen3.5-4B-4bit"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Fills a Hugging Face cache with ``{repo: config}`` beside the target, and returns a resolver
    for the target."""
    snapshots = {}

    def fill(repos, builtin = False):
        for repo, config in {_TARGET: {"vocab_size": 10}, **repos}.items():
            snapshots[repo] = tmp_path / repo.replace("/", "--")
            snapshots[repo].mkdir(parents = True, exist_ok = True)
            (snapshots[repo] / "config.json").write_text(json.dumps(config))
            (snapshots[repo] / "model.safetensors").touch()
        monkeypatch.setattr(spec, "has_builtin_head", lambda model_dir: builtin)
        return lambda mode, named = None: spec.resolve_speculation(
            mode, named, model_dir = str(snapshots[_TARGET]), target_name = _TARGET
        )

    monkeypatch.setattr(spec, "_cached_repos", lambda: iter(list(snapshots)))
    monkeypatch.setattr(
        "utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: snapshots.get(repo)
    )
    return fill


def test_an_explicit_kind_tries_the_named_drafter_then_the_head_then_cached_companions(cache):
    _resolve = cache(
        {
            "a/Qwen3.5-4B-DFlash": _DFLASH,
            "b/Qwen3.5-4B-Eagle3": _EAGLE3,
            "e/Qwen3.5-4B-Eagle3": {
                "architectures": ["LlamaForCausalLMEagle3"],
                "model_type": "llama",
            },
            "c/Qwen3.5-9B-DFlash": _DFLASH,  # another model
            "d/Qwen3.5-4B-DFlash-bf16": {**_DFLASH, "vocab_size": 11},  # another vocabulary
            "g/Qwen3.5-4B-assistant": _ASSISTANT,
        },
        builtin = True,
    )
    assert [os.path.basename(s.path) for s in _resolve("dflash").sources] == [
        "a--Qwen3.5-4B-DFlash"
    ]
    assert _resolve("dflash").copies and spec.mlx_spec_mode("mtp+ngram") == "mtp"
    assert spec.mlx_spec_mode("eagle3+ngram") == "auto"
    assert [os.path.basename(s.path) for s in _resolve("eagle3").sources] == [
        "b--Qwen3.5-4B-Eagle3"
    ]
    named = _resolve("eagle3", "b/Qwen3.5-4B-Eagle3")
    assert [s.kind for s in named.sources] == ["eagle3"] and named.copies and named.speculative
    mtp = [(os.path.basename(s.path), s.builtin) for s in _resolve("draft-mtp").sources]
    assert mtp == [("org--Qwen3.5-4B-4bit", True), ("g--Qwen3.5-4B-assistant", False)]
    assert _resolve("mtp", "g/Qwen3.5-4B-assistant").sources[0].builtin is False
    # A named drafter of another kind is refused with a reason, never substituted silently.
    assert _resolve("dflash", "b/Qwen3.5-4B-Eagle3").reason == spec.DRAFTER_INCOMPATIBLE
    assert _resolve("dspark").reason == spec.DRAFTER_NOT_FOUND
    assert _resolve("dspark").speculative


def test_auto_takes_every_cached_kind_in_preference_order(cache):
    _resolve = cache({"b/Qwen3.5-4B-Eagle3": _EAGLE3, "a/Qwen3.5-4B-DFlash": _DFLASH}, True)
    assert [s.kind for s in _resolve("auto").sources] == ["mtp", "dflash", "eagle3"] and not (
        _resolve("auto").reason or _resolve("off").speculative
    )
    assert [spec.speculates_on_route("auto", vision) for vision in (False, True)] == [False, True]
    listed = [(repo, s.kind) for repo, s, _ in spec.cached_drafters(_TARGET, {"vocab_size": 10})]
    assert listed == [("a/Qwen3.5-4B-DFlash", "dflash"), ("b/Qwen3.5-4B-Eagle3", "eagle3")]
    mtp_head = {"h/Qwen3.5-4B-MTP-bf16": {"model_type": "qwen3_5_mtp"}}
    _resolve = cache({"g/Qwen3.5-4B-assistant": _ASSISTANT, **mtp_head}, True)
    for mode in (None, "bogus", "auto"):
        assert [(s.builtin, s.kind) for s in _resolve(mode).sources] == [(True, "mtp")] + [
            (False, "mtp")
        ] * 2 + [
            (False, "dflash"),
            (False, "eagle3"),
        ]
    assert _resolve("ngram-mod").speculative


def test_cached_drafters_are_matched_by_shape_and_weights_then_marked_by_name(cache, tmp_path):
    shaped = {**_DFLASH, "hidden_size": 8, "num_target_layers": 4}
    narrow = {"target_hidden_size": 8, "transformer_layer_config": {"hidden_size": 2}}
    cache(
        {
            "a/Qwen3.5-4B-DFlash": shaped,
            "n/Other-DFlash": shaped,  # fits, but is not named for the target
            "h/Qwen3.5-4B-DFlash": {**shaped, "hidden_size": 9},
            "t/Qwen3.5-4B-DFlash": {**shaped, "num_target_layers": 5},
            "l/Qwen3.5-4B-DFlash": {**shaped, "dflash_config": {"target_layer_ids": [1, 5]}},
            "w/Qwen3.5-4B-DFlash": shaped,
            "b/Qwen3.5-4B-assistant": {**_ASSISTANT, "backbone_hidden_size": 9},
            "m/Qwen3.5-4B-MTP": {"model_type": "x_mtp", "text_config": {"num_hidden_layers": 5}},
            "e/Qwen3.5-4B-Eagle3": {
                **_EAGLE3,
                **narrow,
                "eagle_aux_hidden_state_layer_ids": [1, 4],
            },
        }
    )
    (tmp_path / "w--Qwen3.5-4B-DFlash" / "model.safetensors").unlink()
    target = {"vocab_size": 10, "text_config": {"hidden_size": 8, "num_hidden_layers": 4}}
    found = spec.cached_drafters("unsloth/Qwen3.5-4B-UD-MLX-4bit", target)
    assert [(repo, named) for repo, _, named in found] == [
        ("a/Qwen3.5-4B-DFlash", True),
        ("e/Qwen3.5-4B-Eagle3", True),
        ("n/Other-DFlash", False),
    ]
    assert len(spec.discover_companions("unsloth/Qwen3.5-4B-UD-MLX-4bit", target)) == 2
    assert len(spec.discover_companions("unsloth/Qwen3.5-4B-UD-MLX-4bit", target, found[1])) == 1
    assert spec._stem("g/gemma-4-31B-it-qat-q4_0-unquantized-assistant") == "gemma431bit"


def test_a_drafter_passed_over_keeps_its_reason_on_the_one_that_attaches(monkeypatch):
    drafters = pytest.importorskip("unsloth_zoo.mlx.speculative")

    built = []

    def factory(path, target):
        if path == "broken":
            raise ValueError("mismatched vocabulary")
        built.append(path)
        return SimpleNamespace(kind = "dflash", max_depth = 15)

    monkeypatch.setattr(drafters, "companion_drafter", factory)
    monkeypatch.setattr(drafters, "native_mtp_drafter", factory)
    sources = tuple(spec.DrafterSource("dflash", path, False) for path in ("big", "broken", "ok"))
    fits = lambda source: (source.path != "big", 4096)

    draft, kind, reason, context = spec.build_draft(
        None, spec.SpecResolution("dflash", sources), fits = fits, draft_n_max = 6
    )
    assert (kind, reason, context, built) == ("dflash", spec.DRAFTER_INCOMPATIBLE, 4096, ["ok"])
    assert (draft.controller.max_depth, draft.controller.max_copy) == (6, 0)
    for d, n, e in ((object(), 3, True), (object(), 4, False), (draft.drafter, 15, True)):
        assert spec._draft(d, False, n).controller.fixed_depth is e
    draft, kind, reason, _ = spec.build_draft(
        None, spec.SpecResolution("dflash", sources[:1], copies = True), fits = fits
    )
    assert (draft.drafter, kind, reason) == (None, "ngram", spec.DRAFTER_NO_MEMORY)
    assert (draft.controller.max_depth, draft.controller.max_copy) == (0, 16)
    auto = spec.SpecResolution("auto", sources[:1], copies = True)
    unfit = [spec.build_draft(None, auto, fits = lambda _: (ok, None)) for ok in (False, None)]
    assert unfit == [(None, None, spec.AUTO_CONTEXT_COST, None), (None,) * 4]
    capped = spec.build_draft(
        None, spec.SpecResolution("ngram", copies = True), fits = fits, draft_n_max = 5
    )
    assert capped[0].controller.max_copy == 5

    def encoder_decoder(inputs, encoder_outputs = None):
        pass

    def cross_attending(inputs, cross_attention_states = None):
        pass

    for target in (encoder_decoder, SimpleNamespace(language_model = cross_attending)):
        refused = spec.build_draft(
            target, spec.SpecResolution("dflash", sources[2:], copies = True), fits = fits
        )
        assert refused == (None, None, spec.RUNTIME_ERROR, None)


def test_a_drafter_carrying_its_own_model_module_is_never_found_or_attached(cache):
    # mlx-vlm imports a config's model_file on load, so such a checkpoint would run its own code.
    custom = {**_DFLASH, "model_file": "model.py"}
    _resolve = cache({"a/Qwen3.5-4B-DFlash": custom})
    assert spec.cached_drafters(_TARGET, {"vocab_size": 10}) == []
    assert not _resolve("auto").sources and not _resolve("dflash").sources
    assert _resolve("dflash", "a/Qwen3.5-4B-DFlash").reason == spec.DRAFTER_NOT_FOUND
