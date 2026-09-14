# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which configurations may tokenize online, and what the lazy view produces.

No GPU and no model: the gate is a pure function of the run's shape, and the
transform runs against a real tokenizer on a real ``datasets.Dataset``. Every
"degrades to the old path" claim is a test here, since a wrong answer is either a
crash (VLM, pre-tokenized) or a run that trains on different rows.
"""

import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, "studio/backend")

from utils.datasets.online_tokenization import (  # noqa: E402
    ENV_FLAG,
    MIN_ROWS_FOR_ONLINE,
    TRUNCATION_ATTESTATION_ATTR,
    OnlineTokenizationDecision,
    attach_online_tokenization,
    build_tokenizing_transform,
    dataset_column_names,
    dataset_supports_with_transform,
    decide_online_tokenization,
    env_override,
    is_processor,
    online_config_args,
    prewarm_batch_count,
    resolve_add_special_tokens,
    text_column_defect,
    trl_supports_skip_prepare_dataset,
)

datasets = pytest.importorskip("datasets")


ROWS = MIN_ROWS_FOR_ONLINE + 5


def _text_dataset(n = ROWS, extra_columns = None):
    data = {
        "text": [f"row {i}" for i in range(n)],
        "conversations": [[{"role": "user", "content": str(i)}] for i in range(n)],
    }
    data.update(extra_columns or {})
    return datasets.Dataset.from_dict(data)


class _Tokenizer:
    """The narrowest thing the online path needs: callable, no ``.tokenizer``."""

    bos_token = "<s>"
    chat_template = "{{ messages }}"

    def __call__(
        self,
        texts,
        truncation = True,
        max_length = 8,
        add_special_tokens = True,
    ):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[len(t)] * min(len(t), max_length if truncation else len(t)) for t in texts]
        return {"input_ids": ids}


class _Processor(_Tokenizer):
    tokenizer = _Tokenizer()


def _base_kwargs(**overrides):
    kwargs = dict(
        dataset = _text_dataset(),
        eval_dataset = None,
        processing_class = _Tokenizer(),
        model = SimpleNamespace(),
        text_field = "text",
        packing = False,
        num_train_epochs = 1,
        max_steps = 0,
        grad_accum = 4,
        workers = 4,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(autouse = True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(ENV_FLAG, raising = False)
    # The gate refuses on spawn platforms; these tests describe Linux behaviour
    # and simulate the other platforms explicitly where that is the point.
    monkeypatch.setattr(sys, "platform", "linux")
    # Same for the TRL hook: the CPU test job installs no TRL, so leaving it
    # ambient makes every gate below report "no skip_prepare_dataset hook".
    # The detector itself is covered separately below.
    monkeypatch.setattr(
        "utils.datasets.online_tokenization.trl_supports_skip_prepare_dataset",
        lambda: True,
    )


# ---------------------------------------------------------------- the happy path


def test_plain_text_single_epoch_run_goes_online():
    decision = decide_online_tokenization(**_base_kwargs())
    assert decision.enabled, decision.reason
    assert decision.workers == 4
    assert decision.prewarm_batches == max(4, 4 * decision.prefetch_factor)


def test_online_config_args_are_the_four_keys_the_mechanism_needs():
    decision = decide_online_tokenization(**_base_kwargs())
    args = online_config_args(decision)
    assert args["dataset_kwargs"] == {"skip_prepare_dataset": True}
    # `Trainer._remove_unused_columns` reads `column_names`, which a transformed
    # split answers from its BACKING table -- it would strip the text column the
    # transform reads.
    assert args["remove_unused_columns"] is False
    assert args["dataloader_num_workers"] == 4
    assert args["dataloader_prefetch_factor"] > 0
    assert args["dataloader_persistent_workers"] is True


# ------------------------------------------------------- degradation, one per gate


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_spawn_platforms_keep_the_eager_path(monkeypatch, platform):
    """Unsloth already forces `dataloader_num_workers = 0` there, because a
    modified `sys.path` does not survive the spawn. Never a crash: just off."""
    monkeypatch.setattr(sys, "platform", platform)
    decision = decide_online_tokenization(**_base_kwargs())
    assert not decision.enabled
    assert platform in decision.reason


@pytest.mark.parametrize(
    "flag, reason_fragment",
    [
        ({"is_vlm": True}, "multimodal"),
        ({"is_audio_vlm": True}, "multimodal"),
        ({"is_deepseek_ocr": True}, "multimodal"),
        ({"is_audio": True}, "audio"),
        ({"is_cpt": True}, "continued pretraining"),
        ({"raw_text_mode": True}, "raw-text"),
        ({"has_custom_collator": True}, "custom data collator"),
        ({"packing": True}, "packing"),
        ({"train_on_completions": True}, "train on completions"),
        ({"dataset_streaming": True}, "streaming"),
    ],
)
def test_each_excluded_shape_takes_the_old_path(flag, reason_fragment):
    decision = decide_online_tokenization(**_base_kwargs(**flag))
    assert not decision.enabled
    assert reason_fragment in decision.reason


def test_streaming_dataset_object_is_refused_even_without_the_flag():
    """`IterableDataset` also has `with_transform` in recent `datasets`, so the
    check is an isinstance and not a `hasattr`."""
    stream = datasets.Dataset.from_dict({"text": ["a", "b"]}).to_iterable_dataset()
    decision = decide_online_tokenization(**_base_kwargs(dataset = stream))
    assert not decision.enabled
    assert "map-style" in decision.reason


def test_a_plain_list_dataset_is_refused():
    decision = decide_online_tokenization(**_base_kwargs(dataset = [{"text": "a"}] * ROWS))
    assert not decision.enabled
    assert "map-style" in decision.reason


@pytest.mark.parametrize("column", ["input_ids", "labels", "prompt", "completion"])
def test_an_already_tokenized_dataset_is_refused(column):
    dataset = _text_dataset(extra_columns = {column: [[1, 2, 3]] * ROWS})
    decision = decide_online_tokenization(**_base_kwargs(dataset = dataset))
    assert not decision.enabled
    assert column in decision.reason


def test_a_processor_is_refused():
    decision = decide_online_tokenization(**_base_kwargs(processing_class = _Processor()))
    assert not decision.enabled
    assert "processor" in decision.reason


def test_a_model_needing_token_type_ids_is_refused(monkeypatch):
    """Gemma-family modules build their causal mask from `token_type_ids`, and
    the zoo's tokenize asks for them. Rather than reproduce that column lazily,
    those models keep the eager path."""
    module = SimpleNamespace(**{"create_" + "causal_mask_mapping": lambda: None})
    monkeypatch.setitem(sys.modules, "fake_gemma_modelling", module)

    class _GemmaLike:
        pass

    _GemmaLike.__module__ = "fake_gemma_modelling"
    decision = decide_online_tokenization(**_base_kwargs(model = _GemmaLike()))
    assert not decision.enabled
    assert "token_type_ids" in decision.reason


def test_missing_text_column_is_refused():
    dataset = datasets.Dataset.from_dict({"conversations": [[]] * ROWS})
    decision = decide_online_tokenization(**_base_kwargs(dataset = dataset))
    assert not decision.enabled
    assert "text" in decision.reason


def test_a_null_text_row_is_refused():
    """The reproduction that motivated this gate: the eager map dies on one None
    inside the constructor, while the lazy view trained past step 20 and would
    have died hours in, at whatever step drew row 137."""
    texts = [f"row {i}" for i in range(ROWS)]
    texts[137] = None
    dataset = datasets.Dataset.from_dict({"text": texts})
    decision = decide_online_tokenization(**_base_kwargs(dataset = dataset))
    assert not decision.enabled
    assert "null" in decision.reason


@pytest.mark.parametrize(
    "column",
    [
        [7] * ROWS,
        [[f"row {i}"] for i in range(ROWS)],
        [{"content": "x"}] * ROWS,
    ],
    ids = ["ints", "lists", "structs"],
)
def test_a_text_column_that_is_not_strings_is_refused(column):
    dataset = datasets.Dataset.from_dict({"text": column})
    decision = decide_online_tokenization(**_base_kwargs(dataset = dataset))
    assert not decision.enabled
    assert "not strings" in decision.reason


def test_a_null_text_row_in_the_eval_split_is_refused():
    texts = [f"row {i}" for i in range(64)]
    texts[7] = None
    eval_dataset = datasets.Dataset.from_dict({"text": texts})
    decision = decide_online_tokenization(**_base_kwargs(eval_dataset = eval_dataset))
    assert not decision.enabled
    assert "eval split" in decision.reason and "null" in decision.reason


def test_the_text_column_check_reads_metadata_and_never_a_row():
    """Both halves come off the schema and Arrow's per-chunk null count, so the
    gate must reach its answer on a split whose rows refuse to be read at all --
    otherwise it is the eager pass it exists to avoid, in miniature."""

    class _Unreadable(type(_text_dataset(16))):
        def __getitem__(self, key):
            raise AssertionError("the gate read a row")

    dataset = _text_dataset()
    unreadable = _Unreadable(dataset.data, info = dataset.info)
    assert text_column_defect(unreadable, "text") is None


def test_a_spawn_start_method_keeps_the_eager_path_on_linux(monkeypatch):
    """The gate is named for Windows and macOS, but the hazard it describes is
    `spawn` re-importing the entry point against a `sys.path` Unsloth modified in
    process. A Linux host set to spawn is the same hazard."""
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none = False: "spawn")
    decision = decide_online_tokenization(**_base_kwargs())
    assert not decision.enabled
    assert "spawn" in decision.reason and "fork" in decision.reason


def test_an_unset_start_method_falls_back_to_the_platform_default(monkeypatch):
    """`get_start_method()` with no argument pins the context and makes a later
    `set_start_method()` raise, so the default is read off the method list."""
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none = False: None)
    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn"])
    assert decide_online_tokenization(**_base_kwargs()).enabled

    monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: ["forkserver"])
    assert not decide_online_tokenization(**_base_kwargs()).enabled


def test_a_trl_without_the_hook_keeps_the_eager_path(monkeypatch):
    """The veto the autouse fixture pins away, exercised on its own: without
    `skip_prepare_dataset` TRL would run its own tokenizing map over the lazy
    view, which is the whole pass the online path exists to avoid."""
    monkeypatch.setattr(
        "utils.datasets.online_tokenization.trl_supports_skip_prepare_dataset",
        lambda: False,
    )
    decision = decide_online_tokenization(**_base_kwargs())
    assert not decision.enabled
    assert "skip_prepare_dataset" in decision.reason


def test_the_hook_detector_reads_the_installed_trl(monkeypatch):
    """No TRL means no SFT run, so the detector says no rather than raising; a
    `SFTConfig` without `dataset_kwargs` says no too, since the key would be
    dropped and TRL would tokenize the view."""
    import builtins

    real_import = builtins.__import__

    def no_trl(name, *args, **kwargs):
        if name == "trl" or name.startswith("trl."):
            raise ImportError("No module named 'trl'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_trl)
    assert trl_supports_skip_prepare_dataset() is False


def test_too_few_workers_is_refused():
    decision = decide_online_tokenization(**_base_kwargs(workers = 1))
    assert not decision.enabled
    assert "workers" in decision.reason


def test_a_small_dataset_keeps_the_eager_path():
    decision = decide_online_tokenization(**_base_kwargs(dataset = _text_dataset(100)))
    assert not decision.enabled
    assert "smaller than" in decision.reason


def test_multi_epoch_runs_keep_the_eager_path():
    """The lazy view re-tokenizes on every pass; the eager one reads Arrow.
    Measured at +2.9% of steady-state training time over 2.4 epochs, against a
    saving that is paid once, so anything past a single pass stays eager."""
    decision = decide_online_tokenization(**_base_kwargs(num_train_epochs = 3))
    assert not decision.enabled
    assert "one pass" in decision.reason


def test_a_step_capped_run_of_unknown_length_keeps_the_eager_path():
    decision = decide_online_tokenization(**_base_kwargs(max_steps = 500))
    assert not decision.enabled
    assert "unknown length" in decision.reason


def test_a_resolved_sub_epoch_step_cap_may_go_online():
    """`max_steps` alone says nothing about passes, but a caller that has
    resolved it to a fraction of an epoch has answered the question."""
    decision = decide_online_tokenization(
        **_base_kwargs(max_steps = 60, resolved_max_steps_epochs = 0.02)
    )
    assert decision.enabled, decision.reason


# ---------------------------------------------------------------- the eval split


def test_a_raw_eval_split_is_transformed_alongside_the_train_split():
    decision = decide_online_tokenization(**_base_kwargs(eval_dataset = _text_dataset(64)))
    assert decision.enabled, decision.reason


def test_an_eval_split_the_transform_cannot_serve_disables_the_feature():
    """`skip_prepare_dataset` skips the EVAL prep too, so an eval split the
    online path cannot tokenize would reach the model as raw text."""
    bad_eval = datasets.Dataset.from_dict({"something_else": ["x"] * 8})
    decision = decide_online_tokenization(**_base_kwargs(eval_dataset = bad_eval))
    assert not decision.enabled
    assert "eval" in decision.reason


def test_an_already_tokenized_eval_split_disables_the_feature():
    tokenized_eval = datasets.Dataset.from_dict({"text": ["x"] * 8, "input_ids": [[1, 2]] * 8})
    decision = decide_online_tokenization(**_base_kwargs(eval_dataset = tokenized_eval))
    assert not decision.enabled
    assert "eval" in decision.reason


# ------------------------------------------------------------------ escape hatch


def test_env_flag_zero_forces_the_eager_path(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "0")
    assert env_override() is False
    decision = decide_online_tokenization(**_base_kwargs())
    assert not decision.enabled
    assert ENV_FLAG in decision.reason


def test_env_flag_one_overrides_the_cost_gates_only(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "1")
    forced = decide_online_tokenization(
        **_base_kwargs(dataset = _text_dataset(10), num_train_epochs = 5)
    )
    assert forced.enabled, forced.reason
    # ...but never a correctness gate: a VLM stays eager however hard it is asked.
    assert not decide_online_tokenization(**_base_kwargs(is_vlm = True)).enabled


def test_an_unrecognised_env_value_is_not_an_override(monkeypatch):
    monkeypatch.setenv(ENV_FLAG, "maybe")
    assert env_override() is None
    assert decide_online_tokenization(**_base_kwargs()).enabled


# ------------------------------------------------------------------- the transform


def test_the_transform_returns_input_ids_for_the_whole_batch():
    transform = build_tokenizing_transform(_Tokenizer(), "text", 8, True)
    out = transform({"text": ["abc", "de"]})
    assert "input_ids" in out
    assert len(out["input_ids"]) == 2


def test_the_transform_passes_the_whole_tokenizer_output_through():
    """The eager map keeps `attention_mask` too (`remove_columns` drops only the
    ORIGINAL columns), and both the collator and the attention dispatcher branch
    on which keys are present."""

    class _WithMask(_Tokenizer):
        def __call__(self, texts, **kwargs):
            out = super().__call__(texts, **kwargs)
            out["attention_mask"] = [[1] * len(ids) for ids in out["input_ids"]]
            return out

    transform = build_tokenizing_transform(_WithMask(), "text", 8, True)
    out = transform({"text": ["abc", "de"]})
    assert sorted(out) == ["attention_mask", "input_ids"]


def test_the_view_is_immutable_and_leaves_the_original_alone():
    """`with_transform`, never `set_transform`: the caller's object is also held
    by the preview and the row-count checks."""
    dataset = _text_dataset(32)
    view = attach_online_tokenization(
        dataset,
        tokenizer = _Tokenizer(),
        text_field = "text",
        max_length = 8,
        add_special_tokens = True,
    )
    assert view is not dataset
    assert "input_ids" in view[0]
    assert "input_ids" not in dataset[0]
    assert dataset[0]["text"] == "row 0"


def test_the_view_yields_the_same_row_count_and_order():
    dataset = _text_dataset(32)
    view = attach_online_tokenization(
        dataset,
        tokenizer = _Tokenizer(),
        text_field = "text",
        max_length = 8,
        add_special_tokens = True,
    )
    assert len(view) == len(dataset)
    assert view[5]["input_ids"] == _Tokenizer()(["row 5"], max_length = 8)["input_ids"][0]


def test_the_view_attests_its_truncation_width():
    """unsloth's `max_length` enforcement reads this instead of scanning every
    row -- and scanning a lazy split is the eager tokenize pass all over again."""
    view = attach_online_tokenization(
        _text_dataset(32),
        tokenizer = _Tokenizer(),
        text_field = "text",
        max_length = 1234,
        add_special_tokens = True,
    )
    assert view.__dict__[TRUNCATION_ATTESTATION_ATTR] == 1234


def test_the_transformed_view_still_reports_its_backing_columns():
    """Pinned because two consumers depend on it: `Trainer._remove_unused_columns`
    (hence `remove_unused_columns = False`) and unsloth's tokenized-split probe,
    which is why that probe reads a row rather than the metadata."""
    view = attach_online_tokenization(
        _text_dataset(32),
        tokenizer = _Tokenizer(),
        text_field = "text",
        max_length = 8,
        add_special_tokens = True,
    )
    assert "text" in dataset_column_names(view)
    assert "input_ids" not in dataset_column_names(view)


# ------------------------------------------------------- the double-BOS rule


def test_add_special_tokens_is_off_when_the_template_emits_a_bos():
    tokenizer = SimpleNamespace(bos_token = "<s>", chat_template = "<s>{{ x }}")
    assert resolve_add_special_tokens(tokenizer, "hello") is False


def test_add_special_tokens_is_off_when_the_text_already_starts_with_bos():
    tokenizer = SimpleNamespace(bos_token = "<s>", chat_template = "{{ x }}")
    assert resolve_add_special_tokens(tokenizer, "<s>hello") is False


def test_add_special_tokens_stays_on_otherwise():
    tokenizer = SimpleNamespace(bos_token = "<s>", chat_template = "{{ x }}")
    assert resolve_add_special_tokens(tokenizer, "hello") is True


def test_no_bos_token_means_add_special_tokens_stays_on():
    tokenizer = SimpleNamespace(bos_token = None, chat_template = "")
    assert resolve_add_special_tokens(tokenizer, "hello") is True


# ----------------------------------------------------------------- small helpers


@pytest.mark.parametrize(
    "grad_accum, workers, prefetch, expected",
    [(4, 4, 4, 16), (32, 2, 2, 32), (1, 0, 0, 1), (0, 0, 0, 1)],
)
def test_prewarm_depth_covers_the_first_step_and_the_queue(grad_accum, workers, prefetch, expected):
    assert prewarm_batch_count(grad_accum, workers, prefetch) == expected


def test_is_processor_spots_a_wrapped_tokenizer():
    assert is_processor(_Processor()) is True
    assert is_processor(_Tokenizer()) is False


def test_dataset_supports_with_transform_rejects_none_and_streams():
    assert dataset_supports_with_transform(None) is False
    assert dataset_supports_with_transform(_text_dataset(4)) is True


def test_a_disabled_decision_never_carries_worker_settings():
    decision = OnlineTokenizationDecision(enabled = False, reason = "test")
    assert decision.workers == 0
    assert decision.prewarm_batches == 0
    assert "off" in decision.as_log_line()
