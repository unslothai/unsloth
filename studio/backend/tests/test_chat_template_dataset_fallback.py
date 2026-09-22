# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dataset template fallback and consistency across splits (#11321)."""

from datasets import Dataset

from utils.datasets import apply_chat_template_to_dataset
from utils.datasets import chat_templates

OWN = "own-template"
OVERRIDE = "unsloth-override"


class _TemplatedTokenizer:
    """Only the checkpoint template supports tool turns."""

    eos_token = "</s>"

    def __init__(self):
        self.chat_template = OWN

    def apply_chat_template(self, conversation, **_kwargs):
        if any(turn["role"] == "broken" for turn in conversation):
            raise ValueError("Invalid content type")
        if self.chat_template == OVERRIDE and any(turn["role"] == "tool" for turn in conversation):
            raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
        # Expose the selected template in the output for assertions.
        turns = "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)
        return f"[{self.chat_template}] {turns}"


def _plain_convo(index):
    return [
        {"role": "user", "content": f"question {index}"},
        {"role": "assistant", "content": f"answer {index}"},
    ]


def _agentic_convo(index):
    return [
        {"role": "user", "content": f"question {index}"},
        {"role": "assistant", "content": "calling a tool"},
        {"role": "tool", "content": "tool result"},
        {"role": "assistant", "content": f"answer {index}"},
    ]


def _dataset_info(convo_factory, rows = 8):
    return {
        "dataset": Dataset.from_dict({"messages": [convo_factory(i) for i in range(rows)]}),
        "detected_format": "chatml_messages",
        "final_format": "chatml_messages",
        "chat_column": "messages",
        "is_standardized": True,
        "warnings": [],
    }


def _apply_override(tokenizer, _model_name):
    tokenizer.chat_template = OVERRIDE
    return tokenizer


def _format(dataset_info, tokenizer, monkeypatch):
    monkeypatch.setattr(chat_templates, "get_tokenizer_chat_template", _apply_override)
    return apply_chat_template_to_dataset(
        dataset_info,
        tokenizer,
        model_name = "unsloth/gemma-4-E4B-it",
        num_proc = 1,
    )


def test_agentic_rows_fall_back_to_the_checkpoints_own_template(monkeypatch):
    tokenizer = _TemplatedTokenizer()

    result = _format(_dataset_info(_agentic_convo), tokenizer, monkeypatch)

    assert result["success"] is True
    assert result["errors"] == []
    assert len(result["dataset"]) == 8
    assert all("tool: tool result" in text for text in result["dataset"]["text"])
    assert tokenizer.chat_template == OWN


def test_override_is_kept_when_it_renders_the_dataset(monkeypatch):
    tokenizer = _TemplatedTokenizer()

    result = _format(_dataset_info(_plain_convo), tokenizer, monkeypatch)

    assert result["success"] is True
    assert len(result["dataset"]) == 8
    assert tokenizer.chat_template == OVERRIDE


def test_override_is_kept_when_neither_template_renders(monkeypatch):
    tokenizer = _TemplatedTokenizer()
    broken = lambda index: [{"role": "broken", "content": f"row {index}"}]

    result = _format(_dataset_info(broken), tokenizer, monkeypatch)

    assert result["success"] is False
    assert "Chat template failed on all 8 rows" in result["errors"][0]
    assert tokenizer.chat_template == OVERRIDE


def test_only_the_rows_the_own_template_rejects_are_dropped(monkeypatch):
    tokenizer = _TemplatedTokenizer()
    mixed = lambda index: (
        [{"role": "broken", "content": f"row {index}"}]
        if index in (2, 5)
        else _agentic_convo(index)
    )

    result = _format(_dataset_info(mixed), tokenizer, monkeypatch)

    assert result["success"] is True
    assert len(result["dataset"]) == 6
    assert tokenizer.chat_template == OWN


def test_plain_rows_at_the_front_do_not_decide_it_for_the_agentic_rows_behind(monkeypatch):
    tokenizer = _TemplatedTokenizer()
    plain_then_agentic = lambda index: (_plain_convo(index) if index < 8 else _agentic_convo(index))

    result = _format(_dataset_info(plain_then_agentic, rows = 20), tokenizer, monkeypatch)

    assert result["success"] is True
    assert len(result["dataset"]) == 20
    assert tokenizer.chat_template == OWN


def test_a_small_dataset_probes_its_last_row(monkeypatch):
    tokenizer = _TemplatedTokenizer()
    agentic_tail = lambda index: (_agentic_convo(index) if index == 11 else _plain_convo(index))

    result = _format(_dataset_info(agentic_tail, rows = 12), tokenizer, monkeypatch)

    assert result["success"] is True
    assert len(result["dataset"]) == 12
    assert tokenizer.chat_template == OWN


def test_tool_rows_between_the_sampled_rows_are_still_dropped(monkeypatch):
    # Row 37 falls between sampled indices.
    tokenizer = _TemplatedTokenizer()
    one_agentic = lambda index: (_agentic_convo(index) if index == 37 else _plain_convo(index))

    result = _format(_dataset_info(one_agentic, rows = 40), tokenizer, monkeypatch)

    assert result["success"] is True
    assert len(result["dataset"]) == 39
    assert tokenizer.chat_template == OVERRIDE
    assert "Dropped 1 of 40 rows" in result["dropped_rows_warning"]


def test_a_separate_eval_split_renders_with_the_template_training_chose(monkeypatch):
    # Training, evaluation, and saving share one tokenizer.
    tokenizer = _TemplatedTokenizer()

    train = _format(_dataset_info(_agentic_convo), tokenizer, monkeypatch)
    evaluation = _format(_dataset_info(_plain_convo), tokenizer, monkeypatch)

    assert all(text.startswith(f"[{OWN}]") for text in train["dataset"]["text"])
    assert all(text.startswith(f"[{OWN}]") for text in evaluation["dataset"]["text"])
    assert tokenizer.chat_template == OWN


def test_an_eval_split_does_not_move_the_template_training_chose(monkeypatch):
    # Evaluation must keep training's choice even when its rows fail.
    tokenizer = _TemplatedTokenizer()

    train = _format(_dataset_info(_plain_convo), tokenizer, monkeypatch)
    evaluation = _format(_dataset_info(_agentic_convo), tokenizer, monkeypatch)

    assert all(text.startswith(f"[{OVERRIDE}]") for text in train["dataset"]["text"])
    assert evaluation["success"] is False
    assert "Chat template failed on all 8 rows" in evaluation["errors"][0]
    assert tokenizer.chat_template == OVERRIDE


def test_fallback_restores_the_template_on_a_processor_and_its_tokenizer():
    # Restore both copies, matching get_chat_template.
    class _Processor(_TemplatedTokenizer):
        def __init__(self):
            super().__init__()
            self.tokenizer = _TemplatedTokenizer()

    processor = _Processor()
    processor.chat_template = OVERRIDE
    processor.tokenizer.chat_template = OVERRIDE
    dataset = Dataset.from_dict({"messages": [_agentic_convo(i) for i in range(4)]})

    note = chat_templates.keep_renderable_chat_template(processor, dataset, "messages", OWN)

    assert note
    assert processor.chat_template == OWN
    assert processor.tokenizer.chat_template == OWN
