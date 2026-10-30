# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import jinja2
import pytest
from pydantic import ValidationError

from core.training import rl_format
from core.training.rl_format import (
    REASONING_START,
    SYSTEM_PROMPT,
    apply_reasoning_template,
    build_warmup_dataset,
)
from models.training import TrainingStartRequest


class FakeTokenizer:
    eos_token = "</s>"
    chat_template = "original"

    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = False,
    ):
        return jinja2.Template(self.chat_template).render(
            messages = messages, eos_token = self.eos_token, add_generation_prompt = add_generation_prompt
        )

    def __call__(
        self,
        text,
        add_special_tokens = False,
    ):
        return {"input_ids": text.split()}


def _tok():
    tok = FakeTokenizer()
    apply_reasoning_template(tok)
    return tok


def test_template_matches_the_notebook_layout():
    tok = _tok()
    text = tok.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}], add_generation_prompt = True
    )
    assert text == SYSTEM_PROMPT + "</s>" + "What is 2+2?" + REASONING_START


def test_a_row_system_prompt_replaces_the_default():
    tok = _tok()
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert text == "Be brief.</s>qa</s>"


def test_processor_and_inner_tokenizer_both_get_the_template():
    class Processor:
        chat_template = "x"

        def __init__(self):
            self.tokenizer = FakeTokenizer()

    proc = Processor()
    apply_reasoning_template(proc)
    assert proc.chat_template == proc.tokenizer.chat_template == rl_format.CHAT_TEMPLATE


ROWS = [
    {"problem": "1+1?", "generated_solution": "<think>add</think>", "expected_answer": "2"},
    {"problem": "name it", "generated_solution": "x", "expected_answer": "a cat"},
    {"problem": "big", "generated_solution": "word " * 500, "expected_answer": "3"},
    {"problem": "2+2?", "generated_solution": "four", "expected_answer": "4"},
]


def test_warmup_rows_are_numeric_short_and_formatted():
    ds = build_warmup_dataset(_tok(), max_seq_length = 400, rows = 10, dataset = ROWS)
    assert len(ds) == 2
    first = ds[0]["text"]
    assert "<think>" not in first
    assert first.endswith(f"1+1?{REASONING_START}add<end_working_out><SOLUTION>2</SOLUTION></s>")


def test_warmup_rows_are_capped():
    assert len(build_warmup_dataset(_tok(), 400, rows = 1, dataset = ROWS)) == 1


def test_warmup_answers_fit_the_grpo_completion_cap():
    # The notebook's cap is the context minus the prompt, which every warm-up answer fits; a
    # tighter cap keeps only the examples whose answer GRPO would not cut off.
    long_answer = {"problem": "q", "generated_solution": "step " * 60, "expected_answer": "5"}
    rows = [ROWS[0], long_answer, ROWS[3]]
    assert len(build_warmup_dataset(_tok(), 400, rows = 10, dataset = rows)) == 3
    kept = build_warmup_dataset(_tok(), 400, rows = 10, dataset = rows, max_completion_length = 20)
    assert len(kept) == 2 and all("step step" not in t for t in kept["text"])
    with pytest.raises(ValueError, match = "max completion length"):
        build_warmup_dataset(_tok(), 400, rows = 10, dataset = [long_answer], max_completion_length = 20)


def test_the_default_completion_cap_fits_every_warmup_example():
    from core.training.rl import rl_lengths
    for context in (2048, 4096, 8192):
        assert rl_lengths("grpo", context, {})[1] >= context // 2


@pytest.mark.parametrize(
    "kw, steps",
    [
        (dict(grpo_reasoning_format = True), 100),
        (dict(grpo_reasoning_format = True, max_seq_length = 1024), 0),
        (dict(), 0),
        (dict(grpo_reasoning_format = True, grpo_format_warmup_steps = 0), 0),
        (dict(grpo_reasoning_format = True, grpo_format_warmup_steps = 30), 30),
    ],
)
def test_the_reasoning_format_defaults_to_the_notebooks_warmup(kw, steps):
    assert _request(**kw).grpo_format_warmup_steps == steps


def test_nothing_fits_is_a_clear_error():
    with pytest.raises(ValueError, match = "max sequence length"):
        build_warmup_dataset(_tok(), max_seq_length = 4, rows = 10, dataset = ROWS)


def _request(**kw):
    base = dict(
        model_name = "unsloth/Qwen3-4B-Base",
        training_type = "LoRA/QLoRA",
        hf_dataset = "openai/gsm8k",
        format_type = "auto",
        objective = "grpo",
        grpo_rewards = [{"name": "r"}],
    )
    return TrainingStartRequest(**{**base, **kw})


def test_warmup_needs_the_reasoning_format():
    with pytest.raises(ValidationError, match = "reasoning format"):
        _request(grpo_format_warmup_steps = 50)
    ok = _request(grpo_format_warmup_steps = 50, grpo_reasoning_format = True, max_seq_length = 2048)
    assert ok.grpo_format_warmup_steps == 50


def test_warmup_needs_room_for_its_examples():
    with pytest.raises(ValidationError, match = "2048 or more"):
        _request(grpo_format_warmup_steps = 50, grpo_reasoning_format = True, max_seq_length = 1024)


def test_warmup_steps_are_bounded():
    with pytest.raises(ValidationError):
        _request(grpo_format_warmup_steps = 5000, grpo_reasoning_format = True)


def test_a_resumed_run_skips_the_warmup():
    settings = {"format_warmup_steps": 50}
    assert rl_format.warmup_steps_to_run(settings, "grpo", resuming = False) == 50
    assert rl_format.warmup_steps_to_run(settings, "grpo", resuming = True) == 0
    assert rl_format.warmup_steps_to_run(settings, "dpo", resuming = False) == 0


def test_the_reasoning_format_refuses_rows_with_tools():
    from datasets import Dataset

    from core.training.rl import reasoning_format_refusal

    user = {"role": "user", "content": "q"}
    plain = Dataset.from_list([{"prompt": [user]}])
    assert reasoning_format_refusal(plain) is None
    tool = {"type": "function", "function": {"name": "add"}}
    assert "tool" in reasoning_format_refusal(
        Dataset.from_list([{"prompt": [user], "tools": [tool]}])
    )
    turns = [user, {"role": "tool", "content": "4"}]
    assert "tool" in reasoning_format_refusal(Dataset.from_list([{"prompt": turns}]))


def test_resume_payload_keeps_the_reasoning_format_and_warmup():
    stored = {"reasoning_format": True, "format_warmup_steps": 50}
    req = _request(rl_settings = stored, max_seq_length = 2048)
    assert req.grpo_reasoning_format and req.grpo_format_warmup_steps == 50


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("add them<end_working_out><SOLUTION>1,234</SOLUTION>", [3.0, 3.0, 3.5]),
        ("add them<end_working_out><SOLUTION>12</SOLUTION>", [3.0, 0.0, -1.5]),
        ("no tags, 1234", [0.0, 0.0, -2.5]),
        ("add them<end_working_out><SOLUTION>1,234</SOLUTION>\n", [3.0, 3.0, 3.5]),
        ("add them<end_working_out><SOLUTION>1,234</SOLUTION>\nmore text", [0.0, 3.0, 3.5]),
    ],
)
def test_bundled_solution_rewards_read_the_reasoning_format(reply, expected):
    from core.training.rewards import get_reward, preview_scores

    specs = [get_reward(n) for n in ("solution-format", "solution-exact", "solution-numeric")]
    scores, _ = preview_scores(specs, reply, reference = "So the total is #### 1234")
    assert [s["score"] for s in scores] == expected


def test_warmup_reports_step_n_of_m_and_stops_on_cancel():
    from types import SimpleNamespace

    from core.training.rl_format import warmup_callback

    seen, stop = [], [False]
    callback = warmup_callback(20, lambda: stop[0], lambda step, steps: seen.append((step, steps)))
    control = SimpleNamespace(should_training_stop = False)
    callback.on_step_end(None, SimpleNamespace(global_step = 7), control)
    assert seen == [(7, 20)] and not control.should_training_stop
    stop[0] = True
    callback.on_step_end(None, SimpleNamespace(global_step = 8), control)
    assert seen[-1] == (8, 20) and control.should_training_stop


def test_the_run_shows_warmup_progress_as_step_n_of_m():
    import inspect

    from core.training import trainer

    source = inspect.getsource(trainer)
    assert 'status_message = f"Format warm-up: step {step} of {steps}..."' in source


def test_the_run_fits_the_warmup_to_its_grpo_completion_cap():
    import inspect

    from core.training import trainer

    source = inspect.getsource(trainer)
    call = source[source.index("run_format_warmup(\n") :]
    call = call[: call.index("\n                    )\n")]
    assert "max_completion_length = rl_lengths(" in call and "rl_settings" in call


def test_the_tool_refusal_and_the_offline_skip_come_before_the_warmup():
    # A refused dataset costs no SFT steps, and an offline run without the warm-up set still trains.
    import inspect

    from core.training import trainer

    src = inspect.getsource(trainer)
    refuse = src.index("refusal = reasoning_format_refusal(rl_train)")
    load = src.index("warmup_source = load_warmup_source()")
    run = src.index("run_format_warmup(\n")
    assert refuse < load < run
    assert "if not _env_offline():\n" in src[load:run]
    assert "self._record_warning(warmup_offline_warning())" in src[load:run]
    assert rl_format.WARMUP_DATASET in rl_format.warmup_offline_warning()


def test_the_warmup_reads_the_source_it_was_handed(monkeypatch):
    monkeypatch.setattr(rl_format, "load_warmup_source", lambda: pytest.fail("loaded again"))
    assert len(build_warmup_dataset(_tok(), 400, rows = 10, dataset = ROWS)) > 0
