# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import copy
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest
from datasets import Dataset, IterableDataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from core.training.dataset_audit import audit_dataset
from core.training.token_preview_worker import _preview


@pytest.fixture
def needs_trl():
    pytest.importorskip("trl", reason = "Dataset audit collation requires TRL")


@pytest.fixture
def tokenizer():
    special = ["<unk>", "<bos>", "<eos>", "<user>", "<assistant>", "<end>"]
    engine = Tokenizer(models.BPE(unk_token = "<unk>"))
    engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    engine.decoder = decoders.ByteLevel()
    engine.train_from_iterator(
        ["hello answer long tail", "### Instruction:\nhello\n### Input:\n\n### Response:\nanswer"],
        trainers.BpeTrainer(
            vocab_size = 280,
            special_tokens = special,
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
            show_progress = False,
        ),
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object = engine,
        unk_token = "<unk>",
        bos_token = "<bos>",
        eos_token = "<eos>",
        pad_token = "<eos>",
        additional_special_tokens = special[3:],
    )
    tok.chat_template = '{{ bos_token }}{% for message in messages %}{{ "<" + message["role"] + "> " + message["content"] + " <end> " }}{% endfor %}'
    tok._unsloth_input_part = "<user>"
    tok._unsloth_output_part = "<assistant>"
    return tok


def config(**kwargs):
    return {
        "model_name": "local-review-tokenizer",
        "format_type": "auto",
        "max_seq_length": 128,
        "train_on_completions": True,
        **kwargs,
    }


def chat(user = "hello", assistant = "answer"):
    return {
        "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    }


@pytest.mark.usefixtures("needs_trl")
def test_full_scan_finds_a_response_lost_beyond_the_preview_pool(tokenizer):
    dataset = Dataset.from_list([chat() for _ in range(100)] + [chat("long " * 100)])
    cfg = config(max_seq_length = 32)
    prefix = audit_dataset(dataset, tokenizer, cfg, max_rows = 100)
    full = audit_dataset(dataset, tokenizer, cfg, max_rows = None)
    assert prefix["coverage"] == {
        "rows_scanned": 100,
        "total_rows": 101,
        "complete": False,
        "mode": "prefix",
        "row_limit": 100,
    }
    assert prefix["status"] == "ok"
    assert full["coverage"]["complete"]
    assert full["rows_losing_all_loss_to_truncation"] == 1
    assert full["rows_without_loss_before_truncation"] == 0
    assert full["rows_without_loss_after_truncation"] == 1
    assert full["examples"][0]["prepared_row"] == 101
    assert full["status"] == "issues_found"


@pytest.mark.usefixtures("needs_trl")
def test_missing_response_is_distinct_from_truncated_response(tokenizer):
    dataset = Dataset.from_list([chat(), {"messages": [{"role": "user", "content": "hello"}]}])
    result = audit_dataset(dataset, tokenizer, config(), max_rows = None)
    assert result["rows_without_loss_before_truncation"] == 1
    assert result["rows_without_loss_after_truncation"] == 1
    assert result["rows_losing_all_loss_to_truncation"] == 0


@pytest.mark.parametrize("format_name", ["chatml", "sharegpt", "alpaca"])
@pytest.mark.usefixtures("needs_trl")
def test_tokens_and_labels_agree_with_existing_preview(tokenizer, format_name):
    rows = {
        "chatml": [chat()],
        "sharegpt": [
            {
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "answer"},
                ]
            }
        ],
        "alpaca": [{"instruction": "hello", "input": "", "output": "answer"}],
    }[format_name]
    cfg = config(max_seq_length = 512)
    report = audit_dataset(Dataset.from_list(rows), tokenizer, cfg, max_rows = None)
    preview = _preview(copy.deepcopy(tokenizer), cfg, {**cfg, "rows": rows})["samples"][0]
    assert report["tokens_after_truncation"] == preview["num_tokens"]
    assert report["loss_targets_after_truncation"] == sum(
        trained for _, _, trained in preview["tokens"][1:]
    )
    assert report["loss_targets_after_truncation"] > 0
    assert report["template"]["text"] and report["template"]["sha256"]


@pytest.mark.parametrize("format_type", ["raw", "auto"])
@pytest.mark.usefixtures("needs_trl")
def test_raw_text_retains_dropped_count_and_ignores_completion_masking(tokenizer, format_type):
    # Auto detection uses the first ten rows, just as training does.
    dataset = Dataset.from_list([{"text": "hello"}] * 10 + [{"text": " "}, {"text": None}])
    report = audit_dataset(dataset, tokenizer, config(format_type = format_type), max_rows = None)
    expected = tokenizer("hello" + (tokenizer.eos_token if format_type == "raw" else ""))[
        "input_ids"
    ]
    assert report["rows_prepared"] == 10
    assert report["rows_dropped_by_formatter"] == 2
    assert report["tokens_after_truncation"] == 10 * len(expected)
    assert report["loss_targets_after_truncation"] == 10 * (len(expected) - 1)
    assert not report["masking"]["applied"]
    assert report["template"]["text"] is None
    assert any("dropped 2" in warning for warning in report["warnings"])


@pytest.mark.usefixtures("needs_trl")
def test_stream_scan_stops_without_consuming_a_lookahead(tokenizer):
    consumed = []

    def rows():
        for i in range(1000000):
            consumed.append(i)
            yield {"text": "hello"}

    report = audit_dataset(
        IterableDataset.from_generator(rows), tokenizer, config(format_type = "raw"), max_rows = 5
    )
    assert consumed == list(range(5))
    assert report["coverage"]["rows_scanned"] == 5
    assert report["coverage"]["total_rows"] is None
    assert not report["coverage"]["complete"]


def test_full_stream_is_rejected_before_iteration(tokenizer):
    def rows():
        pytest.fail("full must refuse an unbounded stream without consuming it")
        yield {}

    with pytest.raises(ValueError, match = "streaming dataset requires a row limit"):
        audit_dataset(IterableDataset.from_generator(rows), tokenizer, config(), max_rows = None)


@pytest.mark.usefixtures("needs_trl")
def test_pretokenized_labels_and_causal_shift_are_authoritative(tokenizer):
    dataset = Dataset.from_list(
        [
            {"input_ids": [1, 10, 11, 2], "labels": [-100, -100, 11, 2]},
            {"input_ids": [1, 12, 2], "labels": [1, -100, -100]},
        ]
    )
    report = audit_dataset(dataset, tokenizer, config(), max_rows = None)
    assert report["format"] == "pretokenized"
    assert not report["masking"]["applied"]
    assert report["loss_targets_after_truncation"] == 2
    assert report["rows_without_loss_after_truncation"] == 1


@pytest.mark.usefixtures("needs_trl")
def test_pretokenized_padding_and_assistant_mask_match_trl_collation(tokenizer):
    from core.training.dataset_audit import _label_rows
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from utils.datasets.pretokenized import prepare_pretokenized_dataset

    dataset = Dataset.from_list(
        [
            {
                "input_ids": [2, 1, 10, 11, 2],
                "attention_mask": [0, 1, 1, 1, 0],
                "assistant_masks": [0, 0, 0, 1, 0],
            }
        ]
    )
    cfg = config(train_on_completions = False)
    prepared = prepare_pretokenized_dataset(dataset, max_seq_length = cfg["max_seq_length"])
    batch = DataCollatorForLanguageModeling(pad_token_id = tokenizer.pad_token_id)(prepared.to_list())
    expected = int((batch["labels"][:, 1:] != -100).sum())
    labels, _ = _label_rows(tokenizer, cfg["model_name"], prepared, "pretokenized", False)
    assert expected == sum(label != -100 for label in labels[0][1:]) == 1


@pytest.mark.usefixtures("needs_trl")
def test_audit_keeps_training_format_detection_even_for_token_named_columns(tokenizer):
    from utils.datasets import format_and_template_dataset

    dataset = Dataset.from_list([{"input_ids": [1, 10, 11], "assistant_masks": [0, 0, 1]}])
    cfg = config(train_on_completions = False)
    training = format_and_template_dataset(
        dataset, tokenizer = copy.deepcopy(tokenizer), model_name = cfg["model_name"], num_proc = 1
    )
    report = audit_dataset(dataset, tokenizer, cfg, max_rows = None)
    assert report["format"] == training["final_format"]


@pytest.mark.usefixtures("needs_trl")
def test_pretokenized_padding_does_not_count_as_truncation(tokenizer):
    dataset = Dataset.from_list(
        [{"input_ids": [2, 1, 10, 11, 2], "attention_mask": [0, 1, 1, 1, 0]}]
    )
    report = audit_dataset(dataset, tokenizer, config(train_on_completions = False), max_rows = None)
    assert report["format"] == "pretokenized"
    assert report["tokens_before_truncation"] == 3
    assert report["rows_over_context"] == 0
    assert report["loss_targets_after_truncation"] == 2


@pytest.mark.parametrize(
    "kind",
    [
        "raw",
        "auto",
        "chat",
        "chat_without_template",
        "alpaca",
        "ids",
        "labels",
        "prompt",
        "prompt_chat",
    ],
)
@pytest.mark.usefixtures("needs_trl")
def test_counts_match_zoo_training_preparation_and_collation(tokenizer, kind):
    from types import SimpleNamespace
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from unsloth_zoo.dataset_utils import sft_prepare_dataset
    from utils.datasets import format_and_template_dataset
    from utils.datasets.chat_templates import get_training_chat_template
    from utils.datasets.pretokenized import prepare_pretokenized_dataset
    from utils.datasets.raw_text import prepare_raw_text_dataset

    if kind == "chat_without_template":
        tokenizer.chat_template = None
    rows = {
        "raw": [{"text": "hello"}],
        "auto": [{"text": "hello"}],
        "chat": [chat()],
        "chat_without_template": [chat()],
        "prompt": [{"prompt": "hello", "completion": "answer"}],
        "prompt_chat": [
            {
                "prompt": [{"role": "user", "content": "hello"}],
                "completion": [{"role": "assistant", "content": "answer"}],
            }
        ],
        "alpaca": [{"instruction": "hello", "input": "", "output": "answer"}],
        "ids": [{"input_ids": [1, 10, tokenizer.eos_token_id]}],
        "labels": [
            {
                "input_ids": [1, 10, tokenizer.eos_token_id],
                "labels": [-100, 10, tokenizer.eos_token_id],
            }
        ],
    }[kind]
    data = Dataset.from_list(rows)
    cfg = config(format_type = "raw" if kind == "raw" else "auto", train_on_completions = False)
    report = audit_dataset(data, tokenizer, cfg, max_rows = None)
    training_tokenizer = copy.deepcopy(tokenizer)
    if kind == "raw":
        prepared = prepare_raw_text_dataset(
            data, eos_token = tokenizer.eos_token, append_eos = True
        ).dataset
        final_format = "raw_text"
    else:
        info = format_and_template_dataset(
            data, tokenizer = training_tokenizer, model_name = cfg["model_name"], num_proc = 1
        )
        prepared, final_format = info["dataset"], info["final_format"]
    training_tokenizer = get_training_chat_template(
        training_tokenizer, cfg["model_name"], final_format
    )
    if final_format == "pretokenized":
        prepared = prepare_pretokenized_dataset(prepared, max_seq_length = cfg["max_seq_length"])
    trainer = SimpleNamespace(
        data_collator = DataCollatorForLanguageModeling(pad_token_id = tokenizer.pad_token_id)
    )
    args = SimpleNamespace(
        max_length = cfg["max_seq_length"], dataset_text_field = "text", dataset_num_proc = 1
    )
    prepared = sft_prepare_dataset(
        trainer, prepared, training_tokenizer, args, False, None, "train"
    )
    batch = trainer.data_collator(prepared.to_list())
    assert report["tokens_after_truncation"] == sum(len(row) for row in prepared["input_ids"])
    assert report["loss_targets_after_truncation"] == int((batch["labels"][:, 1:] != -100).sum())
    if kind == "chat_without_template":
        assert report["template"]["text"] == training_tokenizer.chat_template
        assert report["template"]["sha256"]


@pytest.mark.parametrize("limit", [1, 2])
@pytest.mark.usefixtures("needs_trl")
def test_pretokenized_truncation_recomputes_loss_targets(tokenizer, limit):
    dataset = Dataset.from_list([{"input_ids": [1, 10, 11], "labels": [-100, -100, 11]}])
    report = audit_dataset(dataset, tokenizer, config(max_seq_length = limit), max_rows = None)
    assert report["rows_losing_all_loss_to_truncation"] == 1


@pytest.mark.parametrize("limit", [2, 128])
@pytest.mark.usefixtures("needs_trl")
def test_retained_chat_tokens_keep_labels_and_training_truncation_policy(tokenizer, limit):
    row = chat()
    row["input_ids"] = tokenizer.apply_chat_template(row["messages"], return_dict = False)[:3]
    row["labels"] = [-100] * len(row["input_ids"])
    report = audit_dataset(
        Dataset.from_list([row]),
        tokenizer,
        config(max_seq_length = limit, train_on_completions = False),
    )
    assert report["format"] == "chatml_messages"
    assert report["tokens_before_truncation"] == report["tokens_after_truncation"] == 3
    assert report["rows_without_loss_after_truncation"] == 1
    assert report["status"] == "issues_found"


@pytest.mark.usefixtures("needs_trl")
def test_mixed_chat_labels_follow_training_response_masking(tokenizer):
    from unsloth_zoo.dataset_utils import train_on_responses_only

    row = chat()
    row["input_ids"] = tokenizer.apply_chat_template(row["messages"], return_dict = False)
    row["labels"] = row["input_ids"].copy()
    row["labels"][-2] = -100
    masker = train_on_responses_only(
        None,
        tokenizer = tokenizer,
        return_function = True,
    )
    expected = masker({"input_ids": [row["input_ids"]], "labels": [row["labels"]]})["labels"][0]
    report = audit_dataset(Dataset.from_list([row]), tokenizer, config())
    assert report["loss_targets_after_truncation"] == sum(value != -100 for value in expected[1:])
    assert report["masking"]["applied"]


@pytest.mark.usefixtures("needs_trl")
def test_audit_loads_the_training_tokenizer_before_counting(tokenizer, tmp_path):
    from core.training.dataset_audit import _load_training_tokenizer
    from unsloth.tokenizer_utils import load_correct_tokenizer
    from transformers import DataCollatorForLanguageModeling

    tokenizer.add_special_tokens({"additional_special_tokens": ["<pad>"]})
    tokenizer.save_pretrained(tmp_path)
    cfg = config(model_name = str(tmp_path), train_on_completions = False)
    audited = _load_training_tokenizer(cfg)
    trained = load_correct_tokenizer(str(tmp_path), model_max_length = 128)
    assert audited.pad_token == trained.pad_token == "<pad>"
    rows = [{"input_ids": [tokenizer.bos_token_id, tokenizer.eos_token_id]}]
    expected = DataCollatorForLanguageModeling(trained, mlm = False)(rows)["labels"]
    report = audit_dataset(Dataset.from_list(rows), audited, cfg)
    assert report["loss_targets_after_truncation"] == int((expected[:, 1:] != -100).sum()) == 1
    assert report["status"] == "ok"


@pytest.mark.parametrize("metadata", [{"filename": "train.py"}, {"speech": "transcript"}])
@pytest.mark.usefixtures("needs_trl")
def test_text_metadata_is_not_rejected_as_media(tokenizer, metadata):
    report = audit_dataset(
        Dataset.from_list([{**metadata, "text": "hello"}]), tokenizer, config(format_type = "raw")
    )
    assert report["format"] == "raw_text"
    assert report["status"] == "ok"


def test_actual_media_is_still_refused(tokenizer):
    with pytest.raises(ValueError, match = "text datasets only"):
        audit_dataset(
            Dataset.from_list([{"image": "sample.png", "text": "hello"}]),
            tokenizer,
            config(format_type = "raw"),
        )


@pytest.mark.usefixtures("needs_trl")
def test_truncation_removes_the_entire_completion_even_without_response_masking(tokenizer):
    report = audit_dataset(
        Dataset.from_list([{"prompt": "hello " * 30, "completion": "answer"}]),
        tokenizer,
        config(train_on_completions = False, max_seq_length = 8),
    )
    assert report["loss_targets_before_truncation"] > 0
    assert report["loss_targets_after_truncation"] == 0
    assert report["rows_losing_all_loss_to_truncation"] == 1
    assert report["status"] == "issues_found"
    assert report["tokenization"]["source"] == "prompt_completion"
    assert report["template"]["source"] == "none"


@pytest.mark.parametrize("include_drop", [False, True])
@pytest.mark.usefixtures("needs_trl")
def test_row_expansion_cannot_invent_or_cancel_formatter_drops(tokenizer, include_drop):
    rows = [{"messages": chat()["messages"] * 2}]
    if include_drop:
        rows.append({"messages": [{"role": "user", "content": "hello"}]})
    report = audit_dataset(
        Dataset.from_list(rows),
        tokenizer,
        config(format_type = "alpaca", max_seq_length = 512, train_on_completions = False),
    )
    assert report["rows_prepared"] == 2
    assert report["rows_expanded_by_formatter"] == 1
    assert report["rows_dropped_by_formatter"] == int(include_drop)
    assert report["source_rows_dropped_during_conversion"] == int(include_drop)
    assert report["status"] == ("issues_found" if include_drop else "ok")


def test_packing_is_not_given_unpacked_loss_statistics(tokenizer):
    with pytest.raises(ValueError, match = "packing=False"):
        audit_dataset(Dataset.from_list([chat()]), tokenizer, config(packing = True))


@pytest.mark.usefixtures("needs_trl")
def test_batch_size_does_not_change_effective_template_or_counts(tokenizer):
    dataset = Dataset.from_list([chat("hello " * i) for i in range(1, 10)])
    assert audit_dataset(dataset, tokenizer, config(), batch_size = 1) == audit_dataset(
        dataset, tokenizer, config(), batch_size = 7
    )


@pytest.mark.parametrize("batch_size", [1, 64, 65])
@pytest.mark.parametrize("head_has_bos", [False, True])
@pytest.mark.usefixtures("needs_trl")
def test_prompt_completion_batches_keep_trainings_global_bos_policy(
    tokenizer, batch_size, head_has_bos
):
    from types import SimpleNamespace
    from tokenizers.processors import TemplateProcessing
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from unsloth_zoo.dataset_utils import sft_prepare_dataset

    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single = "<bos> $A", special_tokens = [("<bos>", tokenizer.bos_token_id)]
    )
    prompt = "hello " * 30
    limit = len(tokenizer(prompt)["input_ids"]) + 1
    first = (tokenizer.bos_token if head_has_bos else "") + prompt
    last = ("" if head_has_bos else tokenizer.bos_token) + prompt
    dataset = Dataset.from_list(
        [{"prompt": first, "completion": "answer"}] * 64
        + [{"prompt": last, "completion": "answer"}]
    )
    cfg = config(train_on_completions = False, max_seq_length = limit)
    report = audit_dataset(dataset, tokenizer, cfg, batch_size = batch_size)
    trainer = SimpleNamespace(
        data_collator = DataCollatorForLanguageModeling(
            pad_token_id = tokenizer.pad_token_id, completion_only_loss = True
        ),
        completion_only_loss = True,
    )
    expected = sft_prepare_dataset(
        trainer,
        dataset,
        tokenizer,
        SimpleNamespace(max_length = limit, dataset_num_proc = 1),
        False,
        None,
        "train",
    )
    targets = [
        int((trainer.data_collator([row])["labels"][:, 1:] != -100).sum()) for row in expected
    ]
    assert report["tokens_after_truncation"] == sum(map(len, expected["input_ids"]))
    assert report["loss_targets_after_truncation"] == sum(targets)
    assert report["rows_without_loss_after_truncation"] == targets.count(0)
    assert report["status"] == ("issues_found" if 0 in targets else "ok")
    if not head_has_bos:
        assert targets.count(0) == 1


@pytest.mark.parametrize(
    "case",
    [
        "full_scan",
        "retained_tokens",
        "pad_eos",
        "text_metadata",
        "prompt_completion",
        "expansion",
        "expansion_and_drop",
        "fallback_template",
        "bos_boundary",
    ],
)
@pytest.mark.usefixtures("needs_trl")
@pytest.mark.skipif(
    platform.system() == "Darwin" and platform.machine() == "arm64",
    reason = "Dataset audit CLI does not support MLX hosts",
)
def test_real_cli_full_audit_uses_only_tokenizer_files(tokenizer, tmp_path, case):
    model = tmp_path / "tokenizer"
    if case == "fallback_template":
        tokenizer.chat_template = None
    if case == "bos_boundary":
        from tokenizers.processors import TemplateProcessing
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single = "<bos> $A", special_tokens = [("<bos>", tokenizer.bos_token_id)]
        )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<pad>"]})
    tokenizer.save_pretrained(model)
    data = tmp_path / "train.jsonl"
    tokenized = chat()
    tokenized["input_ids"] = [tokenizer.bos_token_id, 10, 11]
    tokenized["labels"] = [-100] * len(tokenized["input_ids"])
    rows = {
        "full_scan": [chat()] * 100 + [chat("long " * 100)],
        "retained_tokens": [tokenized],
        "pad_eos": [{"input_ids": [tokenizer.bos_token_id, tokenizer.eos_token_id]}],
        "text_metadata": [{"filename": "train.py", "text": "hello"}],
        "prompt_completion": [{"prompt": "hello " * 30, "completion": "answer"}],
        "expansion": [{"messages": chat()["messages"] * 2}],
        "expansion_and_drop": [
            {"messages": chat()["messages"] * 2},
            {"messages": [{"role": "user", "content": "hello"}]},
        ],
        "fallback_template": [chat()],
        "bos_boundary": [{"prompt": "hello " * 30, "completion": "answer"}] * 64
        + [{"prompt": tokenizer.bos_token + "hello " * 30, "completion": "answer"}],
    }[case]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import atexit, json, os, sys\n"
        "from pathlib import Path\n"
        "attempts = []\n"
        "def audit(event, args):\n"
        "    if event == 'open' and isinstance(args[0], str) and (args[0].endswith('.safetensors') or args[0].endswith('pytorch_model.bin')):\n"
        "        attempts.append(args[0])\n"
        "        raise AssertionError('audit must not read model weights')\n"
        "sys.addaudithook(audit)\n"
        "def finish():\n"
        "    torch = sys.modules.get('torch')\n"
        "    result = {'weights': attempts, 'cuda': bool(torch and torch.cuda.is_initialized())}\n"
        "    Path(os.environ['AUDIT_TEST_GUARD_DIR'], str(os.getpid()) + '.json').write_text(json.dumps(result))\n"
        "atexit.register(finish)\n"
    )
    root = Path(__file__).resolve().parents[3]
    env = {
        **os.environ,
        "AUDIT_TEST_GUARD_DIR": str(guard),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join(
            [str(guard), str(root), str(root / "studio/backend"), os.environ.get("PYTHONPATH", "")]
        ),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "unsloth_cli",
            "dataset",
            "check",
            str(data),
            "--model",
            str(model),
            "--train-on-completions" if case == "full_scan" else "--no-train-on-completions",
            "--format-type",
            "raw"
            if case == "text_metadata"
            else ("alpaca" if case.startswith("expansion") else "auto"),
            "--max-seq-length",
            str(len(tokenizer("hello " * 30)["input_ids"]) + 1) if case == "bos_boundary" else "32",
            "--full",
            "--json",
        ],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        env = env,
        timeout = 60,
    )
    assert result.returncode == (
        1
        if case
        in (
            "full_scan",
            "retained_tokens",
            "prompt_completion",
            "expansion_and_drop",
            "bos_boundary",
        )
        else 0
    ), (
        result.stdout,
        result.stderr,
    )
    report = json.loads(result.stdout)
    assert report["coverage"]["complete"]
    if case == "full_scan":
        assert report["rows_losing_all_loss_to_truncation"] == 1
    elif case == "retained_tokens":
        assert report["loss_targets_after_truncation"] == 0
    elif case == "pad_eos":
        assert report["loss_targets_after_truncation"] == 1
    elif case == "prompt_completion":
        assert report["rows_losing_all_loss_to_truncation"] == 1
    elif case.startswith("expansion"):
        assert report["rows_dropped_by_formatter"] == int(case == "expansion_and_drop")
    elif case == "fallback_template":
        assert report["template"]["text"] and report["template"]["sha256"]
    elif case == "bos_boundary":
        assert report["loss_targets_after_truncation"] == 64
        assert report["rows_without_loss_after_truncation"] == 1
    receipts = [json.loads(path.read_text()) for path in guard.glob("*.json")]
    assert len(receipts) >= 2  # CLI parent and tokenizer worker.
    assert all(not receipt["weights"] and not receipt["cuda"] for receipt in receipts)
