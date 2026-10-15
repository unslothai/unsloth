# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Token preview subprocess: loads only a model's tokenizer and shows how training would tokenize and
mask a few dataset rows or a typed text.

Runs the same formatting, chat template and completion masking calls as the training worker, so
what the preview marks as trained is what the run trains on. It never touches a GPU: the parent
starts it with every accelerator hidden, and it exits on its own once idle.
"""

from __future__ import annotations

import copy
import os
import queue
import sys
import traceback
import types
from pathlib import Path
from typing import Any

# Long enough to type between edits, short enough that an abandoned dialog does not keep torch in memory.
IDLE_TIMEOUT_S = 300
# A token list this long already fills the dialog; the counts below still cover the whole row.
MAX_TOKENS_SHOWN = 4096

_EXAMPLE_CONVERSATION = [
    {"role": "user", "content": "What is the capital of Hungary?"},
    {"role": "assistant", "content": "The capital of Hungary is Budapest."},
]


def _load_tokenizer(config: dict):
    from transformers import AutoTokenizer

    source = config.get("tokenizer_source") or config["model_name"]
    token = config.get("hf_token")
    try:
        return AutoTokenizer.from_pretrained(source, token = token, trust_remote_code = False)
    except Exception:
        pass
    try:
        # Spark-TTS keeps its tokenizer under LLM/, as _spark_tts_tokenizer_kwargs reads it for training.
        return AutoTokenizer.from_pretrained(
            source, token = token, trust_remote_code = False, subfolder = "LLM"
        )
    except Exception:
        # A LoRA adapter saved without tokenizer files: read the base it names, as training does.
        from utils.models.model_config import get_base_model_from_lora_identifier

        base = get_base_model_from_lora_identifier(source, token)
        if not base:
            raise
        return AutoTokenizer.from_pretrained(base, token = token, trust_remote_code = False)


def _byte_level_decoder() -> dict:
    """GPT-2's byte-to-unicode table, inverted: byte-level BPE vocabularies spell every byte as one of these."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    printable += list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = {}, 0
    for byte in range(256):
        if byte in printable:
            chars[chr(byte)] = byte
        else:
            chars[chr(256 + extra)] = byte
            extra += 1
    return chars


_BYTE_LEVEL = _byte_level_decoder()


def _token_text(tokenizer, token_id: int, raw: str) -> str:
    """The token as text. A piece holding part of a multi-byte character decodes to U+FFFD on its
    own, so its bytes are spelled out instead: Llama's "ű" after a space is "Ġ" + 0xC5, then 0xB1."""
    text = tokenizer.decode([token_id])
    # Legacy SentencePiece decoding (Mistral, Llama 2) drops the word-initial space of a lone "▁" piece.
    if raw.startswith("▁") and not text.startswith(" "):
        text = " " + text
    if "\ufffd" not in text or not raw:
        return text
    if all(char in _BYTE_LEVEL for char in raw):
        data = bytes(_BYTE_LEVEL[char] for char in raw)
        return "".join(
            char if char.isprintable() or char in " \n\t" else f"<0x{ord(char) - 0xDC00:02X}>"
            for char in data.decode("utf-8", errors = "surrogateescape")
        )
    return raw


def _render_tokens(tokenizer, input_ids: list, labels: list | None) -> dict:
    shown = input_ids[:MAX_TOKENS_SHOWN]
    raw_pieces = tokenizer.convert_ids_to_tokens(shown)
    tokens = []
    for index, token_id in enumerate(shown):
        trained = labels is None or labels[index] != -100
        tokens.append(
            [int(token_id), _token_text(tokenizer, token_id, raw_pieces[index] or ""), trained]
        )
    trained_count = (
        len(input_ids) if labels is None else sum(1 for label in labels if label != -100)
    )
    return {
        "tokens": tokens,
        "input_ids": [int(token_id) for token_id in input_ids],
        "num_tokens": len(input_ids),
        "num_trained": trained_count,
        "num_hidden": max(0, len(input_ids) - len(shown)),
    }


def _mask(tokenizer, model_name: str, encoded: dict, final_format: str) -> tuple[list, dict]:
    """Labels for each row via apply_completion_masking, the call the training worker makes."""
    from datasets import Dataset
    from unsloth.chat_templates import train_on_responses_only
    from utils.datasets.completion_masking import apply_completion_masking

    markers: dict[str, Any] = {}
    notes: list[str] = []

    def preview_train_fn(
        trainer,
        instruction_part = None,
        response_part = None,
        **_ignored,
    ):
        inner = getattr(trainer.processing_class, "tokenizer", trainer.processing_class)
        markers["instruction_part"] = instruction_part or getattr(
            inner, "_unsloth_input_part", None
        )
        markers["response_part"] = response_part or getattr(inner, "_unsloth_output_part", None)
        label_fn = train_on_responses_only(
            None,
            instruction_part = instruction_part,
            response_part = response_part,
            tokenizer = trainer.processing_class,
            return_function = True,
        )
        trainer.train_dataset = trainer.train_dataset.map(label_fn, batched = True)
        return trainer

    trainer = types.SimpleNamespace(
        processing_class = tokenizer,
        train_dataset = Dataset.from_dict({"input_ids": encoded["input_ids"]}),
        eval_dataset = None,
    )
    trainer, applied = apply_completion_masking(
        trainer,
        model_name,
        preview_train_fn,
        notify = lambda level, message: notes.append(message),
        dataset_template = "alpaca" if final_format == "alpaca" else None,
    )
    masking = {"applied": bool(applied), "notes": notes, **markers}
    if not applied:
        return [None] * len(encoded["input_ids"]), masking
    return list(trainer.train_dataset["labels"]), masking


def _format_rows(tokenizer, config: dict, rows: list) -> tuple[list, str]:
    from datasets import Dataset
    from utils.datasets import format_and_template_dataset
    from utils.datasets.raw_text import prepare_raw_text_dataset

    dataset = Dataset.from_list(rows)
    if config.get("raw_text"):
        result = prepare_raw_text_dataset(
            dataset,
            mode_label = "CPT" if config.get("is_cpt") else "raw text",
            split_name = "train",
            eos_token = getattr(tokenizer, "eos_token", None),
            append_eos = True,
        )
        return list(result.dataset["text"]), "raw_text"

    info = format_and_template_dataset(
        dataset,
        model_name = config["model_name"],
        tokenizer = tokenizer,
        is_vlm = False,
        format_type = config.get("format_type") or "auto",
        dataset_name = config.get("dataset_name"),
        custom_format_mapping = config.get("custom_format_mapping"),
        # In-process: a pool for three rows costs seconds and changes nothing in the output.
        num_proc = 1,
    )
    if not info.get("success", True):
        raise ValueError("; ".join(info.get("errors") or []) or "Dataset formatting failed")
    return list(info["dataset"]["text"]), str(info.get("final_format", "")).lower()


def _sample(texts: list, count: int, seed: int) -> list:
    import random
    if len(texts) <= count:
        return texts
    return random.Random(seed).sample(texts, count)


def _preview(tokenizer, config: dict, request: dict) -> dict:
    from utils.datasets.chat_templates import get_training_chat_template
    from utils.datasets.online_tokenization import resolve_add_special_tokens

    model_name = config["model_name"]
    if request.get("text") and request.get("raw_text"):
        texts, final_format = _format_rows(tokenizer, request, [{"text": request["text"]}])
        first_text = texts[0]
    elif request.get("text") is not None:
        texts, final_format = [request["text"]], str(request.get("final_format") or "").lower()
        first_text = texts[0]
    else:
        # Formatting may set a template on the tokenizer it is given; training formats with a throwaway
        # tokenizer too, and masks with the one the model load returns. The whole pool is formatted so a
        # raw-text column and the BOS rule are chosen from the dataset's head, as training does.
        texts, final_format = _format_rows(
            copy.deepcopy(tokenizer), {**config, **request}, request["rows"]
        )
        first_text = texts[0] if texts else None
        texts = _sample(texts, int(request.get("num_samples") or 3), request.get("seed") or 0)
    if not getattr(tokenizer, "chat_template", None):
        tokenizer = copy.deepcopy(tokenizer)
    tokenizer = get_training_chat_template(tokenizer, model_name, final_format)

    max_length = int(request.get("max_seq_length") or 2048)
    add_special_tokens = resolve_add_special_tokens(tokenizer, first_text)
    encoded = tokenizer(
        texts, truncation = True, max_length = max_length, add_special_tokens = add_special_tokens
    )
    full_lengths = [
        len(ids) for ids in tokenizer(texts, add_special_tokens = add_special_tokens)["input_ids"]
    ]

    masking = {"applied": False, "notes": []}
    labels = [None] * len(texts)
    if request.get("train_on_completions") and final_format != "raw_text" and any(texts):
        labels, masking = _mask(tokenizer, model_name, dict(encoded), final_format)

    samples = []
    for text, input_ids, row_labels, full_length in zip(
        texts, encoded["input_ids"], labels, full_lengths
    ):
        sample = _render_tokens(tokenizer, list(input_ids), row_labels)
        sample["text"] = text
        sample["truncated_from"] = full_length if full_length > len(input_ids) else None
        samples.append(sample)

    example_text = None
    if getattr(tokenizer, "chat_template", None):
        try:
            example_text = tokenizer.apply_chat_template(_EXAMPLE_CONVERSATION, tokenize = False)
        except Exception:
            example_text = None

    return {
        "samples": samples,
        "masking": masking,
        "final_format": final_format or None,
        "add_special_tokens": add_special_tokens,
        "vocab_size": len(tokenizer),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "example_text": example_text,
    }


def run_token_preview_process(*, cmd_queue: Any, resp_queue: Any, config: dict) -> None:
    """Subprocess entrypoint. ``config`` names the model; each command on ``cmd_queue`` is a preview
    request answered on ``resp_queue`` with the same ``id``."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONWARNINGS"] = "ignore"
    backend_path = str(Path(__file__).resolve().parent.parent.parent)
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from utils.security.hub_data_files import install as guard_hub_data_files

    guard_hub_data_files()

    tokenizer = None
    load_error = None
    try:
        from utils.transformers_version import activate_transformers_for_subprocess
        activate_transformers_for_subprocess(config["model_name"], config.get("hf_token"))
        tokenizer = _load_tokenizer(config)
    except Exception as exc:  # noqa: BLE001 - reported on every request instead
        load_error = f"Could not load the tokenizer for {config['model_name']}: {exc}"
    if tokenizer is not None:
        try:
            # Paid once here rather than on the first preview; both are needed by most requests.
            import unsloth.chat_templates  # noqa: F401
            import unsloth_zoo.dataset_utils  # noqa: F401
        except Exception:  # noqa: BLE001 - the request that needs them reports the failure
            pass

    while True:
        try:
            request = cmd_queue.get(timeout = IDLE_TIMEOUT_S)
        except queue.Empty:
            return
        if not isinstance(request, dict) or request.get("op") == "shutdown":
            return
        reply: dict[str, Any] = {"id": request.get("id")}
        if load_error is not None:
            # Fatal: the parent starts a fresh worker for the next request, so a retry can succeed.
            reply.update(ok = False, error = load_error, fatal = True)
            resp_queue.put(reply)
            return
        try:
            reply.update(ok = True, result = _preview(tokenizer, config, request))
        except Exception as exc:  # noqa: BLE001 - one bad request must not end the worker
            traceback.print_exc()
            reply.update(ok = False, error = str(exc) or type(exc).__name__)
        resp_queue.put(reply)
