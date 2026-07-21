# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""TEMPLATE_TO_RESPONSES_MAPPER markers must match what the templates render.

The manual instruction/response markers are the fallback for
train_on_completions when auto-detection is unavailable, so a marker that
never matches the rendered chat template masks every assistant token and the
run dies on the all-labels-masked safety net. Six template families shipped
such markers:

  mistral           - "[INST] " / " [/INST]": the surrounding spaces fold into
                      the neighbouring tokens ("[INST]" is a single special
                      token in Mistral v0.3), so the padded strings never match.
  llama             - same space folding, plus llama-2 tokenizes [INST] after
                      <s> as bare "[" on transformers 5.x while the standalone
                      encoding gives "▁[", so the marker must anchor on <s>.
  starling          - trailing space after "GPT4 Correct Assistant:" folds
                      into the next content token ("▁Hello").
  glm               - "[gMASK]<sop>" renders once at text start, never before
                      later user turns; "<think>" is generation scaffolding
                      that non-final turns render as a lone "</think>".
  qwen3-thinking    - "<think>" is stripped from non-final assistant turns
                      (Qwen3-Thinking-2507) or never rendered (QwQ).
  zephyr            - role tags are plain text, and SentencePiece tokenizes
                      "<|assistant|>" differently at text start than after
                      "</s>\\n" mid-conversation; the markers need the leading
                      newline anchor to tokenize like a real turn boundary.

Literal assertions run everywhere; the token-level masking checks need the
representative tokenizers plus unsloth_zoo and skip when either is
unavailable (offline CI).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# model_mappings is dependency-free: load it directly so these tests run
# without the studio venv / package import side effects.
_MM_PATH = Path(_BACKEND_DIR) / "utils" / "datasets" / "model_mappings.py"
_mm_spec = importlib.util.spec_from_file_location("_marker_test_mm", _MM_PATH)
model_mappings = importlib.util.module_from_spec(_mm_spec)
_mm_spec.loader.exec_module(model_mappings)

T2R = model_mappings.TEMPLATE_TO_RESPONSES_MAPPER


# ── Fixed entries: markers derived from what each representative tokenizer
#    actually renders (see PR for the token-level derivation). ──
EXPECTED_FIXED = {
    "mistral": {"instruction": "[INST]", "response": "[/INST]"},
    "llama": {"instruction": "<s>[INST]", "response": "[/INST]"},
    "starling": {
        "instruction": "GPT4 Correct User:",
        "response": "GPT4 Correct Assistant:",
    },
    "glm": {"instruction": "<|user|>", "response": "<|assistant|>"},
    "qwen3-thinking": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n",
    },
    "zephyr": {"instruction": "\n<|user|>\n", "response": "\n<|assistant|>\n"},
}

# Spot-pin some known-good entries so a refactor cannot silently change them.
EXPECTED_UNCHANGED = {
    "qwen3": {
        "instruction": "<|im_start|>user\n",
        "response": "<|im_start|>assistant\n",
    },
    "llama-3.1": {
        "instruction": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "phi-4": {
        "instruction": "<|im_start|>user<|im_sep|>",
        "response": "<|im_start|>assistant<|im_sep|>",
    },
    "gemma-3": {
        "instruction": "<start_of_turn>user\n",
        "response": "<start_of_turn>model\n",
    },
    "gpt-oss": {
        "instruction": "<|start|>user<|message|>",
        "response": "<|start|>assistant<|channel|>final<|message|>",
    },
}


@pytest.mark.parametrize("template", sorted(EXPECTED_FIXED))
def test_fixed_marker_literals(template):
    assert T2R[template] == EXPECTED_FIXED[template]


@pytest.mark.parametrize("template", sorted(EXPECTED_UNCHANGED))
def test_unchanged_marker_literals(template):
    assert T2R[template] == EXPECTED_UNCHANGED[template]


def test_no_marker_is_empty_or_whitespace():
    for template, parts in T2R.items():
        assert parts["instruction"].strip(), template
        assert parts["response"].strip(), template


# ── Token-level checks: markers must select exactly the assistant turns on a
#    rendered two-turn fixture, and the final EOS label must never be -100. ──

REPRESENTATIVES = {
    "mistral": ["unsloth/mistral-7b-instruct-v0.3"],
    "llama": ["unsloth/llama-2-7b-chat"],
    "starling": ["unsloth/Starling-LM-7B-beta"],
    "glm": ["unsloth/GLM-4.7-Flash"],
    "qwen3-thinking": ["unsloth/Qwen3-4B-Thinking-2507", "Qwen/QwQ-32B"],
    "zephyr": ["unsloth/zephyr-sft"],
}

FIXTURE = [
    {"role": "user", "content": "zebra alpha question one?"},
    {"role": "assistant", "content": "grape reply number one."},
    {"role": "user", "content": "zebra beta question two?"},
    {"role": "assistant", "content": "grape reply number two."},
]


def _load_tokenizer(repo):
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # pragma: no cover
        pytest.skip(f"transformers unavailable: {e}")
    try:
        return AutoTokenizer.from_pretrained(repo)
    except OSError as e:
        pytest.skip(f"tokenizer {repo} unavailable (offline?): {e}")
    except Exception:
        # Tokenizer class newer than this transformers (e.g. GLM-4.7's
        # TokenizersBackend): build directly from tokenizer.json.
        try:
            import json as _json
            from huggingface_hub import hf_hub_download
            from transformers import PreTrainedTokenizerFast

            with open(
                hf_hub_download(repo, "tokenizer_config.json"), encoding = "utf-8"
            ) as f:
                cfg = _json.load(f)
            tok_file = hf_hub_download(repo, "tokenizer.json")

            def _tokval(v):
                return v["content"] if isinstance(v, dict) else v

            return PreTrainedTokenizerFast(
                tokenizer_file = tok_file,
                chat_template = cfg.get("chat_template"),
                **{
                    k: _tokval(cfg[k])
                    for k in ("bos_token", "eos_token", "pad_token", "unk_token")
                    if cfg.get(k) is not None
                },
            )
        except Exception as e:
            pytest.skip(f"tokenizer {repo} unavailable (offline?): {e}")


def _train_on_responses_only():
    try:
        from unsloth_zoo.dataset_utils import train_on_responses_only
    except Exception as e:
        pytest.skip(f"unsloth_zoo unavailable: {e}")
    return train_on_responses_only


@pytest.mark.parametrize(
    "template,repo",
    [(t, r) for t, repos in sorted(REPRESENTATIVES.items()) for r in repos],
)
def test_fixed_markers_token_level(template, repo):
    tor = _train_on_responses_only()
    tok = _load_tokenizer(repo)
    parts = T2R[template]

    msgs = [{"role": "system", "content": "You are a terse assistant."}] + FIXTURE
    try:
        ids = tok.apply_chat_template(msgs, tokenize = True, add_generation_prompt = False)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]  # transformers 5.x returns a BatchEncoding
    except Exception:
        ids = tok.apply_chat_template(
            FIXTURE, tokenize = True, add_generation_prompt = False
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]

    fn = tor(
        None,
        instruction_part = parts["instruction"],
        response_part = parts["response"],
        tokenizer = tok,
        return_function = True,
    )
    labels = fn({"input_ids": [list(ids)]})["labels"][0]

    n = len(ids)
    trained = tok.decode([ids[i] for i in range(n) if labels[i] != -100])
    masked = tok.decode([ids[i] for i in range(n) if labels[i] == -100])

    # User and system content fully masked
    assert "question one" not in trained and "question one" in masked
    assert "question two" not in trained and "question two" in masked
    assert "terse assistant" not in trained
    # EVERY assistant turn trained, not just the last
    assert "reply number one" in trained
    assert "reply number two" in trained
    # The final EOS (last non-whitespace token) must never be -100, or the
    # fine-tuned model never learns to stop generating.
    i = n - 1
    while i > 0 and tok.decode([ids[i]]).strip() == "":
        i -= 1
    assert (
        labels[i] != -100
    ), f"final token {tok.convert_ids_to_tokens(int(ids[i]))!r} is masked"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
