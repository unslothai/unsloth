"""A Mistral-format checkpoint (params.json, no config.json) gets a message that
says what it is, instead of the generic "both configs failed" one.

Mistral-Large-3 is published that way: `library_name: vllm`, loaded through
mistral-common, and its card says transformers support did not make it in.
AutoConfig and PeftConfig both fail on it, and the old message pointed the user
at upgrading transformers, which cannot help.
"""

import json
import os

import pytest
import torch


def _write(tmp_path, names):
    for name in names:
        (tmp_path / name).write_text(json.dumps({"dim": 8}))
    return str(tmp_path)


def test_params_json_without_config_json_is_mistral_format(tmp_path):
    from unsloth.models.loader import _is_mistral_format_checkpoint
    assert _is_mistral_format_checkpoint(_write(tmp_path, ["params.json", "tekken.json"])) is True


def test_a_params_json_without_a_mistral_marker_is_not_claimed(tmp_path):
    """Meta's original Llama layout: params.json next to consolidated.00.pth and
    tokenizer.model. Not Mistral's format, so the generic message must stay."""
    from unsloth.models.loader import _is_mistral_format_checkpoint

    assert _is_mistral_format_checkpoint(_write(tmp_path, ["params.json", "consolidated.00.pth", "tokenizer.model"])) is False
    assert _is_mistral_format_checkpoint(_write(tmp_path, ["params.json", "consolidated.safetensors.index.json"])) is True


def test_a_config_json_next_to_params_json_is_a_transformers_repo(tmp_path):
    from unsloth.models.loader import _is_mistral_format_checkpoint
    assert _is_mistral_format_checkpoint(_write(tmp_path, ["params.json", "config.json"])) is False


def test_an_empty_or_unknown_directory_is_not_claimed(tmp_path):
    from unsloth.models.loader import _is_mistral_format_checkpoint
    assert _is_mistral_format_checkpoint(str(tmp_path)) is False
    assert (
        _is_mistral_format_checkpoint(str(tmp_path / "does-not-exist"), local_files_only = True)
        is False
    )


def test_the_message_names_the_file_and_the_route():
    from unsloth.models.loader import _mistral_format_error

    text = _mistral_format_error("mistralai/Mistral-Large-3-675B-Instruct-2512")
    assert "params.json" in text and "config.json" in text
    assert "vLLM" in text and "mistral-common" in text
    assert "Mistral-Large-3-675B-Instruct-2512" in text


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "import unsloth needs an accelerator")
def test_loader_raises_the_specific_message(tmp_path):
    """The arm that fails on main: the loader used to raise the generic message."""
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    path = _write(tmp_path, ["params.json", "tekken.json"])
    with pytest.raises(RuntimeError) as info:
        FastLanguageModel.from_pretrained(path, max_seq_length = 64)
    assert "Mistral's own format" in str(info.value), str(info.value)[:400]
    assert "Both AutoConfig and PeftConfig" not in str(info.value)
