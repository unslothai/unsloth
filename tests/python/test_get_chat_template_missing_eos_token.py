# Test for https://github.com/unslothai/unsloth - named error when a tokenizer has no eos_token
import pytest

from unsloth.chat_templates import get_chat_template


class _TokenizerWithoutEos:
    eos_token = None
    chat_template = None
    name_or_path = "base-model-without-eos"


def test_missing_eos_token_is_named():
    with pytest.raises(RuntimeError, match="has no eos_token"):
        get_chat_template(_TokenizerWithoutEos(), "chatml")
