from unittest.mock import MagicMock

from transformers import PreTrainedTokenizerBase

from unsloth.save import patch_saving_functions


class _ProcessorWithNoneTokenizer:
    tokenizer = None

    def push_to_hub(self, *args, **kwargs):
        return None

    push_to_hub.__doc__ = "stub"

    def save_pretrained(self, *args, **kwargs):
        return None


def test_patch_saving_functions_no_crash_on_none_tokenizer():
    proc = _ProcessorWithNoneTokenizer()
    patch_saving_functions(proc)


def test_patch_saving_functions_still_patches_non_none_tokenizer():
    inner = MagicMock(spec = PreTrainedTokenizerBase)
    inner.save_pretrained = MagicMock()
    inner.save_pretrained.__name__ = "save_pretrained"
    inner.push_to_hub = MagicMock()
    inner.push_to_hub.__name__ = "push_to_hub"
    inner.push_to_hub.__doc__ = "tokenizer doc"

    class _Proc:
        def __init__(self, tok):
            self.tokenizer = tok

        def push_to_hub(self, *args, **kwargs):
            return None

        push_to_hub.__doc__ = "proc doc"

        def save_pretrained(self, *args, **kwargs):
            return None

    proc = _Proc(inner)
    patch_saving_functions(proc)
    assert hasattr(inner, "original_save_pretrained")
