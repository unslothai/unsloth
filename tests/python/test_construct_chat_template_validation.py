"""Negative-path validation tests for unsloth.chat_templates.construct_chat_template.

Regression coverage for the no-match guards added in the PR #5763 follow-up:
missing placeholders or unrecoverable two-example structures must raise
RuntimeError with a clear message (not IndexError/AttributeError) and must
not silently drop the last char via s[:-1]. A minimal fake tokenizer keeps
the cases CPU-only (no HF_TOKEN, no gated download).
"""

import pytest

from unsloth.chat_templates import construct_chat_template


class _FakeTokenizer:
    """Minimal surface construct_chat_template touches before the guards fire."""

    name_or_path = "fake/tokenizer"
    eos_token = "</s>"

    def get_vocab(self):
        return {"</s>": 0}


@pytest.mark.parametrize(
    "template, expected_in_message",
    [
        ("only {INPUT} here, no output marker", "{OUTPUT}"),
        ("only {OUTPUT} here, no input marker", "{INPUT}"),
        ("neither sentinel here, just literal text", "{INPUT}"),
        ("neither sentinel here, just literal text", "{OUTPUT}"),
    ],
)
def test_missing_placeholder_in_chat_template_raises(template, expected_in_message):
    with pytest.raises(RuntimeError) as exc_info:
        construct_chat_template(
            tokenizer = _FakeTokenizer(),
            chat_template = template,
            extra_eos_tokens = ["</s>"],
        )
    assert expected_in_message in str(exc_info.value)


def test_single_pair_template_raises_clear_error_not_attribute_error():
    """A single {INPUT}/{OUTPUT} pair must raise RuntimeError, not the old
    AttributeError on `found.group(1)` when the loop broke without setting `found`."""
    template = "user: {INPUT}\nassistant: {OUTPUT}\n"
    with pytest.raises(RuntimeError):
        construct_chat_template(
            tokenizer = _FakeTokenizer(),
            chat_template = template,
            extra_eos_tokens = ["</s>"],
        )


def test_error_message_excerpt_is_bounded():
    """Error messages must include a bounded excerpt of the offending
    template, not dump arbitrarily large content into the traceback."""
    huge = ("garbage " * 5000) + "{INPUT}"  # ~40 KB, missing {OUTPUT}
    with pytest.raises(RuntimeError) as exc_info:
        construct_chat_template(
            tokenizer = _FakeTokenizer(),
            chat_template = huge,
            extra_eos_tokens = ["</s>"],
        )
    msg = str(exc_info.value)
    # Excerpt is capped well under the template length.
    assert len(msg) < 1000
    assert "{OUTPUT}" in msg
