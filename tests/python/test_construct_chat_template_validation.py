"""Negative-path validation tests for unsloth.chat_templates.construct_chat_template.

Regression coverage for the str.find() / regex no-match guards added in
PR #5763 follow-up: missing placeholders or unrecoverable two-example
structures must raise RuntimeError with a clear message, not IndexError
or AttributeError, and must never silently drop the last character via
s[:-1].

Uses a minimal fake tokenizer so the cases run on CPU-only CI without
HF_TOKEN and without downloading a gated model. The validation paths
exercised here fail before construct_chat_template reaches any heavy
tokenizer interaction, so the stub stays small.
"""

import pytest

from unsloth.chat_templates import construct_chat_template


class _FakeTokenizer:
    """Minimum surface construct_chat_template touches before the
    validation guards fire."""

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
    """One {INPUT}/{OUTPUT} pair (rather than the required two) used to
    crash with AttributeError on `found.group(1)` after the for-loop
    broke without setting `found`. Must raise RuntimeError now."""
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
    # Excerpt is repr-quoted and capped; total message should stay well
    # under the template length.
    assert len(msg) < 1000
    assert "{OUTPUT}" in msg
