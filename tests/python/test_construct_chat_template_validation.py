"""Negative-path validation tests for unsloth.chat_templates.construct_chat_template.

Regression coverage for the no-match guards added in the PR #5763 follow-up:
missing placeholders or unrecoverable two-example structures must raise
RuntimeError with a clear message (not IndexError/AttributeError) and must
not silently drop the last char via s[:-1]. A minimal fake tokenizer keeps
the cases CPU-only (no HF_TOKEN, no gated download).
"""

from types import SimpleNamespace

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


class _SuccessFakeTokenizer(_FakeTokenizer):
    """Adds the surface construct_chat_template touches on the success path."""

    bos_token = "<s>"
    bos_token_id = 1
    added_tokens_decoder: dict = {}

    def __call__(self, text):
        # input_ids[0] must differ from bos_token_id so the BOS-handling branch is skipped.
        return SimpleNamespace(input_ids = [5])


@pytest.mark.parametrize(
    "chat_template",
    [
        # User turn begins with {INPUT} (no prefix before the sentinel).
        "{INPUT} [/INST] {OUTPUT}</s>{INPUT} [/INST] {OUTPUT}</s>",
        # Assistant turn begins with {OUTPUT} (no prefix before the sentinel).
        "User: {INPUT}\n{OUTPUT}</s>User: {INPUT}\n{OUTPUT}</s>",
    ],
)
def test_chat_template_does_not_leak_sentinel_when_section_starts_with_it(chat_template):
    """When an input/output section begins with the {INPUT}/{OUTPUT} sentinel, the
    generated Jinja template must not keep the literal sentinel text. The `startswith`
    branch in the internal `process()` helper used to slice from `find()` (which is 0
    here) instead of past the sentinel, re-including the literal `{INPUT}`/`{OUTPUT}`."""
    _, jinja_template, _, _ = construct_chat_template(
        tokenizer = _SuccessFakeTokenizer(),
        chat_template = chat_template,
        extra_eos_tokens = ["</s>"],
    )
    assert "{INPUT}" not in jinja_template
    assert "{OUTPUT}" not in jinja_template


_SYSTEM_CHAT_TEMPLATE = (
    "{SYSTEM}\n"
    "### User: {INPUT}\n### Assistant: {OUTPUT}</s>"
    "### User: {INPUT}\n### Assistant: {OUTPUT}</s>"
)


def _render(jinja_template, messages):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment()
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(RuntimeError(message))
    return env.from_string(jinja_template).render(
        messages = messages,
        bos_token = "<s>",
        eos_token = "</s>",
        add_generation_prompt = False,
    )


@pytest.mark.parametrize("default_system_message", [None, "You are helpful."])
def test_system_message_is_consumed_by_the_system_part(default_system_message):
    """A caller-supplied system message must be rendered by the system part and
    skipped by the message loop, whatever `default_system_message` is.

    With `default_system_message = None` the generated template used to bind
    `loop_messages` only inside the `{% if %}` arm. The `Fix missing
    loop_messages` step then saw no unconditional binding, rewrote the loop back
    to `messages`, and the system message reached the loop and tripped
    `raise_exception`.
    """
    _, jinja_template, _, _ = construct_chat_template(
        tokenizer = _SuccessFakeTokenizer(),
        chat_template = _SYSTEM_CHAT_TEMPLATE,
        default_system_message = default_system_message,
        extra_eos_tokens = ["</s>"],
    )
    rendered = _render(
        jinja_template,
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
        ],
    )
    assert "Be terse." in rendered
    assert rendered.count("Hi") == 1


def test_absent_system_message_still_renders_without_default():
    """`default_system_message = None` with no system message in the input must
    keep working -- the `{% else %}` arm has to bind `loop_messages = messages`."""
    _, jinja_template, _, _ = construct_chat_template(
        tokenizer = _SuccessFakeTokenizer(),
        chat_template = _SYSTEM_CHAT_TEMPLATE,
        default_system_message = None,
        extra_eos_tokens = ["</s>"],
    )
    rendered = _render(jinja_template, [{"role": "user", "content": "Hi"}])
    assert "Hi" in rendered
