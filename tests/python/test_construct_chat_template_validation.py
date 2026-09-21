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
        "{INPUT} [/INST] {OUTPUT}</s>{INPUT} [/INST] {OUTPUT}</s>",
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


def _render(
    jinja_template,
    messages,
    add_generation_prompt = False,
    bos_token = "<s>",
):
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment()
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(RuntimeError(message))
    return env.from_string(jinja_template).render(
        messages = messages,
        bos_token = bos_token,
        eos_token = "</s>",
        add_generation_prompt = add_generation_prompt,
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
    assert rendered.count("Be terse.") == 1
    assert rendered.count("Hi") == 1
    # A caller system message overrides the default; the default must not leak in.
    if default_system_message is not None:
        assert default_system_message not in rendered


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


_NO_SYSTEM_CHAT_TEMPLATE = (
    "PREAMBLE\n"
    "### User: {INPUT}\n### Assistant: {OUTPUT}</s>"
    "### User: {INPUT}\n### Assistant: {OUTPUT}</s>"
)


def test_static_prefix_without_system_still_rejects_system_message():
    """A template with a static prefix but no {SYSTEM} placeholder cannot render a
    caller system message, so it must still raise rather than silently drop it."""
    _, jinja_template, _, _ = construct_chat_template(
        tokenizer = _SuccessFakeTokenizer(),
        chat_template = _NO_SYSTEM_CHAT_TEMPLATE,
        default_system_message = None,
        extra_eos_tokens = ["</s>"],
    )
    with pytest.raises(RuntimeError, match = "Only user and assistant roles are supported!"):
        _render(
            jinja_template,
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
            ],
        )


@pytest.mark.parametrize("default_system_message", [None, "You are helpful."])
def test_static_prefix_without_system_renders_in_every_conversation(default_system_message):
    """A static prefix must render regardless of the default system message."""
    modelfile, jinja_template, _, _ = construct_chat_template(
        tokenizer = _SuccessFakeTokenizer(),
        chat_template = _NO_SYSTEM_CHAT_TEMPLATE,
        default_system_message = default_system_message,
        extra_eos_tokens = ["</s>"],
    )
    rendered = _render(jinja_template, [{"role": "user", "content": "Hi"}])
    assert rendered.startswith("PREAMBLE\n"), rendered
    assert "PREAMBLE\n" in modelfile.split("TEMPLATE ")[1]


def test_auto_appended_eos_prefers_the_tokenizer_eos_deterministically():
    """When the template has no EOS after {OUTPUT}, construct_chat_template appends one
    itself and picks `extra_eos_tokens[0]`. `extra_eos_tokens.insert(0, tokenizer.eos_token)`
    exists to make that the tokenizer's own EOS, so the choice must not depend on set
    ordering: de-duplicating through `set()` made the appended token, and therefore the
    token ending every formatted training sample, vary with PYTHONHASHSEED."""

    class _TwoEosTokenizer(_SuccessFakeTokenizer):
        def get_vocab(self):
            return {"</s>": 0, "<|myeos|>": 2}

    _, jinja_template, _, _ = construct_chat_template(
        tokenizer = _TwoEosTokenizer(),
        chat_template = (
            "### User: {INPUT}\n### Assistant: {OUTPUT}\n"
            "### User: {INPUT}\n### Assistant: {OUTPUT}\n"
        ),
        default_system_message = None,
        extra_eos_tokens = ["<|myeos|>"],
    )
    assistant_turn = jinja_template.split("'assistant' %}")[1].split("{% else %}")[0]
    assert _TwoEosTokenizer.eos_token in assistant_turn
    assert "<|myeos|>" not in assistant_turn


def test_input_boundary_prefers_the_longest_eos_token():
    class _PrefixEosTokenizer(_SuccessFakeTokenizer):
        def get_vocab(self):
            return {"</s>": 0, "</s>extra": 2}

    _, jinja_template, _, _ = construct_chat_template(
        tokenizer = _PrefixEosTokenizer(),
        chat_template = (
            "### User: {INPUT}</s>extra\n### Assistant: {OUTPUT}</s>\n"
            "### User: {INPUT}</s>extra\n### Assistant: {OUTPUT}</s>\n"
        ),
        default_system_message = None,
        extra_eos_tokens = ["</s>extra"],
    )

    rendered_user_turn = _render(jinja_template, [{"role": "user", "content": "Hi"}])
    assert rendered_user_turn == "### User: Hi</s>extra"


_APOSTROPHE_CHAT_TEMPLATE = (
    "{SYSTEM}\n"
    "### User's turn: {INPUT}\n### Bot's reply: {OUTPUT}</s>"
    "### User's turn: {INPUT}\n### Bot's reply: {OUTPUT}</s>"
)


@pytest.mark.parametrize(
    "default_system_message",
    [
        "Answer the user's question.",
        r"Put the answer in \boxed{}.",
        r"Files live in C:\Users\me",
        # Windows CRLF: Jinja rewrites a raw \r to \n.
        "Answer briefly.\r\nBe polite.",
    ],
)
def test_quotes_and_backslashes_survive_into_the_jinja_template(default_system_message):
    """Template text is concatenated into Jinja `'...'` literals, so it has to be
    escaped on the way in. An apostrophe used to close the literal early
    (TemplateSyntaxError: expected token 'end of print statement'), and a backslash was
    read as a Jinja escape, so `\\boxed` silently became a backspace character and
    `C:\\Users` raised `truncated \\UXXXXXXXX escape`. Covers the system message and
    the instruction/response sections, which are spliced by three separate call sites."""
    _, jinja_template, _, _ = construct_chat_template(
        tokenizer = _SuccessFakeTokenizer(),
        chat_template = _APOSTROPHE_CHAT_TEMPLATE,
        default_system_message = default_system_message,
        extra_eos_tokens = ["</s>"],
    )

    rendered = _render(jinja_template, [{"role": "user", "content": "Hi"}])
    assert default_system_message in rendered
    assert "### User's turn: Hi" in rendered

    prompted = _render(
        jinja_template,
        [{"role": "user", "content": "Hi"}],
        add_generation_prompt = True,
    )
    assert prompted.endswith("### Bot's reply: ")


class _BosFakeTokenizer(_SuccessFakeTokenizer):
    """A bos_token carrying characters the Jinja escaper rewrites."""

    bos_token = "<s'\\a>"

    def __call__(self, text):
        # input_ids[0] == bos_token_id takes the BOS-handling branch.
        return SimpleNamespace(input_ids = [1])


def test_bos_token_with_quote_or_backslash_is_not_emitted_twice():
    """The BOS is stripped from the system section so it is only rendered once, via
    `{{ bos_token }}`. Stripping it after process() had escaped the section left it
    unmatched, and the caller-system branch then emitted it a second time."""
    bos = _BosFakeTokenizer.bos_token
    _, jinja_template, _, _ = construct_chat_template(
        tokenizer = _BosFakeTokenizer(),
        chat_template = bos + _APOSTROPHE_CHAT_TEMPLATE,
        default_system_message = "Be helpful.",
        extra_eos_tokens = ["</s>"],
    )

    for messages in (
        [{"role": "user", "content": "Hi"}],
        [{"role": "system", "content": "Sysmsg"}, {"role": "user", "content": "Hi"}],
    ):
        rendered = _render(jinja_template, messages, bos_token = bos)
        assert rendered.count(bos) == 1, rendered


def test_bos_only_prefix_still_rejects_system_message():
    bos = _BosFakeTokenizer.bos_token
    _, jinja_template, _, _ = construct_chat_template(
        tokenizer = _BosFakeTokenizer(),
        chat_template = bos + _NO_SYSTEM_CHAT_TEMPLATE.removeprefix("PREAMBLE\n"),
        default_system_message = None,
        extra_eos_tokens = ["</s>"],
    )
    with pytest.raises(RuntimeError, match = "Only user and assistant roles are supported!"):
        _render(
            jinja_template,
            [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Hi"},
            ],
            bos_token = bos,
        )
