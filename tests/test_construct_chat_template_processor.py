"""construct_chat_template accepts a multimodal processor.

A multimodal checkpoint loads as a processor, which carries the text tokenizer in
`.tokenizer`. construct_chat_template is tokenizer-shaped throughout -- `get_vocab()`,
`name_or_path`, `bos_token`, and calling the object on a string -- and a processor has
none of that, so `apply_chat_template(dataset, tokenizer = processor, ...)`, which
reaches it, died on `vocab = tokenizer.get_vocab()`.

It now unwraps once at the top. Unlike get_chat_template there is nothing to re-attach:
this returns a template tuple, never the tokenizer. The unwrap therefore only has to
happen before the first tokenizer-shaped use, which is what the ordering test pins.

Importing unsloth needs a GPU, so the statement is pulled out of the source with ast and
run over stand-ins, as tests/test_map_eos_token.py does.
"""

import ast
import os

CHAT_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "chat_templates.py",
)

_BODY = next(
    node.body
    for node in ast.parse(open(CHAT_TEMPLATES_PATH, encoding = "utf-8").read()).body
    if isinstance(node, ast.FunctionDef) and node.name == "construct_chat_template"
)


def _unwrap_branch():
    """The one top-level `if ...:` that rebinds tokenizer to its `.tokenizer`."""
    found = [
        node
        for node in _BODY
        if isinstance(node, ast.If)
        and any(
            isinstance(stmt, ast.Assign)
            and any(getattr(target, "id", None) == "tokenizer" for target in stmt.targets)
            for stmt in node.body
        )
    ]
    assert len(found) == 1, "could not find the processor unwrap in construct_chat_template"
    return found[0]


def _run(tokenizer):
    namespace = {"tokenizer": tokenizer, "ProcessorMixin": _ProcessorMixin}
    exec(
        compile(ast.Module(body = [_unwrap_branch()], type_ignores = []), CHAT_TEMPLATES_PATH, "exec"),
        namespace,
    )
    return namespace["tokenizer"]


class _ProcessorMixin:
    pass


class _FakeTokenizer:
    """A tokenizer as this function uses one: it can produce a vocab, and has no `.tokenizer`."""

    def get_vocab(self):
        return {"<eos>": 0}


class _FakeProcessor(_ProcessorMixin):
    """A processor as this function sees one: a `.tokenizer`, and no get_vocab."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _FakeTokenizerBackend(_FakeTokenizer):
    """A tokenizer backend may expose `.tokenizer` without being a processor."""

    def __init__(self):
        self.tokenizer = object()


def test_processor_is_unwrapped_to_its_inner_tokenizer():
    tokenizer = _FakeTokenizer()
    unwrapped = _run(_FakeProcessor(tokenizer))
    assert unwrapped is tokenizer
    # The get_vocab() call that used to raise now lands on the tokenizer.
    assert unwrapped.get_vocab() == {"<eos>": 0}


def test_a_plain_tokenizer_is_left_alone():
    tokenizer = _FakeTokenizer()
    assert _run(tokenizer) is tokenizer


def test_a_tokenizer_backend_with_a_tokenizer_attribute_is_left_alone():
    tokenizer = _FakeTokenizerBackend()
    assert _run(tokenizer) is tokenizer


def test_the_unwrap_precedes_every_tokenizer_shaped_use():
    # get_vocab() is the first of them and the one that raised. If the unwrap ever drifts
    # below it, the processor reaches get_vocab again and the AttributeError is back.
    unwrap = _unwrap_branch()
    uses = [
        node.lineno
        for node in ast.walk(ast.Module(body = _BODY, type_ignores = []))
        if isinstance(node, ast.Attribute)
        and getattr(node.value, "id", None) == "tokenizer"
        and node.attr in ("get_vocab", "name_or_path", "bos_token", "eos_token")
    ]
    assert uses, "construct_chat_template no longer uses the tokenizer as a tokenizer"
    assert unwrap.lineno < min(uses)
