"""get_chat_template accepts a multimodal processor. See issue #10146.

gemma-4-E2B-it and other multimodal checkpoints load as a processor, not a tokenizer:
the text tokenizer sits in `processor.tokenizer`. On MLX the processor has no
padding_side of its own, so `old_padding_side = tokenizer.padding_side` raised
AttributeError before the function did anything. (On CUDA the loader sets padding_side
on the processor at models/vision.py, which is why this only bit Apple Silicon, but the
processor still has no Rust backend, so the vocab-editing templates broke there too.)

It now unwraps the processor, works on the inner tokenizer and re-attaches on return.
Three parts of that are quiet if broken: `old_tokenizer` must be bound after the unwrap,
or the pad/bos/unk restore near the end reads None off the processor and blanks all
three; the processor must come back carrying the new chat_template, since
ProcessorMixin.apply_chat_template renders off its own attribute; and the bos/eos/pad
copies the loader mirrored onto the processor must be refreshed, since chatml and
gemma_chatml rebuild the tokenizer with a remapped eos.

Importing unsloth needs a GPU, so the statements are pulled out of the source with ast
and run over stand-ins, as tests/test_map_eos_token.py does.
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
    if isinstance(node, ast.FunctionDef) and node.name == "get_chat_template"
)


def _only(found, what):
    assert len(found) == 1, f"could not find {what} in get_chat_template"
    return found[0]


def _assign(name):
    """The one top-level `<name> = ...` in get_chat_template."""
    return _only(
        [
            node
            for node in _BODY
            if isinstance(node, ast.Assign)
            and any(getattr(target, "id", None) == name for target in node.targets)
        ],
        f"the {name} binding",
    )


def _unwrap_branch():
    """The top-level `if ...` whose body claims the processor."""
    return _only(
        [
            node
            for node in _BODY
            if isinstance(node, ast.If)
            and any(
                isinstance(stmt, ast.Assign)
                and any(getattr(target, "id", None) == "_processor" for target in stmt.targets)
                for stmt in node.body
            )
        ],
        "the processor unwrap",
    )


def _reattach_branch():
    """The top-level `if _processor is not None:` that restores it before return."""
    return _only(
        [
            node
            for node in _BODY
            if isinstance(node, ast.If) and "_processor" in ast.dump(node.test)
        ],
        "the processor re-attach",
    )


def _run(statements, **namespace):
    namespace.setdefault("ProcessorMixin", _ProcessorMixin)
    exec(
        compile(ast.Module(body = statements, type_ignores = []), CHAT_TEMPLATES_PATH, "exec"),
        namespace,
    )
    return namespace


class _ProcessorMixin:
    pass


class _FakeTokenizer:
    """A tokenizer as this function sees one: special tokens, no `.tokenizer`."""

    def __init__(
        self,
        eos_token = "<eos>",
        eos_token_id = 1,
    ):
        self.padding_side = "right"
        self.bos_token, self.bos_token_id = "<bos>", 0
        self.eos_token, self.eos_token_id = eos_token, eos_token_id
        self.pad_token, self.pad_token_id = "<pad>", 2


class _FakeProcessor(_ProcessorMixin):
    """A processor as this function sees one: a `.tokenizer`, plus the loader's mirror."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        for token in ("bos_token", "eos_token", "pad_token"):
            setattr(self, token, getattr(tokenizer, token))
            setattr(self, token + "_id", getattr(tokenizer, token + "_id"))


class _FakeTokenizerBackend(_FakeTokenizer):
    """A tokenizer backend may expose `.tokenizer` without being a processor."""

    def __init__(self):
        super().__init__()
        self.tokenizer = object()


def _unwrap(tokenizer):
    return _run(
        [_assign("_processor"), _unwrap_branch(), _assign("old_tokenizer")],
        tokenizer = tokenizer,
    )


def _reattach(processor, tokenizer):
    return _run(
        [_reattach_branch()],
        _processor = processor,
        tokenizer = tokenizer,
        chat_template = "TEMPLATE",
    )


def test_processor_is_unwrapped_to_its_inner_tokenizer():
    tokenizer = _FakeTokenizer()
    namespace = _unwrap(_FakeProcessor(tokenizer))
    assert namespace["tokenizer"] is tokenizer
    # The padding_side read that used to raise now lands on the tokenizer.
    assert namespace["tokenizer"].padding_side == "right"


def test_old_tokenizer_is_the_inner_tokenizer_not_the_processor():
    # getattr(processor, "pad_token", None) is None on MLX, so binding old_tokenizer
    # before the unwrap makes the restore at the end blank pad/bos/unk on the tokenizer.
    tokenizer = _FakeTokenizer()
    assert _unwrap(_FakeProcessor(tokenizer))["old_tokenizer"] is tokenizer


def test_a_plain_tokenizer_is_left_alone():
    tokenizer = _FakeTokenizer()
    namespace = _unwrap(tokenizer)
    assert namespace["_processor"] is None
    assert namespace["tokenizer"] is tokenizer
    assert namespace["old_tokenizer"] is tokenizer


def test_a_tokenizer_backend_with_a_tokenizer_attribute_is_left_alone():
    tokenizer = _FakeTokenizerBackend()
    namespace = _unwrap(tokenizer)
    assert namespace["_processor"] is None
    assert namespace["tokenizer"] is tokenizer
    assert namespace["old_tokenizer"] is tokenizer


def test_processor_is_returned_carrying_the_new_tokenizer_and_template():
    processor = _FakeProcessor(_FakeTokenizer())
    rebuilt = _FakeTokenizer()  # the vocab surgery replaces the tokenizer object
    namespace = _reattach(processor, rebuilt)
    assert namespace["tokenizer"] is processor
    assert processor.tokenizer is rebuilt
    assert processor.chat_template == "TEMPLATE"


def test_a_remapped_eos_reaches_the_processors_mirrored_copy():
    # chatml/gemma_chatml rebuild the tokenizer with eos remapped to the stop word. The
    # loader copied the old eos onto the processor, and that copy is what the collators
    # and save paths read, so it has to follow the rebuild.
    processor = _FakeProcessor(_FakeTokenizer(eos_token = "<eos>", eos_token_id = 1))
    rebuilt = _FakeTokenizer(eos_token = "<|im_end|>", eos_token_id = 107)
    _reattach(processor, rebuilt)
    assert processor.eos_token == "<|im_end|>"
    assert processor.eos_token_id == 107
