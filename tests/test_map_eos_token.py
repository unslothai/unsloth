"""Regression test for the map_eos_token argument of get_chat_template.

get_chat_template accepts a public, type-asserted `map_eos_token` argument, but on
the string-template path it was overwritten by the template's own flag, so an
explicit map_eos_token = False was silently ignored for every template that sets
yes_map_eos_token = True (chatml, gemma, gemma_chatml, gemma2, gemma2_chatml).

Honouring the opt-out is only coherent when the template does not rewrite the vocab
to build its stop word. gemma_chatml and gemma2_chatml *create* their stop word by
renaming the tokenizer's own eos piece (`{"<eos>": "<|im_end|>"}`), and that rename
happens whether or not the caller opts out, while the rebuilt tokenizer only carries
eos_token = stop_word when the mapping is on. Letting the opt-out through for those
two renames <eos> away and then lets the tokenizer class default re-add it as a new
out-of-range id. So those keep forcing the mapping, with a warning.

Importing unsloth needs a GPU, so both halves pull the shipped statements out of
the source with ast, in the same spirit as tests/test_gemma4_chat_template.py
which extracts the templates from the same file rather than importing unsloth.
The first half runs the resolution statements over fake tokenizers and checks the
decision. The second half runs the vocab surgery those decisions gate over a real
fast tokenizer built in memory, and checks the vocabulary and the eos metadata
before and after, since the decision is only interesting for what it does to them.
"""

import ast
import os
import sys
import types

CHAT_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "chat_templates.py",
)


def _source():
    return open(CHAT_TEMPLATES_PATH, encoding = "utf-8").read()


def _resolution_statements():
    """The `if ...: map_eos_token = ...` statements inside get_chat_template, in source order."""
    tree = ast.parse(_source())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_chat_template"
    )
    statements = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.If)
        and any(
            isinstance(stmt, ast.Assign)
            and any(getattr(target, "id", None) == "map_eos_token" for target in stmt.targets)
            for stmt in node.body
        )
    ]
    assert statements, "could not find the map_eos_token resolution in get_chat_template"
    return sorted(statements, key = lambda node: node.lineno)


class _FakeTokenizer:
    def __init__(self, eos_token):
        self.eos_token = eos_token


class _FakeLogger:
    def __init__(self):
        self.messages = []

    def warning_once(self, message):
        self.messages.append(message)


def _resolve(
    map_eos_token,
    yes_map_eos_token,
    token_mapping = None,
    eos_token = "<eos>",
):
    """Run the shipped resolution statements over one (caller, template) combination."""
    module = ast.Module(body = _resolution_statements(), type_ignores = [])
    logger = _FakeLogger()
    namespace = {
        "map_eos_token": map_eos_token,
        "yes_map_eos_token": yes_map_eos_token,
        "token_mapping": token_mapping,
        "tokenizer": _FakeTokenizer(eos_token),
        "logger": logger,
        "type_chat_template": "gemma_chatml",
        "stop_word": "<|im_end|>",
    }
    exec(compile(module, CHAT_TEMPLATES_PATH, "exec"), namespace)
    return namespace["map_eos_token"], logger.messages


def test_explicit_map_eos_token_false_is_honored():
    # A template asking for eos mapping must not override an explicit opt-out.
    resolved, _ = _resolve(map_eos_token = False, yes_map_eos_token = True)
    assert resolved is False


def test_other_map_eos_token_combinations_are_unchanged():
    # The default is map_eos_token = True, so these three paths must not move.
    assert _resolve(map_eos_token = True, yes_map_eos_token = True)[0] is True
    # A template that does not use eos mapping still vetoes it.
    assert _resolve(map_eos_token = True, yes_map_eos_token = False)[0] is False
    assert _resolve(map_eos_token = False, yes_map_eos_token = False)[0] is False


def test_opt_out_is_refused_when_the_template_rewrites_the_vocab():
    # gemma_chatml / gemma2_chatml: <|im_end|> only exists because <eos> is renamed to it, so the opt-out cannot be
    # honored without leaving eos_token dangling.
    # the next test pins the checkpoints where it is not, which is the case keying on tokenizer.eos_token used to miss.
    resolved, messages = _resolve(
        map_eos_token = False,
        yes_map_eos_token = True,
        token_mapping = {"<start_of_turn>": "<|im_start|>", "<eos>": "<|im_end|>"},
        eos_token = "<eos>",
    )
    assert resolved is True
    assert messages, "forcing the mapping back on must not be silent"


def test_opt_out_is_refused_when_eos_token_is_not_the_renamed_piece():
    # gemma-3-270m-it and gemma-3-1b-it ship eos_token = "<end_of_turn>", not "<eos>", yet gemma_chatml still renames
    # <eos> away to build <|im_end|>.
    # Keying the guard on tokenizer.eos_token misses these and rebuilds the tokenizer with no eos_token, so the class
    resolved, messages = _resolve(
        map_eos_token = False,
        yes_map_eos_token = True,
        token_mapping = {"<start_of_turn>": "<|im_start|>", "<eos>": "<|im_end|>"},
        eos_token = "<end_of_turn>",
    )
    assert resolved is True
    assert messages, "forcing the mapping back on must not be silent"


def test_a_template_veto_still_wins_over_the_refusal():
    resolved, messages = _resolve(
        map_eos_token = True,
        yes_map_eos_token = False,
        token_mapping = {"<start_of_turn>": "<|im_start|>", "<eos>": "<|im_end|>"},
        eos_token = "<eos>",
    )
    assert resolved is False
    assert not messages


def test_opt_out_still_honored_when_the_template_leaves_the_vocab_alone():
    # The refusal only overrides the caller.
    # chatml / gemma / gemma2 carry no token_mapping, so nothing is half-applied when the mapping is skipped and the
    resolved, messages = _resolve(
        map_eos_token = False,
        yes_map_eos_token = True,
        token_mapping = None,
        eos_token = "<eos>",
    )
    assert resolved is False
    assert not messages


def test_shipped_templates_still_have_the_shape_the_guard_keys_on():
    """The guard matches on the template carrying a token_mapping, not on template names.

    If gemma_chatml ever stopped renaming <eos>, the guard would quietly stop firing and
    the opt-out would start building the broken tokenizer again, so pin the shape here.
    """
    namespace = {}
    for node in ast.parse(_source()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
    mapping, stop_word = namespace["gemma_chatml_eos_token"]
    assert mapping["<eos>"] == stop_word == "<|im_end|>"


# The resolved flag only matters for what the vocab surgery below it then does, so run that surgery against a real fast
GEMMA_CHATML_MAPPING = {"<start_of_turn>": "<|im_start|>", "<eos>": "<|im_end|>"}
STOP_WORD = "<|im_end|>"
VOCAB = {
    "<unk>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<pad>": 3,
    "<start_of_turn>": 4,
    "<end_of_turn>": 5,
    "hello": 6,
    "world": 7,
}


def _vocab_surgery_block():
    """The `if not is_fast_tokenizer: ... elif token_mapping ... elif map_eos_token ...` chain."""
    tree = ast.parse(_source())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_chat_template"
    )
    blocks = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and getattr(node.test.operand, "id", None) == "is_fast_tokenizer"
    ]
    assert len(blocks) == 1, "could not find the fast-tokenizer vocab surgery in get_chat_template"
    return blocks[0]


def _tiny_fast_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(dict(VOCAB), unk_token = "<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object = backend,
        bos_token = "<bos>",
        eos_token = "<eos>",
        pad_token = "<pad>",
        unk_token = "<unk>",
    )


def _map_tokens(monkeypatch, map_eos_token, token_mapping):
    """Run the shipped surgery over a fresh tiny tokenizer and hand back the result."""
    # The block ends in `from .tokenizer_utils import fix_sentencepiece_tokenizer`, and that import would drag in the
    # GPU-bound unsloth package.
    # The function mirrors the rename into a tokenizer.model file and returns new_tokenizer untouched when there is
    package = types.ModuleType("_unsloth_map_eos_stub")
    package.__path__ = []
    tokenizer_utils = types.ModuleType("_unsloth_map_eos_stub.tokenizer_utils")
    tokenizer_utils.fix_sentencepiece_tokenizer = (
        lambda old_tokenizer, new_tokenizer, mapping, **kwargs: new_tokenizer
    )
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, tokenizer_utils.__name__, tokenizer_utils)

    namespace = {
        "__name__": package.__name__ + ".chat_templates",
        "__package__": package.__name__,
        "is_fast_tokenizer": True,
        "tokenizer": _tiny_fast_tokenizer(),
        "token_mapping": token_mapping,
        "stop_word": STOP_WORD,
        "map_eos_token": map_eos_token,
        "logger": _FakeLogger(),
    }
    module = ast.Module(body = [_vocab_surgery_block()], type_ignores = [])
    exec(compile(module, CHAT_TEMPLATES_PATH, "exec"), namespace)
    return namespace["tokenizer"]


def test_forced_mapping_renames_eos_in_the_vocab_and_takes_eos_token_with_it(monkeypatch):
    # gemma_chatml shape, with the flag the guard forces back on.
    tokenizer = _map_tokens(monkeypatch, map_eos_token = True, token_mapping = GEMMA_CHATML_MAPPING)
    vocab = tokenizer.get_vocab()

    assert "<eos>" not in vocab, "the rename must remove the old piece"
    assert vocab[STOP_WORD] == VOCAB["<eos>"], "the stop word takes over the <eos> id"
    assert vocab["<|im_start|>"] == VOCAB["<start_of_turn>"]
    assert len(vocab) == len(VOCAB), "renaming pieces must not grow the vocab"

    assert tokenizer.eos_token == STOP_WORD
    assert tokenizer.eos_token_id == vocab[STOP_WORD]
    assert tokenizer(STOP_WORD, add_special_tokens = False)["input_ids"] == [vocab[STOP_WORD]]


def test_honoring_the_opt_out_here_would_leave_the_tokenizer_without_an_eos(monkeypatch):
    """Why the guard refuses the opt-out for this shape, rather than an argument about it."""
    tokenizer = _map_tokens(monkeypatch, map_eos_token = False, token_mapping = GEMMA_CHATML_MAPPING)
    vocab = tokenizer.get_vocab()

    assert "<eos>" not in vocab, "map_eos_token does not gate the rename, only the eos metadata"
    assert vocab[STOP_WORD] == VOCAB["<eos>"]

    assert tokenizer.eos_token is None or tokenizer.eos_token not in vocab, (
        f"eos_token = {tokenizer.eos_token!r} now survives the rename, so honouring the "
        f"opt-out here is no longer harmful and the guard should be revisited"
    )


def test_opt_out_on_the_plain_stop_word_path_leaves_the_tokenizer_untouched(monkeypatch):
    # chatml / gemma / gemma2: no token_mapping, so the opt-out skips the surgery outright.
    tokenizer = _map_tokens(monkeypatch, map_eos_token = False, token_mapping = None)
    vocab = tokenizer.get_vocab()

    assert vocab == VOCAB
    assert STOP_WORD not in vocab
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.eos_token_id == VOCAB["<eos>"]


def test_mapping_on_the_plain_stop_word_path_still_swaps_eos_for_the_stop_word(monkeypatch):
    # The default, map_eos_token = True, must keep doing the swap on that same path.
    tokenizer = _map_tokens(monkeypatch, map_eos_token = True, token_mapping = None)
    vocab = tokenizer.get_vocab()

    assert "<eos>" not in vocab
    assert vocab[STOP_WORD] == VOCAB["<eos>"]
    assert tokenizer.eos_token == STOP_WORD
    assert tokenizer.eos_token_id == vocab[STOP_WORD]
    # The other three specials are re-passed by hand on this path, so pin that they survive.
    assert (tokenizer.bos_token, tokenizer.pad_token, tokenizer.unk_token) == (
        "<bos>",
        "<pad>",
        "<unk>",
    )
