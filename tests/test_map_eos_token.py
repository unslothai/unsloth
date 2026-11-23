"""Honour map_eos_token=False, except gemma_chatml and gemma2_chatml, which rename <eos> regardless."""

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
    assert _resolve(map_eos_token = True, yes_map_eos_token = True)[0] is True
    assert _resolve(map_eos_token = True, yes_map_eos_token = False)[0] is False
    assert _resolve(map_eos_token = False, yes_map_eos_token = False)[0] is False


def test_opt_out_is_refused_when_the_template_rewrites_the_vocab():
    # gemma_chatml / gemma2_chatml: <|im_end|> exists only because <eos> is renamed to it,
    # so the opt-out cannot be honored.
    resolved, messages = _resolve(
        map_eos_token = False,
        yes_map_eos_token = True,
        token_mapping = {"<start_of_turn>": "<|im_start|>", "<eos>": "<|im_end|>"},
        eos_token = "<eos>",
    )
    assert resolved is True
    assert messages, "forcing the mapping back on must not be silent"


def test_opt_out_is_refused_when_eos_token_is_not_the_renamed_piece():
    # gemma-3-270m/1b-it ship eos_token "<end_of_turn>" yet gemma_chatml still renames <eos>;
    # keying on tokenizer.eos_token would re-add <eos> past the embeddings.
    resolved, messages = _resolve(
        map_eos_token = False,
        yes_map_eos_token = True,
        token_mapping = {"<start_of_turn>": "<|im_start|>", "<eos>": "<|im_end|>"},
        eos_token = "<end_of_turn>",
    )
    assert resolved is True
    assert messages, "forcing the mapping back on must not be silent"


def test_a_template_veto_still_wins_over_the_refusal():
    # A template that does not want eos mapping keeps map_eos_token = False even with a mapping.
    resolved, messages = _resolve(
        map_eos_token = True,
        yes_map_eos_token = False,
        token_mapping = {"<start_of_turn>": "<|im_start|>", "<eos>": "<|im_end|>"},
        eos_token = "<eos>",
    )
    assert resolved is False
    assert not messages


def test_opt_out_still_honored_when_the_template_leaves_the_vocab_alone():
    resolved, messages = _resolve(
        map_eos_token = False,
        yes_map_eos_token = True,
        token_mapping = None,
        eos_token = "<eos>",
    )
    assert resolved is False
    assert not messages


def test_shipped_templates_still_have_the_shape_the_guard_keys_on():
    """The guard keys on a template's token_mapping, not its name, so pin that shape."""
    namespace = {}
    for node in ast.parse(_source()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                namespace[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                continue
    mapping, stop_word = namespace["gemma_chatml_eos_token"]
    assert mapping["<eos>"] == stop_word == "<|im_end|>"


# Runs the real vocab surgery on an in-memory word-level fast tokenizer (no download/GPU).
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
    # Stub fix_sentencepiece_tokenizer as identity: importing it drags in GPU-bound unsloth,
    # and it returns the tokenizer unchanged without a tokenizer.model.
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
    tokenizer = _map_tokens(monkeypatch, map_eos_token = False, token_mapping = None)
    vocab = tokenizer.get_vocab()

    assert vocab == VOCAB
    assert STOP_WORD not in vocab
    assert tokenizer.eos_token == "<eos>"
    assert tokenizer.eos_token_id == VOCAB["<eos>"]


def test_mapping_on_the_plain_stop_word_path_still_swaps_eos_for_the_stop_word(monkeypatch):
    tokenizer = _map_tokens(monkeypatch, map_eos_token = True, token_mapping = None)
    vocab = tokenizer.get_vocab()

    assert "<eos>" not in vocab
    assert vocab[STOP_WORD] == VOCAB["<eos>"]
    assert tokenizer.eos_token == STOP_WORD
    assert tokenizer.eos_token_id == vocab[STOP_WORD]
    # The other three specials are re-passed by hand on this path.
    assert (tokenizer.bos_token, tokenizer.pad_token, tokenizer.unk_token) == (
        "<bos>",
        "<pad>",
        "<unk>",
    )
