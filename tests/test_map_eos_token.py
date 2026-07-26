"""Regression test for the map_eos_token argument of get_chat_template.

get_chat_template accepts a public, type-asserted `map_eos_token` argument, but on
the string-template path it was overwritten by the template's own flag, so an
explicit map_eos_token = False was silently ignored for every template that sets
yes_map_eos_token = True (chatml, gemma, gemma_chatml, gemma2, gemma2_chatml).

Honouring the opt-out is only coherent when the stop word already exists in the
vocab. gemma_chatml and gemma2_chatml instead *create* their stop word by renaming
the tokenizer's own eos piece (`{"<eos>": "<|im_end|>"}`), and that rename happens
whether or not the caller opts out. Letting the opt-out through for those two would
rename <eos> away and then skip setting eos_token to the replacement, leaving
eos_token pointing at a token no longer in the vocab. So those keep forcing the
mapping, with a warning.

The eos mapping itself needs a real fast tokenizer, so this pulls the resolution
statements out of the shipped source with ast and checks the decision directly,
in the same spirit as tests/test_gemma4_chat_template.py which extracts the
templates from the same file rather than importing unsloth.
"""

import ast
import os

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


def test_opt_out_is_refused_when_the_mapping_renames_the_eos_piece():
    # gemma_chatml / gemma2_chatml: <|im_end|> only exists because <eos> is renamed to it,
    # so the opt-out cannot be honored without leaving eos_token dangling.
    resolved, messages = _resolve(
        map_eos_token = False,
        yes_map_eos_token = True,
        token_mapping = {"<start_of_turn>": "<|im_start|>", "<eos>": "<|im_end|>"},
        eos_token = "<eos>",
    )
    assert resolved is True
    assert messages, "forcing the mapping back on must not be silent"


def test_opt_out_still_honored_when_the_mapping_leaves_eos_alone():
    # A token_mapping that does not touch the tokenizer's eos piece has no such
    # constraint, so the caller's choice stands.
    resolved, messages = _resolve(
        map_eos_token = False,
        yes_map_eos_token = True,
        token_mapping = {"<start_of_turn>": "<|im_start|>"},
        eos_token = "<eos>",
    )
    assert resolved is False
    assert not messages


def test_shipped_templates_still_have_the_shape_the_guard_keys_on():
    """The guard matches on `tokenizer.eos_token in token_mapping`, not on template names.

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
