"""Regression test for the map_eos_token argument of get_chat_template.

get_chat_template accepts a public, type-asserted `map_eos_token` argument, but on
the string-template path it was overwritten by the template's own flag, so an
explicit map_eos_token = False was silently ignored for every template that sets
yes_map_eos_token = True (chatml, gemma, gemma_chatml, gemma2, gemma2_chatml).

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


def _resolution_statements():
    """The `if ...: map_eos_token = ...` statements inside get_chat_template, in source order."""
    tree = ast.parse(open(CHAT_TEMPLATES_PATH, encoding = "utf-8").read())
    func = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_chat_template"
    )
    statements = [
        node for node in ast.walk(func)
        if isinstance(node, ast.If) and any(
            isinstance(stmt, ast.Assign)
            and any(getattr(target, "id", None) == "map_eos_token" for target in stmt.targets)
            for stmt in node.body
        )
    ]
    assert statements, "could not find the map_eos_token resolution in get_chat_template"
    return sorted(statements, key = lambda node: node.lineno)


def _resolve(map_eos_token, yes_map_eos_token):
    module = ast.Module(body = _resolution_statements(), type_ignores = [])
    namespace = {"map_eos_token": map_eos_token, "yes_map_eos_token": yes_map_eos_token}
    exec(compile(module, CHAT_TEMPLATES_PATH, "exec"), namespace)
    return namespace["map_eos_token"]


def test_explicit_map_eos_token_false_is_honored():
    # A template asking for eos mapping must not override an explicit opt-out.
    assert _resolve(map_eos_token = False, yes_map_eos_token = True) is False


def test_other_map_eos_token_combinations_are_unchanged():
    # The default is map_eos_token = True, so these three paths must not move.
    assert _resolve(map_eos_token = True, yes_map_eos_token = True) is True
    # A template that does not use eos mapping still vetoes it.
    assert _resolve(map_eos_token = True, yes_map_eos_token = False) is False
    assert _resolve(map_eos_token = False, yes_map_eos_token = False) is False
