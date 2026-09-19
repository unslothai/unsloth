"""A merged prompt may contain literal braces.

`to_sharegpt(dataset, merged_prompt = ...)` renders the prompt with `str.format`, where a
literal brace is written `{{` or `}}`. The column scan was a plain `re.findall(r"\\{(.+?)\\}")`
over the raw prompt, so it read the inside of an escaped pair as a column name: a prompt that
asks the model to answer in JSON reported its own text as a missing dataset column.

    merged_prompt = 'Task: {instruction}\\nReply as {{"answer": <number>}}'
    KeyError: Unsloth: Your prompt includes '"answer": <number>' but this does not exist in
    the dataset.

Importing unsloth needs a GPU, so the three functions are pulled out of the source with ast and
run over stand-ins, as tests/test_construct_chat_template_processor.py does.
"""

import ast
import os
import re
import types

import pytest

CHAT_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "chat_templates.py",
)

_WANTED = ("_ESCAPED_BRACES_RE", "_column_names_in", "_parse_combined_prompt", "_create_formatter")


def _load():
    tree = ast.parse(open(CHAT_TEMPLATES_PATH, encoding = "utf-8").read())
    namespace = {"re": re}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANTED:
            exec(
                compile(ast.Module(body = [node], type_ignores = []), CHAT_TEMPLATES_PATH, "exec"),
                namespace,
            )
        elif isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) in _WANTED for target in node.targets
        ):
            exec(
                compile(ast.Module(body = [node], type_ignores = []), CHAT_TEMPLATES_PATH, "exec"),
                namespace,
            )
    for name in ("_parse_combined_prompt", "_create_formatter"):
        if name not in namespace:
            pytest.skip(f"{name} not found in unsloth/chat_templates.py")
    return namespace


def _render(
    prompt,
    rows,
    columns = ("instruction", "input"),
):
    ns = _load()
    dataset = types.SimpleNamespace(column_names = list(columns))
    possible_columns, optional_prompts = ns["_parse_combined_prompt"](prompt, dataset)
    formatter = ns["_create_formatter"](possible_columns, optional_prompts, "text")
    return formatter(rows)["text"]


ROWS = {"instruction": ["Sum 1 and 2"], "input": ["1 2"]}


def test_escaped_braces_are_not_column_names():
    assert _render('Task: {instruction}\nReply as {{"answer": <number>}}', ROWS) == [
        'Task: Sum 1 and 2\nReply as {"answer": <number>}'
    ]


def test_escaped_braces_beside_an_optional_block():
    assert _render('{instruction}[[ / {input}]]\nFormat: {{"a": 1}}', ROWS) == [
        'Sum 1 and 2 / 1 2\nFormat: {"a": 1}'
    ]


def test_escaped_braces_wrapping_a_real_column():
    # `{{` + `{input}` + `}}`: the escapes are literal, the inner pair is still a column.
    assert _render("{instruction} -> {{{input}}}", ROWS) == ["Sum 1 and 2 -> {1 2}"]


def test_empty_escaped_pair():
    assert _render("Set is {{}} and task is {instruction}", ROWS) == [
        "Set is {} and task is Sum 1 and 2"
    ]


def test_a_real_missing_column_is_still_rejected():
    # The control: blanking escaped pairs must not blunt the error for a genuine typo.
    with pytest.raises(KeyError, match = "instrution"):
        _render("Task: {instrution}", ROWS)
