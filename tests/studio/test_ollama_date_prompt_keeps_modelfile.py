"""A date-only system turn replaces Ollama's Modelfile SYSTEM (#10436).

``_prepend_current_date_to_messages`` used to invent ``{role: system, content: date}``
when the payload had no system turn. The chat adapter only unshifts a system turn when
``combinedSystemPrompt`` is non-empty, so the date setting was enough to put one on the
wire. For ``preserve_server_system=True`` the date goes on the first user turn instead.

inference.py cannot be imported here (no fastapi in this venv), so the two functions are
ast-extracted and exec'd against ``utils.current_date_prompt_settings``.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"
INFERENCE = BACKEND / "routes" / "inference.py"
SETTINGS = BACKEND / "utils" / "current_date_prompt_settings.py"
DATE_LINE = "The current date is 2026-08-15."

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def _extract_func(source: str, tree: ast.AST, name: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(source, node)
            assert segment is not None
            return segment
    raise AssertionError(f"{name} not found")


def _load_prepend(date_line: str = DATE_LINE):
    from utils import current_date_prompt_settings as settings

    inference_src = INFERENCE.read_text(encoding = "utf-8")
    inference_tree = ast.parse(inference_src)
    settings_src = SETTINGS.read_text(encoding = "utf-8")
    settings_tree = ast.parse(settings_src)
    namespace = {
        "Any": Any,
        "current_date_prompt_line": lambda **_kwargs: date_line,
        "contains_current_date_prompt_line": settings.contains_current_date_prompt_line,
        "replace_current_date_prompt_lines": settings.replace_current_date_prompt_lines,
        "attach_date_line_to_first_user_message": lambda messages, _date_line: messages,
    }
    try:
        exec(
            compile(
                _extract_func(
                    settings_src, settings_tree, "attach_date_line_to_first_user_message"
                ),
                "attach_date_line_to_first_user_message",
                "exec",
            ),
            namespace,
        )
    except AssertionError:
        pass
    for name in ("_refresh_stated_date", "_prepend_current_date_to_messages"):
        exec(
            compile(_extract_func(inference_src, inference_tree, name), name, "exec"),
            namespace,
        )
    return namespace["_prepend_current_date_to_messages"]


def _prepend_preserving(messages):
    prepend = _load_prepend()
    assert "preserve_server_system" in inspect.signature(prepend).parameters
    return prepend(messages, preserve_server_system = True)


def test_ollama_without_a_system_turn_puts_the_date_on_the_user():
    messages = [{"role": "user", "content": "hi"}]
    out = _prepend_preserving(messages)
    assert [msg["role"] for msg in out] == ["user"]
    assert out[0]["content"] == f"{DATE_LINE}\n\nhi"
    assert messages == [{"role": "user", "content": "hi"}]


def test_ollama_still_prefixes_a_user_supplied_system_turn():
    out = _prepend_preserving(
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}],
    )
    assert out[0]["content"] == f"{DATE_LINE}\n\nBe terse."
    assert out[1] == {"role": "user", "content": "hi"}


def test_other_providers_still_invent_a_system_turn():
    prepend = _load_prepend()
    out = prepend([{"role": "user", "content": "hi"}])
    assert out[0] == {"role": "system", "content": DATE_LINE}
    assert out[1] == {"role": "user", "content": "hi"}


def test_ollama_prefixes_structured_user_text():
    out = _prepend_preserving(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png,x"}},
                    {"type": "text", "text": "what is this"},
                ],
            }
        ],
    )
    assert out[0]["role"] == "user"
    assert out[0]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png,x"},
    }
    assert out[0]["content"][1]["text"] == f"{DATE_LINE}\n\nwhat is this"


def test_ollama_without_a_user_turn_does_not_invent_a_system_turn():
    out = _prepend_preserving([{"role": "assistant", "content": "ready"}])
    assert out == [{"role": "assistant", "content": "ready"}]
