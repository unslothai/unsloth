"""
Wiring tests for the per-run cancel_id field.

A chat-thread-scoped session_id is not safe as a cancel key because a
late stop POST can match a subsequent run on the same thread. The fix
adds cancel_id (a fresh UUID per generation) that is sent both in the
completion payload and in the /api/inference/cancel body.

Verifies:
  - ChatCompletionRequest exposes an Optional[str] `cancel_id` field.
  - /api/inference/cancel accepts `cancel_id` as the first-preferred key.
  - OpenAIChatCompletionsRequest (frontend type) includes cancel_id.
  - chat-adapter.ts generates a per-run cancelId (crypto.randomUUID
    with a Math.random fallback), sends it in the completion payload,
    and includes it in the /inference/cancel body on abort.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
MODELS_SRC = (WORKSPACE / "studio/backend/models/inference.py").read_text()
ROUTES_SRC = (WORKSPACE / "studio/backend/routes/inference.py").read_text()
ADAPTER_SRC = (
    WORKSPACE / "studio/frontend/src/features/chat/api/chat-adapter.ts"
).read_text()
API_TYPES_SRC = (
    WORKSPACE / "studio/frontend/src/features/chat/types/api.ts"
).read_text()


def _find_class(tree: ast.AST, name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def test_chat_completion_request_has_cancel_id_field():
    tree = ast.parse(MODELS_SRC)
    cls = _find_class(tree, "ChatCompletionRequest")
    assert cls is not None
    fields = {
        n.target.id
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }
    assert "cancel_id" in fields, (
        "ChatCompletionRequest must expose a cancel_id field for per-run "
        "cancellation routing"
    )


def test_cancel_route_accepts_cancel_id_as_first_key():
    # The cancel endpoint must try cancel_id first; session_id/completion_id
    # are broader / fallback keys.
    for node in ast.walk(ast.parse(ROUTES_SRC)):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == "cancel_inference"
        ):
            break
    else:
        raise AssertionError("cancel_inference handler missing")

    # Find `for k in (...)` tuple with string literals.
    found_cancel_id_first = False
    for sub in ast.walk(node):
        if not isinstance(sub, ast.For):
            continue
        it = sub.iter
        if not isinstance(it, ast.Tuple):
            continue
        vals = [e.value for e in it.elts if isinstance(e, ast.Constant)]
        if vals and vals[0] == "cancel_id":
            found_cancel_id_first = True
            assert "session_id" in vals or "completion_id" in vals
            break
    assert found_cancel_id_first, (
        "cancel_inference must iterate ('cancel_id', ...) so a per-run "
        "cancel id takes precedence over thread-scoped keys"
    )


def test_frontend_request_type_has_cancel_id():
    assert re.search(r"cancel_id\?\s*:\s*string\s*;", API_TYPES_SRC), (
        "OpenAIChatCompletionsRequest must expose an optional cancel_id"
    )


def test_chat_adapter_generates_cancel_id_per_run():
    m = re.search(
        r"const\s+cancelId\s*=\s*([^;]+);",
        ADAPTER_SRC,
    )
    assert m, "chat-adapter.ts must declare a per-run `cancelId` constant"
    rhs = m.group(1)
    assert "randomUUID" in rhs, (
        "cancelId should prefer crypto.randomUUID() for uniqueness"
    )


def test_chat_adapter_sends_cancel_id_in_completion_payload():
    assert "cancel_id: cancelId" in ADAPTER_SRC, (
        "chat-adapter.ts must include cancel_id in the streamChatCompletions "
        "request payload so the backend registers under that key"
    )


def test_chat_adapter_sends_cancel_id_in_abort_cancel_post():
    m = re.search(
        r"const\s+onAbortCancel\s*=\s*\(\)\s*=>\s*\{(.*?)\};",
        ADAPTER_SRC,
        flags = re.DOTALL,
    )
    assert m, "onAbortCancel arrow function missing"
    body = m.group(1)
    assert re.search(r"cancel_id\s*:\s*cancelId", body), (
        "onAbortCancel must include cancel_id in the /inference/cancel body "
        "so a stop POST matches the specific run, not the whole thread"
    )
