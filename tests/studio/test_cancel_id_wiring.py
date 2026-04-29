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


def test_cancel_route_matches_cancel_id_exclusively_when_present():
    # A stale cancel POST carrying cancel_id AND session_id must not
    # cancel a later run on the same thread via the shared session_id.
    # Enforce this by requiring the handler to early-return through an
    # exclusive-cancel_id path -- either an atomic helper or a keys
    # list containing ONLY cancel_id (never session_id).
    for node in ast.walk(ast.parse(ROUTES_SRC)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "cancel_inference":
            break
    else:
        raise AssertionError("cancel_inference handler missing")

    cancel_id_exclusive_branch = False
    for sub in ast.walk(node):
        if not isinstance(sub, ast.If):
            continue
        test_src = ast.unparse(sub.test)
        if "cancel_id" not in test_src or "isinstance" not in test_src:
            continue
        branch_src = "\n".join(ast.unparse(s) for s in sub.body)
        before_return = branch_src.split("return", 1)[0]
        matches_cancel_id_only = (
            "_cancel_by_cancel_id_or_stash(cancel_id)" in branch_src
            or "_cancel_by_keys([cancel_id])" in branch_src
        )
        if matches_cancel_id_only and "session_id" not in before_return:
            cancel_id_exclusive_branch = True
            break
    assert cancel_id_exclusive_branch, (
        "cancel_inference must early-return with an exclusive cancel_id "
        "match when a cancel_id is supplied, so a stale stop POST "
        "cannot cancel a later run on the same thread via session_id"
    )


def test_cancel_route_falls_back_to_session_or_completion_when_no_cancel_id():
    for node in ast.walk(ast.parse(ROUTES_SRC)):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "cancel_inference":
            break
    else:
        raise AssertionError("cancel_inference handler missing")

    src = ast.unparse(node)
    assert "session_id" in src and "completion_id" in src, (
        "cancel_inference must still accept session_id / completion_id as "
        "fallback keys when cancel_id is absent"
    )


def test_frontend_request_type_has_cancel_id():
    assert re.search(
        r"cancel_id\?\s*:\s*string\s*;", API_TYPES_SRC
    ), "OpenAIChatCompletionsRequest must expose an optional cancel_id"


def test_chat_adapter_generates_cancel_id_per_run():
    m = re.search(
        r"const\s+cancelId\s*=\s*([^;]+);",
        ADAPTER_SRC,
    )
    assert m, "chat-adapter.ts must declare a per-run `cancelId` constant"
    rhs = m.group(1)
    assert (
        "randomUUID" in rhs
    ), "cancelId should prefer crypto.randomUUID() for uniqueness"


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


def test_abort_cancel_post_uses_plain_fetch_with_manual_auth_header():
    # authFetch redirects to login on 401, which would kick the user to
    # the login page mid-stop if the access token expired during a long
    # stream. Use plain fetch + manual Authorization header for a
    # best-effort cancel that never triggers the refresh/redirect flow.
    start = ADAPTER_SRC.find("const onAbortCancel")
    assert start >= 0, "onAbortCancel handler missing"
    rest = ADAPTER_SRC[start:]
    end = rest.find("\n      try {")
    body = rest if end < 0 else rest[:end]
    assert "/api/inference/cancel" in body
    assert "authFetch(" not in body, (
        "onAbortCancel must NOT call authFetch; a 401 from it would "
        "redirect the user to the login page during a stop click"
    )
    assert "fetch(" in body, "onAbortCancel must use plain fetch(...)"
    assert "getAuthToken" in body, (
        "onAbortCancel must read the bearer token via getAuthToken() "
        "rather than relying on authFetch's 401 flow"
    )
    assert "Authorization" in body
    assert (
        "keepalive: true" in body
    ), "keepalive is required so the fetch survives page unload during stop"
