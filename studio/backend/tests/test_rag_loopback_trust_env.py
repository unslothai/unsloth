"""AST test locking in the RAG loopback trust_env fix: every httpx client/call in the RAG
package (all target the local 127.0.0.1 llama-server) must set trust_env=False."""
import ast
import os

RAG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "rag")
HTTPX_CALLEES = {"get", "post", "stream", "request", "Client", "AsyncClient"}


def _httpx_calls(path):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # match httpx.<callee>(...)
        if (
            isinstance(func, ast.Attribute)
            and func.attr in HTTPX_CALLEES
            and isinstance(func.value, ast.Name)
            and func.value.id == "httpx"
        ):
            calls.append(node)
    return calls


def _sets_trust_env_false(call):
    for kw in call.keywords:
        if kw.arg == "trust_env" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
            return True
    return False


def test_rag_loopback_httpx_clients_disable_trust_env():
    checked = 0
    for fname in ("embed_llama_server.py", "captioner.py"):
        path = os.path.join(RAG_DIR, fname)
        assert os.path.exists(path), f"missing {path}"
        calls = _httpx_calls(path)
        assert calls, f"expected httpx calls in {fname}"
        for call in calls:
            checked += 1
            assert _sets_trust_env_false(call), (
                f"httpx.{call.func.attr} at {fname}:{call.lineno} must set trust_env=False "
                f"(loopback llama-server client must not honor ambient HTTP(S)_PROXY)"
            )
    assert checked >= 3, f"expected at least 3 loopback httpx calls, found {checked}"


if __name__ == "__main__":
    test_rag_loopback_httpx_clients_disable_trust_env()
    print("OK: all RAG loopback httpx clients set trust_env=False")
