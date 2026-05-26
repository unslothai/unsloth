"""AST checks for Studio API concurrency CLI wiring."""

from __future__ import annotations

import ast
from pathlib import Path

_STUDIO_CLI = (
    Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands" / "studio.py"
)


def _source() -> str:
    return _STUDIO_CLI.read_text()


def test_studio_default_normalizes_queue_policy_before_reexec():
    source = _source()
    tree = ast.parse(source)
    studio_default = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "studio_default"
    )

    validation_line = None
    normalization_line = None
    forwarding_line = None
    for node in ast.walk(studio_default):
        if isinstance(node, ast.Compare) and "api_queue_policy.lower()" in ast.unparse(
            node
        ):
            validation_line = node.lineno
        if isinstance(node, ast.Assign) and ast.unparse(node).strip() == (
            "api_queue_policy = api_queue_policy.lower()"
        ):
            normalization_line = node.lineno
        if isinstance(node, ast.Call) and ast.unparse(node).strip() in {
            'args.extend(["--api-queue-policy", api_queue_policy])',
            "args.extend(['--api-queue-policy', api_queue_policy])",
        }:
            forwarding_line = node.lineno

    assert validation_line is not None
    assert normalization_line is not None
    assert forwarding_line is not None
    assert validation_line < normalization_line < forwarding_line


def test_studio_run_preserves_default_llama_parallel_slots():
    source = _source()
    assert "else 4" in source
    assert "UNSLOTH_API_MAX_CONCURRENCY" in source
    assert "api_concurrency_configured" in source


def test_studio_run_derives_concurrency_from_passthrough_parallel():
    """`unsloth studio run --parallel N` should set both llama slots and the
    API gate to N so they cannot disagree."""
    source = _source()
    assert "_passthrough_parallel" in source
    assert '"--parallel"' in source or "'--parallel'" in source
    assert '"-np"' in source or "'-np'" in source


_BACKEND_MAIN = Path(__file__).resolve().parents[2] / "studio" / "backend" / "main.py"


def test_body_protected_prefixes_cover_every_v1_generation_route():
    """Each /v1 path gated by InferenceConcurrencyMiddleware must also be in
    _BODY_PROTECTED_PREFIXES so MaxBody rejects oversize before the gate."""
    src = _BACKEND_MAIN.read_text()
    for prefix in (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/generate/stream",
        "/v1/audio/generate",
    ):
        assert f'"{prefix}"' in src, f"_BODY_PROTECTED_PREFIXES missing {prefix}"


_BACKEND_RUN = Path(__file__).resolve().parents[2] / "studio" / "backend" / "run.py"


def test_run_server_derives_llama_parallel_slots_from_api_max_concurrency():
    """run_server() must derive llama_parallel_slots from api_max_concurrency
    when the slot count is not explicitly passed."""
    src = _BACKEND_RUN.read_text()
    assert "llama_parallel_slots: int | None = None" in src
    assert "if llama_parallel_slots is None:" in src
    assert "effective_api_max_concurrency if api_concurrency_configured else 1" in src


def test_run_server_reconciles_already_registered_middleware():
    """run_server() must update an already-registered InferenceConcurrencyMiddleware
    so embedders that imported main first do not get stale kwargs."""
    src = _BACKEND_RUN.read_text()
    assert "mw.cls is InferenceConcurrencyMiddleware" in src
    assert 'mw.kwargs["max_concurrency"]' in src
    assert "app.middleware_stack = None" in src
