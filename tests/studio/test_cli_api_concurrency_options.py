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
