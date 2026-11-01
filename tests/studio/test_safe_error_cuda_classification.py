# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the client-safe mapper without importing the model or CUDA stack."""

import ast
import os
from pathlib import Path
from types import ModuleType
from typing import Union
from unittest.mock import patch

import pytest


OOM = "Ran out of memory. Try a smaller model or shorter input."


class _DeviceAssertMappedAsOOM(AssertionError):
    pass


class _FriendlyStreamError(RuntimeError):
    friendly = "The model server stopped generating."


class _ContextRefusal(RuntimeError):
    pass


def _source():
    configured = os.environ.get("UNSLOTH_REPO_ROOT")
    roots = [Path(configured)] if configured else []
    roots.extend(Path(__file__).resolve().parents)
    for root in roots:
        source = root / "studio/backend/utils/utils.py"
        if source.is_file():
            return source
    pytest.skip("Could not locate studio/backend/utils/utils.py")


@pytest.fixture
def mapper():
    tree = ast.parse(_source().read_text(encoding = "utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"safe_error_detail", "is_metal_queue_dead"}
    ]
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_METAL_QUEUE_DEAD_MARKERS"
            for target in node.targets
        )
    ]
    assert len(definitions) == 2 and len(constants) == 1
    namespace = {"Union": Union}
    exec(
        compile(
            ast.Module(body = constants + definitions, type_ignores = []), "<safe-error-mapper>", "exec"
        ),
        namespace,
    )
    stream = ModuleType("core.inference.stream_errors")
    stream.LlamaStreamError = _FriendlyStreamError
    context = ModuleType("core.inference.context_refusal")
    context.ContextBudgetExceeded = _ContextRefusal
    with patch.dict("sys.modules", {stream.__name__: stream, context.__name__: context}):
        yield namespace["safe_error_detail"]


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory",
        "Unable to allocate 1024 bytes",
        "Insufficient Memory",
        "Out of memory",
    ],
)
def test_explicit_memory_errors_keep_memory_guidance(mapper, message):
    assert mapper(RuntimeError(message)) == OOM


def test_connection_refusal_is_not_a_memory_error(mapper):
    assert mapper(ConnectionError("synthetic refusal")) == (
        "Could not reach an upstream service. Please try again."
    )


def test_unknown_error_keeps_fallback_and_hides_internal_path(mapper):
    error = RuntimeError(r"synthetic failure in C:\private\example.py")
    assert mapper(error, "Generation failed.") == "Generation failed."


def test_dead_metal_queue_takes_priority_over_memory_marker(mapper):
    error = RuntimeError("GPU Timeout Error: Insufficient Memory")
    assert mapper(error) == "The GPU stopped responding. Reload the model to recover."


def test_curated_stream_error_preserves_its_friendly_message(mapper):
    assert mapper(_FriendlyStreamError("synthetic internal detail")) == (
        _FriendlyStreamError.friendly
    )


def test_typed_context_refusal_preserves_its_public_message(mapper):
    assert mapper(_ContextRefusal("Request exceeds the context budget.")) == (
        "Request exceeds the context budget."
    )


@pytest.mark.xfail(
    strict = True,
    raises = _DeviceAssertMappedAsOOM,
    reason = "Issue #13468: a non-OOM CUDA assertion receives memory guidance",
)
def test_device_side_assertion_is_not_reported_as_out_of_memory(mapper):
    error = RuntimeError("CUDA error: device-side assert triggered")
    message = mapper(error)
    assert message != str(error)
    if message == OOM:
        raise _DeviceAssertMappedAsOOM("A CUDA assertion was labeled as out-of-memory")
