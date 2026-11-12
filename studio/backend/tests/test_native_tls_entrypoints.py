# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Injection is process-local and lost on spawn, so every spawned HTTPS interpreter must activate it."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent

_ENTRYPOINTS = [
    "main.py",
    "hub/workers/hf_download.py",
    "core/inference/stt_download_worker.py",
    "core/inference/worker.py",
    "core/export/worker.py",
    "core/training/worker.py",
    "core/data_recipe/jobs/worker.py",
]

# Read the assembled script: it is concatenated, so AST literal scraping misses parts.
_PROBE_SCRIPTS = [
    ("utils.transformers_version", "_PROBE_CONFIG_SCRIPT"),
    ("utils.models.model_config", "_VISION_CHECK_SCRIPT"),
]


def _import_time_calls(body):
    """Call names reachable at import: module level, including if/try/with bodies."""
    names = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            names.add(getattr(func, "id", None) or getattr(func, "attr", None))
        for field in ("body", "orelse", "finalbody"):
            names |= _import_time_calls(getattr(node, field, []) or [])
        for handler in getattr(node, "handlers", []) or []:
            names |= _import_time_calls(handler.body)
    return names


@pytest.mark.parametrize("relative", _ENTRYPOINTS)
def test_entrypoint_activates_native_tls_at_module_level(relative):
    tree = ast.parse((_BACKEND / relative).read_text(encoding = "utf-8"))
    assert "activate_native_tls" in _import_time_calls(
        tree.body
    ), f"{relative} spawns a fresh interpreter but never calls activate_native_tls()"


@pytest.mark.parametrize(("module", "attr"), _PROBE_SCRIPTS)
def test_probe_script_activates_before_it_downloads(module, attr):
    script = getattr(importlib.import_module(module), attr)
    ast.parse(script)
    activate = max(script.find("activate_native_tls"), script.find("inject_into_ssl"))
    assert activate != -1, f"{attr} lost its native TLS activation"
    if "inject_into_ssl" in script:
        assert "UNSLOTH_STUDIO_NATIVE_TLS" in script, f"{attr} injects without the opt-out"
    assert activate < script.find(".from_pretrained("), f"{attr} downloads before activating"


def test_prebuilt_core_gate_matches_the_generated_source():
    """prebuilt_core.py's gate is a pasted copy, and drift is silent, so compare parsed AST, not text."""
    from utils.native_tls import inline_gate_source

    source = (_BACKEND.parent / "prebuilt_core.py").read_text(encoding = "utf-8")
    gate = [ast.dump(node) for node in ast.parse(inline_gate_source()).body]
    body = [ast.dump(node) for node in ast.parse(source).body]
    assert any(body[i : i + len(gate)] == gate for i in range(len(body) - len(gate) + 1)), (
        "prebuilt_core.py's gate has drifted from native_tls.inline_gate_source(); "
        "paste the current output of that function over it"
    )


# Builds the server context with truststore's peer-chain hook removed; test_studio_tls.py proves the handshake.
_TLS_SERVER_MODULE = Path("utils") / "studio_tls.py"
_UVICORN_SERVERS = {"Config", "run"}


def _builds_tls_server(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in (
            "PROTOCOL_TLS_SERVER",
            "load_cert_chain",
        ):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in _UVICORN_SERVERS and any(
                kw.arg in ("ssl_certfile", "ssl_keyfile") for kw in node.keywords
            ):
                return True
    return False


def test_backend_serves_no_tls_in_process():
    """In-process HTTPS listeners must use utils/studio_tls.py; truststore only affects clients."""
    offenders = []
    for path in _BACKEND.rglob("*.py"):
        if "tests" in path.parts or "vendor" in path.parts:
            continue
        if path.relative_to(_BACKEND) == _TLS_SERVER_MODULE:
            continue
        text = path.read_text(encoding = "utf-8", errors = "ignore")
        if _builds_tls_server(ast.parse(text)):
            offenders.append(str(path.relative_to(_BACKEND)))
    assert not offenders, (
        "in-process TLS server found, which the native TLS injection breaks: "
        + ", ".join(offenders)
    )


def test_prebuilt_installer_core_injects_at_import():
    """The llama.cpp / whisper.cpp installers are vendored standalone: no backend import."""
    source = (_BACKEND.parent / "prebuilt_core.py").read_text(encoding = "utf-8")
    assert "UNSLOTH_STUDIO_NATIVE_TLS" in source
    assert "inject_into_ssl" in _import_time_calls(ast.parse(source).body)
