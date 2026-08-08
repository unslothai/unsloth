# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every spawned interpreter that talks HTTPS activates the OS trust store.

truststore.inject_into_ssl() is process-local and does not survive a spawn, so a
missing call is invisible until someone behind a TLS-inspecting proxy hits that
one code path. These are static checks (no import, no network): deleting an
activation, or letting a probe's fetch drift ahead of its gating, fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent

# Modules whose fresh interpreter must activate at import, before any Hub call.
_ENTRYPOINTS = [
    "main.py",
    "hub/workers/hf_download.py",
    "core/inference/stt_download_worker.py",
    "core/inference/worker.py",
    "core/export/worker.py",
    "core/training/worker.py",
    "core/data_recipe/jobs/worker.py",
]

# `python -c` children cannot import backend modules, so they carry an inline
# copy of the gating. Named by the function or assignment that builds the script.
_INLINE_SCRIPTS = [
    ("utils/transformers_version.py", "_PROBE_CONFIG_SCRIPT"),
    ("utils/models/model_config.py", "_build_vision_check_script"),
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


def _script_text(tree, name):
    """Concatenate, in source order, the string literals that build a `-c` script."""
    for node in ast.walk(tree):
        named = (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == name for t in node.targets)
        ) or (isinstance(node, ast.FunctionDef) and node.name == name)
        if named:
            parts = sorted(
                (
                    (child.lineno, child.col_offset, child.value)
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant) and isinstance(child.value, str)
                ),
            )
            return "\n".join(part[2] for part in parts)
    raise AssertionError(f"{name} not found")


@pytest.mark.parametrize("relative", _ENTRYPOINTS)
def test_entrypoint_activates_native_tls_at_module_level(relative):
    tree = ast.parse((_BACKEND / relative).read_text(encoding = "utf-8"))
    assert "activate_native_tls" in _import_time_calls(
        tree.body
    ), f"{relative} spawns a fresh interpreter but never calls activate_native_tls()"


@pytest.mark.parametrize(("relative", "name"), _INLINE_SCRIPTS)
def test_probe_script_injects_before_it_downloads(relative, name):
    script = _script_text(ast.parse((_BACKEND / relative).read_text(encoding = "utf-8")), name)
    # Either the shared helper (when the child can reach the backend dir) or an
    # inline copy of it, which then has to honour the opt-out on its own.
    activate = max(script.find("activate_native_tls"), script.find("inject_into_ssl"))
    assert activate != -1, f"{name} lost its native TLS activation"
    if "inject_into_ssl" in script:
        assert (
            "UNSLOTH_STUDIO_NATIVE_TLS" in script
        ), f"{name} injects without honouring the opt-out"
    assert activate < script.find(".from_pretrained("), f"{name} downloads before activating"


def test_backend_serves_no_tls_in_process():
    """truststore's injection is client-side: a context built after it cannot serve TLS.

    Studio serves plain HTTP on loopback, so this never bites, but an in-process
    HTTPS listener added later would fail at handshake on macOS and Windows,
    where activation is default-on. Catch it here instead.
    """
    server_side = ("PROTOCOL_TLS_SERVER", "ssl_certfile", "ssl_keyfile")
    offenders = []
    for path in _BACKEND.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding = "utf-8", errors = "ignore")
        if any(marker in text for marker in server_side):
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
