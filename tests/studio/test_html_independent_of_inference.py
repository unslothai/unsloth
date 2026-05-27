# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
Tests that the HTML extraction path in
``core.chat.document_extractor._extract_html`` does not depend on the
inference backend.

Failure mode the test pins:
    HTML extraction lives in ``core.chat`` and uses the stdlib-only
    ``_html_to_md`` converter to strip ``<script>``/``<style>`` and
    produce Markdown. The PR's first cut imported it via
    ``from core.inference._html_to_md import html_to_markdown``, which
    triggers ``core.inference/__init__.py`` and -- before the lazy
    PEP-562 patch -- pulled in the entire orchestrator + llama-server
    backend.

    In an extraction-only environment (CI without inference extras,
    a Studio install with a broken transformers, a partial
    package) any failure inside that eager import chain would be
    swallowed by the ``except Exception`` fallback in ``_extract_html``
    and the user would get *raw HTML with scripts/styles spliced into
    the prompt*.

After the patch, ``core.inference.__init__`` uses ``__getattr__`` for
lazy resolution. Importing the stdlib-only ``_html_to_md`` no longer
drags in the orchestrator. We assert:

    1. ``import core.inference`` does NOT eagerly load
       ``core.inference.orchestrator`` or ``core.inference.llama_cpp``.
    2. ``_extract_html`` strips ``<script>``/``<style>`` *even when*
       ``core.inference.orchestrator`` is poisoned in ``sys.modules``
       so that any eager import would raise.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parents[2] / "studio" / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


_HEADER = "import sys, importlib\n" f"sys.path.insert(0, {str(_BACKEND)!r})\n"


def _run_subprocess(body: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh Python so module purges don't pollute
    the parent pytest process. Returns the CompletedProcess for the
    caller to assert against ``stdout`` / ``returncode``."""
    return subprocess.run(
        [sys.executable, "-c", _HEADER + body],
        capture_output = True,
        text = True,
        timeout = 60,
    )


def test_importing_core_inference_does_not_eager_load_orchestrator():
    """Importing the package alone must NOT pull in the orchestrator
    or the llama-server backend -- if it does, every consumer that
    only wants ``core.inference._html_to_md`` has to drag in the
    entire inference stack."""
    body = (
        "import core.inference\n"
        "loaded = sorted(n for n in sys.modules\n"
        "                 if n.startswith('core.inference'))\n"
        "print(','.join(loaded))\n"
    )
    proc = _run_subprocess(body)
    assert proc.returncode == 0, proc.stderr
    loaded = set(proc.stdout.strip().split(","))
    assert (
        "core.inference.orchestrator" not in loaded
    ), f"core.inference eagerly imported .orchestrator -- loaded={loaded}"
    assert (
        "core.inference.llama_cpp" not in loaded
    ), f"core.inference eagerly imported .llama_cpp -- loaded={loaded}"


def test_html_extraction_strips_scripts_when_inference_is_broken():
    """The smoking-gun: extract a dirty HTML payload while the
    inference orchestrator/llama_cpp imports are poisoned. If the
    HTML path is properly decoupled, the result is sanitized
    Markdown; if it falls back to the silent-raw-HTML branch, the
    ``<script>`` content survives into the prompt."""
    body = (
        "sys.modules['core.inference.orchestrator'] = None\n"
        "sys.modules['core.inference.llama_cpp'] = None\n"
        "from core.chat import document_extractor as mod\n"
        'dirty = (b"<html><head><style>body{display:none}</style>"\n'
        "         b\"<script>alert('xss')</script></head>\"\n"
        '         b"<body><h1>hello</h1></body></html>")\n'
        "out, *_rest = mod._extract_html(dirty)\n"
        "import json\n"
        "print(json.dumps({'out': out}))\n"
    )
    proc = _run_subprocess(body)
    assert proc.returncode == 0, proc.stderr

    import json

    parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    out = parsed["out"]
    # Pre-fix this returns the raw HTML because the fallback branch
    # in _extract_html swallows the ImportError.
    assert (
        "alert" not in out
    ), f"<script>alert(...)</script> survived into the prompt; raw output:\n{out}"
    assert "<script" not in out.lower()
    assert "<style" not in out.lower()
    assert "hello" in out
