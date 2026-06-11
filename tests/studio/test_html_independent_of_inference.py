# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""HTML extraction (_extract_html) must not depend on the inference backend.

Pinned regression: importing the stdlib-only ``_html_to_md`` through
``core.inference`` eagerly pulled in the orchestrator + llama-server; in an
extraction-only env any failure there was swallowed by _extract_html's
fallback and raw HTML (scripts/styles included) was spliced into the prompt.
core.inference now resolves lazily (PEP 562). Assert: (1) importing
core.inference does not load orchestrator/llama_cpp; (2) _extract_html still
strips script/style even with core.inference.orchestrator poisoned.
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


_HEADER = f"import sys, importlib\nsys.path.insert(0, {str(_BACKEND)!r})\n"


def _run_subprocess(body: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh Python so module purges don't pollute the
    pytest process; returns the CompletedProcess for the caller to assert against ``stdout`` / ``returncode``."""
    return subprocess.run(
        [sys.executable, "-c", _HEADER + body],
        capture_output = True,
        text = True,
        timeout = 60,
    )


def test_importing_core_inference_does_not_eager_load_orchestrator():
    """Importing the package must NOT pull in the orchestrator or the
    llama-server backend -- otherwise every consumer that only wants ``core.inference._html_to_md`` has to drag in the
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
    """Extract dirty HTML while orchestrator/llama_cpp imports are
    poisoned. If the HTML path is properly decoupled, the result is sanitized
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
    # Pre-fix: raw HTML came back because the fallback swallowed the
    # ImportError.
    assert (
        "alert" not in out
    ), f"<script>alert(...)</script> survived into the prompt; raw output:\n{out}"
    assert "<script" not in out.lower()
    assert "<style" not in out.lower()
    assert "hello" in out
