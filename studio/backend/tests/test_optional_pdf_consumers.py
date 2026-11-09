# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Consumers outside RAG parsing reach PDF support through the first-use installer, not a bare import."""

from email.message import Message
import io
import subprocess
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
import core.chat_pdf as chat_pdf
from core.inference import tools
import routes.chat_export as chat_export
from utils.optional_pdf import INSTALL_COMMAND, PDFDependencyUnavailable

_GUIDANCE = f"PDF support is not installed and automatic downloads are disabled. Install it with: {INSTALL_COMMAND}"


def _missing(*_args, **_kwargs):
    raise PDFDependencyUnavailable(_GUIDANCE)


def test_chat_pdf_child_reports_missing_support_with_its_own_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(chat_pdf, "load_pdf_module", _missing)
    monkeypatch.setattr(
        chat_pdf.sys, "stdin", SimpleNamespace(buffer = io.BytesIO(b'{"html": "<p>x</p>"}'))
    )
    assert chat_pdf.main() == chat_pdf.MISSING_PDF_SUPPORT_EXIT_CODE
    assert INSTALL_COMMAND in capsys.readouterr().err


def test_chat_export_returns_install_guidance_as_503(monkeypatch):
    def child(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], chat_pdf.MISSING_PDF_SUPPORT_EXIT_CODE, b"", f"log line\n{_GUIDANCE}\n".encode()
        )

    monkeypatch.setattr(chat_export.subprocess, "run", child)
    app = FastAPI()
    app.include_router(chat_export.router, prefix = "/api/chat-export")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    response = TestClient(app).post("/api/chat-export/pdf", json = {"title": "x", "html": "<p>x</p>"})
    assert response.status_code == 503
    assert response.json()["detail"] == _GUIDANCE


@pytest.mark.parametrize(
    "code, expected",
    [
        ("import fitz\nprint(fitz.open('a.pdf').page_count)", True),
        ("import os, pymupdf", True),
        ("from pymupdf4llm import to_markdown", True),
        ("from fitz.utils import getColor", True),
        ("import pandas as pd\nprint(pd.__version__)", False),
        ("fitz = 1\nprint('import fitz')", False),
        ("from . import fitz", False),
        ("import fitz(", False),
    ],
)
def test_python_tool_detects_pdf_parser_imports(code, expected):
    assert tools._imports_pdf_parser(code) is expected


def test_python_tool_returns_install_guidance_before_spawning(monkeypatch):
    monkeypatch.setattr(tools, "ensure_pdf_support", _missing)
    monkeypatch.setattr(
        tools,
        "_get_workdir",
        lambda *_a: pytest.fail("must not start the sandbox without PDF support"),
    )
    result = tools._python_exec("import fitz\nprint(fitz.__doc__)", session_id = "pdf-check")
    assert result.startswith("Execution error: ")
    assert INSTALL_COMMAND in result


def test_python_tool_skips_the_installer_for_other_code(monkeypatch):
    monkeypatch.setattr(tools, "ensure_pdf_support", lambda: pytest.fail("no PDF import"))

    class Stop(Exception):
        pass

    def stop(*_args):
        raise Stop

    monkeypatch.setattr(tools, "_get_workdir", stop)
    with pytest.raises(Stop):
        tools._python_exec("print(1 + 1)", session_id = "pdf-check")


class _PdfResponse:
    def __init__(self):
        self._body = b"%PDF-1.4\n%%EOF\n"
        self.headers = Message()
        self.headers["Content-Type"] = "application/pdf"

    def read(self, n = None):
        chunk, self._body = (self._body, b"") if n is None else (self._body[:n], self._body[n:])
        return chunk


def _fetch_pdf(monkeypatch):
    monkeypatch.setattr(
        tools, "_validate_and_resolve_host", lambda host, port: (True, "", ["93.184.216.34"])
    )
    opener = type("Opener", (), {"open": lambda self, req, timeout = None: _PdfResponse()})()
    monkeypatch.setattr(tools.urllib.request, "build_opener", lambda *a, **k: opener)
    return tools._fetch_page_text("https://example.com/paper.pdf", timeout = 60)


def test_web_pdf_read_keeps_install_guidance(monkeypatch):
    monkeypatch.setattr(tools, "ensure_pdf_support", _missing)
    out = _fetch_pdf(monkeypatch)
    assert out.startswith("(PDF content could not be read as text: ")
    assert INSTALL_COMMAND in out


def test_web_pdf_first_use_install_is_not_charged_to_the_read(monkeypatch):
    clock = {"time": 1000.0}
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock["time"])

    def slow_install():
        clock["time"] += 90.0

    monkeypatch.setattr(tools, "ensure_pdf_support", slow_install)
    monkeypatch.setattr(tools, "_extract_pdf_text", lambda data: "## Page 1\n\nreadable")
    out = _fetch_pdf(monkeypatch)
    assert "readable" in out
    assert "timed out" not in out
