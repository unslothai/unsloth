# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import re
from pathlib import Path

import pytest
from fastapi import UploadFile

from core.rag import config, folder_sync, parsers
from routes.rag import _save_upload

_TEXT_ACCEPT_TS = (
    Path(__file__).resolve().parents[2] / "frontend/src/features/chat/text-attachment-accept.ts"
)


def test_source_exts_mirror_chat_text_attachments():
    source = _TEXT_ACCEPT_TS.read_text(encoding = "utf-8")
    block = re.search(r"TEXT_ATTACHMENT_EXTENSIONS = \[(.*?)\];", source, re.S).group(1)
    chat = set(re.findall(r'"(\.[^"]+)"', block))
    assert chat - config.SUPPORTED_UPLOAD_EXTS == config.SOURCE_TEXT_EXTS


@pytest.mark.parametrize(
    "name, content",
    [
        ("index.php", "<?php echo 'hello'; ?>"),
        ("Program.cs", "class Program { static void Main() {} }"),
        ("deploy.yaml", "apiVersion: v1\nkind: Pod"),
    ],
)
def test_parse_source_file(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding = "utf-8")
    assert [page.text for page in parsers.parse(str(path))] == [content]


def test_parse_source_file_strips_bom(tmp_path):
    path = tmp_path / "script.py"
    path.write_bytes(b"\xef\xbb\xbfdef test(): pass")
    assert [page.text for page in parsers.parse(str(path))] == ["def test(): pass"]


def test_parse_rejects_binary_under_source_extension(tmp_path):
    path = tmp_path / "module.mod"
    path.write_bytes(b"GFORTRAN\x00\x01\x02")
    with pytest.raises(ValueError, match = "unsupported binary content"):
        parsers.parse(str(path))


def test_scan_indexes_source_and_skips_dependencies_and_secrets(tmp_path):
    files = {
        "src/app.js": "a",
        "src/index.php": "a",
        "build/out.js": "a",
        "infra/env/main.tf": "a",
        "infra/env/terraform.tfstate": "a",
        "src/unsupported.exe": "a",
        "src/.env": "a",
        "src/.env.local": "a",
        "src/prod.env": "a",
        "src/package-lock.json": "a",
        "src/Cargo.lock": "a",
        "node_modules/dep.js": "a",
        ".git/config.json": "a",
        "pyenv/pyvenv.cfg": "a",
        "pyenv/lib/site.py": "a",
    }
    for rel, content in files.items():
        (tmp_path / rel).parent.mkdir(parents = True, exist_ok = True)
        (tmp_path / rel).write_text(content, encoding = "utf-8")
    found, _ = folder_sync._scan(str(tmp_path))
    assert {key.replace("\\", "/") for key in found} == {
        "src/app.js",
        "src/index.php",
        "build/out.js",
        "infra/env/main.tf",
    }


def test_upload_accepts_source_file(rag_home):
    upload = UploadFile(file = io.BytesIO(b"<?php echo 1; ?>"), filename = "index.php")
    _, filename, _ = _save_upload(upload)
    assert filename == "index.php"
