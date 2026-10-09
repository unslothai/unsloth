# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The llm-compressor-main shadow installs from a checksummed source archive, never git."""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from utils import transformers_version as tv


def _archive_bytes() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj = buf, mode = "w:gz") as tar:
        data = b"from setuptools import setup\n"
        info = tarfile.TarInfo(f"llm-compressor-{tv._LLMC_MAIN_SHA}/setup.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def shadow(monkeypatch, tmp_path):
    monkeypatch.setattr(tv, "_llmcompressor_shadow_is_valid", lambda: False)
    monkeypatch.setattr(tv, "_llmcompressor_main_disabled", lambda: False)
    monkeypatch.setattr(tv, "_env_offline", lambda: False)
    monkeypatch.setattr(tv, "_VENV_LLMCOMPRESSOR_DIR", str(tmp_path / "shadow"))
    return tmp_path


def _serve(
    monkeypatch,
    payload: bytes,
    digest: str | None = None,
):
    monkeypatch.setattr(
        tv, "_LLMC_MAIN_ARCHIVE_SHA256", digest or hashlib.sha256(payload).hexdigest()
    )
    urls = []

    def fake_urlopen(request, timeout = None):
        urls.append(request.full_url)
        return _Response(payload)

    monkeypatch.setattr(tv.urllib.request, "urlopen", fake_urlopen)
    return urls


def test_installs_the_extracted_source_with_no_git_requirement(monkeypatch, shadow):
    urls = _serve(monkeypatch, _archive_bytes())
    seen = []

    class _Done:
        returncode = 0
        stdout = ""

    def fake_run(cmd, **kwargs):
        source = next(a for a in cmd if a.endswith(f"llm-compressor-{tv._LLMC_MAIN_SHA}"))
        archival = (tv.Path(source) / ".git_archival.txt").read_text(encoding = "utf-8")
        seen.append((cmd, archival))
        return _Done()

    monkeypatch.setattr(tv.subprocess, "run", fake_run)
    assert tv._ensure_venv_llmcompressor_exists() is True
    assert urls == [
        f"https://github.com/vllm-project/llm-compressor/archive/{tv._LLMC_MAIN_SHA}.tar.gz"
    ]
    cmd, archival = seen[0]
    assert not any("git+" in a for a in cmd)
    assert f"node: {tv._LLMC_MAIN_SHA}" in archival
    assert f"describe-name: {tv._LLMC_MAIN_DESCRIBE}-g{tv._LLMC_MAIN_SHA[:8]}" in archival
    assert not list(shadow.glob(".llmc-src-*"))


def test_checksum_mismatch_installs_nothing(monkeypatch, shadow):
    _serve(monkeypatch, _archive_bytes(), digest = "0" * 64)
    monkeypatch.setattr(tv.subprocess, "run", lambda *a, **k: pytest.fail("ran an install"))
    assert tv._ensure_venv_llmcompressor_exists() is False


def test_network_error_fails_cleanly(monkeypatch, shadow):
    def boom(*a, **k):
        raise OSError("connection reset")

    monkeypatch.setattr(tv.urllib.request, "urlopen", boom)
    monkeypatch.setattr(tv.subprocess, "run", lambda *a, **k: pytest.fail("ran an install"))
    assert tv._ensure_venv_llmcompressor_exists() is False


@pytest.mark.allow_network
def test_pinned_archive_matches_its_checksum(tmp_path):
    source = tv._download_llmcompressor_source(tmp_path)
    # Namespace dirs the archive build keeps only via .git_archival.txt.
    assert (source / "src/llmcompressor/modeling/moe/context.py").is_file()
    assert (source / ".git_archival.txt").is_file()


@pytest.mark.allow_network
def test_the_archive_builds_the_whole_package_without_git(tmp_path, monkeypatch):
    import subprocess
    import sys

    monkeypatch.setenv("PATH", "")  # no git for setuptools_scm to fall back on
    source = tv._download_llmcompressor_source(tmp_path)
    target = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--target",
            str(target),
            str(source),
        ],
        check = True,
    )
    package = target / "llmcompressor"
    assert (package / "modeling" / "moe" / "context.py").is_file()
    assert (package / "modeling" / "patch").is_dir()
    version = (package / "version.py").read_text(encoding = "utf-8")
    assert f"+g{tv._LLMC_MAIN_SHA[:8]}" in version
