# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The llm-compressor-main shadow installs from a checksummed source archive, never git."""

from __future__ import annotations

import io
import tarfile

import pytest

from utils import transformers_version as tv


def _archive_bytes(
    root = None,
    files = None,
    level = 9,
) -> bytes:
    root = root or f"llm-compressor-{tv._LLMC_MAIN_SHA}"
    files = files or {"setup.py": b"from setuptools import setup\n"}
    buf = io.BytesIO()
    with tarfile.open(fileobj = buf, mode = "w:gz", compresslevel = level) as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{root}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _digest(tmp_path, payload: bytes) -> str:
    archive = tmp_path / "digest.tgz"
    archive.write_bytes(payload)
    return tv._archive_content_digest(archive)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def shadow(monkeypatch, tmp_path):
    from utils import utils as studio_utils

    # No ambient proxy: these tests stub urlopen, a proxy opener would reach GitHub for real.
    monkeypatch.setattr(tv, "_hf_proxy_opener", lambda url: None)
    monkeypatch.setattr(studio_utils, "hf_proxy_for_endpoint", lambda url = None: None)
    monkeypatch.setattr(tv, "_llmcompressor_shadow_is_valid", lambda: False)
    monkeypatch.setattr(tv, "_llmcompressor_main_disabled", lambda: False)
    monkeypatch.setattr(tv, "_env_offline", lambda: False)
    monkeypatch.setattr(tv, "_VENV_LLMCOMPRESSOR_DIR", str(tmp_path / "shadow"))
    return tmp_path


def _serve(
    monkeypatch,
    tmp_path,
    payload: bytes,
    digest: str | None = None,
):
    monkeypatch.setattr(tv, "_LLMC_MAIN_ARCHIVE_DIGEST", digest or _digest(tmp_path, payload))
    monkeypatch.setattr(tv, "_hf_proxy_opener", lambda url: None)
    urls = []

    def fake_urlopen(request, timeout = None):
        urls.append(request.full_url)
        return _Response(payload)

    monkeypatch.setattr(tv.urllib.request, "urlopen", fake_urlopen)
    return urls


def test_installs_the_extracted_source_with_no_git_requirement(monkeypatch, shadow):
    urls = _serve(monkeypatch, shadow, _archive_bytes())
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
    _serve(monkeypatch, shadow, _archive_bytes(), digest = "0" * 64)
    monkeypatch.setattr(tv.subprocess, "run", lambda *a, **k: pytest.fail("ran an install"))
    assert tv._ensure_venv_llmcompressor_exists() is False


def test_network_error_fails_cleanly(monkeypatch, shadow):
    def boom(*a, **k):
        raise OSError("connection reset")

    monkeypatch.setattr(tv.urllib.request, "urlopen", boom)
    monkeypatch.setattr(tv.subprocess, "run", lambda *a, **k: pytest.fail("ran an install"))
    assert tv._ensure_venv_llmcompressor_exists() is False


def test_content_digest_ignores_compression_order_and_root_name(tmp_path):
    files = {"a.py": b"a = 1\n", "pkg/b.py": b"b = 2\n"}
    base = _digest(tmp_path, _archive_bytes(files = files))
    reordered = dict(reversed(list(files.items())))
    assert _digest(tmp_path, _archive_bytes("renamed-root", reordered, level = 1)) == base
    assert _digest(tmp_path, _archive_bytes(files = {**files, "a.py": b"a = 3\n"})) != base


def test_the_download_goes_through_the_proxy_aware_opener(monkeypatch, shadow):
    payload = _archive_bytes()
    _serve(monkeypatch, shadow, payload)
    opened = []

    class _Opener:
        def open(
            self,
            request,
            timeout = None,
        ):
            opened.append(request.full_url)
            return _Response(payload)

    monkeypatch.setattr(tv, "_hf_proxy_opener", lambda url: _Opener())
    monkeypatch.setattr(
        tv.urllib.request, "urlopen", lambda *a, **k: pytest.fail("bypassed the proxy")
    )
    assert tv._download_llmcompressor_source(shadow).is_dir()
    assert opened


def test_a_socks_proxy_is_used_through_httpx_never_bypassed(monkeypatch, shadow):
    import httpx

    from utils import utils as studio_utils

    payload = _archive_bytes()
    _serve(monkeypatch, shadow, payload)
    monkeypatch.setattr(
        studio_utils, "hf_proxy_for_endpoint", lambda url = None: "socks5://proxy:1080"
    )
    monkeypatch.setattr(
        tv.urllib.request, "urlopen", lambda *a, **k: pytest.fail("bypassed the socks proxy")
    )
    streamed = []

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_bytes(self, size):
            yield payload

    def fake_stream(method, url, **kwargs):
        streamed.append(url)
        return _Stream()

    monkeypatch.setattr(httpx, "stream", fake_stream)
    assert tv._download_llmcompressor_source(shadow).is_dir()
    assert streamed


def test_a_socks_proxy_without_socksio_fails_closed_with_the_fix(monkeypatch, shadow):
    import httpx

    from utils import utils as studio_utils

    monkeypatch.setattr(
        studio_utils, "hf_proxy_for_endpoint", lambda url = None: "socks5://proxy:1080"
    )
    monkeypatch.setattr(
        tv.urllib.request, "urlopen", lambda *a, **k: pytest.fail("bypassed the socks proxy")
    )

    def no_socksio(*a, **k):
        raise ImportError("Using SOCKS proxy, but the 'socksio' package is not installed.")

    monkeypatch.setattr(httpx, "stream", no_socksio)
    with pytest.raises(RuntimeError, match = "socksio"):
        tv._download_llmcompressor_source(shadow)


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
