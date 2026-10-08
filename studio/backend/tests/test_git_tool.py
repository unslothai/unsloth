# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Git is provided lazily by the feature that needs it, never by the installer."""

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from utils import git_tool


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def windows_without_git(monkeypatch, tmp_path):
    monkeypatch.setattr(git_tool, "_IS_WINDOWS", True)
    monkeypatch.setattr(git_tool.shutil, "which", lambda name: None)
    monkeypatch.setattr(git_tool, "studio_root", lambda: tmp_path)
    monkeypatch.setattr(git_tool, "_mingit_flavor", lambda: "64-bit")
    # git.exe cannot run here: a present, non-empty launcher stands in for a working one.
    monkeypatch.setattr(
        git_tool,
        "_git_works",
        lambda exe: git_tool.Path(exe).is_file() and git_tool.Path(exe).read_bytes() == b"MZ",
    )
    return tmp_path


def _serve(
    monkeypatch,
    payload: bytes,
    *,
    digest: str | None = None,
):
    calls = []
    monkeypatch.setitem(
        git_tool._MINGIT_SHA256, "64-bit", digest or hashlib.sha256(payload).hexdigest()
    )

    def fake_urlopen(request, timeout = None):
        calls.append(request.full_url)
        return _Response(payload)

    monkeypatch.setattr(git_tool.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_working_git_on_path_is_used_as_is(monkeypatch):
    monkeypatch.setattr(git_tool.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(git_tool, "_git_works", lambda exe: True)
    monkeypatch.setattr(
        git_tool.urllib.request, "urlopen", lambda *a, **k: pytest.fail("downloaded")
    )
    assert git_tool.ensure_git() is None


def test_missing_git_off_windows_explains_how_to_install(monkeypatch):
    monkeypatch.setattr(git_tool, "_IS_WINDOWS", False)
    monkeypatch.setattr(git_tool.shutil, "which", lambda name: None)
    with pytest.raises(git_tool.GitUnavailable, match = "package manager"):
        git_tool.ensure_git()


def test_windows_fetches_pinned_mingit_once_into_studio_home(monkeypatch, windows_without_git):
    calls = _serve(monkeypatch, _zip_bytes({"cmd/git.exe": b"MZ", "mingw64/bin/git.exe": b"MZ"}))
    first = git_tool.ensure_git()
    second = git_tool.ensure_git()
    root = windows_without_git / "tools" / f"mingit-{git_tool.MINGIT_VERSION}-64-bit"
    assert first == second == str(root / "cmd")
    assert (root / "cmd" / "git.exe").is_file()
    assert calls == [
        "https://github.com/git-for-windows/git/releases/download/"
        f"{git_tool._MINGIT_TAG}/MinGit-{git_tool.MINGIT_VERSION}-64-bit.zip"
    ]


def test_checksum_mismatch_installs_nothing(monkeypatch, windows_without_git):
    _serve(monkeypatch, _zip_bytes({"cmd/git.exe": b"MZ"}), digest = "0" * 64)
    with pytest.raises(git_tool.GitUnavailable, match = "sha256"):
        git_tool.ensure_git()
    assert not any((windows_without_git / "tools").glob("mingit-*"))


def test_zip_entry_escaping_its_root_is_refused(monkeypatch, windows_without_git):
    _serve(monkeypatch, _zip_bytes({"cmd/git.exe": b"MZ", "../evil.txt": b"x"}))
    with pytest.raises(git_tool.GitUnavailable, match = "outside its root"):
        git_tool.ensure_git()
    assert not (windows_without_git / "tools" / "evil.txt").exists()


def test_offline_windows_without_git_does_not_download(monkeypatch, windows_without_git):
    monkeypatch.setattr(
        git_tool.urllib.request, "urlopen", lambda *a, **k: pytest.fail("downloaded")
    )
    with pytest.raises(git_tool.GitUnavailable, match = "offline"):
        git_tool.ensure_git(allow_download = False)


def test_network_error_becomes_git_unavailable(monkeypatch, windows_without_git):
    def boom(*a, **k):
        raise OSError("connection reset")

    monkeypatch.setattr(git_tool.urllib.request, "urlopen", boom)
    with pytest.raises(git_tool.GitUnavailable, match = "connection reset"):
        git_tool.ensure_git()


def test_with_git_on_path_prepends_for_the_child_only():
    env = {"Path": r"C:\Windows"}
    out = git_tool.with_git_on_path(env, r"C:\s\tools\mingit\cmd")
    assert out["Path"].startswith(r"C:\s\tools\mingit\cmd" + git_tool.os.pathsep)
    assert env == {"Path": r"C:\Windows"}
    assert git_tool.with_git_on_path(env, None) is env


def test_llmcompressor_shadow_gets_git_on_the_install_path(monkeypatch, tmp_path):
    from utils import transformers_version as tv

    monkeypatch.setattr(tv, "_llmcompressor_shadow_is_valid", lambda: False)
    monkeypatch.setattr(tv, "_llmcompressor_main_disabled", lambda: False)
    monkeypatch.setattr(tv, "_env_offline", lambda: False)
    monkeypatch.setattr(tv, "_VENV_LLMCOMPRESSOR_DIR", str(tmp_path / "shadow"))
    monkeypatch.setattr(git_tool, "ensure_git", lambda **k: "/studio/tools/mingit/cmd")
    seen = []

    class _Done:
        returncode = 0
        stdout = ""

    def fake_run(cmd, **kwargs):
        env = kwargs["env"]
        seen.append(next(v for k, v in env.items() if k.upper() == "PATH"))
        return _Done()

    monkeypatch.setattr(tv.subprocess, "run", fake_run)
    assert tv._ensure_venv_llmcompressor_exists() is True
    assert seen and seen[0].startswith("/studio/tools/mingit/cmd" + git_tool.os.pathsep)


def test_llmcompressor_shadow_fails_cleanly_without_git(monkeypatch, tmp_path):
    from utils import transformers_version as tv

    monkeypatch.setattr(tv, "_llmcompressor_shadow_is_valid", lambda: False)
    monkeypatch.setattr(tv, "_llmcompressor_main_disabled", lambda: False)
    monkeypatch.setattr(tv, "_env_offline", lambda: False)
    monkeypatch.setattr(tv, "_VENV_LLMCOMPRESSOR_DIR", str(tmp_path / "shadow"))

    def no_git(**k):
        raise git_tool.GitUnavailable("Git is required")

    monkeypatch.setattr(git_tool, "ensure_git", no_git)
    monkeypatch.setattr(tv.subprocess, "run", lambda *a, **k: pytest.fail("ran an install"))
    assert tv._ensure_venv_llmcompressor_exists() is False


@pytest.mark.allow_network
@pytest.mark.skipif(git_tool.os.name != "nt", reason = "runs the real MinGit git.exe")
def test_real_mingit_runs_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(git_tool.shutil, "which", lambda name: None)
    monkeypatch.setattr(git_tool, "studio_root", lambda: tmp_path)
    git_dir = git_tool.ensure_git()
    out = git_tool.subprocess.run(
        [str(git_tool.Path(git_dir) / "git.exe"), "--version"],
        capture_output = True,
        text = True,
        check = True,
    )
    assert git_tool.MINGIT_VERSION.rsplit(".", 1)[0] in out.stdout


def test_bundled_git_uses_the_windows_certificate_store():
    env = git_tool.with_git_on_path({"PATH": "x", "GIT_CONFIG_COUNT": "1"}, r"C:\s\mingit\cmd")
    assert env["GIT_CONFIG_COUNT"] == "2"
    assert (env["GIT_CONFIG_KEY_1"], env["GIT_CONFIG_VALUE_1"]) == ("http.sslBackend", "schannel")
    assert "GIT_CONFIG_COUNT" not in git_tool.with_git_on_path({"PATH": "x"}, None)
