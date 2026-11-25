# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The latest-transformers install skips the confirmation only for a PyPI release the package scan
passes; every other verdict, an offline session and transformers main keep it."""

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from utils import transformers_latest as tl
from utils import transformers_version as tv
from utils.security import package_scan as ps

VERSION = "5.30.0"
PLAN = (f"transformers=={VERSION}", "huggingface_hub==1.8.0", "hf_xet==1.4.2", "tiktoken")
WHEEL_URL = f"https://files.pythonhosted.org/packages/xx/transformers-{VERSION}-py3-none-any.whl"


def _wheel(members: dict | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, body in (
            members
            or {
                "transformers/__init__.py": "__version__ = '5.30.0'\n",
                f"transformers-{VERSION}.dist-info/METADATA": "Name: transformers\n",
            }
        ).items():
            archive.writestr(name, body)
    return buf.getvalue()


class FakeIndex:
    """PyPI JSON + OSV querybatch + the wheel host, all in memory."""

    def __init__(self, wheel: bytes | None = None):
        self.wheel = wheel if wheel is not None else _wheel()
        self.published_sha = hashlib.sha256(self.wheel).hexdigest()
        self.requires = ["numpy>=1.17", "huggingface-hub>=1.0", "tokenizers>=0.22", "pyyaml"]
        self.vulns: dict = {}
        self.osv_down = False
        self.pypi_down = False
        self.yanked = False
        self.answer_name = None
        self.wheel_url = WHEEL_URL
        self.downloads = 0

    def http_json(
        self,
        url,
        payload = None,
    ):
        if url == ps._OSV_BATCH_URL:
            if self.osv_down:
                return None
            return {
                "results": [
                    {"vulns": [{"id": i} for i in self.vulns.get(q["package"]["name"], [])]}
                    for q in payload["queries"]
                ]
            }
        if self.pypi_down:
            return None
        name, version = url.split("/pypi/", 1)[1].split("/")[:2]
        files = [
            {
                "filename": f"{name}-{version}.tar.gz",
                "packagetype": "sdist",
                "url": f"https://files.pythonhosted.org/{name}.tar.gz",
                "digests": {"sha256": "a" * 64},
                "yanked": False,
            }
        ]
        if name == "transformers":
            files.insert(
                0,
                {
                    "filename": f"transformers-{version}-py3-none-any.whl",
                    "packagetype": "bdist_wheel",
                    "url": self.wheel_url,
                    "digests": {"sha256": self.published_sha},
                    "yanked": self.yanked,
                },
            )
        return {
            "info": {
                "name": self.answer_name or name,
                "version": version,
                "requires_dist": self.requires + ['pytest; extra == "testing"'],
            },
            "urls": files,
        }

    def download(self, url, dest):
        self.downloads += 1
        with open(dest, "wb") as handle:
            handle.write(self.wheel)
        return True


@pytest.fixture
def index(monkeypatch):
    fake = FakeIndex()
    ps.clear_cache()
    monkeypatch.setattr(ps, "_http_json", fake.http_json)
    monkeypatch.setattr(ps, "_download", fake.download)
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "UNSLOTH_OFFLINE"):
        monkeypatch.delenv(var, raising = False)
    yield fake
    ps.clear_cache()


# ---- the scanner itself ----


def test_clean_release_returns_published_hashes(index):
    result = ps.scan_pypi_install(PLAN)
    assert result.status == ps.CLEAN, result.reason
    assert result.hashes[f"transformers=={VERSION}"][0] == index.published_sha
    assert "huggingface_hub==1.8.0" in result.hashes and "tiktoken" not in result.hashes
    assert index.downloads == 1


def test_clean_verdict_is_cached(index):
    ps.scan_pypi_install(PLAN)
    ps.scan_pypi_install(PLAN)
    assert index.downloads == 1


def test_tampered_wheel_is_flagged(index):
    index.published_sha = "0" * 64
    result = ps.scan_pypi_install(PLAN)
    assert result.status == ps.FLAGGED and "sha256" in result.reason


@pytest.mark.parametrize(
    "members,needle",
    [
        ({"transformers/__init__.py": "", "evil.pth": "import os"}, ".pth"),
        ({"transformers/__init__.py": "", "sitecustomize.py": ""}, "top-level"),
        ({"transformers/../../x.py": ""}, "unsafe path"),
        ({f"transformers-{VERSION}.data/scripts/x": ""}, "top-level"),
    ],
)
def test_wheel_content_red_flags(index, members, needle):
    index.wheel = _wheel(members)
    index.published_sha = hashlib.sha256(index.wheel).hexdigest()
    result = ps.scan_pypi_install(PLAN)
    assert result.status == ps.FLAGGED and needle in result.reason


def test_malicious_osv_entry_is_flagged(index):
    index.vulns = {"transformers": ["MAL-2026-1"]}
    assert ps.scan_pypi_install(PLAN).status == ps.FLAGGED


@pytest.mark.parametrize(
    "setup",
    [
        # An ordinary advisory on one package must not hide a MAL entry on another.
        lambda i: setattr(i, "vulns", {"transformers": ["GHSA-x"], "huggingface-hub": ["MAL-1"]}),
        # Nor may an unreachable release check skip the advisory lookup.
        lambda i: (setattr(i, "pypi_down", True), setattr(i, "vulns", {"hf-xet": ["MAL-2"]})),
    ],
)
def test_malicious_entry_outranks_a_review_elsewhere(index, setup):
    setup(index)
    assert ps.scan_pypi_install(PLAN).status == ps.FLAGGED


def test_known_advisory_needs_review(index):
    index.vulns = {"huggingface-hub": ["GHSA-aaaa-bbbb-cccc"]}
    result = ps.scan_pypi_install(PLAN)
    assert result.status == ps.REVIEW and "GHSA-aaaa-bbbb-cccc" in result.reason


@pytest.mark.parametrize("attr", ["osv_down", "pypi_down", "yanked"])
def test_unreachable_or_yanked_is_inconclusive(index, attr):
    setattr(index, attr, True)
    assert ps.scan_pypi_install(PLAN).status == ps.REVIEW


def test_offline_is_inconclusive_without_network(index, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(ps, "_http_json", lambda *a, **k: pytest.fail("fetched while offline"))
    assert ps.scan_pypi_install(PLAN).status == ps.REVIEW


def test_typosquatted_plan_package_is_flagged(index):
    result = ps.scan_pypi_install((f"transformers=={VERSION}", "tokenizerz==0.23.0"))
    assert result.status == ps.FLAGGED and "tokenizers" in result.reason


def test_typosquatted_dependency_is_flagged(index):
    index.requires.append("requestss>=2")
    result = ps.scan_pypi_install(PLAN)
    assert result.status == ps.FLAGGED and "requests" in result.reason


def test_new_dependency_needs_review(index):
    index.requires.append("brand-new-thing>=1")
    result = ps.scan_pypi_install(PLAN)
    assert result.status == ps.REVIEW and "brand-new-thing" in result.reason


def test_name_mismatch_and_foreign_host_are_flagged(index):
    index.answer_name = "transformerz"
    assert ps.scan_pypi_install(PLAN).status == ps.FLAGGED
    index.answer_name = None
    ps.clear_cache()
    index.wheel_url = "https://evil.example/transformers.whl"
    assert ps.scan_pypi_install(PLAN).status == ps.FLAGGED


def test_scanner_never_raises(index, monkeypatch):
    monkeypatch.setattr(ps, "_check_release", lambda *a, **k: 1 / 0)
    assert ps.scan_pypi_install(PLAN).status == ps.REVIEW


# ---- the install path ----


@pytest.fixture
def install_env(index, monkeypatch, tmp_path):
    tl.clear_caches()
    monkeypatch.setattr(tl, "_cache_file", lambda: tmp_path / "transformers_latest_check.json")
    monkeypatch.setattr(tv, "_VENV_T5_LATEST_DIR", str(tmp_path / "venv_t5_latest"))
    monkeypatch.delenv("UNSLOTH_STUDIO_NO_LATEST_TRANSFORMERS", raising = False)
    snapshot = {
        "pypi_version": VERSION,
        "main_checked": True,
        "main_version": "5.31.0.dev0",
    }
    monkeypatch.setattr(tl, "_refresh_snapshot", lambda: None)
    monkeypatch.setattr(tl, "_get_snapshot", lambda: snapshot)
    monkeypatch.setattr(tl, "_save_snapshot_file", lambda s: None)
    monkeypatch.setattr(tl, "compat_plan", lambda v: ((), []))
    monkeypatch.setattr(tl, "latest_venv_pinned_version", lambda: VERSION)
    calls = []

    def _fake_ensure(
        version,
        extra_packages = (),
        before_swap = None,
    ):
        calls.append((version, dict(tv._install_hashes)))
        return True

    monkeypatch.setattr(tl, "ensure_latest_transformers_venv", _fake_ensure)
    yield calls
    tl.clear_caches()


def test_clean_pypi_release_auto_installs_hash_pinned(install_env, index):
    assert tl.auto_install_preflight(VERSION) is None
    result = tl.install_latest_transformers(VERSION, auto = True)
    assert result["success"] is True and result["scan_status"] == ps.CLEAN
    ((version, hashes),) = install_env
    assert version == VERSION
    # Only the inspected wheel, never the sdist beside it.
    assert hashes[f"transformers=={VERSION}"] == (index.published_sha,)
    # The context is scoped to this install.
    assert tv._install_hashes == {}


@pytest.mark.parametrize(
    "setup,status",
    [
        (lambda i: setattr(i, "osv_down", True), ps.REVIEW),
        (lambda i: setattr(i, "vulns", {"transformers": ["GHSA-x"]}), ps.REVIEW),
        (lambda i: setattr(i, "published_sha", "0" * 64), ps.FLAGGED),
    ],
)
def test_unclean_scan_never_auto_installs(install_env, index, setup, status):
    setup(index)
    blocked = tl.auto_install_preflight(VERSION)
    assert blocked is not None and blocked["success"] is False
    assert blocked["scan_status"] == status
    assert blocked.get("requires_consent", False) is (status == ps.REVIEW)
    result = tl.install_latest_transformers(VERSION, auto = True)
    assert result["success"] is False and install_env == []


def test_offline_keeps_the_dialog(install_env, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    blocked = tl.auto_install_preflight(VERSION)
    assert blocked["requires_consent"] is True and "offline" in blocked["message"]
    assert install_env == []


def test_git_main_keeps_the_dialog(install_env, monkeypatch):
    blocked = tl.auto_install_preflight("5.31.0.dev0")
    assert blocked["requires_consent"] is True
    monkeypatch.setattr(tl, "_resolve_main_commit", lambda: "f" * 40)
    monkeypatch.setattr(tl, "_fetch_main_version", lambda ref = "main": "5.31.0.dev0")
    result = tl.install_latest_transformers("5.31.0.dev0", auto = True)
    assert result["requires_consent"] is True and install_env == []


def test_stale_version_is_not_auto_installed(install_env):
    assert tl.auto_install_preflight("5.29.0")["requires_consent"] is True


def test_confirmed_install_proceeds_past_review_but_not_flagged(install_env, index):
    index.osv_down = True
    assert tl.install_latest_transformers(VERSION)["success"] is True
    ps.clear_cache()
    index.osv_down = False
    index.vulns = {"transformers": ["MAL-2026-9"]}
    result = tl.install_latest_transformers(VERSION)
    assert result["success"] is False and result["scan_status"] == ps.FLAGGED
    assert len(install_env) == 1


def test_confirmed_main_install_is_not_scanned(install_env, index, monkeypatch):
    monkeypatch.setattr(tl, "_resolve_main_commit", lambda: "f" * 40)
    monkeypatch.setattr(tl, "_fetch_main_version", lambda ref = "main": "5.31.0.dev0")
    monkeypatch.setattr(tl, "_scan_release_plan", lambda *a: pytest.fail("main is not on PyPI"))
    assert tl.install_latest_transformers("5.31.0.dev0")["success"] is True


# ---- hash pinning reaches the installer ----


def test_install_to_dir_requires_hashes(monkeypatch, tmp_path):
    seen = []

    def _run(cmd, **kwargs):
        if "-r" in cmd:
            seen.append(open(cmd[cmd.index("-r") + 1], encoding = "utf-8").read())
        seen.append(cmd)

        class _R:
            returncode = 0
            stdout = ""

        return _R()

    monkeypatch.setattr(tv.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(tv.subprocess, "run", _run)
    with tv.install_with_hashes({"transformers==5.30.0": ("b" * 64,)}):
        assert tv._install_to_dir("transformers==5.30.0", str(tmp_path))
        assert tv._install_to_dir("tiktoken", str(tmp_path))
    req, pinned_cmd, plain_cmd = seen
    assert req.strip() == f"transformers==5.30.0 --hash=sha256:{'b' * 64}"
    assert "--require-hashes" in pinned_cmd and "transformers==5.30.0" not in pinned_cmd
    assert plain_cmd[-1] == "tiktoken" and "--require-hashes" not in plain_cmd


# ---- the route ----


def test_route_returns_requires_consent_before_reserving(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import inference

    monkeypatch.setattr(
        tl,
        "auto_install_preflight",
        lambda v: tl._consent_required(v, "Automatic install skipped: offline.", ps.REVIEW),
    )
    monkeypatch.setattr(
        tv, "try_begin_sidecar_swap", lambda *a, **k: pytest.fail("reserved before the scan")
    )
    app = FastAPI()
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    with TestClient(app) as client:
        response = client.post(
            "/api/inference/install-latest-transformers",
            json = {"version": VERSION, "auto": True},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["requires_consent"] is True and body["success"] is False
    assert body["scan_status"] == ps.REVIEW
