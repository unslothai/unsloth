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


@pytest.mark.parametrize("order", [1, -1])
def test_typosquat_is_flagged_whatever_the_dependency_order(index, order):
    # An unfamiliar dependency (review) listed first must not stop the typosquat check.
    index.requires += ["brand-new-thing>=1", "requestss>=2"][::order]
    assert ps.scan_pypi_install(PLAN).status == ps.FLAGGED


def test_review_still_inspects_the_wheel(index):
    index.requires.append("brand-new-thing>=1")
    index.wheel = _wheel({"transformers/__init__.py": "", "evil.pth": "import os"})
    index.published_sha = hashlib.sha256(index.wheel).hexdigest()
    assert ps.scan_pypi_install(PLAN).status == ps.FLAGGED


def test_new_dependency_needs_review(index):
    index.requires.append("brand-new-thing>=1")
    result = ps.scan_pypi_install(PLAN)
    assert result.status == ps.REVIEW and "brand-new-thing" in result.reason
    # The inspected wheel stays pinned for a confirmed install past the review.
    assert result.hashes[f"transformers=={VERSION}"] == (index.published_sha,)


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
    # A host mirror (PIP_INDEX_URL, uv.toml) must not change what these tests see.
    for var in (
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "PIP_FIND_LINKS",
        "UV_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_INDEX",
        "UV_EXTRA_INDEX_URL",
        "UV_FIND_LINKS",
        "UV_CONFIG_FILE",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("UV_NO_CONFIG", "1")
    monkeypatch.setattr(tv, "_pip_effective_settings", lambda: {})
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
    # uv: --verify-hashes, since --require-hashes also rejects unpinned UV_CONSTRAINT entries.
    assert "--verify-hashes" in pinned_cmd and "transformers==5.30.0" not in pinned_cmd
    assert plain_cmd[-1] == "tiktoken" and "--verify-hashes" not in plain_cmd


def test_pip_fallback_requires_hashes(monkeypatch, tmp_path):
    cmds = []

    def _run(cmd, **kwargs):
        cmds.append(cmd)

        class _R:
            returncode = 0 if cmd[0] != "uv" else 1
            stdout = ""

        return _R()

    monkeypatch.setattr(tv.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(tv.subprocess, "run", _run)
    with tv.install_with_hashes({"transformers==5.30.0": ("b" * 64,)}):
        assert tv._install_to_dir("transformers==5.30.0", str(tmp_path))
    assert cmds[0][0] == "uv" and "--require-hashes" in cmds[1]


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


# ---- custom package index ----


_INDEX_ENVS = (
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "UV_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_INDEX",
    "UV_EXTRA_INDEX_URL",
    "UV_FIND_LINKS",
    "UV_CONFIG_FILE",
)


@pytest.fixture
def no_index_config(monkeypatch, tmp_path):
    """No index env, an empty pip config, and no uv.toml anywhere uv would look."""
    for var in _INDEX_ENVS:
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("UV_NO_CONFIG", "1")
    monkeypatch.setattr(tv, "_pip_effective_settings", lambda: {})
    return tmp_path


@pytest.mark.parametrize(
    "var,value",
    [
        ("PIP_INDEX_URL", "https://mirror.example/simple"),
        ("UV_INDEX_URL", "https://mirror.example/simple"),
        ("UV_DEFAULT_INDEX", "https://mirror.example/simple"),
        ("UV_INDEX", "corp=https://corp.example/simple"),
        ("PIP_EXTRA_INDEX_URL", "https://extra.example/simple"),
        ("UV_FIND_LINKS", "/srv/wheels"),
    ],
)
def test_custom_index_env_is_detected(no_index_config, monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    assert ps.custom_package_index() == var


def test_pypi_index_env_is_not_custom(no_index_config, monkeypatch):
    monkeypatch.setenv("PIP_INDEX_URL", "https://pypi.org/simple/")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://pypi.org/simple")
    assert ps.custom_package_index() is None


def test_custom_index_in_pip_config(no_index_config, monkeypatch):
    monkeypatch.setattr(
        tv, "_pip_effective_settings", lambda: {"global.index-url": "https://mirror.example/simple"}
    )
    assert ps.custom_package_index() == "pip config global.index-url"


@pytest.mark.parametrize(
    "body,custom",
    [
        ('index-url = "https://mirror.example/simple"\n', True),
        ('[pip]\nextra-index-url = ["https://extra.example/simple"]\n', True),
        ('[[index]]\nname = "corp"\nurl = "https://corp.example/simple"\n', True),
        ('[[index]]\nurl = "https://pypi.org/simple"\ndefault = true\n', False),
        ("not = [valid toml\n", True),
        ("cache-dir = '/tmp/x'\n", False),
    ],
)
def test_custom_index_in_uv_toml(no_index_config, monkeypatch, body, custom):
    cfg = no_index_config / "uv.toml"
    cfg.write_text(body)
    monkeypatch.setenv("UV_CONFIG_FILE", str(cfg))
    assert (ps.custom_package_index() is not None) is custom


def test_custom_index_skips_auto_and_installs_unpinned_on_consent(
    install_env, index, no_index_config, monkeypatch
):
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.example/simple")
    blocked = tl.auto_install_preflight(VERSION)
    assert blocked["requires_consent"] is True
    assert "custom package index, not verified against PyPI" in blocked["message"]
    result = tl.install_latest_transformers(VERSION, auto = True)
    assert result["requires_consent"] is True and install_env == []
    assert index.downloads == 0, "a mirror is never checked against PyPI"
    # Confirmed: the install exactly as before, no scan and no hash pins.
    monkeypatch.setattr(tl, "_scan_release_plan", lambda *a: pytest.fail("scanned a mirror"))
    result = tl.install_latest_transformers(VERSION)
    assert result["success"] is True
    assert install_env == [(VERSION, {})]


def test_pypi_auto_path_unchanged_without_custom_index(install_env, index, no_index_config):
    assert tl.auto_install_preflight(VERSION) is None
    result = tl.install_latest_transformers(VERSION, auto = True)
    assert result["success"] is True and result["scan_status"] == ps.CLEAN
    assert install_env[0][1][f"transformers=={VERSION}"] == (index.published_sha,)


# ---- hash mismatch ----


def _runner(outputs: dict):
    calls = []

    def _run(cmd, **kwargs):
        calls.append(cmd)
        code, out = outputs[cmd[0]]

        class _R:
            returncode = code
            stdout = out

        return _R()

    return calls, _run


def test_uv_hash_mismatch_is_not_retried(monkeypatch, tmp_path):
    calls, run = _runner(
        {"uv": (1, "error: Hash mismatch for `transformers==5.30.0`\n  Expected: sha256:bbb")}
    )
    monkeypatch.setattr(tv.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(tv.subprocess, "run", run)
    with tv.install_with_hashes({"transformers==5.30.0": ("b" * 64,)}):
        assert tv._install_to_dir("transformers==5.30.0", str(tmp_path)) is False
        assert tv.install_hash_mismatch() == "transformers==5.30.0"
    assert len(calls) == 1, "a hash mismatch must not fall through to another install"


def test_pip_hash_mismatch_is_recorded(monkeypatch, tmp_path):
    calls, run = _runner(
        {
            "uv": (2, "error: some unrelated failure"),
            tv.sys.executable: (
                1,
                "ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE.",
            ),
        }
    )
    monkeypatch.setattr(tv.shutil, "which", lambda name: "/usr/bin/uv")
    monkeypatch.setattr(tv.subprocess, "run", run)
    with tv.install_with_hashes({"transformers==5.30.0": ("b" * 64,)}):
        assert tv._install_to_dir("transformers==5.30.0", str(tmp_path)) is False
        assert tv.install_hash_mismatch() == "transformers==5.30.0"
    assert "--require-hashes" in calls[1]


def test_hash_mismatch_surfaces_a_clear_error(install_env, index, no_index_config, monkeypatch):
    def _mismatching_ensure(
        version,
        extra_packages = (),
        before_swap = None,
    ):
        tv._hash_mismatch = f"transformers=={version}"
        return False

    monkeypatch.setattr(tl, "ensure_latest_transformers_venv", _mismatching_ensure)
    result = tl.install_latest_transformers(VERSION, auto = True)
    assert result["success"] is False
    assert "hash mismatch" in result["message"] and f"transformers=={VERSION}" in result["message"]
