# SPDX-License-Identifier: AGPL-3.0-only
"""Persistent MXC read grants against real Windows ACLs and icacls (#12941).

A Microsoft Store Python lives under Program Files\\WindowsApps, owned by TrustedInstaller with
read and execute only for everyone else, so icacls can neither add nor remove an entry there. The
locked folder below reproduces that with the same owner and DACL.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from core.inference import mxc_read_grants, mxc_runtime, os_sandbox

pytestmark = pytest.mark.skipif(os.name != "nt", reason = "real Windows ACLs")

TRUSTED_INSTALLER = "*S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
SYSTEM, ADMINISTRATORS, USERS = "*S-1-5-18", "*S-1-5-32-544", "*S-1-5-32-545"


def _run(*argv: str) -> None:
    done = subprocess.run(argv, capture_output = True, text = True, errors = "replace")
    assert done.returncode == 0, f"{argv}: {done.stdout}{done.stderr}"


def _is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


@pytest.fixture
def studio(monkeypatch, tmp_path):
    # Long canonical spelling: _identity refuses an 8.3 alias such as RUNNER~1.
    base = Path(os.path.realpath(tmp_path))
    studio_home = base / "home" / ".unsloth" / "studio"
    studio_home.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.delenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, raising = False)
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: True)
    monkeypatch.setattr(os_sandbox, "studio_state_roots", lambda: (str(studio_home),))
    monkeypatch.setattr(
        mxc_runtime, "dacl_state_path", lambda: studio_home / "mxc-runtime" / "dacl-restore"
    )
    mxc_read_grants._scanned.clear()
    return base


@pytest.fixture
def store_python(studio):
    if not _is_admin():
        if os.environ.get("CI"):
            pytest.fail("CI must run as an administrator to build the locked folder")
        pytest.skip("building a TrustedInstaller-owned folder needs an administrator")
    root = (
        studio
        / "WindowsApps"
        / "PythonSoftwareFoundation.Python.3.12_3.12.2800.0_x64__qbz5n2kfra8p0"
    )
    (root / "Lib" / "encodings").mkdir(parents = True)
    (root / "python.exe").write_bytes(b"")
    (root / "Lib" / "encodings" / "__init__.py").write_text("")
    _run(
        "icacls",
        str(root),
        "/inheritance:r",
        "/grant:r",
        f"{TRUSTED_INSTALLER}:(OI)(CI)(F)",
        f"{SYSTEM}:(OI)(CI)(RX)",
        f"{ADMINISTRATORS}:(OI)(CI)(RX)",
        f"{USERS}:(OI)(CI)(RX)",
    )
    _run("icacls", str(root), "/setowner", TRUSTED_INSTALLER, "/T", "/C")
    yield str(root)
    subprocess.run(["takeown", "/F", str(root), "/R", "/D", "Y"], capture_output = True)
    subprocess.run(
        ["icacls", str(root), "/grant", f"{ADMINISTRATORS}:(OI)(CI)(F)", "/T", "/C"],
        capture_output = True,
    )
    shutil.rmtree(root, ignore_errors = True)


def _record():
    path = mxc_read_grants.record_path()
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text(encoding = "utf-8"))["grants"]


def test_the_locked_folder_matches_the_store_python_acl(store_python):
    # The fixture is only evidence if Windows really refuses this account a DACL change.
    assert mxc_read_grants._can_change_permissions(store_python) is False
    ok, output = mxc_read_grants._grant(store_python)
    assert not ok, output


def test_a_store_python_folder_keeps_the_per_launch_grant_on_every_launch(store_python):
    for _ in range(3):
        assert mxc_read_grants.ensure([store_python]) == ()
    assert _record() == {}


def test_a_stuck_pending_grant_from_an_older_studio_is_dropped(store_python):
    key = os.path.normcase(store_python)
    pending = {key: {"state": "pending", "identity": mxc_read_grants._identity(store_python)}}
    mxc_read_grants._save_record(pending)
    assert mxc_read_grants.ensure([store_python]) == ()
    assert _record() == {}
    mxc_read_grants._save_record(pending)
    assert mxc_read_grants.revoke_recorded() == ()
    assert _record() == {}


def test_a_folder_studio_owns_is_still_granted_once_and_revoked(studio):
    venv = studio / "home" / ".unsloth" / "studio" / "unsloth_studio"
    (venv / "Lib" / "site-packages").mkdir(parents = True)
    (venv / "Lib" / "site-packages" / "six.py").write_text("")
    venv = str(venv)
    assert mxc_read_grants._can_change_permissions(venv) is True
    assert mxc_read_grants.ensure([venv]) == (venv,)
    assert mxc_read_grants._package_aces(venv) == (True, True)
    assert {v["state"] for v in _record().values()} == {"complete"}
    assert mxc_read_grants.ensure([venv]) == (venv,)
    assert mxc_read_grants.revoke_recorded() == (os.path.normcase(venv),)
    assert mxc_read_grants._package_aces(venv)[1] is False
    assert _record() == {}


def _installed_packages() -> list[str]:
    windows_apps = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "WindowsApps"
    try:
        return sorted(str(p) for p in windows_apps.iterdir() if p.is_dir())[:20]
    except OSError:
        return []


def test_real_windowsapps_packages_never_wedge_a_launch(studio):
    packages = _installed_packages()
    if not packages:
        pytest.skip("no readable Program Files\\WindowsApps packages on this host")
    for package in packages:
        mxc_read_grants.ensure([package])  # must not raise ReadGrantError
    assert {v.get("state") for v in _record().values()} <= {"complete"}
