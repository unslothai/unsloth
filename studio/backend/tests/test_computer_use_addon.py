# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Computer use is an add-on: what Unsloth does without it, and how it is installed and removed."""

import re
import sys
from pathlib import Path

import pytest

from core.inference import computer_use
from core.inference.computer_use import addon
from state import tool_policy
from utils import host_policy

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "studio" / "addons" / "computer_use"


@pytest.fixture
def bare(monkeypatch, tmp_path):
    """The stub as a fresh install has it: no add-on, and an empty Studio home to install into."""
    prefix = computer_use.__name__ + "."
    path = list(computer_use.__path__)
    modules = {name: mod for name, mod in sys.modules.items() if name.startswith(prefix)}
    attrs = dict(vars(computer_use))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    host_policy._reset_loopback_default_state()
    host_policy.apply_stdio_mcp_loopback_default("127.0.0.1")
    computer_use.__path__[:] = computer_use._OWN_PATH
    yield tmp_path / "addons" / "computer_use"
    computer_use.__path__[:] = path
    sys.modules.update(modules)
    vars(computer_use).update(attrs)
    host_policy._reset_loopback_default_state()


@pytest.fixture
def client(bare):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes.computer_use import router

    app = FastAPI()
    app.include_router(router, prefix = "/api/computer-use")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    return TestClient(app)


def test_the_versions_named_by_unsloth_and_the_addon_agree():
    pyproject = (SOURCE / "pyproject.toml").read_text(encoding = "utf-8")
    assert re.search(r'^version = "(.+)"$', pyproject, re.M).group(1) == computer_use.ADDON_VERSION
    assert f'name = "{addon.DISTRIBUTION}"' in pyproject


def test_the_unsloth_wheel_leaves_the_addon_out():
    pyproject = (REPO / "pyproject.toml").read_text(encoding = "utf-8")
    excluded = pyproject.split("[tool.setuptools.exclude-package-data]")[1]
    assert '"addons/**"' in excluded.split('"studio" = [')[1].split("]")[0]
    shipped = {path.name for path in Path(computer_use._OWN_PATH[0]).glob("*.py")}
    assert shipped == {"__init__.py", "addon.py", "budget.py"}


def test_nothing_of_the_addon_loads_or_runs_without_it(bare):
    assert not computer_use.installed()
    assert computer_use.is_computer_tool("computer")
    result = computer_use.execute_computer_tool({"action": "screenshot"}, approved = True)
    assert result == f"Error: {computer_use.NOT_INSTALLED}"
    assert computer_use.status_text({"action": "screenshot"}) == "Using the computer"
    assert not tool_policy.needs_tool_confirmation(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = "auto",
        name = "computer",
        arguments = {"action": "click"},
        is_high_risk = lambda *args, **kwargs: True,
        never_needs = lambda name: False,
    )


def test_the_tool_is_never_offered_without_it(bare):
    from routes import inference as route
    assert route._computer_use_tool(object(), True) is None


def test_the_status_says_it_can_be_installed_and_the_other_routes_refuse(client):
    status = client.get("/api/computer-use/status").json()
    assert status["installed"] is False and status["available"] is False
    assert status["reason"] is None and status["enabled"] is False
    refused = client.put("/api/computer-use/settings", json = {"enabled": True})
    assert refused.status_code == 409 and "not installed" in refused.json()["detail"]
    assert client.post("/api/computer-use/resume").status_code == 409


def test_a_server_others_can_reach_is_told_why_and_cannot_install(client, bare):
    host_policy.apply_stdio_mcp_loopback_default("0.0.0.0")
    status = client.get("/api/computer-use/status").json()
    assert status["installed"] is False and "other machines" in status["reason"]
    assert client.post("/api/computer-use/install").status_code == 403
    assert not bare.exists()


def test_installing_copies_the_checkout_and_removing_deletes_it(client, bare, monkeypatch):
    from core.inference.computer_use import policy

    # On, as an update from a build with computer use built in leaves it.
    chosen = {"enabled": True, "coordinates": "auto", "allowed_apps": [], "blocked_apps": []}
    monkeypatch.setenv("UNSLOTH_STUDIO_COMPUTER_USE_BACKEND", "virtual")
    monkeypatch.setattr(policy, "settings", lambda: dict(chosen))
    monkeypatch.setattr(policy, "update_settings", lambda **changes: chosen.update(changes))
    installed = client.post("/api/computer-use/install")
    assert installed.status_code == 200 and installed.json()["installed"] is True
    # Installing never switches it on.
    assert installed.json()["enabled"] is False
    package = bare / computer_use.ADDON_VERSION / computer_use.ADDON_PACKAGE
    assert (package / "tool.py").is_file() and (package / "backends" / "base.py").is_file()
    assert computer_use.__path__[-1] == str(package)
    assert [entry.name for entry in bare.iterdir() if entry.is_dir()] == [
        computer_use.ADDON_VERSION
    ]
    # A second press changes nothing.
    assert client.post("/api/computer-use/install").status_code == 200

    removed = client.delete("/api/computer-use/install")
    assert removed.status_code == 200 and removed.json()["installed"] is False
    assert not package.exists() and not computer_use.installed()
    assert not any(name.startswith(computer_use.__name__ + ".tool") for name in sys.modules)


def test_an_older_version_is_not_loaded_and_is_replaced(bare):
    old = bare / "0.0.1" / computer_use.ADDON_PACKAGE
    old.mkdir(parents = True)
    (old / "tool.py").write_text("raise SystemExit('stale')\n", encoding = "utf-8")
    assert not computer_use.installed()
    addon.install()
    assert computer_use.installed() and not old.exists()


def test_a_download_is_one_hash_pinned_wheel_and_nothing_else(bare, monkeypatch, tmp_path):
    monkeypatch.setattr(addon, "ADDON_SHA256", "ab" * 32)
    requirements = tmp_path / "requirements.txt"
    command = addon._download_command(tmp_path / "staging", requirements)
    pinned = f"{addon.DISTRIBUTION}=={computer_use.ADDON_VERSION} --hash=sha256:{'ab' * 32}"
    assert requirements.read_text(encoding = "utf-8").strip() == pinned
    assert command[-2:] == ["--requirement", str(requirements)]
    for flag in (
        "--require-hashes",
        "--no-deps",
        "--only-binary",
        ":all:",
        "https://pypi.org/simple",
    ):
        assert flag in command
    assert command[command.index("--target") + 1] == str(tmp_path / "staging")


def test_nothing_is_downloaded_without_a_pinned_hash(bare, monkeypatch):
    def ran(*args, **kwargs):
        raise AssertionError("the installer must not run")

    monkeypatch.setattr(addon, "_checkout_source", lambda: None)
    monkeypatch.setattr(addon, "ADDON_SHA256", "")
    monkeypatch.setattr(addon.subprocess, "run", ran)
    with pytest.raises(addon.AddonInstallError, match = "this build"):
        addon.install()
    assert not computer_use.installed()


def test_a_removal_that_cannot_finish_leaves_nothing_that_loads(bare, monkeypatch):
    addon.install()
    monkeypatch.setattr(addon.shutil, "rmtree", lambda *args, **kwargs: None)
    addon.uninstall()
    assert not computer_use.installed()
    assert not (bare / computer_use.ADDON_VERSION).exists()


def test_a_failed_download_installs_nothing_and_says_so_plainly(bare, monkeypatch):
    def fail(*args, **kwargs):
        raise addon.subprocess.CalledProcessError(1, "uv", stderr = b"https://user:secret@proxy")

    monkeypatch.setattr(addon, "_checkout_source", lambda: None)
    monkeypatch.setattr(addon, "ADDON_SHA256", "ab" * 32)
    monkeypatch.setattr(addon.subprocess, "run", fail)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    with pytest.raises(addon.AddonInstallError) as caught:
        addon.install()
    assert "secret" not in str(caught.value)
    assert not computer_use.installed()
    assert not [entry for entry in bare.iterdir() if entry.is_dir()]


def test_offline_mode_refuses_the_download(bare, monkeypatch):
    monkeypatch.setattr(addon, "_checkout_source", lambda: None)
    monkeypatch.setattr(addon, "ADDON_SHA256", "ab" * 32)
    monkeypatch.setenv("UV_OFFLINE", "1")
    with pytest.raises(addon.AddonInstallError, match = "offline"):
        addon.install()
