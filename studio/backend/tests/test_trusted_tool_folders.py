# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trusted tool folders: read and run there without a prompt, never write, never escape by spelling."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

from core.inference import os_sandbox, tool_path_approval as gate, tools
from state.tool_policy import needs_tool_confirmation
from utils import trusted_tool_folders_settings as trusted
from utils.account_context import OWNER, AccountContext, as_api_caller, bind_account, reset_account


@pytest.fixture
def scratch(linkable_temp_base):
    # Under the home dir: a macOS temp path is a denied system folder, which saving refuses.
    return Path(tempfile.mkdtemp(dir = linkable_temp_base))


@pytest.fixture
def folder(scratch, monkeypatch):
    """A trusted folder holding run.sh, registered as if saved, with the other DB-backed roots empty."""
    tools_dir = scratch / "tools"
    tools_dir.mkdir()
    (tools_dir / "run.sh").write_text("echo hi\n")
    stored = [os.path.realpath(tools_dir)]
    monkeypatch.setattr(trusted, "_stored", lambda: list(stored))
    monkeypatch.setattr(gate, "_studio_db_exists", lambda: True)
    monkeypatch.setattr(gate, "_scan_folder_roots", lambda: ())
    monkeypatch.setattr(gate, "_hf_cache_dirs", lambda: ())
    monkeypatch.setattr(gate, "_silent_roots_cache", None)
    yield tools_dir
    gate._silent_roots_cache = None


def _asks(path, writing = False):
    gate._silent_roots_cache = None
    return gate._path_needs_approval(str(path), writing = writing)


def test_inside_reads_silently_but_writes_still_ask(folder):
    assert _asks(folder / "run.sh") is False
    assert _asks(folder / "sub" / "tool.exe") is False
    assert _asks(folder) is False
    assert _asks(folder / "run.sh", writing = True) is True


def test_a_sibling_sharing_the_prefix_still_asks(folder):
    sibling = folder.parent / (folder.name + "x")
    sibling.mkdir()
    assert _asks(sibling / "run.sh") is True
    assert _asks(str(folder) + "-old/run.sh") is True


def test_dotdot_out_of_the_folder_still_asks(folder):
    (folder.parent / "secret.txt").write_text("s")
    assert _asks(f"{folder}/../secret.txt") is True
    assert _asks(f"{folder}/sub/../../secret.txt") is True


def test_a_symlink_out_of_the_folder_still_asks(folder):
    outside = folder.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    try:
        (folder / "link").symlink_to(outside, target_is_directory = True)
    except OSError:
        pytest.skip("this platform does not allow creating a symlink here")
    assert _asks(folder / "link" / "secret.txt") is True


def test_a_credential_inside_the_folder_still_asks(folder):
    (folder / "sub").mkdir()
    (folder / "sub" / ".git-credentials").write_text("x")
    assert _asks(folder / "sub" / ".git-credentials") is True


@pytest.mark.parametrize(
    "relative", ["sub/.git-credentials", "deep/er/token", ".ssh/config", "a/.env"]
)
def test_a_folder_with_a_nested_credential_is_not_granted(folder, relative):
    # The sandbox binds the folder whole, so a script there could read it without any prompt.
    path = folder / relative
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text("x")
    assert os_sandbox.trusted_tool_folder_roots() == ()
    assert _asks(folder / "run.sh") is True


def test_a_hard_link_to_a_file_elsewhere_is_not_granted(folder):
    secret = folder.parent / "secret.txt"
    secret.write_text("s")
    try:
        os.link(secret, folder / "notes.txt")
    except OSError:
        pytest.skip("this volume does not allow hard links here")
    assert os_sandbox.trusted_tool_folder_roots() == ()


def test_a_folder_too_large_to_check_is_not_granted(folder, monkeypatch):
    monkeypatch.setattr(os_sandbox, "_TRUSTED_FOLDER_SCAN_LIMIT", 3)
    for index in range(4):
        (folder / f"f{index}.txt").write_text("x")
    assert os_sandbox.trusted_tool_folder_roots() == ()


def test_the_scan_stops_at_the_cap_inside_one_directory(folder, monkeypatch):
    monkeypatch.setattr(os_sandbox, "_TRUSTED_FOLDER_SCAN_LIMIT", 5)
    for index in range(50):
        (folder / f"f{index}.txt").write_text("x")
    seen = []
    real = gate._references_sensitive_path
    monkeypatch.setattr(gate, "_references_sensitive_path", lambda p: seen.append(p) or real(p))
    assert os_sandbox._holds_a_nested_credential(str(folder)) is True
    assert len(seen) <= 5


def test_a_folder_with_a_credential_at_its_top_is_not_trusted(folder):
    (folder / ".env").write_text("KEY=x")
    assert os_sandbox.trusted_tool_folder_roots() == ()
    assert _asks(folder / "run.sh") is True


def test_the_os_sandbox_grants_the_folder(folder, monkeypatch):
    monkeypatch.setattr(
        "utils.paths.storage_roots.well_known_model_dirs", lambda: (), raising = False
    )
    assert os.path.realpath(folder) in os_sandbox.model_library_roots()


def test_no_folders_is_todays_behaviour(folder, monkeypatch):
    monkeypatch.setattr(trusted, "_stored", lambda: [])
    assert _asks(folder / "run.sh") is True


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows drive, UNC and case spellings")
def test_windows_spellings(folder):
    drive, rest = os.path.splitdrive(str(folder / "run.sh"))
    assert _asks(str(folder / "run.sh").upper()) is False
    assert _asks(str(folder / "run.sh").lower()) is False
    assert _asks((folder / "run.sh").as_posix()) is False
    assert _asks(str(folder / "run.sh").lower(), writing = True) is True
    # Other spellings of the same file are not matched lexically, so they keep asking.
    for spelling in (
        "\\\\?\\" + str(folder / "run.sh"),
        f"\\\\localhost\\{drive[0]}${rest}",
        drive + rest.lstrip("\\"),
        rest,
    ):
        assert _asks(spelling) is True, spelling


# Forward slashes: an unquoted backslash is an escape to Git Bash, so `C:\a\b` runs `C:ab`.
@pytest.mark.parametrize(
    "command",
    ["{t}/run.sh", "bash {t}/run.sh --flag", "php {t}/script.php", "cat {t}/run.sh"],
)
def test_terminal_runs_from_the_folder_without_a_prompt(folder, command):
    command = command.format(t = folder.as_posix())
    gate._silent_roots_cache = None
    assert tools.is_high_risk_tool_call("terminal", {"command": command}) is False
    for mode in ("auto", "off"):
        assert not needs_tool_confirmation(
            confirm_tool_calls = True,
            bypass_permissions = False,
            permission_mode = mode,
            name = "terminal",
            arguments = {"command": command},
            sandbox_level = "low",
        ), mode


@pytest.mark.parametrize(
    "command",
    ["echo x > {t}/run.sh", "cp payload {t}/run.sh", "{t}/../escape.sh", "{t}x/run.sh"],
)
def test_terminal_writes_and_escapes_still_ask(folder, command):
    command = command.format(t = folder.as_posix())
    gate._silent_roots_cache = None
    assert tools.is_high_risk_tool_call("terminal", {"command": command}) is True
    assert needs_tool_confirmation(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = "off",
        name = "terminal",
        arguments = {"command": command},
        sandbox_level = "low",
    )


def test_python_runs_from_the_folder_but_writing_there_asks(folder):
    gate._silent_roots_cache = None
    script = (folder / "run.sh").as_posix()
    run = f"import subprocess\nsubprocess.run([{script!r}])"
    write = f"open({script!r}, 'w').write('x')"
    assert tools.is_high_risk_tool_call("python", {"code": run}) is False
    assert tools.is_high_risk_tool_call("python", {"code": write}) is True


@pytest.mark.parametrize(
    "account",
    [as_api_caller(OWNER), AccountContext("acct-1", "alice")],
    ids = ["owner-api-key", "managed-account"],
)
def test_only_the_owners_own_chats_get_the_folders(folder, account):
    token = bind_account(account)
    try:
        assert trusted.trusted_tool_folders_for_caller() == ()
        gate._silent_roots_cache = None
        command = f"{folder.as_posix()}/run.sh"
        assert tools.is_high_risk_tool_call("terminal", {"command": command}) is True
    finally:
        reset_account(token)
        gate._silent_roots_cache = None


def test_runtime_reads_fail_closed(monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr("storage.studio_db.get_app_setting", broken)
    with pytest.raises(OSError):
        trusted.get_trusted_tool_folders()
    assert trusted.trusted_tool_folders_for_caller() == ()


class TestSaving:
    @pytest.fixture(autouse = True)
    def store(self, monkeypatch):
        saved = {}
        monkeypatch.setattr(
            "storage.studio_db.upsert_app_settings", lambda values, **_: saved.update(values)
        )
        return saved

    def test_stores_the_real_path_once(self, scratch, store):
        real = scratch / "real"
        real.mkdir()
        spellings = [str(real), f"{scratch}/real/../real/", f"  {real}  "]
        try:
            (scratch / "alias").symlink_to(real, target_is_directory = True)
            spellings.append(str(scratch / "alias"))
        except OSError:
            pass
        assert trusted.set_trusted_tool_folders(spellings) == [os.path.realpath(real)]
        assert store[trusted.TRUSTED_TOOL_FOLDERS_SETTING_KEY] == [os.path.realpath(real)]

    def test_empty_list_clears(self, store):
        assert trusted.set_trusted_tool_folders([]) == []
        assert store[trusted.TRUSTED_TOOL_FOLDERS_SETTING_KEY] == []

    @pytest.mark.parametrize(
        "value",
        [
            "relative/tools",
            os.path.abspath(os.sep),
            os.path.expanduser("~"),
            os.path.dirname(os.path.expanduser("~")),
            "/definitely/not/here",
        ],
        ids = ["relative", "fs-root", "home", "homes-parent", "missing"],
    )
    def test_refused(self, value, store):
        with pytest.raises(ValueError):
            trusted.set_trusted_tool_folders([value])
        assert store == {}

    def test_refuses_the_studio_home_and_its_ancestors(self, scratch, monkeypatch, store):
        home = scratch / "studio-home"
        home.mkdir()
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
        for value in (home, scratch):
            with pytest.raises(ValueError, match = "Unsloth"):
                trusted.set_trusted_tool_folders([str(value)])
        assert store == {}

    @pytest.mark.parametrize("name", [".env", "token", ".git-credentials", "sub/.ssh/id_rsa"])
    def test_refuses_a_folder_the_sandbox_would_not_grant(self, scratch, store, name):
        folder = scratch / "tools"
        (folder / name).parent.mkdir(parents = True)
        (folder / name).write_text("x")
        with pytest.raises(ValueError, match = "credential"):
            trusted.set_trusted_tool_folders([str(folder)])
        assert store == {}

    @pytest.mark.skipif(sys.platform not in ("darwin", "win32"), reason = "case-insensitive volumes")
    def test_refuses_the_studio_home_in_another_case(self, scratch, monkeypatch, store):
        home = scratch / "studio-home"
        (home / "auth").mkdir(parents = True)
        monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
        with pytest.raises(ValueError, match = "Unsloth"):
            trusted.set_trusted_tool_folders([str(scratch / "STUDIO-HOME")])
        assert os_sandbox.grantable_trusted_roots((str(scratch / "STUDIO-HOME"),)) == ()
        assert store == {}

    def test_refuses_a_file(self, scratch, store):
        path = scratch / "tool.sh"
        path.write_text("x")
        with pytest.raises(ValueError):
            trusted.set_trusted_tool_folders([str(path)])

    def test_refuses_a_non_list(self, store):
        with pytest.raises(ValueError):
            trusted.set_trusted_tool_folders("/opt/tools")


class TestRoute:
    @pytest.fixture
    def build(self, monkeypatch):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import routes.settings as settings

        state = {"folders": []}
        monkeypatch.setattr(settings, "get_trusted_tool_folders", lambda: list(state["folders"]))

        def _set(value):
            if any(not item.startswith("/") for item in value):
                raise ValueError("Folder 1 is not an absolute path.")
            state["folders"] = list(value)
            return list(value)

        monkeypatch.setattr(settings, "set_trusted_tool_folders", _set)

        def make(account, via_api_key = False):
            app = FastAPI()
            app.include_router(settings.router)

            async def subject():
                token = bind_account(account)
                try:
                    yield account.username
                finally:
                    reset_account(token)

            app.dependency_overrides[settings.get_current_subject] = subject
            app.dependency_overrides[settings.authenticated_via_api_key] = lambda: via_api_key
            return TestClient(app, raise_server_exceptions = False)

        return make, state

    def test_owner_saves_and_reads(self, build):
        make, state = build
        with make(OWNER) as client:
            put = client.put("/trusted-tool-folders", json = {"folders": ["/opt/tools"]})
            assert put.status_code == 200, put.text
            assert client.get("/trusted-tool-folders").json() == {"folders": ["/opt/tools"]}

    def test_an_unreadable_store_is_an_error_not_an_empty_list(self, build, monkeypatch):
        import routes.settings as settings

        make, _ = build

        def broken():
            raise OSError("database is locked")

        monkeypatch.setattr(settings, "get_trusted_tool_folders", broken)
        with make(OWNER) as client:
            assert client.get("/trusted-tool-folders").status_code == 500

    def test_a_refused_folder_is_a_400_with_the_reason(self, build):
        make, state = build
        with make(OWNER) as client:
            put = client.put("/trusted-tool-folders", json = {"folders": ["tools"]})
        assert put.status_code == 400
        assert "absolute" in put.json()["detail"]
        assert state["folders"] == []

    @pytest.mark.parametrize(
        "account, via_api_key",
        [(AccountContext("a" * 32, "alice"), False), (OWNER, True)],
        ids = ["managed-account", "owner-api-key"],
    )
    def test_only_the_owners_ui_session_may_read_or_write(self, build, account, via_api_key):
        make, state = build
        state["folders"] = ["/opt/saved"]
        with make(account, via_api_key) as client:
            put = client.put("/trusted-tool-folders", json = {"folders": ["/opt/tools"]})
            get = client.get("/trusted-tool-folders")
        assert put.status_code == 403, put.text
        assert get.status_code == 403, get.text
        assert state["folders"] == ["/opt/saved"]
