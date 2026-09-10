# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import subprocess
import sys
from pathlib import Path

import pytest

from auth import policy, storage
from core.inference import tool_confinement, tools
from utils.account_context import OWNER, AccountContext, run_as
from utils.paths import storage_roots

from .test_account_lifecycle import auth_env, matrix  # noqa: F401

ALICE = AccountContext("alice-id", "alice")
BOB = AccountContext("bob-id", "bob")

LANDLOCK = (
    sys.platform == "linux"
    and tool_confinement.landlock_abi() >= tool_confinement._MIN_LANDLOCK_ABI
)


@pytest.fixture(autouse = True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects"))
    monkeypatch.delenv("UNSLOTH_STUDIO_SANDBOX_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS", raising = False)
    # The sandbox caps the child at 10000 processes per user; a busy CI host is often already above that.
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_NPROC", "4000000")
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(tools, "_workdirs", {})
    monkeypatch.setattr(tools, "_active_sessions", {})
    monkeypatch.setattr(tools, "_pending_removals", {})
    monkeypatch.setattr(tools, "_removing_sessions", set())
    monkeypatch.setattr(tools, "_legacy_sandbox_migrated", True)
    monkeypatch.setattr(tools, "_start_detached_sweep", lambda: None)
    monkeypatch.setattr(tools, "_legacy_sandbox_root", lambda: str(tmp_path / "legacy"))


def _seed(tmp_path: Path) -> dict[str, Path]:
    files = {}
    for account in (OWNER, ALICE, BOB):
        workdir = Path(run_as(account, tools._get_workdir, "chat"))
        secret = workdir / f"{account.username}-secret.txt"
        secret.write_text(f"{account.username.upper()}_PRIVATE")
        files[account.username] = secret
    auth_dir = tmp_path / "studio" / "auth"
    auth_dir.mkdir(parents = True, exist_ok = True)
    files["auth"] = auth_dir / "auth.db"
    files["auth"].write_text("OWNER_AUTH_DB")
    return files


def test_owner_spawns_exactly_as_before():
    assert run_as(OWNER, tools._account_confinement) is None
    kwargs = {"preexec_fn": tools._sandbox_preexec}
    argv = ["bash", "-c", "true"]
    assert tools._apply_confinement(None, kwargs, argv) is argv
    assert kwargs["preexec_fn"] is tools._sandbox_preexec


@pytest.mark.skipif(sys.platform == "linux" and LANDLOCK, reason = "host confines with Landlock")
@pytest.mark.skipif(sys.platform == "darwin", reason = "host confines with sandbox-exec")
def test_managed_account_is_refused_without_a_mechanism(monkeypatch):
    monkeypatch.setattr(tool_confinement, "landlock_abi", lambda: 0)
    with pytest.raises(tool_confinement.ToolConfinementUnavailable):
        run_as(ALICE, tools._account_confinement)
    out = run_as(ALICE, tools._bash_exec, "echo hi", session_id = "chat")
    assert out.startswith("Execution error: Code execution is unavailable for this account")


def test_owner_opt_out_runs_unconfined_when_no_mechanism(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(tool_confinement.ToolConfinementUnavailable):
        run_as(ALICE, tools._account_confinement)
    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_UNCONFINED_TOOLS", "1")
    confinement = run_as(ALICE, tools._account_confinement)
    assert confinement.mechanism == "unconfined-by-owner"
    assert confinement.preexec is None and confinement.wrap(["x"]) == ["x"]


def test_macos_profile_hides_install_root_then_allows_own_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    run_as(ALICE, tools._get_workdir, "chat")
    confinement = run_as(ALICE, tools._account_confinement)
    assert confinement.mechanism == "sandbox-exec"
    argv = confinement.wrap(["bash", "-c", "true"])
    assert argv[:2] == ["/usr/bin/sandbox-exec", "-p"]
    profile = argv[2]
    studio = str((tmp_path / "studio").resolve())
    alice_root = str((tmp_path / "studio" / "accounts" / "alice-id").resolve())
    assert profile.startswith("(version 1)\n(deny default)")
    deny = profile.index(f'(deny file-read* file-write* (subpath "{studio}"))')
    allow = profile.index(f'(allow file-read* (subpath "{alice_root}"))')
    writable = profile.index(f'(allow file-read* file-write* (subpath "{alice_root}/sandbox"))')
    assert (
        deny < allow < writable
    ), "the account roots must be allowed after the install root is denied"
    assert argv[3:] == ["bash", "-c", "true"]


def test_macos_profile_keeps_the_interpreter_readable_under_a_hidden_root(tmp_path, monkeypatch):
    """Later rules win: an interpreter prefix inside the install root must be allowed again."""
    venv = tmp_path / "studio" / "unsloth_studio"
    (venv / "lib").mkdir(parents = True)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(tool_confinement, "_interpreter_roots", lambda: [str(venv)])
    run_as(ALICE, tools._get_workdir, "chat")
    profile = run_as(ALICE, tools._account_confinement).wrap(["bash"])[2]

    studio = str((tmp_path / "studio").resolve())
    deny = profile.rindex(f'(deny file-read* file-write* (subpath "{studio}"))')
    allow = profile.rindex(f'(allow file-read* (subpath "{venv}"))')
    assert deny < allow, "the interpreter must be readable after the install root is denied"
    assert f'(allow file-read* (subpath "{studio}"))' not in profile


def test_macos_profile_hides_other_accounts_temporary_roots(tmp_path, monkeypatch):
    import tempfile as _tempfile

    from utils.paths import storage_roots

    shared_tmp = tmp_path / "tmp"
    shared_tmp.mkdir()
    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(shared_tmp))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    run_as(BOB, storage_roots.tmp_root).mkdir(parents = True, exist_ok = True)
    (run_as(BOB, storage_roots.tmp_root) / "dataset.jsonl").write_text("BOB_PRIVATE")

    profile = run_as(ALICE, tools._account_confinement).wrap(["bash"])[2]
    base = str((shared_tmp / "unsloth-studio").resolve())
    alice_tmp = str(run_as(ALICE, storage_roots.tmp_root).resolve())
    bob_tmp = str(run_as(BOB, storage_roots.tmp_root).resolve())

    deny = profile.index(f'(deny file-read* file-write* (subpath "{base}"))')
    allow = profile.index(f'(allow file-read* file-write* (subpath "{alice_tmp}"))')
    # The server's own bare tempfile output sits beside the Studio subtree, outside every account root.
    assert '(allow file-read* (subpath "/var/folders"))' not in profile
    assert '(subpath "/private/var/folders") (subpath "/var/folders"))' in profile
    folders = profile.index('(deny file-read* file-write* (subpath "/private/var/folders")')
    assert folders < allow
    assert deny < allow, "the account's own temporary root must be allowed after the deny"
    assert f'(subpath "{bob_tmp}")' not in profile


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
@pytest.mark.parametrize("tool", ["terminal", "python"])
def test_managed_child_cannot_read_or_write_other_accounts(tool, tmp_path):
    files = _seed(tmp_path)
    bob_dir = Path(run_as(BOB, tools._get_workdir, "chat"))
    foreign = {
        "owner": files["unsloth"] if "unsloth" in files else files[OWNER.username],
        "alice": files["alice"],
        "auth": files["auth"],
    }

    def run(command_or_code: str) -> str:
        if tool == "terminal":
            return run_as(BOB, tools._bash_exec, command_or_code, session_id = "chat")
        return run_as(BOB, tools._python_exec, command_or_code, session_id = "chat")

    if tool == "terminal":
        assert "BOB_PRIVATE" in run("cat bob-secret.txt")
        assert "written" in run("echo written > note.txt && cat note.txt")
    else:
        assert "BOB_PRIVATE" in run("print(open('bob-secret.txt').read())")
        assert "written" in run(
            "open('note.txt','w').write('written'); print(open('note.txt').read())"
        )
    assert (bob_dir / "note.txt").read_text().strip() == "written"

    for name, path in foreign.items():
        rel = os.path.relpath(path, bob_dir)
        for target in (rel, str(path)):
            if tool == "terminal":
                out = run(f"cat {target}; echo rc=$?")
            else:
                out = run(
                    "try:\n"
                    f"    print(open({target!r}).read())\n"
                    "except OSError as e:\n"
                    "    print('DENIED', e.__class__.__name__)\n"
                )
            assert "PRIVATE" not in out and "AUTH_DB" not in out, (name, target, out)
            assert "rc=0" not in out
            if tool == "terminal":
                out = run(f"echo BOB_OVERWROTE > {target}; echo rc=$?")
            else:
                out = run(
                    "try:\n"
                    f"    open({target!r}, 'w').write('BOB_OVERWROTE')\n"
                    "except OSError as e:\n"
                    "    print('DENIED', e.__class__.__name__)\n"
                )
            assert "rc=0" not in out
        assert path.read_text() != "BOB_OVERWROTE", (name, path)

    studio = tmp_path / "studio"
    if tool == "terminal":
        out = run(f"ls {studio}; echo rc=$?; ls {studio / 'accounts' / 'alice-id'}; echo rc=$?")
    else:
        out = run(
            "import os\n"
            f"for p in [{str(studio)!r}, {str(studio / 'accounts' / 'alice-id')!r}]:\n"
            "    try:\n"
            "        print(os.listdir(p))\n"
            "    except OSError as e:\n"
            "        print('DENIED', e.__class__.__name__)\n"
        )
    assert "rc=0" not in out
    assert "auth" not in out.replace("DENIED", "") or "PermissionError" in out


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_confinement_survives_nested_processes(tmp_path):
    files = _seed(tmp_path)
    alice = files["alice"]
    out = run_as(
        BOB,
        tools._bash_exec,
        f"python -c \"import subprocess; print(subprocess.run(['cat', {str(alice)!r}], "
        'capture_output=True, text=True).stderr)"',
        session_id = "chat",
    )
    assert "PRIVATE" not in out
    assert "Permission denied" in out or "denied" in out.lower()


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_confined_child_keeps_interpreter_and_system_tools(tmp_path):
    _seed(tmp_path)
    out = run_as(
        BOB,
        tools._python_exec,
        "import json, sqlite3, tempfile, os\n"
        "with tempfile.NamedTemporaryFile('w', delete=False) as f:\n"
        "    f.write('x')\n"
        "print(json.dumps({'tmp': os.path.exists(f.name), 'sqlite': sqlite3.sqlite_version != ''}))\n",
        session_id = "chat",
    )
    assert '"tmp": true' in out and '"sqlite": true' in out
    out = run_as(
        BOB, tools._bash_exec, "ls /usr/bin | head -1; python --version", session_id = "chat"
    )
    assert "Python" in out


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_owner_child_remains_unconfined(tmp_path):
    files = _seed(tmp_path)
    out = run_as(OWNER, tools._bash_exec, f"cat {files['alice']}", session_id = "chat")
    assert "ALICE_PRIVATE" in out


def test_landlock_rules_cover_own_roots_only(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux rule builder")
    run_as(ALICE, tools._get_workdir, "chat")
    rules = run_as(ALICE, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
    handled = tool_confinement._handled_mask(3)
    writable = [p for p, access in rules if access == handled & ~tool_confinement._FS_MAKE_SYM]
    alice_root = str((tmp_path / "studio" / "accounts" / "alice-id").resolve())
    assert alice_root not in writable
    assert f"{alice_root}/sandbox" in writable
    assert str((tmp_path / "projects" / "Accounts" / "alice-id" / "Projects").resolve()) in writable
    assert alice_root in [
        p for p, access in rules if access != handled & ~tool_confinement._FS_MAKE_SYM
    ]
    assert str((tmp_path / "studio").resolve()) not in [p for p, _ in rules]
    assert all(
        not p.startswith(str((tmp_path / "studio").resolve()) + os.sep) or p.startswith(alice_root)
        for p, _ in rules
    )
    read_only = [
        p
        for p, access in rules
        if access not in (handled, handled & ~tool_confinement._FS_MAKE_SYM)
    ]
    assert any(
        p.startswith(os.path.realpath(sys.prefix)) or os.path.realpath(sys.prefix).startswith(p)
        for p in read_only
    )


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_managed_child_cannot_plant_a_link_in_its_own_tree(tmp_path):
    """The server follows links for the owner, so a managed child gets no way to make one."""
    files = _seed(tmp_path)
    bob_dir = Path(run_as(BOB, tools._get_workdir, "chat"))
    out = run_as(
        BOB,
        tools._bash_exec,
        f"ln -s {files['alice'].parent} linked; echo rc=$?; mkdir made && echo made > made/f && cat made/f",
        session_id = "chat",
    )
    assert "rc=0" not in out
    assert not (bob_dir / "linked").is_symlink()
    assert "made" in out


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_managed_child_cannot_signal_another_accounts_process(tmp_path):
    """Two accounts' tools run as one Unix user; signals stay inside the child's own domain."""
    _seed(tmp_path)
    alice_job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        out = run_as(
            BOB,
            tools._python_exec,
            "import os, signal, subprocess\n"
            f"pid = {alice_job.pid}\n"
            "try:\n"
            "    os.kill(pid, signal.SIGTERM); print('SIGNALLED')\n"
            "except OSError as e:\n"
            "    print('signal denied', e.__class__.__name__)\n"
            "child = subprocess.Popen(['sleep', '30'])\n"
            "child.terminate(); print('own child rc', child.wait())\n",
            session_id = "chat",
        )
        assert "SIGNALLED" not in out and "signal denied" in out, out
        assert "own child rc -15" in out, out
        assert alice_job.poll() is None
    finally:
        alice_job.kill()


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_managed_child_writes_only_its_sandbox_and_projects(tmp_path):
    files = _seed(tmp_path)
    bob_root = Path(run_as(BOB, storage_roots_workspace))
    (bob_root / "studio.db").write_text("BOB-DB")
    (bob_root / "assets").mkdir(exist_ok = True)
    (bob_root / "assets" / "data.jsonl").write_text('{"text": "BOB-DATA"}')
    out = run_as(
        BOB,
        tools._bash_exec,
        f"cat {bob_root}/assets/data.jsonl; echo rc=$?; "
        f"echo TAMPERED > {bob_root}/studio.db; echo write_rc=$?; "
        f"echo x > {bob_root}/assets/new.txt; echo assets_rc=$?; "
        "echo mine > own.txt; echo sandbox_rc=$?",
        session_id = "chat",
    )
    assert "BOB-DATA" in out and "rc=0" in out, out
    assert "write_rc=0" not in out and "assets_rc=0" not in out, out
    assert "sandbox_rc=0" in out, out
    assert (bob_root / "studio.db").read_text() == "BOB-DB"
    assert files["unsloth"].read_text() == "UNSLOTH_PRIVATE"


def storage_roots_workspace():
    from utils.paths.storage_roots import workspace_root
    return workspace_root()


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_install_under_a_granted_root_stays_hidden(tmp_path, monkeypatch):
    home = Path(sys.prefix) / f"mu-confinement-home-{os.getpid()}"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    try:
        files = _seed(home.parent)
        auth = home / "auth" / "auth.db"
        auth.parent.mkdir(parents = True, exist_ok = True)
        auth.write_text("OWNER_AUTH_DB", encoding = "utf-8")
        out = run_as(
            BOB,
            tools._bash_exec,
            f"cat {auth}; echo rc=$?; ls {home}; echo ls_rc=$?; python -c 'import sys; print(sys.prefix)'",
            session_id = "chat",
        )
        assert "OWNER_AUTH_DB" not in out and "rc=0" not in out and "ls_rc=0" not in out, out
        assert sys.prefix in out, out
        rules = run_as(BOB, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
        assert all(not tool_confinement._contains(p, str(home.resolve())) for p, _ in rules)
    finally:
        import shutil
        shutil.rmtree(home, ignore_errors = True)


def test_sandbox_home_override_is_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(tmp_path / "custom-sandboxes"))
    roots = run_as(BOB, tool_confinement._writable_roots)
    assert str((tmp_path / "custom-sandboxes" / "accounts" / "bob-id").resolve()) in roots


def test_landlock_below_abi_3_is_refused(monkeypatch):
    monkeypatch.setattr(tool_confinement, "landlock_abi", lambda: 2)
    assert tool_confinement._linux_confinement(tools._SANDBOX_SITE_DIR) is None


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_sandbox_home_under_a_granted_root_hides_other_accounts(tmp_path, monkeypatch):
    base = Path(sys.prefix) / f"mu-shared-sandboxes-{os.getpid()}"
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(base))
    try:
        alice_dir = Path(run_as(ALICE, tools._get_workdir, "chat"))
        (alice_dir / "alice-secret.txt").write_text("ALICE_PRIVATE")
        out = run_as(
            BOB,
            tools._bash_exec,
            f"cat {alice_dir}/alice-secret.txt; echo rc=$?; "
            "echo mine > own.txt; cat own.txt; echo own_rc=$?",
            session_id = "chat",
        )
        assert "ALICE_PRIVATE" not in out and "rc=0" not in out.replace("own_rc=0", ""), out
        assert "own_rc=0" in out and "mine" in out, out
        rules = run_as(BOB, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
        assert all(not tool_confinement._contains(p, str(alice_dir)) for p, _ in rules)
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors = True)


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_projects_home_under_a_granted_root_hides_other_accounts(tmp_path, monkeypatch):
    from utils.paths.storage_roots import project_workspaces_root

    base = Path(sys.prefix) / f"mu-shared-projects-{os.getpid()}"
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(base))
    try:
        alice_projects = run_as(ALICE, project_workspaces_root)
        alice_projects.mkdir(parents = True, exist_ok = True)
        (alice_projects / "plan.md").write_text("ALICE_PRIVATE")
        bob_projects = run_as(BOB, project_workspaces_root)
        out = run_as(
            BOB,
            tools._bash_exec,
            f"cat {alice_projects}/plan.md; echo rc=$?; "
            f"echo mine > {bob_projects}/own.md; echo own_rc=$?",
            session_id = "chat",
        )
        assert "ALICE_PRIVATE" not in out and "rc=0" not in out.replace("own_rc=0", ""), out
        assert "own_rc=0" in out, out
        assert (bob_projects / "own.md").read_text().strip() == "mine"
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors = True)


def test_macos_profile_hides_shared_sandbox_and_project_bases(tmp_path, monkeypatch):
    prefix = tmp_path / "prefix"
    sandboxes = prefix / "sandboxes"
    projects = prefix / "projects"
    sandboxes.mkdir(parents = True)
    projects.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_HOME", str(sandboxes))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(projects))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(tool_confinement, "_interpreter_roots", lambda: [str(prefix)])
    run_as(ALICE, tools._get_workdir, "chat")

    profile = run_as(BOB, tools._account_confinement).wrap(["bash"])[2]
    bob_sandbox = str(Path(run_as(BOB, tools.sandbox_root)).resolve())
    alice_sandbox = str(Path(run_as(ALICE, tools.sandbox_root)).resolve())
    for base in (sandboxes, projects):
        deny = profile.index(f'(deny file-read* file-write* (subpath "{base.resolve()}"))')
        assert profile.index(f'(allow file-read* (subpath "{prefix.resolve()}"))') < deny
    allow_own = profile.index(f'(allow file-read* file-write* (subpath "{bob_sandbox}"))')
    assert (
        profile.rindex(f'(deny file-read* file-write* (subpath "{sandboxes.resolve()}"))')
        < allow_own
    )
    assert f'(subpath "{alice_sandbox}")' not in profile


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_hf_cache_under_a_granted_root_is_hidden(monkeypatch):
    from utils import hf_cache_settings

    cache = Path(sys.prefix) / f"mu-shared-hf-{os.getpid()}"
    secret = cache / "models--acme--private" / "snapshots" / "x" / "config.json"
    secret.parent.mkdir(parents = True)
    secret.write_text("ACME_PRIVATE", encoding = "utf-8")
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {"HF_HUB_CACHE": str(cache)})
    try:
        run_as(BOB, tools._get_workdir, "chat")
        rules = run_as(BOB, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
        assert all(not tool_confinement._contains(p, str(secret)) for p, _ in rules)
        out = run_as(BOB, tools._bash_exec, f"cat {secret}; echo rc=$?", session_id = "chat")
        assert "ACME_PRIVATE" not in out and "rc=0" not in out, out
    finally:
        import shutil
        shutil.rmtree(cache, ignore_errors = True)


def test_macos_profile_hides_the_hf_cache(tmp_path, monkeypatch):
    from utils import hf_cache_settings

    prefix = tmp_path / "prefix"
    cache = prefix / "hf" / "hub"
    cache.mkdir(parents = True)
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {"HF_HUB_CACHE": str(cache)})
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(tool_confinement, "_interpreter_roots", lambda: [str(prefix)])

    profile = run_as(BOB, tools._account_confinement).wrap(["bash"])[2]
    deny = profile.index(f'(deny file-read* file-write* (subpath "{cache.resolve()}"))')
    assert profile.index(f'(allow file-read* (subpath "{prefix.resolve()}"))') < deny


def test_bases_outside_a_granted_root_leave_the_landlock_rules_unchanged(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux rule builder")
    from utils.paths.storage_roots import studio_root

    run_as(ALICE, tools._get_workdir, "chat")
    rules = run_as(ALICE, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
    install_only = tool_confinement._existing((run_as(ALICE, studio_root),))
    original = tool_confinement._protected_roots
    tool_confinement._protected_roots = lambda: install_only
    try:
        baseline = run_as(ALICE, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
    finally:
        tool_confinement._protected_roots = original
    assert rules == baseline


def test_default_layout_denies_exactly_the_previous_macos_roots(tmp_path, monkeypatch):
    import tempfile as _tempfile

    home = tmp_path / "home"
    (home / "Documents").mkdir(parents = True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(home / "Documents" / "Unsloth Studio"))
    shared_tmp = tmp_path / "tmp"
    shared_tmp.mkdir()
    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(shared_tmp))
    from utils import hf_cache_settings

    monkeypatch.delenv("XDG_CACHE_HOME", raising = False)
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    monkeypatch.setattr(hf_cache_settings, "_stored_cache_home", lambda: None)
    monkeypatch.setattr(hf_cache_settings, "_stored_history", lambda: [])
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    profile = run_as(ALICE, tools._account_confinement).wrap(["bash"])[2]
    denied = {
        line.split('"')[1] for line in profile.splitlines() if line.startswith("(deny file-read*")
    }
    assert denied == {
        "/private/var/folders",
        str((tmp_path / "studio").resolve()),
        str((shared_tmp / "unsloth-studio").resolve()),
        str(home.resolve()),
    }


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_link_to_an_ancestor_of_the_install_root_stays_hidden(tmp_path, monkeypatch):
    """A symlink inside a read root pointing at an ancestor of the install root is opened as that ancestor, so it must not become a rule of its own."""
    prefix = Path(os.path.realpath(sys.prefix))
    parent = prefix / f"mu-ancestor-{os.getpid()}"
    home = parent / "unsloth-studio"
    link = prefix / f"mu-root-link-{os.getpid()}"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    try:
        auth = home / "auth" / "auth.db"
        auth.parent.mkdir(parents = True, exist_ok = True)
        auth.write_text("OWNER_AUTH_DB", encoding = "utf-8")
        link.symlink_to(parent, target_is_directory = True)
        out = run_as(
            BOB,
            tools._bash_exec,
            f"cat {link}/unsloth-studio/auth/auth.db; echo rc=$?; "
            "echo mine > own.txt; cat own.txt; echo own_rc=$?",
            session_id = "chat",
        )
        assert "OWNER_AUTH_DB" not in out and "rc=0" not in out.replace("own_rc=0", ""), out
        assert "own_rc=0" in out and "mine" in out, out
        rules = run_as(BOB, tool_confinement._landlock_rules, 3, tools._SANDBOX_SITE_DIR)
        assert all(not tool_confinement._contains(p, str(link)) for p, _ in rules), rules
    finally:
        import shutil
        link.unlink(missing_ok = True)
        shutil.rmtree(parent, ignore_errors = True)


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_shared_temporary_base_under_a_granted_root_hides_other_accounts(tmp_path, monkeypatch):
    import tempfile as _tempfile

    from utils.paths import storage_roots

    base = Path(os.path.realpath(sys.prefix)) / f"mu-shared-tmp-{os.getpid()}"
    base.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(base))
    try:
        alice_tmp = run_as(ALICE, storage_roots.tmp_root)
        alice_tmp.mkdir(parents = True, exist_ok = True)
        (alice_tmp / "dataset.jsonl").write_text("ALICE_PRIVATE", encoding = "utf-8")
        bob_tmp = run_as(BOB, storage_roots.tmp_root)
        out = run_as(
            BOB,
            tools._bash_exec,
            f"cat {alice_tmp}/dataset.jsonl; echo rc=$?; "
            f"echo mine > {bob_tmp}/own.txt; echo own_rc=$?",
            session_id = "chat",
        )
        assert "ALICE_PRIVATE" not in out and "rc=0" not in out.replace("own_rc=0", ""), out
        assert "own_rc=0" in out, out
        assert (bob_tmp / "own.txt").read_text().strip() == "mine"
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors = True)


def test_macos_profile_keeps_the_user_cache_dir_readable(tmp_path, monkeypatch):
    """dyld reads its closure cache under the per-user cache dir, so that one subtree is allowed back after the /var/folders deny."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(tool_confinement.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    cache = "/var/folders/ab/cd/C"
    monkeypatch.setattr(
        tool_confinement, "_darwin_user_cache_dirs", lambda: (cache, "/private" + cache)
    )
    profile = run_as(ALICE, tools._account_confinement).wrap(["bash"])[2]
    deny = profile.index('(deny file-read* file-write* (subpath "/private/var/folders")')
    assert deny < profile.index(f'(allow file-read* (subpath "{cache}"))')
    assert deny < profile.index(f'(allow file-read* (subpath "/private{cache}"))')


@pytest.mark.skipif(not LANDLOCK, reason = "Landlock not available on this kernel")
def test_another_accounts_command_text_is_not_on_the_process_list(tmp_path):
    """Landlock cannot deny /proc/<pid>/cmdline, so a confined command runs from a file."""
    import threading

    _seed(tmp_path)
    marker = "ALICE_PROMPT_9f3a"
    alice = {}

    def run_alice():
        alice["out"] = run_as(
            ALICE, tools._bash_exec, f"echo begin; sleep 4; echo {marker}", session_id = "chat"
        )

    thread = threading.Thread(target = run_alice)
    thread.start()
    try:
        snoop = (
            "import os, time\n"
            "hits = []\n"
            "for _ in range(20):\n"
            "    for pid in os.listdir('/proc'):\n"
            "        if not pid.isdigit():\n"
            "            continue\n"
            "        try:\n"
            "            text = open(os.path.join('/proc', pid, 'cmdline'), 'rb').read()\n"
            "        except OSError:\n"
            "            continue\n"
            f"        if b'{marker}' in text and b'python' not in text:\n"
            "            hits.append(text)\n"
            "    time.sleep(0.1)\n"
            "print('HITS', len(hits))\n"
        )
        out = run_as(BOB, tools._python_exec, snoop, session_id = "chat")
    finally:
        thread.join()
    assert "HITS 0" in out, out
    assert marker in alice["out"]
    assert not list(Path(run_as(ALICE, tools._get_workdir, "chat")).glob(".studio_cmd_*"))


def _make_private_roots(account):
    for root in (
        storage_roots.workspace_root,
        storage_roots.project_workspaces_root,
        storage_roots.tmp_root,
    ):
        run_as(account, root).mkdir(parents = True, exist_ok = True)


def test_a_tool_launch_after_deletion_refuses_instead_of_recreating_the_roots(matrix):  # noqa: F811
    """A chat authenticated before the delete must not rematerialize the private roots.

    ``_ensure_dirs`` used to create the workspace, sandbox, temporary and project roots
    with a raw ``Path.mkdir``, so a tool process launched after ``delete_account`` returned
    rebuilt an orphaned account tree and ran in it.
    """
    _, _, accounts = matrix
    alice = storage.get_account("alice")
    _make_private_roots(alice)
    Path(run_as(alice, tools.sandbox_root)).mkdir(parents = True, exist_ok = True)

    storage.delete_account(alice.account_id, accounts.retire_account_roots)

    workspace = run_as(alice, storage_roots.workspace_root)
    tmp = run_as(alice, storage_roots.tmp_root)
    projects = run_as(alice, storage_roots.project_workspaces_root)
    sandbox = Path(run_as(alice, tools.sandbox_root))
    assert not any(root.exists() for root in (workspace, tmp, projects, sandbox))

    for helper in (
        tool_confinement._readable_account_roots,
        tool_confinement._writable_roots,
    ):
        with pytest.raises(storage_roots.RetiredAccountError):
            run_as(alice, helper)

    # The public entry a tool call uses refuses too, as a returned error rather than a raise.
    with pytest.raises(storage_roots.RetiredAccountError):
        run_as(alice, tools._get_workdir, "chat")
    result = run_as(alice, tools._bash_exec, "echo hi", "chat")
    assert "account has been deleted" in result

    assert not any(root.exists() for root in (workspace, tmp, projects, sandbox))


def test_runtime_secret_mounts_are_excluded_from_the_system_grant(tmp_path, monkeypatch):
    """A managed tool must not read /run/secrets or /run/credentials; the rest of /run stays readable."""
    if sys.platform != "linux":
        pytest.skip("Linux rule builder")
    run = tmp_path / "run"
    for name in ("user/1000", "secrets", "credentials", "lock"):
        (run / name).mkdir(parents = True)
    (run / "secrets" / "db_password").write_text("hunter2")
    (run / "credentials" / "svc" / "token").parent.mkdir()
    (run / "credentials" / "svc" / "token").write_text("token")
    monkeypatch.setattr(tool_confinement, "_SYSTEM_READ_ROOTS", (str(run),))
    # The shipped list, relocated under the fake /run, so the test reads the real constant.
    monkeypatch.setattr(
        tool_confinement,
        "_PRIVATE_RUNTIME_ROOTS",
        tuple(
            str(run / os.path.relpath(p, "/run")) for p in tool_confinement._PRIVATE_RUNTIME_ROOTS
        ),
    )
    run_as(ALICE, tools._get_workdir, "chat")
    rules = run_as(ALICE, tool_confinement._landlock_rules, 6, tools._SANDBOX_SITE_DIR)
    granted = [p for p, _ in rules]
    resolved = str(run.resolve())
    print(f"granted under {resolved}: {[p for p in granted if p.startswith(resolved)]}")
    assert f"{resolved}/lock" in granted
    assert resolved not in granted
    for private in ("secrets", "credentials", "user"):
        assert not any(
            p == f"{resolved}/{private}" or p.startswith(f"{resolved}/{private}/") for p in granted
        ), private


def test_privileged_etc_secrets_are_excluded_from_the_system_grant(tmp_path, monkeypatch):
    """A root-run Studio must not hand a managed tool /etc/shadow, sudoers, host keys or private TLS keys."""
    if sys.platform != "linux":
        pytest.skip("Linux rule builder")
    etc = tmp_path / "etc"
    for name in ("ssl/private", "ssl/certs", "ssh", "sudoers.d", "security"):
        (etc / name).mkdir(parents = True)
    for name in (
        "shadow",
        "sudoers",
        "sudoers.d/admins",
        "security/opasswd",
        "ssl/private/server.key",
        "ssh/ssh_host_ed25519_key",
        "hosts",
        "ssh/ssh_config",
        "ssh/ssh_host_ed25519_key.pub",
        "ssl/certs/ca.pem",
    ):
        (etc / name).write_text(name)
    monkeypatch.setattr(tool_confinement, "_SYSTEM_READ_ROOTS", (str(etc),))
    monkeypatch.setattr(
        tool_confinement,
        "_PRIVATE_SYSTEM_PATHS",
        tuple(
            str(etc / os.path.relpath(p, "/etc")) for p in tool_confinement._PRIVATE_SYSTEM_PATHS
        ),
    )
    run_as(ALICE, tools._get_workdir, "chat")
    rules = run_as(ALICE, tool_confinement._landlock_rules, 6, tools._SANDBOX_SITE_DIR)
    granted = [p for p, _ in rules]
    resolved = str(etc.resolve())
    assert resolved not in granted
    for public in ("hosts", "ssh/ssh_config", "ssh/ssh_host_ed25519_key.pub", "ssl/certs"):
        assert f"{resolved}/{public}" in granted, public
    for private in (
        "shadow",
        "sudoers",
        "sudoers.d",
        "security/opasswd",
        "ssl/private",
        "ssh/ssh_host_ed25519_key",
    ):
        assert not any(
            p == f"{resolved}/{private}" or p.startswith(f"{resolved}/{private}/") for p in granted
        ), private
