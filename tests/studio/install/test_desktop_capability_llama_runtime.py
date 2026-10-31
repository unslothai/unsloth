# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A wheel missing studio reports null silently, and a payload change blocks desktop launch."""

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import typing
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = PACKAGE_ROOT / "studio" / "install_llama_prebuilt.py"
SPEC = importlib.util.spec_from_file_location("studio_install_llama_prebuilt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ILP = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ILP
SPEC.loader.exec_module(ILP)

MANAGED_RS = PACKAGE_ROOT / "studio" / "src-tauri" / "src" / "preflight" / "managed.rs"

# Spelled out rather than derived from the command's dict, which could not notice a dropped key.
PRE_PR_KEYS: dict[str, type | tuple[type, ...]] = {
    "desktop_protocol_version": int,
    "desktop_manageability_version": int,
    "supports_provision_desktop_auth": bool,
    "supports_api_only": bool,
    "supports_desktop_backend_ownership": bool,
    "studio_install_ok": bool,
    "studio_install_reason": str,
    "version": str,
}
NEW_KEYS = ("llama_runtime_ok", "llama_runtime_reason")

# Bumping either tells older desktops the protocol changed; additive keys must not.
EXPECTED_PROTOCOL_VERSION = 1
EXPECTED_MANAGEABILITY_VERSION = 2


def _venv_python() -> Path | None:
    """The interpreter of the prepared venv, or None when it was never built."""
    override = os.environ.get("UNSLOTH_DESKTOP_CAP_VENV")
    root = (
        Path(override).expanduser()
        if override
        else Path(os.environ.get("UNSLOTH_WORKSPACE") or PACKAGE_ROOT.parent)
        / "temp"
        / "venv_desktop_cap"
    )
    for candidate in (root / "bin" / "python", root / "Scripts" / "python.exe"):
        if candidate.is_file():
            return candidate
    return None


VENV_PYTHON = _venv_python()
NEEDS_VENV = pytest.mark.skipif(
    VENV_PYTHON is None,
    reason = "no prepared venv with the CLI installed; see this module's docstring",
)


def _console_script() -> Path:
    assert VENV_PYTHON is not None
    name = "unsloth.exe" if os.name == "nt" else "unsloth"
    return VENV_PYTHON.parent / name


def _capabilities(
    install_dir: Path,
    tmp_path: Path,
    *,
    json_output: bool = True,
):
    """Runs the installed CLI outside the checkout, with Studio and llama.cpp paths pointed at fixtures."""
    env = dict(os.environ)
    env["UNSLOTH_LLAMA_CPP_PATH"] = str(install_dir)
    env["UNSLOTH_STUDIO_HOME"] = str(tmp_path / "studio_home")
    # cwd matters: from the checkout `studio` would resolve to the source tree, not site-packages.
    args = [str(_console_script()), "studio", "desktop-capabilities"]
    if json_output:
        args.append("--json")
    result = subprocess.run(
        args,
        cwd = str(tmp_path),
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    return result.returncode, result.stdout


def _shared_health_groups() -> list[list[str]]:
    """Runtime file groups every install kind on this platform shares, read from the health tables."""
    host = ILP.platform_only_host()
    prefix = "windows-" if host.is_windows else "macos-" if host.is_macos else "linux-"
    kinds = sorted(k for k in ILP.INSTALL_KIND_BACKENDS if k.startswith(prefix))
    assert kinds, f"no install kinds for {prefix!r}"
    shared = set.intersection(
        *(
            {
                tuple(group)
                for group in ILP.runtime_payload_health_groups(
                    kind,
                    source_label = None,
                    runtime_name = None,
                    tag = "b10830",
                )
            }
            for kind in kinds
        )
    )
    assert shared, "the shared payload must not be empty, or 'complete' means nothing"
    return [list(group) for group in sorted(shared)]


def _complete_tree(root: Path) -> Path:
    """Empty files suffice: installed_runtime_health only looks and never executes."""
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    runtime_dir.mkdir(parents = True, exist_ok = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"release_tag": "b10830-mix-d5c17a0", "tag": "b10830"}) + "\n",
        encoding = "utf-8",
    )
    for group in _shared_health_groups():
        (runtime_dir / group[0].replace("*", "")).write_text("x", encoding = "utf-8")
    ext = ".exe" if host.is_windows else ""
    for name in ("server", "quantize"):
        (runtime_dir / f"llama-{name}{ext}").write_text("x", encoding = "utf-8")
    return runtime_dir


def _assert_pre_pr_payload_intact(payload: dict) -> None:
    """Every key an older desktop already reads, still present and still its old type."""
    for key, expected in PRE_PR_KEYS.items():
        assert key in payload, f"{key} disappeared from the capability payload"
        # bool subclasses int, so an int field must not accept a bool.
        if expected is int:
            assert isinstance(payload[key], int) and not isinstance(
                payload[key], bool
            ), f"{key} is {payload[key]!r}, not an int"
        else:
            assert isinstance(payload[key], expected), f"{key} is {payload[key]!r}"
    assert payload["desktop_protocol_version"] == EXPECTED_PROTOCOL_VERSION
    assert payload["desktop_manageability_version"] == EXPECTED_MANAGEABILITY_VERSION


@NEEDS_VENV
def test_studio_is_importable_from_an_installed_wheel():
    """The whole feature hangs off this import succeeding outside a source checkout.

    ``desktop_capabilities`` swallows every probe exception, so a ``studio`` package
    missing from the wheel would report null for every user with no error. Run from a cwd
    that is not the checkout, or the tree beside the tests answers instead.
    """
    assert VENV_PYTHON is not None
    probe = (
        "import studio.install_llama_prebuilt as m; "
        "print(m.__file__); print(callable(m.installed_runtime_health))"
    )
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", probe],
        cwd = str(VENV_PYTHON.parent),
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr
    module_file, callable_flag = result.stdout.strip().splitlines()
    assert callable_flag == "True"
    assert (
        "site-packages" in module_file
    ), f"resolved to {module_file}, not the installed package; the checkout shadowed it"


@NEEDS_VENV
def test_nothing_installed_reports_null_rather_than_false(tmp_path):
    """The desktop repairs on an explicit false, so NotInstalled must not report one."""
    empty = tmp_path / "empty"
    empty.mkdir()
    rc, out = _capabilities(empty, tmp_path)
    assert rc == 0
    payload = json.loads(out)
    assert payload["llama_runtime_ok"] is None
    assert payload["llama_runtime_reason"] == ""
    _assert_pre_pr_payload_intact(payload)


@NEEDS_VENV
def test_a_complete_tree_reports_true_with_an_empty_reason(tmp_path):
    root = tmp_path / "llama.cpp"
    _complete_tree(root)
    rc, out = _capabilities(root, tmp_path)
    assert rc == 0
    payload = json.loads(out)
    assert payload["llama_runtime_ok"] is True
    assert payload["llama_runtime_reason"] == ""
    _assert_pre_pr_payload_intact(payload)


@NEEDS_VENV
def test_a_quarantined_library_reports_false_with_a_reason(tmp_path):
    """The shape antivirus leaves behind; the reason string is what the desktop shows."""
    root = tmp_path / "llama.cpp"
    runtime_dir = _complete_tree(root)
    victim = sorted(
        path
        for path in runtime_dir.iterdir()
        if not path.name.startswith("llama-server") and not path.name.startswith("llama-quantize")
    )[0]
    victim.unlink()
    rc, out = _capabilities(root, tmp_path)
    assert rc == 0
    payload = json.loads(out)
    assert payload["llama_runtime_ok"] is False
    assert payload["llama_runtime_reason"], "a false verdict with no reason tells nobody anything"
    _assert_pre_pr_payload_intact(payload)


@NEEDS_VENV
def test_a_missing_llama_server_reports_binaries_missing(tmp_path):
    """The payload groups name libraries only, so on Linux and macOS a quarantined
    llama-server would otherwise read as a complete install."""
    root = tmp_path / "llama.cpp"
    runtime_dir = _complete_tree(root)
    ext = ".exe" if ILP.platform_only_host().is_windows else ""
    (runtime_dir / f"llama-server{ext}").unlink()
    rc, out = _capabilities(root, tmp_path)
    assert rc == 0
    payload = json.loads(out)
    assert payload["llama_runtime_ok"] is False
    assert payload["llama_runtime_reason"] == "llama_runtime_binaries_missing"
    _assert_pre_pr_payload_intact(payload)


@NEEDS_VENV
def test_the_human_readable_form_still_prints_every_key(tmp_path):
    """The desktop reads --json; a support request pastes the bare form, same dict."""
    root = tmp_path / "llama.cpp"
    _complete_tree(root)
    rc, out = _capabilities(root, tmp_path, json_output = False)
    assert rc == 0
    printed = {line.split(":", 1)[0] for line in out.splitlines() if ":" in line}
    for key in (*PRE_PR_KEYS, *NEW_KEYS):
        assert key in printed


@NEEDS_VENV
def test_the_probe_stays_off_the_critical_path_budget(tmp_path):
    """The command runs at every launch under a desktop timeout, so its cost is a product
    constraint. Measured as the import plus the call, which is all the try block does.

    The bound is loose (half a second against a measured ~30ms), so it catches a probe that
    grew a network call or a GPU detection rather than CI jitter.
    """
    assert VENV_PYTHON is not None
    root = tmp_path / "llama.cpp"
    _complete_tree(root)
    script = (
        "import json, time\n"
        "import unsloth_cli.commands.studio\n"
        "t0 = time.perf_counter()\n"
        "from studio.install_llama_prebuilt import installed_runtime_health\n"
        "t1 = time.perf_counter()\n"
        "health = installed_runtime_health()\n"
        "t2 = time.perf_counter()\n"
        "print(json.dumps({'import': t1 - t0, 'call': t2 - t1, 'health': health}))\n"
    )
    env = dict(os.environ)
    env["UNSLOTH_LLAMA_CPP_PATH"] = str(root)
    env["UNSLOTH_STUDIO_HOME"] = str(tmp_path / "studio_home")
    started = time.perf_counter()
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", script],
        cwd = str(tmp_path),
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr
    measured = json.loads(result.stdout)
    assert measured["health"] == [True, ""]
    total = measured["import"] + measured["call"]
    assert total < 0.5, f"the probe added {total:.3f}s to every launch"
    assert time.perf_counter() - started < 120


@NEEDS_VENV
def test_an_unimportable_probe_leaves_the_verdict_null(tmp_path):
    """The ``except Exception`` arm, exercised rather than read.

    A denied tree, a corrupt marker or a missing ``studio`` package must all land on null
    with the rest of the payload unchanged. ImportError is the one that would hit every
    user at once, so it is the one simulated: a meta_path hook refusing the module while a
    complete tree, which would otherwise answer true, sits on disk.
    """
    assert VENV_PYTHON is not None
    root = tmp_path / "llama.cpp"
    _complete_tree(root)
    driver = tmp_path / "blocked.py"
    driver.write_text(
        "import sys\n"
        "class _Blocker:\n"
        "    def find_spec(self, name, path = None, target = None):\n"
        "        if name == 'studio.install_llama_prebuilt':\n"
        "            raise ImportError('simulated: studio not shipped')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        "from unsloth_cli import app\n"
        "sys.argv = ['unsloth', 'studio', 'desktop-capabilities', '--json']\n"
        "app()\n",
        encoding = "utf-8",
    )
    env = dict(os.environ)
    env["UNSLOTH_LLAMA_CPP_PATH"] = str(root)
    env["UNSLOTH_STUDIO_HOME"] = str(tmp_path / "studio_home")
    result = subprocess.run(
        [str(VENV_PYTHON), str(driver)],
        cwd = str(tmp_path),
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["llama_runtime_ok"] is None
    assert payload["llama_runtime_reason"] == ""
    _assert_pre_pr_payload_intact(payload)


def test_the_command_emits_exactly_the_pre_pr_keys_plus_the_two_new_ones():
    """Read off the source, so it holds without the venv. A key added without a matching
    Option<T> in managed.rs is invisible to the desktop; a key removed breaks it.
    """
    source = (PACKAGE_ROOT / "unsloth_cli" / "commands" / "studio.py").read_text(encoding = "utf-8")
    body = source.split("def desktop_capabilities(", 1)[1]
    body = body.split("if json_output:", 1)[0]
    emitted = set(re.findall(r'^\s+"([a-z_]+)":', body, flags = re.MULTILINE))
    emitted |= set(re.findall(r'payload\["([a-z_]+)"\]', body))
    assert emitted == set(PRE_PR_KEYS) | set(NEW_KEYS), sorted(emitted)


@NEEDS_VENV
def test_dropping_the_new_keys_yields_the_pre_pr_payload(tmp_path):
    """Additive, proven by subtraction: strip the two keys and what is left is a payload the
    pre-PR desktop already accepted. Neither version constant may move, since bumping one
    tells a desktop to treat the CLI as stale.
    """
    root = tmp_path / "llama.cpp"
    _complete_tree(root)
    rc, out = _capabilities(root, tmp_path)
    assert rc == 0
    payload = json.loads(out)
    stripped = {key: value for key, value in payload.items() if key not in NEW_KEYS}
    assert set(stripped) == set(PRE_PR_KEYS)
    _assert_pre_pr_payload_intact(stripped)


def test_the_desktop_reads_every_emitted_key_as_optional():
    """A desktop built from this tree must survive a pre-PR CLI sending neither new key.
    serde fills an absent Option with None and managed.rs only treats Some(false) as broken,
    so absent is safe only while every field stays an Option.
    """
    source = MANAGED_RS.read_text(encoding = "utf-8")
    struct_body = source.split("struct DesktopCapability {", 1)[1].split("\n}", 1)[0]
    fields = dict(re.findall(r"^\s+([a-z_]+):\s*(.+),$", struct_body, flags = re.MULTILINE))
    for key in (*PRE_PR_KEYS, *NEW_KEYS):
        assert key in fields, f"the desktop struct has no field for {key}"
        assert fields[key].startswith(
            "Option<"
        ), f"{key} is {fields[key]}, so a CLI that omits it fails the whole parse"


def test_unknown_keys_do_not_break_the_desktop_parse():
    """Keeps the desktop parse tolerant: the capability struct must never gain deny_unknown_fields."""
    source = MANAGED_RS.read_text(encoding = "utf-8")
    assert "deny_unknown_fields" not in source
    prologue = source.split("struct DesktopCapability {", 1)[0]
    assert "deny_unknown_fields" not in prologue.rsplit("#[derive", 1)[-1]


def test_unknown_keys_do_not_break_the_cli_side_consumer():
    """The CI probe reads named keys only, so extra keys in the payload must stay inert."""
    probe = PACKAGE_ROOT / ".github" / "scripts" / "interrupted_install_probe.py"
    if not probe.is_file():
        pytest.skip("CI probe script not present in this tree")
    source = probe.read_text(encoding = "utf-8")
    assert 'parsed.get("studio_install_ok")' in source
    assert not re.search(r"set\(parsed", source)
    payload = {key: ("" if kind is str else kind()) for key, kind in PRE_PR_KEYS.items()}
    payload["studio_install_ok"] = True
    payload.update({key: None for key in NEW_KEYS})
    payload["some_future_key"] = {"nested": [1, 2, 3]}
    payload["another_future_key"] = "ignored"
    parsed = json.loads(json.dumps(payload))
    assert isinstance(parsed, dict)
    value = parsed.get("studio_install_ok")
    assert isinstance(value, bool) and value is True


def test_the_managed_probe_is_skipped_when_a_custom_runtime_is_active(monkeypatch):
    """Skip the managed probe under a user's own runtime; the backend would never open that tree."""
    active = _active_helper()
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    assert active() is True

    pinned = pathlib.Path(__file__).resolve().parents[3] / "studio" / "install_llama_prebuilt.py"
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(pinned))
    assert active() is False

    monkeypatch.setenv("LLAMA_SERVER_PATH", "   ")
    assert active() is True


def test_the_managed_runtime_path_override_is_not_treated_as_a_custom_runtime(monkeypatch):
    """UNSLOTH_LLAMA_CPP_PATH moves the managed root itself, so default_managed_llama_dir
    already grades exactly the tree that variable names. Skipping on it would drop the
    coverage for every user who relocated their install."""
    active = _active_helper()
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", "/opt/relocated/llama.cpp")
    assert active() is True


def _helper_namespace(studio_home = None):
    """Reads the helper block from the CLI source, since importing the module pulls in typer."""
    text = (
        pathlib.Path(__file__).resolve().parents[3] / "unsloth_cli" / "commands" / "studio.py"
    ).read_text(encoding = "utf-8")
    start = text.index("def _managed_llama_runtime_is_the_active_one")
    end = text.index('@studio_app.command("desktop-capabilities"', start)
    master_start = text.index("def _master_root_llama_dir")
    master_end = text.index("def _ensure_studio_env_exported", master_start)
    namespace = {
        "os": __import__("os"),
        "sys": __import__("sys"),
        "Optional": typing.Optional,
        "Path": pathlib.Path,
        "_PACKAGE_ROOT": pathlib.Path(__file__).resolve().parents[3],
        "STUDIO_HOME": pathlib.Path(studio_home)
        if studio_home is not None
        else pathlib.Path.home() / ".unsloth" / "studio",
        "_STUDIO_HOME_IS_CUSTOM": studio_home is not None,
    }
    exec(compile(text[master_start:master_end], "<helper>", "exec"), namespace)
    exec(compile(text[start:end], "<helper>", "exec"), namespace)
    return namespace


def _active_helper():
    return _helper_namespace()["_managed_llama_runtime_is_the_active_one"]


def test_an_inferred_studio_root_is_graded_not_the_legacy_tree(tmp_path, monkeypatch):
    """Grade the studio root inferred from sys.prefix; the desktop scrubs STUDIO_HOME before spawning."""
    for name in (
        "LLAMA_SERVER_PATH",
        "UNSLOTH_LLAMA_CPP_PATH",
        "UNSLOTH_STUDIO_HOME",
        "STUDIO_HOME",
        "UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH",
    ):
        monkeypatch.delenv(name, raising = False)
    root = tmp_path / "custom-studio"
    graded = _helper_namespace(root)["_llama_runtime_to_grade"]()
    assert graded == root / "llama.cpp"
    assert "UNSLOTH_STUDIO_HOME" not in os.environ


def test_a_legacy_install_still_grades_the_legacy_tree(tmp_path, monkeypatch):
    """The other direction: nothing was inferred, so nothing is exported and the answer is
    the tree every ordinary install has."""
    for name in (
        "LLAMA_SERVER_PATH",
        "UNSLOTH_LLAMA_CPP_PATH",
        "UNSLOTH_STUDIO_HOME",
        "STUDIO_HOME",
        "UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH",
    ):
        monkeypatch.delenv(name, raising = False)
    graded = _helper_namespace()["_llama_runtime_to_grade"]()
    assert graded == pathlib.Path.home() / ".unsloth" / "llama.cpp"


def test_an_explicit_studio_home_is_left_alone(tmp_path, monkeypatch):
    """An ambient UNSLOTH_STUDIO_HOME is the user's, not an inference, and it must survive
    the call unchanged rather than being replaced by the inferred root."""
    for name in (
        "LLAMA_SERVER_PATH",
        "UNSLOTH_LLAMA_CPP_PATH",
        "UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH",
        "STUDIO_HOME",
    ):
        monkeypatch.delenv(name, raising = False)
    theirs = tmp_path / "theirs"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(theirs))
    graded = _helper_namespace(tmp_path / "inferred")["_llama_runtime_to_grade"]()
    assert graded == theirs / "llama.cpp"
    assert os.environ["UNSLOTH_STUDIO_HOME"] == str(theirs)


def test_a_deleted_llama_server_path_does_not_suppress_the_managed_verdict(tmp_path, monkeypatch):
    """A deleted LLAMA_SERVER_PATH falls through to the managed tree, which must still be graded."""
    active = _active_helper()
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(tmp_path / "gone" / "llama-server"))
    assert active() is True

    present = tmp_path / "llama-server"
    present.write_text("", encoding = "utf-8")
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(present))
    assert active() is False


def test_a_dangling_symlink_pin_falls_through_like_any_absent_pin(tmp_path, monkeypatch):
    """A dangling symlink pin is absent, so the finder falls through and the managed tree is graded."""
    if os.name == "nt":
        pytest.skip("POSIX symlink semantics")
    active = _active_helper()

    link = tmp_path / "pinned"
    os.symlink(tmp_path / "never-existed", link)
    assert os.path.lexists(link) and not link.is_file()
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(link))
    assert active() is True

    (tmp_path / "never-existed").write_text("", encoding = "utf-8")
    assert active() is False


def test_a_user_set_runtime_override_is_not_ours_to_repair(tmp_path, monkeypatch):
    """A user's UNSLOTH_LLAMA_CPP_PATH override is not graded, since setup cannot repair that tree."""
    active = _active_helper()
    override = tmp_path / "relocated" / "llama.cpp"
    server = (
        override / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    )
    server.parent.mkdir(parents = True)
    server.write_text("x", encoding = "utf-8")
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(override))
    _stub_stored_selection(monkeypatch, "/home/someone/older-build")
    assert active() is False

    _stub_stored_selection(monkeypatch, None)
    assert active() is False

    monkeypatch.setenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", "1")
    assert active() is True
    _stub_stored_selection(monkeypatch, "/home/someone/older-build")
    assert active() is False


def test_the_cli_s_own_inferred_override_is_not_mistaken_for_a_user_pin(tmp_path, monkeypatch):
    """A CLI-inferred UNSLOTH_LLAMA_CPP_PATH is not a user pin, and the backend walks past it."""
    active = _active_helper()
    studio_home = tmp_path / "custom-studio"
    managed = studio_home / "llama.cpp"
    server = managed / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    server.parent.mkdir(parents = True)
    server.write_text("", encoding = "utf-8")
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(managed))
    _stub_stored_selection(monkeypatch, "/home/someone/older-build")
    assert active() is False, (
        "the CLI's own inferred override names the managed tree, so the finder skips "
        "it and the stored folder is what the backend opens"
    )

    elsewhere = tmp_path / "hand-built" / "llama.cpp"
    pinned = (
        elsewhere / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    )
    pinned.parent.mkdir(parents = True)
    pinned.write_text("x", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(elsewhere))
    assert active() is False
    _stub_stored_selection(monkeypatch, None)
    assert active() is False
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(managed))
    assert active() is True


def test_a_master_root_grades_the_runtime_beside_studio_not_the_one_under_it(tmp_path, monkeypatch):
    """A master root grades llama.cpp beside studio/, not under it, to match the installer's export."""
    active = _active_helper()
    master = tmp_path / "portable"
    managed = master / "llama.cpp"
    server = managed / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    server.parent.mkdir(parents = True)
    server.write_text("", encoding = "utf-8")
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(master / "studio"))
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(managed))
    _stub_stored_selection(monkeypatch, None)
    assert (
        active() is True
    ), "the exported <master>/llama.cpp is the tree this install owns, so it is graded"

    elsewhere = tmp_path / "hand-built" / "llama.cpp"
    pinned = (
        elsewhere / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    )
    pinned.parent.mkdir(parents = True)
    pinned.write_text("x", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(elsewhere))
    assert active() is False


def test_the_master_root_rule_is_the_one_the_export_writes(tmp_path, monkeypatch):
    """One rule, not two that can drift: the value graded has to be the value exported."""
    namespace = _helper_namespace()
    master_dir = namespace["_master_root_llama_dir"]
    monkeypatch.delenv("UNSLOTH_HOME", raising = False)
    assert master_dir() is None, "no master root means the studio home decides, as before"
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    assert master_dir() == master.resolve() / "llama.cpp"
    monkeypatch.setenv("UNSLOTH_HOME", "   ")
    assert master_dir() is None


def test_an_override_that_holds_no_server_does_not_outrank_the_stored_folder(tmp_path, monkeypatch):
    """An override with no server yields to the stored folder, since the backend never loads it."""
    active = _active_helper()
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path / "never-installed"))
    _stub_stored_selection(monkeypatch, "/home/someone/older-build")
    assert active() is False, "the finder walks past an empty override to the stored folder"

    _stub_stored_selection(monkeypatch, None)
    assert active() is True


@pytest.mark.skipif(os.name == "nt", reason = "POSIX ~name expansion")
def test_an_override_naming_no_account_answers_instead_of_raising(tmp_path, monkeypatch):
    """Codex 3962938521, P2. Path.expanduser raises RuntimeError for a "~name" that resolves
    to no account, which an override left in a service unit or a .env after a rename does, and
    this doctor's whole job is to answer. Every other reader of the variable goes through
    expanded_user_path, which hands an unresolvable name back unchanged, so the override then
    reaches the search as an ordinary path, finds nothing, and the documented order continues."""
    active = _active_helper()
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", "~no-such-account-9d3f/llama.cpp")
    with pytest.raises(RuntimeError):
        pathlib.Path("~no-such-account-9d3f/llama.cpp").expanduser()
    _stub_stored_selection(monkeypatch, None)
    assert active() is True, "an unexpandable override holds no server, so the search walks on"


def _stub_stored_selection(monkeypatch, selected):
    """The helper imports by name, so parent packages must be in sys.modules or real ones get pulled in."""
    import types

    for name in ("studio", "studio.backend", "studio.backend.utils"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    settings = types.ModuleType("studio.backend.utils.llama_cpp_path_settings")
    settings.get_stored_custom_llama_cpp_path = lambda: selected
    settings.llama_server_candidates = _real_llama_server_candidates
    # The real reader: without it the guarded import failed and tests graded the fallback.
    settings.expanded_user_path = lambda value: pathlib.Path(os.path.expanduser(str(value)))
    monkeypatch.setitem(sys.modules, "studio.backend.utils.llama_cpp_path_settings", settings)
    prebuilt = types.ModuleType("studio.install_llama_prebuilt")
    prebuilt.default_managed_llama_dir = _managed_dir_rule
    monkeypatch.setitem(sys.modules, "studio.install_llama_prebuilt", prebuilt)


def _managed_dir_rule():
    """Mirrors default_managed_llama_dir: override, then the custom studio home, then legacy root."""
    override = (os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or "").strip()
    if override:
        return pathlib.Path(override).expanduser()
    home = (os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or "").strip()
    if home:
        root = pathlib.Path(home).expanduser()
        if root != pathlib.Path.home() / ".unsloth" / "studio":
            return root / "llama.cpp"
    return pathlib.Path.home() / ".unsloth" / "llama.cpp"


def _real_llama_server_candidates(directory):
    """The shipped layouts, read off llama_cpp_path_settings rather than retyped."""
    root = pathlib.Path(directory)
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    candidates = [root / name, root / "build" / "bin" / name]
    if sys.platform == "win32":
        candidates.append(root / "build" / "bin" / "Release" / name)
    return tuple(candidates)


def test_a_skipped_runtime_verdict_says_so_in_its_reason(monkeypatch):
    """A skipped verdict must name the skip in its reason, so the desktop does not cache it."""
    source = (
        pathlib.Path(__file__).resolve().parents[3] / "unsloth_cli" / "commands" / "studio.py"
    ).read_text(encoding = "utf-8")
    body = source.split("def desktop_capabilities(", 1)[1].split("if json_output:", 1)[0]
    assert 'payload["llama_runtime_reason"] = "llama_runtime_not_managed"' in body
    assert 'payload["llama_runtime_reason"] = "llama_runtime_probe_failed"' in body
    managed_rs = MANAGED_RS.read_text(encoding = "utf-8")
    for reason in ("llama_runtime_not_managed", "llama_runtime_probe_failed"):
        assert reason in managed_rs, "the desktop must know the reason the CLI emits"


def test_the_stored_settings_lookup_can_reach_its_own_database_module(monkeypatch):
    """The stored-folder lookup silently reads as absent unless studio/backend is on sys.path."""
    import subprocess
    import sys as _sys

    root = pathlib.Path(__file__).resolve().parents[3]
    without = subprocess.run(
        [_sys.executable, "-c", "import storage.studio_db"],
        cwd = root,
        capture_output = True,
        text = True,
    )
    assert without.returncode != 0, "storage must not already be importable from the repo root"
    assert "No module named 'storage'" in without.stderr

    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    _active_helper()()
    assert str(root / "studio" / "backend") in _sys.path, (
        "the helper must put the backend on the path itself, or the settings lookup "
        "silently answers None and the skip never happens"
    )


@pytest.mark.skipif(os.name == "nt", reason = "POSIX ~ expansion")
def test_the_finder_expands_the_override_the_way_every_other_reader_does(tmp_path, monkeypatch):
    """Codex 3960401528, P1. ``default_managed_llama_dir``, ``get_stored_custom_llama_cpp_path``
    and the desktop's own pinning all expand UNSLOTH_LLAMA_CPP_PATH; the finder's
    ``Path(custom_llama_cpp)`` was the one literal read. A "~/llama.cpp" written into a
    service unit, a .env or the Windows environment dialog reaches the process unexpanded, so
    the finder searched a folder named ~ beside the working directory, walked past it and
    loaded a different runtime than the probe graded and the desktop fingerprinted.

    Driven against the real finder rather than read off the source: importing it costs about
    a third of a second."""
    backend_dir = PACKAGE_ROOT / "studio" / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    from core.inference.llama_cpp import LlamaCppBackend

    home = tmp_path / "home"
    build = home / "llama.cpp" / "build" / "bin"
    build.mkdir(parents = True)
    server = build / "llama-server"
    server.write_text("", encoding = "utf-8")
    os.chmod(server, 0o755)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", "~/llama.cpp")
    monkeypatch.chdir(tmp_path)
    assert LlamaCppBackend._find_llama_server_binary() == str(server)
