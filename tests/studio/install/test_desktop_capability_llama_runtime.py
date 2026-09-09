# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end contract for the ``llama_runtime_ok`` keys in ``desktop-capabilities``.

The probe function and the payload tables are covered by
``test_installed_runtime_health`` and ``test_keep_install_backcompat_9979``. This file
owns the wiring between them: the command the desktop shells out to must emit those keys
from a real pip install and keep emitting the payload older desktops already parse.

Two failure modes justify the subprocess cost, since an in-process import cannot see them:

* ``studio`` not being packaged. The probe sits inside a bare ``except Exception``, so a
  wheel without ``studio`` would report null forever with no error and no failure from a
  source checkout, which has ``studio/`` on sys.path.
* The command exiting non-zero or dropping a key. The desktop hands stdout to serde_json
  and treats a failed parse as Stale, so a payload change is launch-blocking.

The subprocess tests need a venv with the CLI installed from this tree, which CI does not
build; they skip when it is absent. Build one with::

    uv venv "$UNSLOTH_WORKSPACE/temp/venv_desktop_cap"
    uv pip install --python "$UNSLOTH_WORKSPACE/temp/venv_desktop_cap/bin/python" .

The rest of the file is pure text and dict work and always runs.
"""

import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time
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

# Every key the payload carried before the two llama_runtime keys, with the type the
# desktop's Option<T> fields require. Spelled out rather than derived from the command's
# own dict, which could not notice a key being dropped or retyped.
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

# Bumping either tells an older desktop the CLI speaks a protocol it does not know, so an
# additive key must not touch them.
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
    """Run the installed console script against ``install_dir`` and return (rc, stdout).

    Both overrides point at fixtures so the developer's real ~/.unsloth and Studio home
    stay out of the run.
    """
    env = dict(os.environ)
    env["UNSLOTH_LLAMA_CPP_PATH"] = str(install_dir)
    env["UNSLOTH_STUDIO_HOME"] = str(tmp_path / "studio_home")
    # cwd matters: from the source checkout `studio` resolves to the tree beside it rather
    # than site-packages, the shadowing this file rules out.
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
    """Required runtime file groups every install kind on this platform shares.

    Mirrors ``_kept_install_payload_is_healthy`` for a marker naming no backend, the shape
    the fixture below writes. Derived from the tables so it holds on any platform.
    """
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
    """A marker plus every file the health tables require; returns the runtime dir.

    Empty files, since ``installed_runtime_health`` only looks and never executes.
    """
    host = ILP.platform_only_host()
    runtime_dir = ILP.install_runtime_dir(root, host)
    runtime_dir.mkdir(parents = True, exist_ok = True)
    (root / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps({"release_tag": "b10830-mix-d5c17a0", "tag": "b10830"}) + "\n",
        encoding = "utf-8",
    )
    for group in _shared_health_groups():
        # Dropping the globs from the first pattern still matches it:
        # libggml-cpu*.so* -> libggml-cpu.so.
        (runtime_dir / group[0].replace("*", "")).write_text("", encoding = "utf-8")
    ext = ".exe" if host.is_windows else ""
    for name in ("server", "quantize"):
        (runtime_dir / f"llama-{name}{ext}").write_text("", encoding = "utf-8")
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


# ── part 1: the real command in a real install ───────────────────────────────


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


# ── part 2: interoperability in both directions ──────────────────────────────


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
    """An older desktop meeting a newer CLI. serde ignores unknown fields unless told
    otherwise, so the guard is that nobody adds deny_unknown_fields to the capability
    struct; without it, today's additive keys would have bricked every shipped desktop.
    """
    source = MANAGED_RS.read_text(encoding = "utf-8")
    assert "deny_unknown_fields" not in source
    prologue = source.split("struct DesktopCapability {", 1)[0]
    assert "deny_unknown_fields" not in prologue.rsplit("#[derive", 1)[-1]


def test_unknown_keys_do_not_break_the_cli_side_consumer():
    """The same question for the Python consumer: CI's interrupted-install probe decides
    HEALTHY or REPAIRABLE from this payload by reading named keys, so extra keys must be
    inert. A consumer comparing key sets would fail the build on the next additive field.
    """
    probe = PACKAGE_ROOT / ".github" / "scripts" / "interrupted_install_probe.py"
    if not probe.is_file():
        pytest.skip("CI probe script not present in this tree")
    source = probe.read_text(encoding = "utf-8")
    # The parse is `json.loads` plus `.get`, never a key-set comparison.
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
    """Codex 3958908987, P2. _find_llama_server_binary prefers LLAMA_SERVER_PATH and the
    folder chosen in Studio's settings ahead of the managed tree, so grading the managed tree
    regardless would send a user who runs their own build into repair over an install their
    backend never opens. Offline that repair cannot even succeed."""
    active = _active_helper()
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    assert active() is True

    pinned = pathlib.Path(__file__).resolve().parents[3] / "studio" / "install_llama_prebuilt.py"
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(pinned))
    assert active() is False

    # Whitespace is not a selection: the finder strips before testing it.
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
    """The helper block, read out of the CLI source: importing the module pulls in typer.

    ``studio_home`` stands in for the root ``_resolve_studio_home`` inferred off
    ``sys.prefix``; None is the ordinary legacy install.
    """
    text = (
        pathlib.Path(__file__).resolve().parents[3] / "unsloth_cli" / "commands" / "studio.py"
    ).read_text(encoding = "utf-8")
    start = text.index("def _managed_llama_runtime_is_the_active_one")
    end = text.index('@studio_app.command("desktop-capabilities"', start)
    namespace = {
        "os": __import__("os"),
        "sys": __import__("sys"),
        "Path": pathlib.Path,
        "_PACKAGE_ROOT": pathlib.Path(__file__).resolve().parents[3],
        "STUDIO_HOME": pathlib.Path(studio_home)
        if studio_home is not None
        else pathlib.Path.home() / ".unsloth" / "studio",
        "_STUDIO_HOME_IS_CUSTOM": studio_home is not None,
    }
    exec(compile(text[start:end], "<helper>", "exec"), namespace)
    return namespace


def _active_helper():
    return _helper_namespace()["_managed_llama_runtime_is_the_active_one"]


def test_an_inferred_studio_root_is_graded_not_the_legacy_tree(tmp_path, monkeypatch):
    """Codex 3971960862, P1. The desktop scrubs UNSLOTH_STUDIO_HOME and STUDIO_HOME before
    it spawns this command (MANAGED_CHILD_SCRUBBED_ENV), so default_managed_llama_dir read
    an empty environment and answered the legacy ~/.unsloth/llama.cpp, while
    preflight::managed::inferred_studio_llama_root fingerprints <root>/llama.cpp off the
    same sys.prefix inference the CLI already made. The two halves graded different trees,
    so quarantine in the runtime actually in use never moved the cached verdict."""
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
    # And the environment is left exactly as it was found, since this runs inside a command
    # that goes on to read it.
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
    """Codex 3958908320, P2. _scan_pinned treats an absent pin as no pin and falls through to
    the managed tree, so a LLAMA_SERVER_PATH naming a file that has since been deleted still
    loads the managed runtime. Suppressing the verdict on the bare string left a quarantined
    managed runtime reporting Ready and failing at model load."""
    active = _active_helper()
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(tmp_path / "gone" / "llama-server"))
    assert active() is True

    present = tmp_path / "llama-server"
    present.write_text("", encoding = "utf-8")
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(present))
    assert active() is False


def test_a_dangling_symlink_pin_falls_through_like_any_absent_pin(tmp_path, monkeypatch):
    """Codex 3959620579, P2, correcting this test's own earlier claim. ``_file_status`` asks
    ``Path.is_file()``, which follows the link, so a pin whose target was deleted or
    quarantined reads as "absent" there and the finder walks on to the managed tree. lexists
    called that a pin and left the tree the backend really loads ungraded, so an incomplete
    managed runtime reported Ready. A pin that resolves to a file, executable or not, does
    stop the finder and is still not ours to grade."""
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


def test_an_explicit_runtime_override_outranks_a_stale_stored_folder(tmp_path, monkeypatch):
    """Codex 3959620607, P2. The finder reads UNSLOTH_LLAMA_CPP_PATH at step 1b and the
    stored folder only at step 2, so an override with an older selection still in the
    settings database is the tree the backend opens. Reading the setting first returned
    False, nothing graded that tree, and preflight stayed Ready over a runtime missing files.
    default_managed_llama_dir points at exactly the override, so it is ours to grade."""
    active = _active_helper()
    override = tmp_path / "relocated" / "llama.cpp"
    server = (
        override / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    )
    server.parent.mkdir(parents = True)
    server.write_text("", encoding = "utf-8")
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(override))
    _stub_stored_selection(monkeypatch, "/home/someone/older-build")
    assert active() is True

    # The desktop's own marker is the exception: the finder skips the override when
    # it set it, so the stored folder wins again and the managed tree is not ours.
    monkeypatch.setenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", "1")
    assert active() is False


def test_the_cli_s_own_inferred_override_is_not_mistaken_for_a_user_pin(tmp_path, monkeypatch):
    """Codex 3962938538, P2. Under a custom STUDIO_HOME the CLI's
    _ensure_studio_env_exported writes STUDIO_HOME/llama.cpp into
    UNSLOTH_LLAMA_CPP_PATH and sets no marker, while the backend calls
    mark_managed_llama_cpp_path on the same value before discovery and its finder
    then walks past the override to the stored folder. Reading the marker alone
    made this grade the managed tree as an explicit pin, so a damaged managed tree
    blocked launch and was sent for repair though the backend would never open it."""
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

    # A pin somewhere else under the same studio home is a real user pin and still
    # outranks the stored folder, which is the case the marker check protects.
    elsewhere = tmp_path / "hand-built" / "llama.cpp"
    pinned = (
        elsewhere / "build" / "bin" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    )
    pinned.parent.mkdir(parents = True)
    pinned.write_text("", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(elsewhere))
    assert active() is True


def test_an_override_that_holds_no_server_does_not_outrank_the_stored_folder(tmp_path, monkeypatch):
    """Codex 3960069962, P2. _scan_pinned finds no candidate under an empty or missing
    UNSLOTH_LLAMA_CPP_PATH and walks on to the stored folder, so treating the override as
    final graded a directory nobody loads, answered "not installed", and left the runtime the
    backend really opens ungraded."""
    active = _active_helper()
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path / "never-installed"))
    _stub_stored_selection(monkeypatch, "/home/someone/older-build")
    assert active() is False, "the finder walks past an empty override to the stored folder"

    # With no stored folder either, the finder reaches the managed tree, so there is
    # something to grade again.
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
    """A stored custom folder, without a settings database or the backend package.

    The helper imports ``studio.backend.utils.llama_cpp_path_settings`` by name, so the
    parents have to be in sys.modules too or the real packages are pulled in.
    """
    import types

    for name in ("studio", "studio.backend", "studio.backend.utils"):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    settings = types.ModuleType("studio.backend.utils.llama_cpp_path_settings")
    settings.get_stored_custom_llama_cpp_path = lambda: selected
    # The real layout contract, so the helper and the finder cannot disagree about
    # which folders hold a server.
    settings.llama_server_candidates = _real_llama_server_candidates
    # The real reader, not Path.expanduser: leaving it off the stub made the
    # helper's import fail, and the import is wrapped in a fallback, so every test
    # here silently graded the except branch instead of the code it names.
    settings.expanded_user_path = lambda value: pathlib.Path(os.path.expanduser(str(value)))
    monkeypatch.setitem(sys.modules, "studio.backend.utils.llama_cpp_path_settings", settings)
    # The helper asks install_llama_prebuilt for the managed root, and the stub
    # package above hides the real module, so it is stubbed to the same rule.
    prebuilt = types.ModuleType("studio.install_llama_prebuilt")
    prebuilt.default_managed_llama_dir = _managed_dir_rule
    monkeypatch.setitem(sys.modules, "studio.install_llama_prebuilt", prebuilt)


def _managed_dir_rule():
    """``default_managed_llama_dir``'s rule, retyped only because the stub package
    above hides the real module: the override, else a custom studio home's
    llama.cpp, else the legacy root. The studio-home arm is not decoration, since
    that is the value the helper compares an override against."""
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
    """Codex 3959620616, P2, the CLI half. Null because another runtime is selected is not
    null because nothing is installed: the first expires when the user clears the selection,
    which the desktop's fingerprint does not watch. Naming it lets managed.rs decline to
    cache it. Read off the source, so it holds without the venv."""
    source = (
        pathlib.Path(__file__).resolve().parents[3] / "unsloth_cli" / "commands" / "studio.py"
    ).read_text(encoding = "utf-8")
    body = source.split("def desktop_capabilities(", 1)[1].split("if json_output:", 1)[0]
    assert 'payload["llama_runtime_reason"] = "llama_runtime_not_managed"' in body
    assert "llama_runtime_not_managed" in (
        pathlib.Path(__file__).resolve().parents[3]
        / "studio"
        / "src-tauri"
        / "src"
        / "preflight"
        / "managed.rs"
    ).read_text(encoding = "utf-8"), "the desktop must know the reason the CLI emits"


def test_the_stored_settings_lookup_can_reach_its_own_database_module(monkeypatch):
    """Codex 3958908340, P2, reproduced before fixing: llama_cpp_path_settings imports
    storage.studio_db as a top level package and swallows the failure, so without
    studio/backend on sys.path the stored selection always read as absent and a user whose
    custom folder is set in Studio would be sent to repair a tree their backend never opens.

    The import is asserted through a fresh interpreter, since sys.modules in this one may
    already carry a storage imported by an earlier test."""
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
