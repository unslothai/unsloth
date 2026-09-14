# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where local models live is one policy, not two.

``lmstudio_model_dirs`` / ``ollama_model_dirs`` / ``well_known_model_dirs`` were implemented
twice: in ``utils.paths.storage_roots`` for the model picker and in ``hub.utils.paths`` for
the folder browser. Only one read ``~/.lmstudio/settings.json`` as utf-8-sig, so a BOM'd file
put the user's custom folder in the picker and nowhere else, with nothing logged (#9748).

Simulation notice: this suite runs on one host, so only Linux is native. Windows, macOS and
WSL are ``sys.platform`` / ``_IS_WSL`` / ``_WSL_AUTOMOUNT_ROOT`` monkeypatches. ``os.name`` is
patched only in the pure-string normalizer tests, never in one touching ``tmp_path``: it swaps
pathlib's flavour mid-run and every filesystem assertion after it becomes a lie. The
authoritative signal for those platforms stays the per-OS CI matrix on real runners.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hub.utils import paths as hub_paths
from utils.paths import path_utils
from utils.paths import storage_roots


BOM_UTF8 = b"\xef\xbb\xbf"
BACKEND_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_home(monkeypatch, tmp_path):
    """Point ``Path.home()`` at an empty tmp dir so the host's real home is never read."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("OLLAMA_MODELS", raising = False)
    return home


class _RecordingLogger:
    """Stands in for the module logger.

    Asserted through this rather than capsys: logging is structlog, hub/tests stubs it,
    and whether a debug record reaches stdout is config this test should not pin. What
    matters is that the code calls debug once, with the cause attached.
    """

    def __init__(self):
        self.debug_calls: list[tuple] = []

    def debug(self, event, *args, **kwargs):
        self.debug_calls.append((event, args))
        # Mirror structlog's own %-interpolation so a broken format string raises here
        # instead of silently rendering "%s" to a user later.
        if args:
            event % args

    def __getattr__(self, _name):
        return lambda *a, **k: None


@pytest.fixture
def recording_logger(monkeypatch):
    logger = _RecordingLogger()
    monkeypatch.setattr(storage_roots, "logger", logger)
    return logger


def write_settings(home: Path, payload: bytes) -> Path:
    settings = home / ".lmstudio" / "settings.json"
    settings.parent.mkdir(parents = True, exist_ok = True)
    settings.write_bytes(payload)
    return settings


def settings_json(downloads: Path | str, *, bom: bool = False) -> bytes:
    body = json.dumps({"downloadsFolder": str(downloads)}).encode("utf-8")
    return (BOM_UTF8 if bom else b"") + body


# ---------------------------------------------------------------------------
# A. Encoding of ~/.lmstudio/settings.json
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bom", [False, True], ids = ["no_bom", "utf8_bom"])
def test_downloads_folder_is_found_with_or_without_a_bom(fake_home, bom):
    """The bug in #9748: a BOM'd settings file used to drop the folder on the hub side."""
    downloads = fake_home / "lmstudio-models"
    downloads.mkdir()
    write_settings(fake_home, settings_json(downloads, bom = bom))

    assert storage_roots.lmstudio_model_dirs() == [downloads]
    assert hub_paths.lmstudio_model_dirs() == [downloads]
    assert downloads.resolve() in storage_roots.well_known_model_dirs()


def test_bom_with_crlf_line_endings_is_still_read(fake_home):
    downloads = fake_home / "lmstudio-models"
    downloads.mkdir()
    body = json.dumps({"downloadsFolder": str(downloads)}, indent = 2).replace("\n", "\r\n")
    write_settings(fake_home, BOM_UTF8 + body.encode("utf-8"))

    assert storage_roots.lmstudio_model_dirs() == [downloads]


@pytest.mark.parametrize(
    "payload, label",
    [
        (b"", "empty_file"),
        (b"   \n\t ", "whitespace_only"),
        (b"{ not json", "malformed"),
        (b"null", "json_null"),
        (b'["a", "b"]', "top_level_array"),
        (b'"just a string"', "top_level_string"),
        (b"123", "top_level_number"),
        (b'{"other": "key"}', "missing_key"),
        (b'{"downloadsFolder": ""}', "empty_value"),
        (b'{"downloadsFolder": null}', "null_value"),
        (b'{"downloadsFolder": 123}', "number_value"),
        (b'{"downloadsFolder": []}', "list_value"),
        (b'{"downloadsFolder": {"path": "/x"}}', "object_value"),
        ("\ufeff".encode("utf-16-le"), "utf16_le_bom"),
        ("\ufeff".encode("utf-16-be"), "utf16_be_bom"),
        ("\ufeff".encode("utf-32-le"), "utf32_le_bom"),
        (b"\xff\xfe" + '{"downloadsFolder": "/x"}'.encode("utf-16-le"), "utf16_le_document"),
        (b'{"downloadsFolder": "/caf\xe9"}', "invalid_utf8_bytes"),
    ],
)
def test_unusable_settings_never_raise_and_never_invent_a_directory(fake_home, payload, label):
    """Every degenerate settings file degrades to "no custom folder", never a crash.

    ``~/.lmstudio/models`` does not exist in this fake home, so a clean degrade is [].
    """
    write_settings(fake_home, payload)

    assert storage_roots.lmstudio_model_dirs() == []
    assert hub_paths.lmstudio_model_dirs() == []
    assert storage_roots.well_known_model_dirs() == []


def test_a_parse_failure_is_logged_once_with_its_cause(fake_home, recording_logger):
    """The old code swallowed this into ``except Exception: pass``."""
    write_settings(fake_home, b"{ not json")

    assert storage_roots.lmstudio_model_dirs() == []

    assert len(recording_logger.debug_calls) == 1
    event, args = recording_logger.debug_calls[0]
    assert "LM Studio settings" in event
    # The path and the underlying exception both have to reach the log line, or the
    # message tells an operator nothing they can act on.
    assert len(args) == 2
    assert str(fake_home) in str(args[0])
    assert isinstance(args[1], Exception)


def test_a_readable_settings_file_logs_nothing(fake_home, recording_logger):
    downloads = fake_home / "lmstudio-models"
    downloads.mkdir()
    write_settings(fake_home, settings_json(downloads, bom = True))

    assert storage_roots.lmstudio_model_dirs() == [downloads]
    assert recording_logger.debug_calls == []


# ---------------------------------------------------------------------------
# B. Filesystem state of the configured folder
# ---------------------------------------------------------------------------


def test_a_configured_folder_that_does_not_exist_is_dropped(fake_home):
    write_settings(fake_home, settings_json(fake_home / "nope"))
    assert storage_roots.lmstudio_model_dirs() == []


def test_a_configured_path_that_is_a_file_is_dropped(fake_home):
    target = fake_home / "a-file"
    target.write_text("not a directory", encoding = "utf-8")
    write_settings(fake_home, settings_json(target))

    assert storage_roots.lmstudio_model_dirs() == []


def test_a_symlinked_folder_keeps_the_spelling_the_user_configured(fake_home, tmp_path):
    """Not the resolved target.

    ``_scan_lmstudio_dir`` mints the user-facing model id from this path, so resolving
    here would rename every model behind a symlinked root and invalidate saved
    selections on an existing install.
    """
    real = tmp_path / "real-models"
    real.mkdir()
    link = fake_home / "linked-models"
    link.symlink_to(real, target_is_directory = True)
    write_settings(fake_home, settings_json(link))

    assert storage_roots.lmstudio_model_dirs() == [link]
    # well_known_model_dirs is the other contract: it feeds a containment check, so it
    # does resolve.
    assert real.resolve() in storage_roots.well_known_model_dirs()


def test_a_broken_symlink_is_dropped(fake_home, tmp_path):
    link = fake_home / "dangling"
    link.symlink_to(tmp_path / "was-never-there", target_is_directory = True)
    write_settings(fake_home, settings_json(link))

    assert storage_roots.lmstudio_model_dirs() == []


def test_a_symlink_loop_is_dropped_without_taking_the_list_with_it(fake_home):
    """A raising resolve() costs one candidate, not the list.

    The old storage_roots copy called ``p.resolve()`` unguarded, so one bad entry
    emptied everything.
    """
    loop = fake_home / "loop"
    loop.symlink_to(loop)
    write_settings(fake_home, settings_json(loop))

    default = fake_home / ".lmstudio" / "models"
    default.mkdir(parents = True, exist_ok = True)

    assert storage_roots.lmstudio_model_dirs() == [default]


def test_a_tilde_in_the_configured_path_is_expanded(fake_home, monkeypatch):
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    downloads = fake_home / "tilde-models"
    downloads.mkdir()
    write_settings(fake_home, settings_json("~/tilde-models"))

    assert storage_roots.lmstudio_model_dirs() == [downloads]


@pytest.mark.parametrize(
    "name",
    [
        "models with spaces",
        "modeles-cafe-eaigu",
        "\u6a21\u578b",
        "models-\U0001f600",
        "m" * 200,
    ],
    ids = ["spaces", "ascii", "cjk", "emoji", "long_name"],
)
def test_awkward_but_legal_directory_names_survive(fake_home, name):
    downloads = fake_home / name
    downloads.mkdir()
    write_settings(fake_home, settings_json(downloads))

    assert storage_roots.lmstudio_model_dirs() == [downloads]


def test_a_trailing_separator_does_not_produce_a_duplicate(fake_home):
    downloads = fake_home / "lmstudio-models"
    downloads.mkdir()
    write_settings(fake_home, settings_json(str(downloads) + os.sep))

    found = storage_roots.lmstudio_model_dirs()
    assert len(found) == 1
    assert found[0].resolve() == downloads.resolve()


def test_settings_path_being_a_directory_is_not_a_crash(fake_home):
    (fake_home / ".lmstudio" / "settings.json").mkdir(parents = True)
    assert storage_roots.lmstudio_model_dirs() == []


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    # geteuid is POSIX-only, and skipif evaluates at import, so calling it unguarded
    # fails collection of this whole module on Windows.
    reason = "needs POSIX mode bits and a non-root user",
)
def test_an_unreadable_settings_file_is_logged_not_raised(fake_home, recording_logger):
    settings = write_settings(fake_home, settings_json(fake_home))
    settings.chmod(0o000)
    try:
        assert storage_roots.lmstudio_model_dirs() == []
        assert len(recording_logger.debug_calls) == 1
    finally:
        settings.chmod(0o600)


# ---------------------------------------------------------------------------
# C. Platform matrix -- pure-string normalizer behaviour
# ---------------------------------------------------------------------------


@pytest.fixture
def as_host(monkeypatch):
    """Apply one simulated host to the normalizers.

    Only the seams ``host_normalize_path`` branches on. No test using this may touch
    tmp_path files, since ``os.name`` is patched here.
    """

    def _apply(label: str, *, automount: str = "/mnt/"):
        is_wsl = label == "wsl"
        monkeypatch.setattr(sys, "platform", "win32" if label == "windows" else "linux")
        if label == "macos":
            monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(path_utils, "_IS_WSL", is_wsl)
        monkeypatch.setattr(path_utils, "_WSL_AUTOMOUNT_ROOT", automount)
        monkeypatch.setattr(os, "name", "nt" if label == "windows" else "posix")

    return _apply


@pytest.mark.parametrize("host", ["linux", "macos", "windows", "wsl"])
def test_a_posix_path_is_returned_unchanged_on_every_host(as_host, host):
    as_host(host)
    assert path_utils.host_normalize_path("/home/u/models") == "/home/u/models"


@pytest.mark.parametrize(
    "host, expected",
    [
        ("linux", "C:/Users/u/models"),
        ("macos", "C:/Users/u/models"),
        ("windows", "C:/Users/u/models"),
    ],
)
def test_a_drive_letter_path_keeps_its_drive_off_wsl(as_host, host, expected):
    as_host(host)
    assert path_utils.host_normalize_path("C:\\Users\\u\\models") == expected


def test_a_drive_letter_path_is_mapped_under_the_default_automount_root(as_host):
    as_host("wsl", automount = "/mnt/")
    assert path_utils.host_normalize_path("C:\\Users\\u\\models") == "/mnt/c/Users/u/models"


def test_a_drive_letter_path_honours_a_custom_automount_root(as_host):
    """The whole reason host_normalize_path is not normalize_path.

    ``[automount] root = /`` puts C: at ``/c/``, while the loader-facing normalize_path
    deliberately keeps predicting ``/mnt/``.
    """
    as_host("wsl", automount = "/c-drive-root/")
    assert path_utils.host_normalize_path("C:\\models") == "/c-drive-root/c/models"
    assert path_utils.normalize_path("C:\\models") == "/mnt/c/models"


def test_a_unc_path_gets_forward_slashes_on_every_host(as_host):
    for host in ("linux", "macos", "windows", "wsl"):
        as_host(host)
        assert path_utils.host_normalize_path("\\\\server\\share\\models") == (
            "//server/share/models"
        )


@pytest.mark.parametrize("host", ["linux", "macos", "wsl"])
def test_a_backslash_in_a_posix_name_is_left_alone(as_host, host):
    """Regression guard.

    A backslash is a legal character in a POSIX filename, WSL included. Rewriting it to
    "/" turns one real directory into a two-segment path that does not exist, and the
    folder vanishes from the picker with nothing logged.
    """
    as_host(host)
    assert path_utils.host_normalize_path("/home/u/models\\backup") == "/home/u/models\\backup"


def test_a_backslash_is_a_separator_on_windows(as_host):
    """Only there can it not be part of a name."""
    as_host("windows")
    assert path_utils.host_normalize_path("relative\\models") == "relative/models"


def test_the_empty_string_is_returned_unchanged(as_host):
    as_host("linux")
    assert path_utils.host_normalize_path("") == ""


# ---------------------------------------------------------------------------
# C2. Platform matrix -- discovery against a real filesystem (os.name untouched)
# ---------------------------------------------------------------------------


@pytest.fixture
def as_wsl(monkeypatch):
    def _apply(automount: str):
        monkeypatch.setattr(path_utils, "_IS_WSL", True)
        monkeypatch.setattr(path_utils, "_WSL_AUTOMOUNT_ROOT", automount)

    return _apply


def test_a_windows_downloads_folder_is_discovered_under_wsl(fake_home, tmp_path, as_wsl):
    """A drive-letter path is meaningless to a WSL process until mapped.

    Only the hub copy used to do this; folding them together gives it to the picker too.
    """
    mount = tmp_path / "mnt"
    models = mount / "c" / "Users" / "u" / "models"
    models.mkdir(parents = True)
    as_wsl(f"{mount}/")
    write_settings(fake_home, settings_json("C:\\Users\\u\\models"))

    assert storage_roots.lmstudio_model_dirs() == [models]


def test_a_windows_downloads_folder_is_dropped_on_plain_linux(fake_home):
    """No WSL, no drive mapping: "C:\\..." is not a path this host can open."""
    write_settings(fake_home, settings_json("C:\\Users\\u\\models"))
    assert storage_roots.lmstudio_model_dirs() == []


@pytest.mark.skipif(
    os.name == "nt",
    # The property under test is POSIX-only: on Windows this name cannot be created,
    # mkdir reads the backslash as a separator and the parent does not exist.
    reason = "a backslash cannot be part of a filename on Windows",
)
def test_a_posix_folder_with_a_backslash_survives_discovery(fake_home):
    """End-to-end form of the normalizer guard above; legal on Linux and macOS."""
    downloads = fake_home / "models\\backup"
    downloads.mkdir()
    write_settings(fake_home, settings_json(downloads))

    assert storage_roots.lmstudio_model_dirs() == [downloads]
    assert hub_paths.lmstudio_model_dirs() == [downloads]


# ---------------------------------------------------------------------------
# D. The two entry points are one implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["lmstudio_model_dirs", "ollama_model_dirs", "well_known_model_dirs"]
)
def test_hub_and_picker_share_one_function_object(name):
    assert getattr(hub_paths, name) is getattr(storage_roots, name), name


def test_the_hub_normalizer_stays_separate_from_the_loader_normalizer():
    """They disagree on a custom-root WSL host, and that difference is load-bearing.

    The hub copy answers "where is this on my disk", the path_utils copy "where will the
    loader look". This pins that folding the discovery copies together did not fold
    these two together as well.
    """
    assert hub_paths.normalize_path is not path_utils.normalize_path
    assert hasattr(hub_paths, "_IS_WSL")
    assert hasattr(hub_paths, "_WSL_AUTOMOUNT_ROOT")


def test_the_hub_normalizer_still_reads_its_own_module_attributes(monkeypatch):
    """test_gguf_variants_local_resolution monkeypatches exactly these two names."""
    monkeypatch.setattr(hub_paths, "_IS_WSL", True)
    monkeypatch.setattr(hub_paths, "_WSL_AUTOMOUNT_ROOT", "/custom-root/")

    assert hub_paths.normalize_path("D:\\models") == "/custom-root/d/models"


# ---------------------------------------------------------------------------
# E. Return-shape contracts (the backwards-compatibility guard)
# ---------------------------------------------------------------------------


def test_per_tool_lists_are_unresolved_and_well_known_is_resolved(fake_home, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    lm_link = fake_home / "lm-link"
    lm_link.symlink_to(real, target_is_directory = True)
    write_settings(fake_home, settings_json(lm_link))

    assert storage_roots.lmstudio_model_dirs() == [lm_link]
    assert all(p == p.resolve() for p in storage_roots.well_known_model_dirs())


def test_candidate_order_is_settings_then_default_then_legacy(fake_home):
    custom = fake_home / "custom"
    custom.mkdir()
    default = fake_home / ".lmstudio" / "models"
    default.mkdir(parents = True)
    legacy = fake_home / ".cache" / "lm-studio" / "models"
    legacy.mkdir(parents = True)
    write_settings(fake_home, settings_json(custom))

    assert storage_roots.lmstudio_model_dirs() == [custom, default, legacy]


def test_lm_studio_sorts_ahead_of_ollama_in_the_quick_picks(fake_home, monkeypatch, tmp_path):
    lm = fake_home / ".lmstudio" / "models"
    lm.mkdir(parents = True)
    ollama = tmp_path / "ollama-models"
    ollama.mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(ollama))

    found = storage_roots.well_known_model_dirs()
    assert found.index(lm.resolve()) < found.index(ollama.resolve())


def test_the_same_directory_reached_two_ways_appears_once(fake_home):
    """First spelling wins; the alias is dropped."""
    default = fake_home / ".lmstudio" / "models"
    default.mkdir(parents = True)
    write_settings(fake_home, settings_json(str(default) + "/./"))

    found = storage_roots.lmstudio_model_dirs()
    assert len(found) == 1


def test_discovery_returns_path_objects_not_strings(fake_home):
    default = fake_home / ".lmstudio" / "models"
    default.mkdir(parents = True)

    for result in (
        storage_roots.lmstudio_model_dirs(),
        storage_roots.ollama_model_dirs(),
        storage_roots.well_known_model_dirs(),
    ):
        assert all(isinstance(p, Path) for p in result)


# ---------------------------------------------------------------------------
# F. $OLLAMA_MODELS -- previously untested anywhere in the repo
# ---------------------------------------------------------------------------


def test_the_ollama_env_override_is_honoured(fake_home, monkeypatch, tmp_path):
    models = tmp_path / "ollama-models"
    models.mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(models))

    assert storage_roots.ollama_model_dirs() == [models]
    assert models.resolve() in storage_roots.well_known_model_dirs()


@pytest.mark.parametrize("value", ["", "   "])
def test_a_blank_ollama_env_is_ignored(fake_home, monkeypatch, value):
    monkeypatch.setenv("OLLAMA_MODELS", value)
    assert storage_roots.ollama_model_dirs() == []


def test_an_ollama_env_pointing_nowhere_is_ignored(fake_home, monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "absent"))
    assert storage_roots.ollama_model_dirs() == []


def test_a_windows_ollama_env_is_mapped_under_wsl(fake_home, tmp_path, as_wsl, monkeypatch):
    mount = tmp_path / "mnt"
    models = mount / "d" / "ollama"
    models.mkdir(parents = True)
    as_wsl(f"{mount}/")
    monkeypatch.setenv("OLLAMA_MODELS", "D:\\ollama")

    assert storage_roots.ollama_model_dirs() == [models]


def test_the_ollama_default_is_found_without_the_env(fake_home, monkeypatch):
    monkeypatch.delenv("OLLAMA_MODELS", raising = False)
    default = fake_home / ".ollama" / "models"
    default.mkdir(parents = True)

    assert storage_roots.ollama_model_dirs() == [default]


# ---------------------------------------------------------------------------
# G. Import graph -- the new hub -> storage_roots edge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "first, second",
    [
        ("hub.utils.paths", "utils.paths"),
        ("utils.paths", "hub.utils.paths"),
        ("utils.paths.storage_roots", "hub.utils.paths"),
        ("hub.utils.paths", "utils.paths.storage_roots"),
    ],
)
def test_the_modules_import_in_either_order_without_a_cycle(first, second):
    """Fresh interpreter: a cycle sys.modules already papered over here would not
    reproduce in-process."""
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import importlib\n"
        "importlib.import_module(%r); importlib.import_module(%r)\n"
        "from hub.utils import paths as h\n"
        "from utils.paths import storage_roots as s\n"
        "assert h.lmstudio_model_dirs is s.lmstudio_model_dirs\n"
        "assert h.ollama_model_dirs is s.ollama_model_dirs\n"
        "assert h.well_known_model_dirs is s.well_known_model_dirs\n"
        "print('ok')\n" % (str(BACKEND_ROOT), first, second)
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output = True, text = True, timeout = 120
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_ollama_model_dirs_is_exported_from_the_paths_package():
    """It was defined only on the hub side before, so the package never exported it."""
    import utils.paths as paths_pkg

    assert paths_pkg.ollama_model_dirs is storage_roots.ollama_model_dirs
    assert "ollama_model_dirs" in paths_pkg.__all__
