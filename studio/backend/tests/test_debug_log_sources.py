# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Discovery decides what the log viewer is allowed to open, so it is also the
whole path-traversal defence. The conftest _isolate_studio_home fixture already
points UNSLOTH_STUDIO_HOME at a tmp dir."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils import debug_log_sources


def _home() -> Path:
    return Path(os.environ["UNSLOTH_STUDIO_HOME"])


def _seed(
    family_dir: str,
    name: str,
    body: str = "hello\n",
) -> Path:
    directory = _home() / "logs" / family_dir
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / name
    path.write_text(body, encoding = "utf-8")
    return path


def test_all_three_families_are_found():
    _seed("server", f"server-20260813-101010-pid{os.getpid()}.log")
    _seed("llama-server", "llama-1765000000-port-8080.log")
    _seed("diffusion-server", "diffusion-1765000000-port-8081.log")
    families = {source.family for source in debug_log_sources.list_sources()}
    assert families == {"server", "llama-server", "diffusion-server"}


def test_ids_are_stable_across_calls_and_unique_per_file():
    _seed("llama-server", "llama-1765000001-port-8080.log")
    _seed("llama-server", "llama-1765000002-port-8081.log")
    first = {s.label: s.id for s in debug_log_sources.list_sources()}
    second = {s.label: s.id for s in debug_log_sources.list_sources()}
    assert first == second
    assert len(set(first.values())) == len(first)


def test_an_id_resolves_back_to_the_file_it_named():
    path = _seed("llama-server", "llama-1765000003-port-8080.log")
    source = next(s for s in debug_log_sources.list_sources() if s.label == path.name)
    assert debug_log_sources.resolve_source_id(source.id) == Path(os.path.realpath(path))


@pytest.mark.parametrize(
    "hostile",
    [
        "server:../../../../etc/passwd",
        "../../etc/passwd",
        "server:" + "0" * 16,
        "nosuchfamily:abcdef0123456789",
        "server",
        "",
        ":",
    ],
)
def test_a_hostile_source_id_resolves_to_nothing(hostile):
    assert debug_log_sources.resolve_source_id(hostile) is None


def test_a_symlink_out_of_the_log_dir_is_not_readable(tmp_path):
    """The one way an attacker could turn this into an arbitrary file reader."""
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY\n")
    directory = _home() / "logs" / "server"
    directory.mkdir(parents = True, exist_ok = True)
    link = directory / "server-20260813-999999-pid1.log"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    labels = [s.label for s in debug_log_sources.list_sources()]
    assert link.name not in labels


def test_a_symlinked_family_directory_outside_the_studio_root_is_not_scanned(tmp_path, monkeypatch):
    root = tmp_path / "studio"
    outside = tmp_path / "unrelated"
    (root / "logs").mkdir(parents = True)
    outside.mkdir()
    stolen = outside / "server-20260813-999999-pid1.log"
    stolen.write_text("not a Studio log\n", encoding = "utf-8")
    try:
        (root / "logs" / "server").symlink_to(outside, target_is_directory = True)
    except (OSError, NotImplementedError):
        # Skip because Windows requires Developer Mode or elevation for this
        # primitive; other hosts still exercise the containment behavior.
        pytest.skip("symlinks unavailable")

    monkeypatch.setattr(debug_log_sources, "candidate_roots", lambda: [root])
    assert stolen.name not in [source.label for source in debug_log_sources.list_sources()]


def test_a_missing_family_directory_is_not_an_error():
    _seed("server", f"server-20260813-101011-pid{os.getpid()}.log")
    sources = debug_log_sources.list_sources()
    assert all(source.family != "diffusion-server" for source in sources)


def test_another_installations_logs_are_not_offered(monkeypatch, tmp_path):
    """With UNSLOTH_STUDIO_HOME set, the runners write under it, so the legacy
    ~/.unsloth/studio belongs to a different install and must stay invisible."""
    legacy = tmp_path / "legacy"
    (legacy / ".unsloth" / "studio" / "logs" / "diffusion-server").mkdir(parents = True)
    (
        legacy / ".unsloth" / "studio" / "logs" / "diffusion-server" / "diffusion-1-port-1.log"
    ).write_text("someone else's log\n", encoding = "utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: legacy))

    _seed("server", f"server-20260813-101014-pid{os.getpid()}.log")
    labels = [source.label for source in debug_log_sources.list_sources()]
    assert "diffusion-1-port-1.log" not in labels


def test_the_running_session_is_marked_current():
    path = _seed("server", f"server-20260813-101012-pid{os.getpid()}.log")
    source = next(s for s in debug_log_sources.list_sources() if s.label == path.name)
    assert source.is_current is True


def test_a_retained_log_whose_pid_merely_starts_with_ours_is_not_current():
    """Retention keeps the newest 20 session logs, so a file from pid 12345 can
    still be sitting there while we are pid 1234. A substring test called both
    of them the running session and could hand the picker the wrong default."""
    ours = _seed("server", f"server-20260813-101012-pid{os.getpid()}.log")
    other = _seed("server", f"server-20260101-090000-pid{os.getpid()}9.log")
    current = {s.label: s.is_current for s in debug_log_sources.list_sources()}
    assert current[ours.name] is True
    assert current[other.name] is False


def test_only_the_newest_files_per_family_are_offered():
    for i in range(debug_log_sources.MAX_SOURCES_PER_FAMILY + 4):
        path = _seed("llama-server", f"llama-17650001{i:02d}-port-8080.log")
        os.utime(path, (1_765_000_000 + i, 1_765_000_000 + i))
    llama = [s for s in debug_log_sources.list_sources() if s.family == "llama-server"]
    assert len(llama) == debug_log_sources.MAX_SOURCES_PER_FAMILY
    assert llama[0].modified_at >= llama[-1].modified_at


def test_a_runtime_log_under_the_other_root_is_still_found(monkeypatch, tmp_path):
    """studio_root() infers a root from the installer venv and the llama runner
    does not, so on such an install the two disagree. Scanning one root only
    would lose exactly the logs a failed model load writes."""
    other_home = tmp_path / "legacy-home"
    (other_home / "logs" / "llama-server").mkdir(parents = True)
    stray = other_home / "logs" / "llama-server" / "llama-1765009999-port-9099.log"
    stray.write_text("child runtime output\n", encoding = "utf-8")

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        debug_log_sources,
        "candidate_roots",
        lambda: [Path(os.environ["UNSLOTH_STUDIO_HOME"]), other_home],
    )
    labels = [s.label for s in debug_log_sources.list_sources()]
    assert stray.name in labels


def test_the_default_source_prefers_the_running_session():
    _seed("llama-server", "llama-1765000500-port-8080.log")
    path = _seed("server", f"server-20260813-101013-pid{os.getpid()}.log")
    default = debug_log_sources.default_source_id()
    assert default is not None
    assert debug_log_sources.resolve_source_id(default) == Path(os.path.realpath(path))


def test_containment_survives_a_windows_extended_length_prefix(monkeypatch):
    """ntpath.realpath decides per call whether to keep the \\\\?\\ prefix, so
    the directory and the file in it can come back spelled differently. pathlib
    reads that as two different drives, and the whole family disappears with no
    error anywhere."""
    import ntpath

    monkeypatch.setattr(os.path, "normcase", ntpath.normcase)
    monkeypatch.setattr(os, "sep", "\\")

    directory = "C:\\Users\\dan\\.unsloth\\studio\\logs\\server"
    entry = "\\\\?\\C:\\Users\\dan\\.unsloth\\studio\\logs\\server\\server-1-pid2.log"
    assert debug_log_sources._is_inside(entry, directory) is True
    # The protection it exists for must still hold under the same spelling.
    assert debug_log_sources._is_inside("\\\\?\\C:\\Users\\dan\\.ssh\\id_rsa", directory) is False
    assert debug_log_sources._is_inside("C:\\Users\\dan\\.ssh\\id_rsa", directory) is False
    # A sibling directory whose name merely starts with ours is not inside it.
    assert debug_log_sources._is_inside(directory + "-old\\x.log", directory) is False


def test_a_case_insensitive_volume_does_not_list_a_file_twice(monkeypatch):
    """On APFS or NTFS a home reached as /Users/Bob and as /Users/bob is one
    directory. Case-sensitive dedup offered every log twice, under two ids."""
    import ntpath

    monkeypatch.setattr(os.path, "normcase", ntpath.normcase)
    lower = "c:\\users\\dan\\studio\\logs\\server\\server-1-pid2.log"
    upper = "C:\\Users\\Dan\\Studio\\Logs\\Server\\Server-1-pid2.log"
    assert debug_log_sources._identity(lower) == debug_log_sources._identity(upper)


def test_posix_identity_stays_case_sensitive():
    """The same fold on Linux would merge two genuinely different files."""
    if os.name != "posix":
        pytest.skip("posix only")
    assert debug_log_sources._identity("/a/Server.log") != debug_log_sources._identity(
        "/a/server.log"
    )


DESKTOP_FIXTURES = [
    ("desktop-backend", "logs", "backend-backend-1786344247254-2-s01.log"),
    ("desktop-install", "logs", "install-1786344247254-2-s01.log"),
    ("desktop-update", "logs", "update-1786344247254-2-s01.log"),
    ("desktop-repair", "logs", "repair-repair-1-2-update-update-3-s01.log"),
    ("desktop-shell", "", "tauri.log"),
    ("desktop-shell", "", "tauri.log.1"),
]


@pytest.mark.parametrize(
    "family,subdir,name", DESKTOP_FIXTURES, ids = [f[2] for f in DESKTOP_FIXTURES]
)
def test_the_desktop_shell_logs_are_offered(family, subdir, name):
    """The Tauri shell writes these beside the Python logs, and backend-*.log is
    the ONLY record when the backend dies before its own file logging starts.
    Without them a user whose app failed to start is told nothing was logged."""
    directory = _home() / subdir if subdir else _home()
    directory.mkdir(parents = True, exist_ok = True)
    (directory / name).write_text("desktop output\n", encoding = "utf-8")
    sources = debug_log_sources.list_sources()
    assert any(
        s.label == name and s.family == family for s in sources
    ), f"{name} not offered; got {[(s.family, s.label) for s in sources]}"


def test_a_desktop_log_resolves_and_reads_back():
    directory = _home() / "logs"
    directory.mkdir(parents = True, exist_ok = True)
    path = directory / "backend-backend-1786344247254-2-s01.log"
    path.write_text("the backend died before it could open its own log\n", encoding = "utf-8")
    source = next(s for s in debug_log_sources.list_sources() if s.label == path.name)
    assert debug_log_sources.resolve_source_id(source.id) == Path(os.path.realpath(path))


def test_a_python_log_is_not_claimed_by_a_desktop_family():
    """logs/ now has both flat desktop files and the per-family subdirectories,
    so the globs must not overlap."""
    _seed("server", f"server-20260813-101015-pid{os.getpid()}.log")
    (_home() / "logs").mkdir(parents = True, exist_ok = True)
    (_home() / "logs" / "backend-backend-1-2-s01.log").write_text("x\n", encoding = "utf-8")
    by_family = {}
    for source in debug_log_sources.list_sources():
        by_family.setdefault(source.family, []).append(source.label)
    assert all(label.startswith("server-") for label in by_family.get("server", []))
    assert all(label.startswith("backend-") for label in by_family.get("desktop-backend", []))


def test_a_huge_directory_does_not_stat_every_file(monkeypatch):
    """logs/llama-server is never pruned and reaches five figures on a real
    install, while this endpoint is polled once a second."""
    directory = _home() / "logs" / "llama-server"
    directory.mkdir(parents = True, exist_ok = True)
    for i in range(400):
        (directory / f"llama-17660000{i:03d}-port-8080.log").write_text("x\n", encoding = "utf-8")

    real_stat = Path.stat
    calls = {"n": 0}

    def _counting_stat(self, *args, **kwargs):
        calls["n"] += 1
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _counting_stat)
    files = debug_log_sources._family_files("llama-server")
    assert len(files) == debug_log_sources.MAX_SOURCES_PER_FAMILY
    # Cost must track the presort slice (MAX * 3 candidates, an is_file plus a
    # stat each), not the 400 files present.
    ceiling = debug_log_sources.MAX_SOURCES_PER_FAMILY * 3 * 2 + 8
    assert (
        calls["n"] <= ceiling
    ), f"stat called {calls['n']} times for 400 files; the presort is not working"
    assert calls["n"] < 400, "cost still scales with the directory"


def test_a_literal_tilde_home_is_scanned_both_ways(tmp_path, monkeypatch):
    """The writer and the reader disagreed about the tilde.

    _swa_cache_path builds Path(home) raw, so a value passed literally (systemd
    EnvironmentFile, dotenv) makes the runners write into a directory NAMED
    "~", while expanduser sent discovery to the real home. Scanning one of the
    two lost the llama logs the viewer exists to reach.
    """
    monkeypatch.chdir(tmp_path)
    literal = tmp_path / "~" / "studio"
    (literal / "logs" / "llama-server").mkdir(parents = True)
    (literal / "logs" / "llama-server" / "llama-1786000000.log").write_text("x", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", "~/studio")

    labels = [source.label for source in debug_log_sources.list_sources()]
    assert "llama-1786000000.log" in labels


def test_no_live_session_defaults_to_the_newest_log_not_an_old_server_one(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    server_dir = tmp_path / "logs" / "server"
    llama_dir = tmp_path / "logs" / "llama-server"
    server_dir.mkdir(parents = True)
    llama_dir.mkdir(parents = True)
    # A retained log from a previous run: a pid that is not ours.
    old = server_dir / "server-20260101-000000-pid1.log"
    old.write_text("previous run\n", encoding = "utf-8")
    newest = llama_dir / "llama-1786000000.log"
    newest.write_text("the failure\n", encoding = "utf-8")
    os.utime(old, (1_000_000, 1_000_000))
    os.utime(newest, (2_000_000, 2_000_000))

    chosen = debug_log_sources.resolve_source_id(debug_log_sources.default_source_id())
    assert chosen == newest


def test_a_live_server_session_still_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    server_dir = tmp_path / "logs" / "server"
    llama_dir = tmp_path / "logs" / "llama-server"
    server_dir.mkdir(parents = True)
    llama_dir.mkdir(parents = True)
    current = server_dir / f"server-20260101-000000-pid{os.getpid()}.log"
    current.write_text("this session\n", encoding = "utf-8")
    newer = llama_dir / "llama-1786000000.log"
    newer.write_text("a runner\n", encoding = "utf-8")
    os.utime(current, (1_000_000, 1_000_000))
    os.utime(newer, (2_000_000, 2_000_000))

    chosen = debug_log_sources.resolve_source_id(debug_log_sources.default_source_id())
    assert chosen == current
