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


def _seed(family_dir: str, name: str, body: str = "hello\n") -> Path:
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
