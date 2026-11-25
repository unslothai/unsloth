# SPDX-License-Identifier: AGPL-3.0-only

"""The per-render spelling memo must leave every Seatbelt profile byte-identical."""

from __future__ import annotations

import os

import pytest

from core.inference import sandbox_macos as backend


def _cases(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    tmp = tmp_path / "ptmp"
    tmp.mkdir()
    link = tmp_path / "work-link"
    link.symlink_to(work, target_is_directory = True)
    runtime = tmp_path / "venv"
    (runtime / "lib").mkdir(parents = True)
    (runtime / "bin").mkdir()
    (runtime / "bin" / "python").write_text("")
    alias = tmp_path / "lib-alias"
    alias.symlink_to(runtime / "lib", target_is_directory = True)
    accented = tmp_path / "données"
    accented.mkdir()
    return [
        dict(workdir = str(work), private_tmp = str(tmp), runtime_paths = ()),
        dict(
            workdir = str(link),
            private_tmp = str(tmp),
            runtime_paths = (str(runtime / "lib"), str(runtime / "bin"), str(alias)),
        ),
        dict(
            workdir = str(work),
            private_tmp = "/tmp/unsloth-render-memo",
            runtime_paths = (str(accented),),
            developer_paths = (str(runtime / "bin"),),
            egress_port = 41234,
        ),
        dict(
            workdir = str(work),
            private_tmp = str(tmp),
            runtime_paths = backend.runtime_read_paths(str(work)),
        ),
    ]


def test_memoized_render_is_byte_identical_to_the_unmemoized_one(tmp_path):
    for case in _cases(tmp_path):
        kwargs = {"developer_paths": (), "egress_port": None, **case}
        unmemoized = backend._build_profile(**kwargs)
        assert backend._render_spellings.get() is None
        assert backend.build_profile(**case) == unmemoized


def test_each_render_resolves_a_path_once_and_the_next_render_resolves_again(tmp_path, monkeypatch):
    case = _cases(tmp_path)[1]
    calls = []
    real = os.path.realpath
    monkeypatch.setattr(os.path, "realpath", lambda p, *a, **k: calls.append(p) or real(p, *a, **k))
    kwargs = {"developer_paths": (), "egress_port": None, **case}
    backend._build_profile(**kwargs)
    unmemoized = len(calls)
    calls.clear()
    backend.build_profile(**case)
    memoized = list(calls)
    assert case["workdir"] in memoized, "the workdir is resolved during a render"
    assert len(memoized) < unmemoized, (len(memoized), unmemoized)
    calls.clear()
    backend.build_profile(**case)
    assert (
        case["workdir"] in calls
    ), "a later render must re-resolve, not reuse the previous render's view"


def test_a_symlink_retargeted_between_renders_is_seen_by_the_next_render(tmp_path):
    case = _cases(tmp_path)[1]
    other = tmp_path / "elsewhere"
    other.mkdir()
    before = backend.build_profile(**case)
    link = tmp_path / "work-link"
    link.unlink()
    link.symlink_to(other, target_is_directory = True)
    after = backend.build_profile(**case)
    assert str(other) not in before and str(other) in after


def test_a_rejected_path_still_refuses_every_render(tmp_path):
    case = dict(_cases(tmp_path)[0], runtime_paths = ("relative/path",))
    for _ in range(2):
        with pytest.raises(backend.SandboxUnavailableError):
            backend.build_profile(**case)
        assert backend._render_spellings.get() is None


def test_runtime_read_paths_dedupe_keeps_the_same_selection(tmp_path, monkeypatch):
    work = str(tmp_path / "w")
    os.makedirs(work)
    expected = backend.runtime_read_paths(work)
    # Duplicate candidates (sys.prefix == sys.base_prefix ...) collapse to the same selection, in order.
    assert backend.runtime_read_paths(work) == expected
    assert len(expected) == len(set(expected))
