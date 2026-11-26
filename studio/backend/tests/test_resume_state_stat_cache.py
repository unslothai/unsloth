# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inside a listing, checkpoint state verdicts are kept per unchanged file; answers never differ."""

import itertools
import json
import os
import sys
import time
import types
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from core.training import resume as resume_mod
from utils.scan_memo import scan_memo


@pytest.fixture(autouse = True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(resume_mod, "_STATE_CACHE", type(resume_mod._STATE_CACHE)())
    monkeypatch.setattr(resume_mod, "outputs_root", lambda: tmp_path)
    monkeypatch.setattr(resume_mod, "resolve_output_dir", lambda value: Path(value))


def _skip_settle_window(monkeypatch) -> None:
    real_time_ns = time.time_ns
    monkeypatch.setattr(time, "time_ns", lambda: real_time_ns() + 3600 * 10**9)


def _reads(monkeypatch) -> list:
    calls: list = []
    real_read = resume_mod._read_state_file
    real_loads = json.loads

    def read(path, require_tensor, io_error):
        if Path(path).is_file() and Path(path).stat().st_size:
            calls.append(Path(path).name)
        return real_read(path, require_tensor, io_error)

    def loads(text, *args, **kwargs):
        calls.append("trainer_state.json")
        return real_loads(text, *args, **kwargs)

    monkeypatch.setattr(resume_mod, "_read_state_file", read)
    monkeypatch.setattr(
        resume_mod,
        "json",
        types.SimpleNamespace(loads = loads, dumps = json.dumps, JSONDecodeError = json.JSONDecodeError),
    )
    return calls


def _pt(path: Path, obj) -> None:
    torch.save(obj, path)


def _no_stop_pickle(path: Path) -> None:
    _pt(path, {"state": {}, "param_groups": []})
    with zipfile.ZipFile(path) as source:
        items = [(info, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(path, "w") as target:
        for info, data in items:
            if info.filename.endswith("data.pkl"):
                data = data[:-1]
            target.writestr(info, data)


_TRAINER = {
    "ok": lambda step: json.dumps({"global_step": step}),
    "mismatch": lambda step: json.dumps({"global_step": step + 1}),
    "bad_json": lambda step: '{"global_step": ',
    "bool": lambda step: json.dumps({"global_step": True}),
    "negative": lambda step: json.dumps({"global_step": -1}),
    "list": lambda step: "[1, 2]",
    "missing": None,
}
_ADAPTER = {
    "ok": lambda p: save_file({"w": torch.ones(2)}, p),
    "no_keys": lambda p: save_file({}, p),
    "corrupt": lambda p: p.write_bytes(b"\x08\x00\x00\x00\x00\x00\x00\x00{bad"),
    "empty": lambda p: p.write_bytes(b""),
    "missing": None,
}
_OPTIMIZER = {
    "ok": lambda p: _pt(
        p, {"state": {0: {"exp_avg": torch.ones(2)}}, "param_groups": [{"lr": 1.0}]}
    ),
    "tensor_free": lambda p: _pt(p, {"state": {}, "param_groups": [{"lr": 1.0}]}),
    "truncated": lambda p: (
        _pt(p, {"state": {0: torch.ones(64)}}),
        p.write_bytes(p.read_bytes()[:200]),
    ),
    "not_zip": lambda p: p.write_bytes(b"not a torch archive"),
    "no_stop": _no_stop_pickle,
    "missing": None,
}
_SCHEDULER = {
    "ok": lambda p: _pt(p, {"last_epoch": 3}),
    "empty": lambda p: p.write_bytes(b""),
    "missing": None,
}


def _write(path: Path, variants: dict, name: str, *args) -> None:
    writer = variants[name]
    if path.is_symlink() or path.exists():
        path.unlink()
    if writer is None:
        return
    if variants is _TRAINER:
        path.write_text(writer(*args), encoding = "utf-8")
    else:
        writer(path)


def _build(checkpoint: Path, step: int, trainer: str, adapter: str, optimizer: str, scheduler: str):
    checkpoint.mkdir(parents = True, exist_ok = True)
    _write(checkpoint / "trainer_state.json", _TRAINER, trainer, step)
    _write(checkpoint / "adapter_model.safetensors", _ADAPTER, adapter)
    _write(checkpoint / "optimizer.pt", _OPTIMIZER, optimizer)
    _write(checkpoint / "scheduler.pt", _SCHEDULER, scheduler)


def _uncached(run: Path):
    return resume_mod.get_resume_checkpoint_path(str(run))


def _listed(run: Path):
    with scan_memo():
        return resume_mod.get_resume_checkpoint_path(str(run))


_SHAPES = list(itertools.product(_TRAINER, _ADAPTER, _OPTIMIZER, _SCHEDULER))


def test_cached_answers_match_uncached_for_every_checkpoint_shape(tmp_path, monkeypatch):
    _skip_settle_window(monkeypatch)
    runs = []
    for index, shape in enumerate(_SHAPES):
        run = tmp_path / f"run-{index}"
        _build(run / "checkpoint-10", 10, *shape)
        # An older valid checkpoint behind it: the listing falls back to it when the newest is bad.
        if index % 3 == 0:
            _build(run / "checkpoint-5", 5, "ok", "ok", "ok", "ok")
        runs.append(run)
    expected = [_uncached(run) for run in runs]
    assert sum(answer is not None for answer in expected) > len(runs) // 4

    reads = _reads(monkeypatch)
    assert [_listed(run) for run in runs] == expected
    cold = len(reads)
    assert [_listed(run) for run in runs] == expected
    assert len(reads) == cold
    assert [_uncached(run) for run in runs] == expected
    assert len(reads) > cold

    def rotate(variants: dict, name: str) -> str:
        names = list(variants)
        return names[(names.index(name) + 1) % len(names)]

    for run, shape in zip(runs, _SHAPES):
        trainer, adapter, optimizer, scheduler = shape
        _build(
            run / "checkpoint-10",
            10,
            rotate(_TRAINER, trainer),
            rotate(_ADAPTER, adapter),
            rotate(_OPTIMIZER, optimizer),
            rotate(_SCHEDULER, scheduler),
        )
    rotated = [_uncached(run) for run in runs]
    assert rotated != expected
    assert [_listed(run) for run in runs] == rotated


def _same_size_rewrite(path: Path, data: bytes) -> None:
    before = path.stat()
    assert len(data) == before.st_size
    with open(path, "r+b") as f:
        f.write(data)
    os.utime(path, ns = (before.st_atime_ns, before.st_mtime_ns))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason = "st_ctime is the creation time on Windows, so it does not move on a rewrite",
)
def test_same_size_rewrite_with_restored_mtime_is_seen(tmp_path, monkeypatch):
    _skip_settle_window(monkeypatch)
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-10"
    _build(checkpoint, 10, "ok", "ok", "ok", "ok")
    assert _listed(run) == str(checkpoint)

    _same_size_rewrite(checkpoint / "trainer_state.json", json.dumps({"global_step": 11}).encode())
    assert _listed(run) is None
    _same_size_rewrite(checkpoint / "trainer_state.json", json.dumps({"global_step": 10}).encode())
    assert _listed(run) == str(checkpoint)

    optimizer = checkpoint / "optimizer.pt"
    _same_size_rewrite(optimizer, b"\0" * optimizer.stat().st_size)
    assert _uncached(run) is None
    assert _listed(run) is None


def test_files_written_within_the_settle_window_are_not_kept(tmp_path, monkeypatch):
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-10"
    _build(checkpoint, 10, "ok", "ok", "ok", "ok")
    reads = _reads(monkeypatch)

    assert _listed(run) == str(checkpoint)
    first = len(reads)
    assert _listed(run) == str(checkpoint)
    assert len(reads) == 2 * first
    assert resume_mod._STATE_CACHE == {}

    _same_size_rewrite(checkpoint / "trainer_state.json", json.dumps({"global_step": 11}).encode())
    assert _listed(run) is None


def test_deleted_and_replaced_files(tmp_path, monkeypatch):
    _skip_settle_window(monkeypatch)
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-10"
    _build(checkpoint, 10, "ok", "ok", "ok", "ok")
    assert _listed(run) == str(checkpoint)

    (checkpoint / "scheduler.pt").unlink()
    assert _listed(run) is None
    _write(checkpoint / "scheduler.pt", _SCHEDULER, "ok")
    assert _listed(run) == str(checkpoint)

    replacement = tmp_path / "replacement.pt"
    _write(replacement, _OPTIMIZER, "not_zip")
    os.replace(replacement, checkpoint / "optimizer.pt")
    assert _listed(run) is None
    _write(replacement, _OPTIMIZER, "ok")
    os.replace(replacement, checkpoint / "optimizer.pt")
    assert _listed(run) == str(checkpoint)

    (checkpoint / "trainer_state.json").unlink()
    assert _listed(run) is None
    for child in checkpoint.iterdir():
        child.unlink()
    checkpoint.rmdir()
    assert _listed(run) is None


@pytest.mark.skipif(sys.platform == "win32", reason = "symlinks need privileges on Windows")
def test_symlinked_state_files_follow_their_target(tmp_path, monkeypatch):
    _skip_settle_window(monkeypatch)
    good = tmp_path / "good.pt"
    bad = tmp_path / "bad.pt"
    _write(good, _OPTIMIZER, "ok")
    _write(bad, _OPTIMIZER, "no_stop")
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-10"
    _build(checkpoint, 10, "ok", "ok", "missing", "ok")
    link = checkpoint / "optimizer.pt"
    link.symlink_to(good)
    assert _listed(run) == str(checkpoint) == _uncached(run)

    link.unlink()
    link.symlink_to(bad)
    assert _listed(run) is None is _uncached(run)

    link.unlink()
    link.symlink_to(tmp_path / "dangling.pt")
    assert _listed(run) is None is _uncached(run)

    other = tmp_path / "other" / "checkpoint-10"
    _build(other, 10, "ok", "ok", "truncated", "ok")
    alias_run = tmp_path / "alias"
    alias_run.mkdir()
    (alias_run / "checkpoint-10").symlink_to(checkpoint)
    link.unlink()
    link.symlink_to(good)
    assert _listed(alias_run) == str(alias_run / "checkpoint-10")
    (alias_run / "checkpoint-10").unlink()
    (alias_run / "checkpoint-10").symlink_to(other)
    assert _listed(alias_run) is None is _uncached(alias_run)


def test_an_io_error_is_not_kept(tmp_path, monkeypatch):
    _skip_settle_window(monkeypatch)
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-10"
    _build(checkpoint, 10, "ok", "ok", "ok", "ok")
    real_zip = zipfile.ZipFile
    failing = {"left": 1}

    def flaky(*args, **kwargs):
        if failing["left"]:
            failing["left"] -= 1
            raise OSError(24, "Too many open files")
        return real_zip(*args, **kwargs)

    monkeypatch.setattr(resume_mod.zipfile, "ZipFile", flaky)
    assert _listed(run) is None
    assert _listed(run) == str(checkpoint)


def test_a_swallowed_stat_error_is_not_kept(tmp_path, monkeypatch):
    # Python 3.14 Path.is_file() is os.path.isfile, which turns any stat OSError into False.
    _skip_settle_window(monkeypatch)
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-10"
    _build(checkpoint, 10, "ok", "ok", "ok", "ok")
    real_is_file = Path.is_file
    failing = {"left": 1}

    def flaky(self, *args, **kwargs):
        if self.name == "optimizer.pt" and failing["left"]:
            failing["left"] -= 1
            return False
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", flaky)
    assert _listed(run) is None
    assert _listed(run) == str(checkpoint)


def test_resume_start_paths_never_consult_the_cache(tmp_path, monkeypatch):
    _skip_settle_window(monkeypatch)
    run = tmp_path / "run"
    checkpoint = run / "checkpoint-10"
    _build(checkpoint, 10, "ok", "ok", "ok", "ok")
    assert _listed(run) == str(checkpoint)
    assert resume_mod._STATE_CACHE

    def refuse(*args, **kwargs):
        raise AssertionError("cache consulted outside a listing")

    monkeypatch.setattr(resume_mod, "_cached_state", refuse)
    monkeypatch.setattr(resume_mod, "_store_state", refuse)
    # What the start route, the trainer and the worker call; none of them opens a scan_memo scope.
    assert resume_mod.get_resume_checkpoint_path(str(run)) == str(checkpoint)
    assert resume_mod.get_resume_checkpoint_path(str(run), expected_step = 10) == str(checkpoint)
    assert resume_mod.is_resume_checkpoint_valid(checkpoint) is True
    assert resume_mod._checkpoint_state(checkpoint) == 10
    assert resume_mod.has_resume_state(str(run)) is True


def test_cache_is_bounded(tmp_path, monkeypatch):
    _skip_settle_window(monkeypatch)
    monkeypatch.setattr(resume_mod, "_STATE_CACHE_MAX", 7)
    runs = []
    for index in range(6):
        run = tmp_path / f"run-{index}"
        _build(run / "checkpoint-10", 10, "ok", "ok", "ok", "ok")
        runs.append(run)
    answers = [_listed(run) for run in runs]
    assert len(resume_mod._STATE_CACHE) == 7
    assert answers == [_uncached(run) for run in runs]
    assert [_listed(run) for run in runs] == answers
    assert len(resume_mod._STATE_CACHE) == 7
