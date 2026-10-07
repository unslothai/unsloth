# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "docker" / "unsloth_run.py"
RUN_SH = REPO_ROOT / "docker" / "run.sh"

NOTEBOOK = {
    "cells": [{"cell_type": "code", "source": ["print(1)\n"], "metadata": {}, "outputs": []}],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}
EXECUTED = {**NOTEBOOK, "metadata": {"executed": True}}


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_NB_TF_MARKER", str(tmp_path / "marker"))
    spec = importlib.util.spec_from_file_location("unsloth_run_url_under_test", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_load", lambda *_a, **_k: json.loads(json.dumps(NOTEBOOK)))
    return mod


@pytest.fixture()
def cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def _run(
    runner,
    monkeypatch,
    argv,
    rc = 0,
):
    seen = {}

    def nbconvert(cmd, env = None):
        src = cmd[cmd.index("--output") - 1]
        seen["input"] = src
        kernel_cwd = os.path.dirname(os.path.abspath(src))
        os.makedirs(os.path.join(kernel_cwd, "lora_model"), exist_ok = True)
        Path(kernel_cwd, "lora_model", "adapter_model.safetensors").write_bytes(b"w")
        if rc == 0:
            name = cmd[cmd.index("--output") + 1]
            if not name.endswith(".ipynb"):
                name += ".ipynb"
            out = Path(cmd[cmd.index("--output-dir") + 1], name)
            out.write_text(json.dumps(EXECUTED), encoding = "utf-8")
            seen["output"] = out
        return rc

    monkeypatch.setattr(runner.subprocess, "call", nbconvert)
    monkeypatch.setattr(runner.sys, "argv", ["unsloth-run", *argv])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == rc
    return seen


def test_a_url_run_keeps_the_executed_notebook_and_its_saves(runner, monkeypatch, cwd):
    seen = _run(runner, monkeypatch, ["https://example.invalid/nb/Llama.ipynb"])

    assert Path(seen["input"]).parent == cwd
    assert (cwd / "lora_model" / "adapter_model.safetensors").is_file()
    assert json.loads((cwd / "Llama.ipynb").read_text(encoding = "utf-8")) == EXECUTED


def test_a_url_run_does_not_overwrite_a_notebook_already_in_cwd(runner, monkeypatch, cwd):
    (cwd / "Llama.ipynb").write_text("mine", encoding = "utf-8")

    _run(runner, monkeypatch, ["https://example.invalid/nb/Llama.ipynb"])

    assert (cwd / "Llama.ipynb").read_text(encoding = "utf-8") == "mine"
    assert json.loads((cwd / "Llama-1.ipynb").read_text(encoding = "utf-8")) == EXECUTED


@pytest.mark.parametrize(
    "url, name",
    [
        (
            "https://example.invalid/nb/Llama3.2_%281B%29-Chat.ipynb?raw=true#scrollTo=x",
            "Llama3.2_(1B)-Chat.ipynb",
        ),
        ("https://example.invalid/get?nb=Llama.ipynb", "get.ipynb"),
        ("https://example.invalid/", "notebook.ipynb"),
    ],
)
def test_a_url_run_names_the_notebook_after_the_url_path(runner, monkeypatch, cwd, url, name):
    seen = _run(runner, monkeypatch, [url])

    assert seen["output"] == cwd / name
    assert sorted(p.name for p in cwd.iterdir()) == sorted(["lora_model", name])


def test_a_failed_url_run_keeps_what_it_saved(runner, monkeypatch, cwd):
    _run(runner, monkeypatch, ["https://example.invalid/nb/Llama.ipynb"], rc = 1)

    assert (cwd / "lora_model" / "adapter_model.safetensors").is_file()
    assert json.loads((cwd / "Llama.ipynb").read_text(encoding = "utf-8")) == NOTEBOOK


def test_a_url_download_takes_the_owner_of_the_directory_it_lands_in(runner, monkeypatch, cwd):
    owners = []
    monkeypatch.setattr(
        runner.os, "fchown", lambda fd, uid, gid: owners.append((uid, gid)), raising = False
    )
    _run(runner, monkeypatch, ["https://example.invalid/nb/Llama.ipynb"])

    st = os.stat(cwd)
    assert owners == [(st.st_uid, st.st_gid)]


def test_unsloth_run_reads_and_hides_the_requested_host_owner(runner, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RUN_UID", "1234")
    monkeypatch.setenv("UNSLOTH_RUN_GID", "5678")

    assert runner._host_run_ids() == (1234, 5678)

    assert "UNSLOTH_RUN_UID" not in os.environ
    assert "UNSLOTH_RUN_GID" not in os.environ


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "inotify requires Linux")
def test_ownership_monitor_tracks_only_changed_paths(runner, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    untouched = existing / "untouched"
    untouched.write_text("old", encoding = "utf-8")
    modified = existing / "modified"
    modified.write_text("old", encoding = "utf-8")
    monitor = runner._OwnershipMonitor(str(tmp_path))
    assert monitor.start()

    modified.write_text("new", encoding = "utf-8")
    created = existing / "new"
    created.mkdir()
    artifact = created / "model.bin"
    artifact.write_bytes(b"model")
    affected, recursive = monitor.stop()

    assert str(artifact) in affected
    assert str(created) in recursive
    assert str(modified) in affected
    assert str(untouched) not in affected


def test_unsloth_run_chowns_only_monitored_root_outputs(runner, monkeypatch, tmp_path):
    changed = str(tmp_path / "changed")
    artifact = str(tmp_path / "new" / "model.bin")
    owners = []
    monkeypatch.setattr(runner.os, "lstat", lambda _path: SimpleNamespace(st_uid = 0))
    monkeypatch.setattr(
        runner.os,
        "chown",
        lambda path, uid, gid, **kwargs: owners.append((path, uid, gid, kwargs)),
    )

    runner._restore_output_ownership({changed, artifact}, set(), 1234, 5678)

    assert owners == [
        (changed, 1234, 5678, {"follow_symlinks": False}),
        (artifact, 1234, 5678, {"follow_symlinks": False}),
    ]


def _run_sh_argv(tmp_path, *command):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "argv"
    docker = bindir / "docker"
    docker.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "{argv_log}"\n', encoding = "utf-8"
    )
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "UNSLOTH_WORKDIR": str(tmp_path),
        "UNSLOTH_GPUS": "none",
    }
    proc = subprocess.run(
        [shutil.which("bash"), str(RUN_SH), *command],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert proc.returncode == 0, proc.stderr
    return argv_log.read_text(encoding = "utf-8").splitlines()


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None, reason = "POSIX shell required"
)
def test_run_sh_starts_unsloth_run_in_the_mounted_host_dir(tmp_path):
    argv = _run_sh_argv(
        tmp_path, "unsloth-run", "--timeout", "60", "https://example.invalid/nb/Llama.ipynb"
    )
    image = argv.index("unsloth/unsloth:latest")
    assert ["-w", "/workspace/host"] in [argv[i : i + 2] for i in range(image)]
    pairs = [argv[i : i + 2] for i in range(image)]
    assert ["-e", f"UNSLOTH_RUN_UID={os.getuid()}"] in pairs
    assert ["-e", f"UNSLOTH_RUN_GID={os.getgid()}"] in pairs
    assert "--user" not in argv[:image]
    assert f"{tmp_path}:/workspace/host" in argv[:image]


@pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None, reason = "POSIX shell required"
)
@pytest.mark.parametrize(
    "command",
    [("jupyter", "lab"), ("unsloth-run", "unsloth-notebooks/nb/Llama.ipynb")],
)
def test_run_sh_leaves_other_commands_in_the_image_workdir(tmp_path, command):
    argv = _run_sh_argv(tmp_path, *command)
    flags = argv[: argv.index("unsloth/unsloth:latest")]
    assert "-w" not in flags
    assert not any(flag.startswith("UNSLOTH_RUN_UID=") for flag in flags)
    assert not any(flag.startswith("UNSLOTH_RUN_GID=") for flag in flags)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
