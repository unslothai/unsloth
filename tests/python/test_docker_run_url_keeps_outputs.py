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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "docker" / "unsloth_run.py"
ROOT_SHIM_PATH = REPO_ROOT / "docker" / "unsloth_root_shim.py"
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
        seen["env"] = env
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


def test_a_url_download_takes_the_callers_mapped_owner(runner, monkeypatch, cwd):
    owners = []
    monkeypatch.setenv("UNSLOTH_RUN_UID", "1234")
    monkeypatch.setenv("UNSLOTH_RUN_GID", "5678")
    monkeypatch.setattr(runner, "_container_run_ids", lambda ids: ids)
    monkeypatch.setattr(
        runner.os, "fchown", lambda fd, uid, gid: owners.append((uid, gid)), raising = False
    )
    _run(runner, monkeypatch, ["https://example.invalid/nb/Llama.ipynb"])

    assert owners == [(1234, 5678)]


def test_a_url_out_path_uses_the_callers_owner_for_new_directories_and_output(
    runner, monkeypatch, cwd
):
    owners = []
    monkeypatch.setenv("UNSLOTH_RUN_UID", "1234")
    monkeypatch.setenv("UNSLOTH_RUN_GID", "5678")
    monkeypatch.setattr(runner, "_container_run_ids", lambda ids: ids)
    monkeypatch.setattr(
        runner.os,
        "chown",
        lambda path, uid, gid: owners.append((Path(path), uid, gid)),
    )

    _run(
        runner,
        monkeypatch,
        ["https://example.invalid/nb/Llama.ipynb", "--out", "new/dir/result.ipynb"],
    )

    expected = {(cwd / "new", 1234, 5678), (cwd / "new" / "dir", 1234, 5678)}
    assert expected.issubset(set(owners))
    assert any(
        path.name.startswith(".unsloth-run-out-") and (uid, gid) == (1234, 5678)
        for path, uid, gid in owners
    )


def test_unsloth_run_reads_and_hides_the_requested_host_owner(runner, monkeypatch):
    monkeypatch.setenv("UNSLOTH_RUN_UID", "1234")
    monkeypatch.setenv("UNSLOTH_RUN_GID", "5678")

    assert runner._host_run_ids() == (1234, 5678)

    assert "UNSLOTH_RUN_UID" not in os.environ
    assert "UNSLOTH_RUN_GID" not in os.environ


def test_rootless_id_maps_translate_the_host_caller_to_container_root(runner, tmp_path):
    uid_map = tmp_path / "uid_map"
    gid_map = tmp_path / "gid_map"
    uid_map.write_text("0 1000 1\n1 100000 65536\n", encoding = "utf-8")
    gid_map.write_text("0 1000 1\n1 100000 65536\n", encoding = "utf-8")

    assert runner._container_run_ids((1000, 1000), uid_map, gid_map) == (0, 0)
    assert runner._container_run_ids((101234, 101234), uid_map, gid_map) == (1235, 1235)


def test_an_unmapped_host_identity_is_rejected(runner, tmp_path):
    uid_map = tmp_path / "uid_map"
    gid_map = tmp_path / "gid_map"
    uid_map.write_text("0 100000 65536\n", encoding = "utf-8")
    gid_map.write_text("0 100000 65536\n", encoding = "utf-8")

    with pytest.raises(SystemExit, match = "host ID 1000 is not mapped"):
        runner._container_run_ids((1000, 1000), uid_map, gid_map)


def test_url_run_uses_host_identity_with_a_privileged_install_path(runner, monkeypatch, cwd):
    monkeypatch.setenv("UNSLOTH_RUN_UID", "1234")
    monkeypatch.setenv("UNSLOTH_RUN_GID", "5678")
    monkeypatch.setattr(runner, "_container_run_ids", lambda ids: ids)

    seen = _run(runner, monkeypatch, ["https://example.invalid/nb/Llama.ipynb"])

    assert seen["env"]["UNSLOTH_NB_ROOT_INSTALL"] == "1"
    assert runner._host_owned_command(["jupyter"], (1234, 5678)) == [
        "/usr/bin/setpriv",
        "--reuid=1234",
        "--regid=5678",
        "--keep-groups",
        "--inh-caps=-all,+chown,+dac_override,+fowner,+setgid,+setuid",
        "--ambient-caps=-all,+chown,+dac_override,+fowner,+setgid,+setuid",
        "jupyter",
    ]


@pytest.mark.parametrize("host_ids", [None, (0, 0)])
def test_run_without_a_nonroot_mapped_identity_needs_no_privilege_wrapper(runner, host_ids):
    cmd = ["jupyter"]
    assert runner._host_owned_command(cmd, host_ids) is cmd


def test_root_shim_restores_root_before_executing_system_tools(monkeypatch):
    spec = importlib.util.spec_from_file_location("unsloth_root_shim_under_test", ROOT_SHIM_PATH)
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    calls = []
    monkeypatch.setenv("UNSLOTH_NB_ROOT_INSTALL", "1")
    monkeypatch.setattr(shim.os, "setgid", lambda gid: calls.append(("gid", gid)))
    monkeypatch.setattr(shim.os, "setuid", lambda uid: calls.append(("uid", uid)))
    monkeypatch.setattr(
        shim.os,
        "execv",
        lambda path, argv: calls.append(("exec", path, argv)),
    )
    monkeypatch.setattr(shim.sys, "argv", ["apt-get", "install", "-y", "git"])

    shim.main()

    assert calls == [
        ("gid", 0),
        ("uid", 0),
        ("exec", "/usr/bin/apt-get", ["/usr/bin/apt-get", "install", "-y", "git"]),
    ]


def test_root_shim_emulates_common_sudo_flags(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "unsloth_root_shim_sudo_under_test", ROOT_SHIM_PATH
    )
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    calls = []
    monkeypatch.setenv("UNSLOTH_NB_ROOT_INSTALL", "1")
    monkeypatch.setattr(shim.os, "setgid", lambda gid: None)
    monkeypatch.setattr(shim.os, "setuid", lambda uid: None)
    monkeypatch.setattr(
        shim.os,
        "execvpe",
        lambda file, argv, env: calls.append((file, argv, env)),
    )
    monkeypatch.setattr(shim.sys, "argv", ["sudo", "-E", "--", "apt-get", "update"])

    shim.main()

    assert calls == [("apt-get", ["apt-get", "update"], shim.os.environ)]


def test_root_shim_does_not_elevate_an_unprivileged_container(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "unsloth_root_shim_unprivileged_under_test", ROOT_SHIM_PATH
    )
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    calls = []
    monkeypatch.delenv("UNSLOTH_NB_ROOT_INSTALL", raising = False)
    monkeypatch.setattr(shim.os, "geteuid", lambda: 1234)
    monkeypatch.setattr(
        shim.os,
        "setuid",
        lambda uid: pytest.fail(f"unexpected privilege restoration to UID {uid}"),
    )
    monkeypatch.setattr(
        shim.os,
        "setgid",
        lambda gid: pytest.fail(f"unexpected privilege restoration to GID {gid}"),
    )
    monkeypatch.setattr(
        shim.os,
        "execv",
        lambda path, argv: calls.append((path, argv)),
    )
    monkeypatch.setattr(shim.sys, "argv", ["dpkg", "--print-architecture"])

    shim.main()

    assert calls == [("/usr/bin/dpkg", ["/usr/bin/dpkg", "--print-architecture"])]


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
