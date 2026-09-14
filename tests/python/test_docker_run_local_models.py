# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import json
import os
import shutil
import stat
import subprocess

import pytest

_RUN_SH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docker",
    "run.sh",
)

pytestmark = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason = "POSIX shell required",
)

_LEAKS = (
    "HF_HOME",
    "HF_TOKEN",
    "OLLAMA_MODELS",
    "TRITON_CACHE_DIR",
    "UNSLOTH_ALLOW_CPU",
    "UNSLOTH_GPUS",
    "UNSLOTH_HERMES_DIR",
    "UNSLOTH_LMSTUDIO_DIR",
    "UNSLOTH_MODELS_DIR",
    "UNSLOTH_OLLAMA_DIR",
    "WANDB_API_KEY",
)


def _run(tmp_path, **env_extra):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok = True)
    argv_log = tmp_path / "argv"
    docker = bindir / "docker"
    docker.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > ' + str(argv_log) + "\n")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)

    env = {k: v for k, v in os.environ.items() if k not in _LEAKS}
    env.update(
        PATH = f"{bindir}:/usr/bin:/bin",
        HOME = str(tmp_path / "home"),
        UNSLOTH_WORKDIR = str(tmp_path),
        UNSLOTH_GPUS = "none",
        **env_extra,
    )
    env.pop("PWD", None)
    proc = subprocess.run(
        [shutil.which("bash"), _RUN_SH, "true"],
        cwd = tmp_path,
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert proc.returncode == 0, proc.stderr

    argv = argv_log.read_text().splitlines()
    mounts = {}
    for flag, spec in zip(argv, argv[1:]):
        if flag != "-v":
            continue
        read_only = spec.endswith(":ro")
        source, target = spec.removesuffix(":ro").rsplit(":", 1)
        mounts[target] = (os.path.realpath(source), read_only)
    return mounts, proc.stderr


def _dir(path):
    path.mkdir(parents = True)
    return os.path.realpath(path)


def test_detected_model_folders_are_mounted_read_only(tmp_path):
    home = tmp_path / "home"
    lmstudio = _dir(home / ".lmstudio" / "models")
    ollama = _dir(home / ".ollama" / "models")
    hermes = _dir(home / ".hermes" / "models")

    mounts, _ = _run(tmp_path)

    assert mounts["/root/.lmstudio/models"] == (lmstudio, True)
    assert mounts["/root/.ollama/models"] == (ollama, True)
    assert mounts["/root/.hermes/models"] == (hermes, True)
    assert "/workspace/models" not in mounts


def test_no_model_folders_adds_no_model_mounts(tmp_path):
    (tmp_path / "home" / ".lmstudio").mkdir(parents = True)
    (tmp_path / "home" / ".lmstudio" / "settings.json").write_text("{}")

    mounts, _ = _run(tmp_path, UNSLOTH_OLLAMA_DIR = "none")

    model_targets = {
        "/root/.lmstudio/models",
        "/root/.ollama/models",
        "/root/.hermes/models",
        "/workspace/models",
    }
    assert not model_targets & mounts.keys()
    assert "/workspace/.cache/huggingface" in mounts


def test_lmstudio_downloads_folder_setting_wins(tmp_path):
    home = tmp_path / "home"
    _dir(home / ".lmstudio" / "models")
    custom = _dir(tmp_path / "lmstudio-custom")
    (home / ".lmstudio" / "settings.json").write_text(
        json.dumps({"downloadsFolder": custom}, indent = 2)
    )

    mounts, _ = _run(tmp_path)

    assert mounts["/root/.lmstudio/models"] == (custom, True)


def test_legacy_lmstudio_cache_is_found(tmp_path):
    legacy = _dir(tmp_path / "home" / ".cache" / "lm-studio" / "models")

    mounts, _ = _run(tmp_path)

    assert mounts["/root/.lmstudio/models"] == (legacy, True)


def test_ollama_models_env_wins(tmp_path):
    _dir(tmp_path / "home" / ".ollama" / "models")
    custom = _dir(tmp_path / "ollama-custom")

    mounts, _ = _run(tmp_path, OLLAMA_MODELS = custom)

    assert mounts["/root/.ollama/models"] == (custom, True)


def test_explicit_dirs_override_detection(tmp_path):
    _dir(tmp_path / "home" / ".lmstudio" / "models")
    lmstudio = _dir(tmp_path / "lmstudio")
    models = _dir(tmp_path / "gguf")

    mounts, _ = _run(tmp_path, UNSLOTH_LMSTUDIO_DIR = lmstudio, UNSLOTH_MODELS_DIR = "gguf")

    assert mounts["/root/.lmstudio/models"] == (lmstudio, True)
    assert mounts["/workspace/models"] == (models, True)


@pytest.mark.parametrize(
    "variable, folder, target",
    [
        ("UNSLOTH_LMSTUDIO_DIR", ".lmstudio/models", "/root/.lmstudio/models"),
        ("UNSLOTH_OLLAMA_DIR", ".ollama/models", "/root/.ollama/models"),
        ("UNSLOTH_HERMES_DIR", ".hermes/models", "/root/.hermes/models"),
    ],
)
def test_none_skips_a_detected_folder(tmp_path, variable, folder, target):
    _dir(tmp_path / "home" / folder)

    mounts, _ = _run(tmp_path, **{variable: "none"})

    assert target not in mounts


def test_a_missing_explicit_dir_warns_and_is_skipped(tmp_path):
    missing = str(tmp_path / "typo")

    mounts, stderr = _run(tmp_path, UNSLOTH_MODELS_DIR = missing)

    assert "/workspace/models" not in mounts
    assert missing in stderr
