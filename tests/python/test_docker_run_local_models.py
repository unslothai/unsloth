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
    "HERMES_HOME",
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


def _fake_wslpath(tmp_path, windows, mapped):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok = True)
    wslpath = bindir / "wslpath"
    wslpath.write_text(
        f'#!/usr/bin/env bash\n[[ "$1" == -u && "$2" == \'{windows}\' ]] || exit 1\n'
        f"printf '%s\\n' '{mapped}'\n"
    )
    wslpath.chmod(wslpath.stat().st_mode | stat.S_IEXEC)


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


def test_lmstudio_downloads_folder_tilde_is_expanded(tmp_path):
    home = tmp_path / "home"
    custom = _dir(home / "tilde-models")
    (home / ".lmstudio").mkdir()
    (home / ".lmstudio" / "settings.json").write_text(
        json.dumps({"downloadsFolder": "~/tilde-models"})
    )

    mounts, _ = _run(tmp_path)

    assert mounts["/root/.lmstudio/models"] == (custom, True)


def test_lmstudio_downloads_folder_with_a_backslash_is_json_decoded(tmp_path):
    home = tmp_path / "home"
    _dir(home / ".lmstudio" / "models")
    custom = _dir(tmp_path / "back\\slash models")
    (home / ".lmstudio" / "settings.json").write_text(json.dumps({"downloadsFolder": custom}))

    mounts, _ = _run(tmp_path)

    assert mounts["/root/.lmstudio/models"] == (custom, True)


def test_lmstudio_windows_downloads_folder_is_mapped_under_wsl(tmp_path):
    home = tmp_path / "home"
    _dir(home / ".lmstudio" / "models")
    windows = _dir(tmp_path / "mnt" / "c" / "Users" / "u" / "models")
    (home / ".lmstudio" / "settings.json").write_text(
        json.dumps({"downloadsFolder": "C:\\Users\\u\\models"})
    )
    _fake_wslpath(tmp_path, "C:\\Users\\u\\models", windows)

    mounts, _ = _run(tmp_path)

    assert mounts["/root/.lmstudio/models"] == (windows, True)


def test_custom_hermes_home_is_the_models_root(tmp_path):
    root = tmp_path / "srv" / "hermes"
    models = _dir(root / "models")

    mounts, _ = _run(tmp_path, HERMES_HOME = str(root))

    assert mounts["/root/.hermes/models"] == (models, True)


def test_hermes_home_tilde_is_expanded(tmp_path):
    models = _dir(tmp_path / "home" / "hermes-root" / "models")

    mounts, _ = _run(tmp_path, HERMES_HOME = "~/hermes-root")

    assert mounts["/root/.hermes/models"] == (models, True)


def test_windows_hermes_home_is_mapped_under_wsl(tmp_path):
    root = _dir(tmp_path / "mnt" / "d" / "hermes")
    models = _dir(tmp_path / "mnt" / "d" / "hermes" / "models")
    _fake_wslpath(tmp_path, "D:\\hermes", root)

    mounts, _ = _run(tmp_path, HERMES_HOME = "D:\\hermes")

    assert mounts["/root/.hermes/models"] == (models, True)


def test_hermes_profile_outside_the_home_uses_its_root(tmp_path):
    root = tmp_path / "data"
    models = _dir(root / "models")
    _dir(root / "profiles" / "coder")

    mounts, _ = _run(tmp_path, HERMES_HOME = str(root / "profiles" / "coder"))

    assert mounts["/root/.hermes/models"] == (models, True)


def test_hermes_session_home_without_models_keeps_native_models(tmp_path):
    native = _dir(tmp_path / "home" / ".hermes" / "models")
    _dir(tmp_path / "session")

    mounts, _ = _run(tmp_path, HERMES_HOME = str(tmp_path / "session"))

    assert mounts["/root/.hermes/models"] == (native, True)


def test_legacy_lmstudio_cache_is_found(tmp_path):
    legacy = _dir(tmp_path / "home" / ".cache" / "lm-studio" / "models")

    mounts, _ = _run(tmp_path)

    assert mounts["/root/.lmstudio/models"] == (legacy, True)


def test_ollama_models_env_wins(tmp_path):
    _dir(tmp_path / "home" / ".ollama" / "models")
    custom = _dir(tmp_path / "ollama-custom")

    mounts, _ = _run(tmp_path, OLLAMA_MODELS = custom)

    assert mounts["/root/.ollama/models"] == (custom, True)


def test_windows_ollama_models_env_is_mapped_under_wsl(tmp_path):
    _dir(tmp_path / "home" / ".ollama" / "models")
    windows = _dir(tmp_path / "mnt" / "d" / "ollama")
    _fake_wslpath(tmp_path, "D:\\ollama", windows)

    mounts, _ = _run(tmp_path, OLLAMA_MODELS = "D:\\ollama")

    assert mounts["/root/.ollama/models"] == (windows, True)


_SERVICE_STORE = "/usr/share/ollama/.ollama/models"
_VAR_LIB_STORE = "/var/lib/ollama/.ollama/models"


def _ollama_mounts(tmp_path, *stores, **env_extra):
    root = tmp_path / "root"
    paths = [_dir(root / store.lstrip("/")) for store in stores]
    mounts, _ = _run(tmp_path, UNSLOTH_DEV_ROOT = str(root), **env_extra)
    argv = (tmp_path / "argv").read_text().splitlines()
    ollama = {t: m for t, m in mounts.items() if "ollama" in t}
    return ollama, paths, argv


def test_home_store_does_not_hide_the_service_store(tmp_path):
    home = _dir(tmp_path / "home" / ".ollama" / "models")

    ollama, (service,), _ = _ollama_mounts(tmp_path, _SERVICE_STORE)

    assert ollama == {"/root/.ollama/models": (home, True), _SERVICE_STORE: (service, True)}


def test_every_ollama_store_is_mounted(tmp_path):
    home = _dir(tmp_path / "home" / ".ollama" / "models")

    ollama, (service, var_lib), _ = _ollama_mounts(tmp_path, _SERVICE_STORE, _VAR_LIB_STORE)

    assert ollama == {
        "/root/.ollama/models": (home, True),
        _SERVICE_STORE: (service, True),
        _VAR_LIB_STORE: (var_lib, True),
    }


def test_lone_service_store_keeps_its_mount_point(tmp_path):
    ollama, (service,), argv = _ollama_mounts(tmp_path, _SERVICE_STORE)

    assert ollama == {"/root/.ollama/models": (service, True)}
    assert not any(a.startswith("OLLAMA_MODELS=") for a in argv)


def test_ollama_models_env_store_is_not_shadowed_by_the_home_store(tmp_path):
    _dir(tmp_path / "home" / ".ollama" / "models")
    custom = _dir(tmp_path / "ollama custom")

    ollama, (service,), argv = _ollama_mounts(tmp_path, _SERVICE_STORE, OLLAMA_MODELS = custom)

    assert ollama == {"/root/.ollama/models": (custom, True), _SERVICE_STORE: (service, True)}
    assert not any(a.startswith("OLLAMA_MODELS=") for a in argv)


@pytest.mark.skipif(os.name == "posix" and os.geteuid() == 0, reason = "root enters any directory")
def test_store_the_launcher_cannot_enter_is_still_mounted(tmp_path):
    store = tmp_path / "service-owned"
    store.mkdir()
    store.chmod(0o600)
    try:
        mounts, _ = _run(tmp_path, OLLAMA_MODELS = str(store))
    finally:
        store.chmod(0o700)

    assert mounts["/root/.ollama/models"] == (os.path.realpath(store), True)


def test_the_same_ollama_store_is_mounted_once(tmp_path):
    root = tmp_path / "root"
    service = _dir(root / _SERVICE_STORE.lstrip("/"))
    (tmp_path / "home" / ".ollama").mkdir(parents = True)
    (tmp_path / "home" / ".ollama" / "models").symlink_to(service)

    ollama, _, argv = _ollama_mounts(tmp_path, OLLAMA_MODELS = service)

    assert list(ollama) == ["/root/.ollama/models"]
    assert ollama["/root/.ollama/models"] == (service, True)
    assert not any(a.startswith("OLLAMA_MODELS=") for a in argv)


@pytest.mark.parametrize("value", ["none", "picked"])
def test_explicit_ollama_dir_mounts_only_that_dir(tmp_path, value):
    _dir(tmp_path / "home" / ".ollama" / "models")
    picked = _dir(tmp_path / "picked")
    dirs = {"none": "none", "picked": picked}

    ollama, _, _ = _ollama_mounts(tmp_path, _SERVICE_STORE, UNSLOTH_OLLAMA_DIR = dirs[value])

    assert ollama == ({} if value == "none" else {"/root/.ollama/models": (picked, True)})


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
