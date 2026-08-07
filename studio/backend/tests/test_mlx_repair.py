# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MLX self-heal: on Apple Silicon with MLX missing, reinstall it by name on a
background thread (off the startup critical path). No-op elsewhere / when present
/ when disabled. Models on core.training.worker's runtime backend self-heal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.mlx_repair as mr  # noqa: E402


@pytest.fixture(autouse = True)
def _reset_attempt_guard(monkeypatch):
    monkeypatch.setattr(mr, "_attempted", False)
    monkeypatch.delenv(mr.DISABLE_ENV_VAR, raising = False)


def test_uv_cmd_targets_this_interpreter_with_mlx_packages(monkeypatch):
    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")
    cmd = mr._uv_install_cmd("--upgrade", *mr.MLX_PACKAGES)
    assert cmd is not None
    assert cmd[:5] == ["/usr/bin/uv", "pip", "install", "--python", sys.executable]
    assert set(mr.MLX_PACKAGES) <= set(cmd)
    # Minimum versions are pinned so the resolver cannot backtrack to an old
    # mlx-vlm that imports but breaks VLM Train/Export.
    assert "mlx-vlm>=0.4.4" in cmd


def test_uv_executable_finds_installer_location_when_path_is_minimal(monkeypatch, tmp_path):
    uv = tmp_path / ".local" / "bin" / "uv"
    uv.parent.mkdir(parents = True)
    uv.write_text("#!/bin/sh\n", encoding = "utf-8")
    uv.chmod(0o755)
    monkeypatch.setattr(mr.shutil, "which", lambda _x: None)
    monkeypatch.setattr(mr.Path, "home", lambda: tmp_path)
    assert mr._uv_executable() == str(uv)


def test_no_uv_repair_stays_chat_only_without_pip(monkeypatch):
    monkeypatch.setattr(mr, "_uv_executable", lambda: None)
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))
    called = {"run": False}

    def _fake_run(*_args, **_kwargs):
        called["run"] = True
        raise AssertionError("plain pip fallback must not run")

    monkeypatch.setattr(mr.subprocess, "run", _fake_run)
    assert mr.attempt_mlx_repair() is False
    assert called["run"] is False


def test_constraint_pins_installed_transformers(monkeypatch):
    transformers = pytest.importorskip("transformers")
    args, path = mr._transformers_constraint_args()
    try:
        assert args[:1] == ["--constraint"]
        assert args[1] == path
        assert Path(path).read_text().strip() == f"transformers=={transformers.__version__}"
    finally:
        if path:
            Path(path).unlink(missing_ok = True)


def test_repair_install_pins_transformers_and_cleans_up(monkeypatch):
    pytest.importorskip("transformers")
    captured = {}
    created_paths = []
    real_args = mr._transformers_constraint_args

    def _spy_args():
        args, path = real_args()
        if path:
            created_paths.append(path)
        return args, path

    monkeypatch.setattr(mr, "_transformers_constraint_args", _spy_args)
    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")

    class _Result:
        returncode = 0
        stdout = ""

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _Result()

    monkeypatch.setattr(mr.subprocess, "run", _fake_run)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)

    assert mr.attempt_mlx_repair() is True
    cmd = captured["cmd"]
    # transformers is pinned via a constraint file so the mlx install cannot
    # upgrade it underneath Unsloth, and the temp constraint file is cleaned up.
    assert "--constraint" in cmd
    assert "--upgrade" in cmd
    reinstall_pairs = set(zip(cmd, cmd[1:]))
    for name in mr._MLX_PACKAGE_NAMES:
        assert ("--reinstall-package", name) in reinstall_pairs
    for pkg in mr.MLX_PACKAGES:
        assert pkg in cmd
    assert created_paths and not Path(created_paths[0]).exists()
    # The install mirrors the main installer by relaxing the transformers pin via
    # UV_OVERRIDE so a current mlx-vlm can coexist with transformers==4.57.6.
    env = captured["env"]
    assert env is not None
    assert env.get("UV_OVERRIDE", "").endswith("overrides-darwin-arm64.txt")


def test_install_requires_prebuilt_wheels(monkeypatch):
    # A source distribution's PEP 517 build backend runs arbitrary code at install
    # time, before the post-install stack check. The unattended self-heal must
    # require pre-built wheels so a malicious resolver-selected sdist cannot execute
    # during ordinary Unsloth startup. mlx/mlx-metal ship wheels only and
    # mlx-lm/mlx-vlm publish py3-none-any wheels, so a healthy self-heal still works.
    pytest.importorskip("transformers")
    captured = {}

    class _Result:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")
    monkeypatch.setattr(
        mr.subprocess, "run", lambda cmd, **k: captured.update(cmd = cmd) or _Result()
    )
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)

    assert mr.attempt_mlx_repair() is True
    assert mr._ONLY_BINARY_ARG in captured["cmd"]


def test_install_env_drops_secrets_and_source_redirects(monkeypatch):
    # The unattended self-heal must not hand resolver/build code the full Unsloth
    # environment: secrets and package-source redirects are dropped, while the
    # variables uv genuinely needs are forwarded.
    monkeypatch.setenv("HF_TOKEN", "secret-hf")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-aws")
    monkeypatch.setenv("WANDB_API_KEY", "secret-wandb")
    monkeypatch.setenv("UV_FIND_LINKS", "/tmp/evil")
    monkeypatch.setenv("UV_DEFAULT_INDEX", "file:///tmp/evil-index")
    monkeypatch.setenv("UV_INDEX_URL", "https://evil.example/simple")
    monkeypatch.setenv("PIP_INDEX_URL", "https://evil.example/simple")
    monkeypatch.setenv("UV_CACHE_DIR", "/tmp/evil-cache")
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/evil-xdg-cache")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/studio")

    env = mr._mlx_install_env()

    # Secrets never reach a (potentially malicious) build/install hook.
    for secret in ("HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "WANDB_API_KEY"):
        assert secret not in env
    # A poisoned process env cannot repoint the install at a hostile source or
    # an attacker-staged cache (cache poisoning / symlink writes).
    for redirect in (
        "UV_FIND_LINKS",
        "UV_DEFAULT_INDEX",
        "UV_INDEX_URL",
        "PIP_INDEX_URL",
        "UV_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        assert redirect not in env
    # What uv genuinely needs is still forwarded.
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"] == "/home/studio"
    # UV_OVERRIDE is set by us (not inherited), so a poisoned one is ignored.
    assert env.get("UV_OVERRIDE", "").endswith("overrides-darwin-arm64.txt")


def test_repair_rejects_inadequate_stack(monkeypatch):
    # A successful uv run that still leaves an old/missing mlx-vlm must NOT clear
    # chat-only: attempt_mlx_repair returns False so Train/Export stay disabled.
    class _Result:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Result())
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)
    assert mr.attempt_mlx_repair() is False


def test_repair_invalidates_import_caches_before_stack_check(monkeypatch):
    events = []

    class _Result:
        returncode = 0
        stdout = ""

    def _stack_available():
        events.append("check")
        assert events == ["invalidate", "check"]
        return True

    monkeypatch.setattr(mr.subprocess, "run", lambda *a, **k: _Result())
    monkeypatch.setattr(mr, "_uv_executable", lambda: "/usr/bin/uv")
    monkeypatch.setattr(mr, "_transformers_constraint_args", lambda: ([], None))
    monkeypatch.setattr(mr.importlib, "invalidate_caches", lambda: events.append("invalidate"))
    monkeypatch.setattr(mr, "mlx_stack_available", _stack_available)

    assert mr.attempt_mlx_repair() is True
    assert events == ["invalidate", "check"]


def test_stack_unavailable_without_mlx(monkeypatch):
    import importlib.metadata as metadata

    def _missing(_name):
        raise metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(metadata, "version", _missing)
    assert mr.mlx_stack_available() is False


def test_stack_unavailable_checks_versions_before_imports(monkeypatch):
    import importlib.metadata as metadata

    def _version(name):
        if name == "mlx":
            return "0.21.0"
        return mr._MLX_MIN_VERSIONS[name]

    def _import_module(_name):
        raise AssertionError("MLX modules must not import before versions pass")

    monkeypatch.setattr(metadata, "version", _version)
    monkeypatch.setattr(mr.importlib, "import_module", _import_module)
    assert mr.mlx_stack_available() is False


def test_stack_unavailable_when_companion_import_fails(monkeypatch):
    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "version", lambda name: mr._MLX_MIN_VERSIONS[name])

    def _import_module(name):
        if name == "mlx_vlm":
            raise ModuleNotFoundError(name)
        return object()

    monkeypatch.setattr(mr.importlib, "import_module", _import_module)
    assert mr.mlx_stack_available() is False


def test_stack_available_requires_runtime_imports_and_versions(monkeypatch):
    import importlib.metadata as metadata

    imported = []

    def _import_module(name):
        imported.append(name)
        return object()

    monkeypatch.setattr(mr.importlib, "import_module", _import_module)
    monkeypatch.setattr(metadata, "version", lambda name: mr._MLX_MIN_VERSIONS[name])

    assert mr.mlx_stack_available() is True
    assert imported == list(mr._MLX_RUNTIME_IMPORTS)


def test_mlx_packages_exclude_known_bad_mlx_lm():
    # mlx-lm 0.31.3 regressed QK-norm archs (gemma4 / qwen3_5); the install spec
    # must exclude it so the resolver picks 0.31.2 or >=0.31.4. See mlx-lm #1242.
    (mlx_lm_spec,) = [p for p in mr.MLX_PACKAGES if p.startswith("mlx-lm")]
    assert mlx_lm_spec == "mlx-lm>=0.22.0,!=0.31.3"


@pytest.mark.parametrize("bad_form", ["0.31.3", "0.31.3.0"])
def test_known_bad_installed_mlx_lm_triggers_repair(monkeypatch, bad_form):
    # An installed 0.31.3 counts as unsatisfied so the self-heal replaces it;
    # parsed-Version compare also catches the trailing-zero form 0.31.3.0.
    import importlib.metadata as metadata

    def _version(name):
        return bad_form if name == "mlx-lm" else mr._MLX_MIN_VERSIONS[name]

    monkeypatch.setattr(metadata, "version", _version)
    monkeypatch.setattr(
        mr.importlib, "import_module", lambda _n: pytest.fail("versions must gate imports")
    )
    assert mr.mlx_stack_available() is False


def test_no_op_off_apple_silicon(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: False)
    called = {"n": 0}
    monkeypatch.setattr(
        mr, "attempt_mlx_repair", lambda **_k: called.__setitem__("n", called["n"] + 1) or True
    )
    assert mr.start_mlx_autorepair_if_needed() is False
    assert called["n"] == 0


def test_no_op_when_mlx_stack_present(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: True)
    started = mr.start_mlx_autorepair_if_needed()
    assert started is False


def test_disable_env_skips(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)
    monkeypatch.setenv(mr.DISABLE_ENV_VAR, "1")
    assert mr.start_mlx_autorepair_if_needed() is False


def test_apple_silicon_missing_mlx_starts_repair_and_redetects(monkeypatch):
    import threading

    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)

    repaired = {"called": False}

    def _fake_repair(**_kw):
        repaired["called"] = True
        return True

    redetected = {"called": False}

    # _run_repair_and_redetect imports utils.hardware.hardware lazily; stub repair
    # and capture that re-detection is invoked on success.
    monkeypatch.setattr(mr, "attempt_mlx_repair", _fake_repair)

    import utils.hardware.hardware as hw

    monkeypatch.setattr(hw, "detect_hardware", lambda: redetected.__setitem__("called", True))

    started = mr.start_mlx_autorepair_if_needed()
    assert started is True

    # Join the daemon thread deterministically.
    for thread in threading.enumerate():
        if thread.name == "mlx-autorepair":
            thread.join(timeout = 5)

    assert repaired["called"] is True
    assert redetected["called"] is True


def test_attempts_only_once_per_process(monkeypatch):
    monkeypatch.setattr(mr, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mr, "mlx_stack_available", lambda: False)
    monkeypatch.setattr(mr, "attempt_mlx_repair", lambda **_k: False)

    first = mr.start_mlx_autorepair_if_needed()
    second = mr.start_mlx_autorepair_if_needed()
    assert first is True
    assert second is False  # guard prevents a second concurrent attempt


def test_mlx_install_env_routes_uv_override_through_safe_path(monkeypatch):
    # uv truncates UV_OVERRIDE at the first space (issue #6503).
    seen = {}

    def _spy(path):
        seen["path"] = path
        return "/space free/marker.txt".replace(" ", "_")

    monkeypatch.setattr(mr, "uv_safe_path", _spy)
    monkeypatch.delenv("UV_OVERRIDE", raising = False)

    env = mr._mlx_install_env()

    # The override file ships in the repo, so the helper must have run.
    assert "path" in seen
    assert str(seen["path"]).endswith("overrides-darwin-arm64.txt")
    assert env["UV_OVERRIDE"] == "/space_free/marker.txt"
