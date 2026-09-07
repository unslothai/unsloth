# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def installer(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "install_srt_runtime.py"
    spec = importlib.util.spec_from_file_location("srt_installer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(tmp_path / "install_srt_runtime.py"))
    monkeypatch.setattr(module.sys, "platform", "linux")
    root = tmp_path / "backend/core/inference/srt_runtime"
    root.mkdir(parents = True)
    (root / "package.json").write_text(
        json.dumps({"dependencies": {"@anthropic-ai/sandbox-runtime": "0.0.75"}})
    )
    (root / "package-lock.json").write_text(
        json.dumps(
            {"packages": {"node_modules/@anthropic-ai/sandbox-runtime": {"version": "0.0.75"}}}
        )
    )
    (root / "apply_patch.mjs").write_text("// fixture patch entrypoint\n")
    monkeypatch.setattr(module.shutil, "which", lambda name: "/trusted/bin/" + name)
    return module, root


def test_unsupported_setup_never_installs_or_changes_policy(installer, monkeypatch):
    module, _ = installer
    monkeypatch.setattr(module.sys, "platform", "freebsd")
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: pytest.fail("Windows invoked a process")
    )
    with pytest.raises(RuntimeError, match = "support"):
        module.install()


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_native_setup_installs_verified_helper_without_privileged_action(
    installer, monkeypatch, platform
):
    module, root = installer
    monkeypatch.setattr(module.sys, "platform", platform)
    npm_dir = root / "node tools"
    npm_cli = npm_dir / "node_modules/npm/bin/npm-cli.js"
    npm_cli.parent.mkdir(parents = True)
    npm_cli.write_text("// npm fixture")
    monkeypatch.setattr(module.shutil, "which", lambda name: str(npm_dir / name))
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(stdout = "v24.13.0"),
    )
    assert module.install() == 0
    assert len(calls) == 4
    assert all("windows-install" not in call for call in calls)
    if platform == "win32":
        assert calls[1][:2] == [str(npm_dir / "node"), str(npm_cli)]
    assert "--ignore-scripts" in calls[1]
    assert "verifyInstallation" in calls[-1][-2]


@pytest.mark.parametrize("integrity_ok", [True, False])
def test_explicit_windows_install_runs_only_after_integrity(installer, monkeypatch, integrity_ok):
    module, root = installer
    monkeypatch.setattr(module.sys, "platform", "win32")
    npm_cli = root / "node_modules/npm/bin/npm-cli.js"
    npm_cli.parent.mkdir(parents = True)
    npm_cli.write_text("// npm fixture")
    monkeypatch.setattr(module.shutil, "which", lambda name: str(root / name))
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if "--input-type=module" in argv and not integrity_ok:
            raise module.subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout = "v24.13.0")

    monkeypatch.setattr(module.subprocess, "run", run)
    if not integrity_ok:
        with pytest.raises(module.subprocess.CalledProcessError):
            module.install(windows_install = True)
        assert all("windows-install" not in call for call in calls)
        return
    assert module.install(windows_install = True) == 0
    assert "verifyInstallation" in calls[-2][-2]
    assert calls[-1] == [
        str(root / "node"),
        str(root / "node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js"),
        "windows-install",
    ]


def test_windows_install_flag_rejects_other_platform_before_process(installer, monkeypatch):
    module, _ = installer
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(RuntimeError, match = "requires Windows"):
        module.install(windows_install = True)


def test_offline_install_uses_lock_and_verifies_without_lifecycle_scripts(installer, monkeypatch):
    module, root = installer
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout = "v24.13.0\n")

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.install(offline = True) == 0
    assert calls[1][0] == [
        "/trusted/bin/npm",
        "ci",
        "--ignore-scripts",
        "--no-audit",
        "--no-fund",
        "--offline",
    ]
    assert calls[1][1]["cwd"] == root
    assert calls[2][0] == ["/trusted/bin/node", str(root / "apply_patch.mjs")]
    assert calls[3][0][-1] == (root / "bridge.mjs").as_uri()
    assert "verifyInstallation" in calls[3][0][-2]


@pytest.mark.parametrize("version", ["v18.20.0", "v20.10.9", "unexpected"])
def test_old_node_refuses_before_npm(installer, monkeypatch, version):
    module, _ = installer
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(stdout = version),
    )
    with pytest.raises(RuntimeError, match = "Node >=20.11"):
        module.install()
    assert len(calls) == 1


def test_changed_lock_refuses_before_npm(installer, monkeypatch):
    module, root = installer
    (root / "package-lock.json").write_text('{"packages": {}}')
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(stdout = "v24.13.0"),
    )
    with pytest.raises(RuntimeError, match = "lock"):
        module.install()
    assert len(calls) == 1


def test_missing_node_refuses_without_downloading(installer, monkeypatch):
    module, _ = installer
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        module.subprocess, "run", lambda *a, **k: pytest.fail("Missing Node launched a process")
    )
    with pytest.raises(RuntimeError, match = "Node >=20.11"):
        module.install()


def test_patch_failure_stops_before_integrity_success(installer, monkeypatch):
    module, _ = installer
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[-1].endswith("apply_patch.mjs"):
            raise module.subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout = "v24.13.0")

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(module.subprocess.CalledProcessError):
        module.install()
    assert not any("--input-type=module" in argv for argv in calls)


def test_integrity_failure_is_not_reported_as_installed(installer, monkeypatch):
    module, _ = installer

    def run(argv, **kwargs):
        if "--input-type=module" in argv:
            raise module.subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(stdout = "v24.13.0")

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(module.subprocess.CalledProcessError):
        module.install()
