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
    (root / "package.json").write_text(json.dumps({"dependencies": {"@anthropic-ai/sandbox-runtime": "0.0.75"}}))
    (root / "package-lock.json").write_text(json.dumps({"packages": {"node_modules/@anthropic-ai/sandbox-runtime": {"version": "0.0.75"}}}))
    (root / "apply_patch.mjs").write_text("// fixture patch entrypoint\n")
    monkeypatch.setattr(module.shutil, "which", lambda name: "/trusted/bin/" + name)
    return module, root


def test_non_linux_setup_never_installs_or_changes_policy(installer, monkeypatch):
    module, _ = installer
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Windows invoked a process"))
    assert module.install() == 0


def test_offline_install_uses_lock_and_verifies_without_lifecycle_scripts(installer, monkeypatch):
    module, root = installer
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(stdout = "v24.13.0\n")
    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.install(offline = True) == 0
    assert calls[1][0] == ["/trusted/bin/npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund", "--offline"]
    assert calls[1][1]["cwd"] == root
    assert calls[2][0] == ["/trusted/bin/node", str(root / "apply_patch.mjs")]
    assert calls[3][0][-1] == (root / "bridge.mjs").as_uri()
    assert "verifyInstallation" in calls[3][0][-2]


@pytest.mark.parametrize("version", ["v18.20.0", "v20.10.9", "unexpected"])
def test_old_node_refuses_before_npm(installer, monkeypatch, version):
    module, _ = installer
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(stdout = version))
    with pytest.raises(RuntimeError, match = "Node >=20.11"):
        module.install()
    assert len(calls) == 1


def test_changed_lock_refuses_before_npm(installer, monkeypatch):
    module, root = installer
    (root / "package-lock.json").write_text('{"packages": {}}')
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(stdout = "v24.13.0"))
    with pytest.raises(RuntimeError, match = "lock"):
        module.install()
    assert len(calls) == 1


def test_missing_node_refuses_without_downloading(installer, monkeypatch):
    module, _ = installer
    monkeypatch.setattr(module.shutil, "which", lambda name: None)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Missing Node launched a process"))
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
