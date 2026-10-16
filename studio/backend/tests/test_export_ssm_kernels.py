# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export installs the SSM kernels a checkpoint needs before Unsloth and its modeling import, as a
training or chat load does: setup no longer installs them, and Nemotron-H's remote modeling code
cannot even import without mamba-ssm."""

import json
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.export import worker as export_worker  # noqa: E402
from utils import ssm_runtime  # noqa: E402


class _Queue(list):
    def put(self, item):
        self.append(item)


class _ClosedQueue:
    def get(self, timeout = None):
        raise EOFError


def test_a_nemotron_h_export_provisions_kernels_before_the_backend_import(monkeypatch, tmp_path):
    checkpoint = tmp_path / "nemotron-export"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    order: list = []

    def _ensure(probe, **kwargs):
        order.append(("provision", kwargs["is_ssm"], kwargs["mamba_ssm_required"]))
        return []

    monkeypatch.setattr(ssm_runtime, "ensure_ssm_runtime", _ensure)
    monkeypatch.setattr(export_worker, "_setup_log_capture", lambda queue: None)
    monkeypatch.setattr(export_worker, "_activate_transformers_version", lambda *a, **k: None)
    monkeypatch.setattr(export_worker, "apply_gpu_ids", lambda *a, **k: None)
    monkeypatch.setattr(
        "core._torchao_stub.install_torchao_windows_rocm_real_or_stub", lambda: None
    )
    monkeypatch.setattr("core.import_guards.ensure_real_packages", lambda *names: None)

    class _Backend:
        def __init__(self):
            order.append(("backend",))

    fake_export = types.ModuleType("core.export.export")

    def _getattr(name):
        if name == "ExportBackend":
            order.append(("import",))
            return _Backend
        raise AttributeError(name)

    fake_export.__getattr__ = _getattr
    monkeypatch.setitem(sys.modules, "core.export.export", fake_export)
    monkeypatch.setattr(export_worker, "_handle_load", lambda *a, **k: order.append(("load",)))
    # run_export_process configures its own process; keep that from leaking into this one.
    for key in ("TQDM_MININTERVAL", "TOKENIZERS_PARALLELISM", "PYTHONWARNINGS", "PYTHONUNBUFFERED"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setenv("ENVIRONMENT_TYPE", "development")
    monkeypatch.setattr("loggers.config.LogConfig.setup_logging", lambda *a, **k: None)
    monkeypatch.setattr("loggers.config.allow_progress_bars", lambda: None)

    queue = _Queue()
    export_worker.run_export_process(
        cmd_queue = _ClosedQueue(),
        resp_queue = queue,
        config = {"checkpoint_path": str(checkpoint), "allow_ambient": True},
    )
    assert order[0] == ("provision", True, True)
    assert order.index(("provision", True, True)) < order.index(("import",))
    assert ("load",) in order


def test_a_failed_required_install_stops_the_export_before_the_import(monkeypatch, tmp_path):
    checkpoint = tmp_path / "falcon"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "falcon_h1"}))
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    queue = _Queue()
    assert export_worker._provision_ssm_kernels(str(checkpoint), None, None, queue) is False
    assert queue[-1]["type"] == "error"
    assert "mamba-ssm" in queue[-1]["error"]


def test_an_optional_ssm_family_exports_on_the_torch_path(monkeypatch, tmp_path):
    checkpoint = tmp_path / "jamba"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "jamba"}))
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    queue = _Queue()
    assert export_worker._provision_ssm_kernels(str(checkpoint), None, None, queue) is True
    assert not any(item.get("type") == "error" for item in queue)


@pytest.mark.parametrize("pinned", [True, False])
def test_a_lora_export_resolves_its_base(monkeypatch, tmp_path, pinned):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "falcon_h1"}))
    adapter = tmp_path / "my-adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    seen = []
    monkeypatch.setattr(
        ssm_runtime, "ensure_ssm_runtime", lambda probe, **k: seen.append(k["is_ssm"]) or []
    )
    monkeypatch.setattr("utils.transformers_version._remote_lora_base", lambda *a, **k: None)
    assert export_worker._provision_ssm_kernels(
        str(adapter), str(base) if pinned else None, None, _Queue()
    )
    assert seen == [True]
