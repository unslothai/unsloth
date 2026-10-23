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
    order = [
        step
        for step in _run_export(monkeypatch, tmp_path, {"device_backend": "cuda"})
        if step[0] != "window"
    ]
    assert order[0] == ("provision", True, True)
    assert order.index(("provision", True, True)) < order.index(("import",))
    assert ("load",) in order


@pytest.mark.parametrize("device_backend", ["cpu", "mlx"])
def test_a_cpu_or_mlx_export_makes_no_extra_connectivity_probe(
    monkeypatch, tmp_path, device_backend
):
    # 9f5b13325 entered no "installing SSM kernels" window; an export that installs nothing must not.
    order = _run_export(monkeypatch, tmp_path, {"device_backend": device_backend})
    assert ("window", "installing SSM kernels") not in order
    assert not any(step[0] == "provision" for step in order)
    assert ("load",) in order


@pytest.mark.parametrize("device_backend", ["cuda", "cpu", "mlx"])
def test_provisioning_adds_no_connectivity_window(monkeypatch, tmp_path, device_backend):
    # Same window count as 9f5b13325: provisioning rides the activation window.
    order = _run_export(monkeypatch, tmp_path, {"device_backend": device_backend})
    windows = [step for step in order if step[0] == "window"]
    assert windows == [("window", "activating transformers"), ("window", None)]


def test_a_cuda_export_installs_inside_the_activation_window(monkeypatch, tmp_path):
    order = _run_export(monkeypatch, tmp_path, {"device_backend": "cuda"})
    activation = order.index(("window", "activating transformers"))
    provision = order.index(("provision", True, True))
    assert activation < provision < order.index(("import",))
    assert not any(step[0] == "window" for step in order[activation + 1 : provision])


def _run_export(monkeypatch, tmp_path, extra_config):
    import contextlib

    monkeypatch.setattr(export_worker.sys, "platform", "linux")
    checkpoint = tmp_path / "nemotron-export"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    order: list = []

    @contextlib.contextmanager
    def _window(step = None):
        order.append(("window", step))
        yield

    monkeypatch.setattr(export_worker, "_offline_window_if_unreachable", _window)

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
        config = {"checkpoint_path": str(checkpoint), "allow_ambient": True, **extra_config},
    )
    return order


def test_a_failed_required_install_stops_the_export_before_the_import(monkeypatch, tmp_path):
    checkpoint = tmp_path / "falcon"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "falcon_h1"}))
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    monkeypatch.setattr(ssm_runtime, "_torch_env", lambda: {"cuda_version": "12.8"})
    # A transformers without the native Falcon-H1 model cannot load it without mamba-ssm.
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda model_type: False)
    queue = _Queue()
    assert export_worker._provision_ssm_kernels(str(checkpoint), None, None, queue, "cuda") is False
    assert queue[-1]["type"] == "error"
    assert "mamba-ssm" in queue[-1]["error"]


def test_a_native_required_hybrid_exports_without_mamba_ssm(monkeypatch, tmp_path):
    checkpoint = tmp_path / "falcon"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "falcon_h1"}))
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    monkeypatch.setattr(ssm_runtime, "_torch_env", lambda: {"cuda_version": "12.8"})
    monkeypatch.setattr(ssm_runtime, "native_modeling_available", lambda model_type: True)
    queue = _Queue()
    assert export_worker._provision_ssm_kernels(str(checkpoint), None, None, queue, "cuda") is True
    assert not any(item.get("type") == "error" for item in queue)
    assert any("Warning: could not install mamba-ssm" in item.get("message", "") for item in queue)


def test_an_optional_ssm_family_exports_on_the_torch_path(monkeypatch, tmp_path):
    checkpoint = tmp_path / "jamba"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "jamba"}))
    monkeypatch.setattr(ssm_runtime, "_install_kernel", lambda **k: False)
    queue = _Queue()
    assert export_worker._provision_ssm_kernels(str(checkpoint), None, None, queue, "cuda") is True
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
        str(adapter), str(base) if pinned else None, None, _Queue(), "cuda"
    )
    assert seen == [True]


@pytest.mark.parametrize(
    "platform, device_backend",
    [("darwin", None), ("darwin", "mlx"), ("linux", "mlx"), ("linux", "cpu")],
)
def test_mlx_cpu_and_macos_exports_install_no_cuda_kernels(
    monkeypatch, tmp_path, platform, device_backend
):
    # causal-conv1d / mamba-ssm are CUDA / ROCm extensions: an MLX or CPU export never loads them,
    # and a source build there would only delay (or fail) a valid export.
    checkpoint = tmp_path / "qwen"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setattr(export_worker.sys, "platform", platform)
    monkeypatch.setattr(
        ssm_runtime,
        "provision_ssm_kernels",
        lambda *a, **k: pytest.fail("no SSM kernel provisioning on MLX / CPU / macOS"),
    )
    queue = _Queue()
    assert (
        export_worker._provision_ssm_kernels(str(checkpoint), None, None, queue, device_backend)
        is True
    )
    assert queue == []


@pytest.mark.parametrize("device_backend", ["cuda", "rocm"])
def test_cuda_exports_still_provision(monkeypatch, tmp_path, device_backend):
    checkpoint = tmp_path / "qwen"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setattr(export_worker.sys, "platform", "linux")
    seen = []
    monkeypatch.setattr(
        ssm_runtime, "provision_ssm_kernels", lambda *a, **k: seen.append(a) or (None, [])
    )
    assert export_worker._provision_ssm_kernels(
        str(checkpoint), None, None, _Queue(), device_backend
    )
    assert seen


def test_both_export_provisioning_sites_pass_the_device_backend():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(export_worker.run_export_process))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_provision_ssm_kernels"
    ]
    assert len(calls) == 2
    for call in calls:
        assert ast.unparse(call.args[4]) == "config.get('device_backend')"


@pytest.mark.parametrize("device_backend", ["xpu", None, ""])
def test_xpu_or_unspecified_exports_install_no_cuda_kernels(monkeypatch, tmp_path, device_backend):
    # Only a CUDA / ROCm export provisions: XPU builds none, and an unknown backend is not CUDA.
    monkeypatch.setattr(export_worker.sys, "platform", "linux")
    monkeypatch.setattr(
        ssm_runtime,
        "provision_ssm_kernels",
        lambda *a, **k: pytest.fail("no SSM kernel provisioning off CUDA / ROCm"),
    )
    assert export_worker._provision_ssm_kernels("acme/hybrid", None, None, _Queue(), device_backend)


# ── Security gates run before any SSM kernel is provisioned ──────────────────


class _CmdQueue:
    def __init__(self, cmds):
        self._cmds = list(cmds)

    def get(self, timeout = None):
        if self._cmds:
            return self._cmds.pop(0)
        raise EOFError


class _Verdict:
    def __init__(self, blocked):
        self.blocked = blocked
        self.reason = "flagged pickle"
        self.model_name = "acme/evil"
        self.max_severity = "CRITICAL"

    def response_payload(self):
        return {"blocked": self.blocked}


def _gated_export(
    monkeypatch,
    tmp_path,
    *,
    malware,
    consent,
    cmds = (),
    trust = True,
):
    """Run the export worker with the Hub security verdicts stubbed: *malware* / *consent* say
    whether each gate blocks, *cmds* are fed to the command loop after startup."""
    import contextlib

    monkeypatch.setattr(export_worker.sys, "platform", "linux")
    order: list = []
    monkeypatch.setattr(
        export_worker, "_offline_window_if_unreachable", lambda step = None: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        export_worker,
        "_provision_ssm_kernels",
        lambda *a, **k: order.append(("provision", a[0])) or True,
    )
    monkeypatch.setattr("utils.transformers_version._remote_lora_base", lambda *a, **k: None)
    monkeypatch.setattr(export_worker, "_nemotron_auto_trust", lambda *a, **k: False)

    def _file_security(target, **kwargs):
        order.append(("malware_scan", target, kwargs["load_subdirs"]))
        return _Verdict(target in malware if isinstance(malware, set) else malware)

    def _consent(targets, **kwargs):
        order.append(("consent_scan", tuple(targets)))
        return _Verdict(consent)

    monkeypatch.setattr("utils.security.evaluate_file_security", _file_security)
    monkeypatch.setattr("utils.security.evaluate_remote_code_consent_for_targets", _consent)
    # The preflight must stay transformers-free: no load-subdir computation.
    monkeypatch.setattr(
        "utils.security.security_load_subdirs",
        lambda *a, **k: pytest.fail("preflight computed load subdirs (imports transformers)"),
    )
    monkeypatch.setattr(export_worker, "_setup_log_capture", lambda queue: None)
    monkeypatch.setattr(export_worker, "_activate_transformers_version", lambda *a, **k: None)
    monkeypatch.setattr(export_worker, "apply_gpu_ids", lambda *a, **k: None)
    monkeypatch.setattr(
        "core._torchao_stub.install_torchao_windows_rocm_real_or_stub", lambda: None
    )
    monkeypatch.setattr("core.import_guards.ensure_real_packages", lambda *names: None)

    class _Backend:
        def cleanup_memory(self):
            pass

    fake_export = types.ModuleType("core.export.export")
    fake_export.ExportBackend = _Backend
    monkeypatch.setitem(sys.modules, "core.export.export", fake_export)
    monkeypatch.setattr(
        export_worker,
        "_handle_load",
        lambda b, cmd, q: order.append(("load", cmd["checkpoint_path"])),
    )
    for key in ("TQDM_MININTERVAL", "TOKENIZERS_PARALLELISM", "PYTHONWARNINGS", "PYTHONUNBUFFERED"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setenv("ENVIRONMENT_TYPE", "development")
    monkeypatch.setattr("loggers.config.LogConfig.setup_logging", lambda *a, **k: None)
    monkeypatch.setattr("loggers.config.allow_progress_bars", lambda: None)

    queue = _Queue()
    export_worker.run_export_process(
        cmd_queue = _CmdQueue(cmds),
        resp_queue = queue,
        config = {
            "checkpoint_path": "acme/evil",
            "base_model": "acme/evil-base",
            "trust_remote_code": trust,
            "allow_ambient": True,
            "device_backend": "cuda",
        },
    )
    return order, queue


def test_a_malware_blocked_export_provisions_no_kernels(monkeypatch, tmp_path):
    order, queue = _gated_export(monkeypatch, tmp_path, malware = True, consent = False)
    assert not any(step[0] == "provision" for step in order)
    assert not any(step[0] == "load" for step in order)
    loaded = [r for r in queue if r.get("type") == "loaded"]
    assert loaded and loaded[0]["error_kind"] == "malware_blocked"
    assert loaded[0]["success"] is False
    # Transformers-free preflight: no subdir scoping.
    assert order[0] == ("malware_scan", "acme/evil", ())


def test_a_consent_blocked_trusted_export_provisions_no_kernels(monkeypatch, tmp_path):
    order, queue = _gated_export(monkeypatch, tmp_path, malware = False, consent = True)
    assert ("consent_scan", ("acme/evil", "acme/evil-base")) in order
    assert not any(step[0] == "provision" for step in order)
    loaded = [r for r in queue if r.get("type") == "loaded"]
    assert loaded and loaded[0]["error_kind"] == "remote_code_blocked"
    assert "remote_code" in loaded[0]


def test_an_untrusted_export_skips_the_consent_scan_and_provisions(monkeypatch, tmp_path):
    order, queue = _gated_export(monkeypatch, tmp_path, malware = False, consent = True, trust = False)
    assert not any(step[0] == "consent_scan" for step in order)
    assert ("provision", "acme/evil") in order
    assert order.index(("malware_scan", "acme/evil", ())) < order.index(("provision", "acme/evil"))


def test_a_blocked_load_command_provisions_no_kernels(monkeypatch, tmp_path):
    # Startup is clear; the later "load" command names a flagged checkpoint.
    order, queue = _gated_export(
        monkeypatch,
        tmp_path,
        malware = {"acme/evil2"},
        consent = False,
        cmds = [{"type": "load", "checkpoint_path": "acme/evil2", "trust_remote_code": True}],
    )
    assert ("provision", "acme/evil") in order  # startup was clear
    assert ("load", "acme/evil") in order
    assert ("provision", "acme/evil2") not in order
    assert ("load", "acme/evil2") not in order
    loaded = [r for r in queue if r.get("type") == "loaded"]
    assert [r["error_kind"] for r in loaded] == ["malware_blocked"]


def test_a_clear_load_command_gates_then_provisions_then_loads(monkeypatch, tmp_path):
    order, queue = _gated_export(
        monkeypatch,
        tmp_path,
        malware = False,
        consent = False,
        cmds = [{"type": "load", "checkpoint_path": "acme/ok2"}],
    )
    scan = order.index(("malware_scan", "acme/ok2", ()))
    assert scan < order.index(("provision", "acme/ok2")) < order.index(("load", "acme/ok2"))
