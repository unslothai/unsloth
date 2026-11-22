# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the persistent export command loop through a disconnect and recovery."""

import os
import queue
import sys
import threading
import types
from unittest.mock import patch


def _run_worker(monkeypatch, commands, unreachable):
    """Run the command loop with a stub export that makes a real Hub client call. Returns
    (offline flag, request reached the transport) per export, and the export responses."""
    import httpx
    from huggingface_hub import HfApi, constants
    from huggingface_hub.errors import OfflineModeIsEnabled
    from core.export import worker
    from core import import_guards
    from loggers import config as log_config
    from loggers.config import LogConfig
    from utils import transformers_version
    from utils.utils import force_hf_offline

    calls = []

    def handle_request(self, request):
        return httpx.Response(200, json = {"id": "org/model", "siblings": []}, request = request)

    class Backend:
        def cleanup_memory(self):
            pass

        def export_merged_model(self, **kwargs):
            offline = constants.HF_HUB_OFFLINE
            try:
                HfApi(token = False).model_info("org/model")
                calls.append((offline, True))
            except OfflineModeIsEnabled:
                calls.append((offline, False))
            return True, "saved", "/tmp/export"

    fake_export = types.ModuleType("core.export.export")
    fake_export.ExportBackend = Backend
    fake_loader = types.ModuleType("unsloth.models.loader_utils")
    fake_loader._force_hf_offline = force_hf_offline
    monkeypatch.setitem(sys.modules, "core.export.export", fake_export)
    monkeypatch.setitem(sys.modules, "unsloth.models.loader_utils", fake_loader)
    monkeypatch.setattr(import_guards, "ensure_real_packages", lambda *args: None)
    monkeypatch.setattr(LogConfig, "setup_logging", lambda **kwargs: None)
    monkeypatch.setattr(worker, "_setup_log_capture", lambda _: None)
    monkeypatch.setattr(worker, "_activate_transformers_version", lambda *args: None)
    monkeypatch.setattr(worker, "_handle_load", lambda *args: None)
    # Worker-process globals; keep them from leaking into later tests.
    monkeypatch.setattr(worker, "_log_forward_gate", threading.Event())
    monkeypatch.setattr(log_config, "_BARS_RESTORED", log_config._BARS_RESTORED)
    # Below the client's request hook, which is where huggingface_hub enforces offline mode.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(transformers_version, "hf_endpoint_unreachable", unreachable)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    cmd_queue = queue.Queue()
    for command in [*commands, {"type": "shutdown"}]:
        cmd_queue.put(command)
    resp_queue = queue.Queue()
    # The worker normally owns its process environment; isolate that mutation here.
    with patch.dict(os.environ):
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ["UNSLOTH_OFFLINE_PROBE"] = "1"
        worker.run_export_process(
            cmd_queue = cmd_queue,
            resp_queue = resp_queue,
            config = {"checkpoint_path": "/tmp/ckpt"},
        )
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert constants.HF_HUB_OFFLINE is False
    return calls, [m for m in resp_queue.queue if m.get("type") == "export_merged_done"]


def test_local_export_rechecks_connectivity_and_restores_hub_state(monkeypatch):
    # Activation and load online, first export disconnected, next export recovered.
    verdicts = [False, False, True, False]
    probes = []

    def unreachable():
        probes.append(None)
        return verdicts[len(probes) - 1]

    export = {"type": "export", "export_type": "merged"}
    calls, done = _run_worker(monkeypatch, [export, export], unreachable)
    assert len(probes) == len(verdicts)
    assert calls == [(True, False), (False, True)]
    assert [m["success"] for m in done] == [True, True]


def test_push_export_is_not_pinned_offline(monkeypatch):
    export = {"type": "export", "export_type": "merged", "push_to_hub": True}
    calls, _ = _run_worker(monkeypatch, [export], lambda: True)
    assert calls == [(False, True)]


# ── An auto-set (Hub unreachable) offline does not block kernel installs ──────

import contextlib  # noqa: E402

import pytest  # noqa: E402


def _isolated_offline_env(monkeypatch, **preset):
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "UV_OFFLINE",
        "_UNSLOTH_HF_OFFLINE_AUTO",
    ):
        monkeypatch.delenv(name, raising = False)
    for name, value in preset.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("UNSLOTH_OFFLINE_PROBE", "1")


@pytest.fixture(params = ["env", "force"])
def window_branch(request, monkeypatch):
    """Both ways the window goes offline: env vars only (huggingface_hub not yet usable) or the
    loader's _force_hf_offline (post-import)."""
    from utils.utils import force_hf_offline

    if request.param == "env":
        monkeypatch.setitem(sys.modules, "unsloth.models.loader_utils", None)  # import fails
    else:
        fake_loader = types.ModuleType("unsloth.models.loader_utils")
        fake_loader._force_hf_offline = force_hf_offline
        monkeypatch.setitem(sys.modules, "unsloth.models.loader_utils", fake_loader)
    return request.param


def _install_kernel_reaches_the_lock(monkeypatch):
    """Whether ssm_runtime._install_kernel gets past its offline guard (to the kernel lock, which
    is stubbed busy so nothing installs)."""
    from utils import kernel_install, ssm_runtime

    reached = []

    @contextlib.contextmanager
    def _busy(name, **kwargs):
        reached.append(name)
        yield False

    monkeypatch.setattr(kernel_install, "kernel_install_lock", _busy)
    monkeypatch.setattr(ssm_runtime, "_is_importable", lambda name: False)
    monkeypatch.setattr(ssm_runtime, "_FAILED_INSTALLS", set())
    monkeypatch.setattr(
        ssm_runtime, "install_wheel", lambda *a, **k: pytest.fail("no install in this test")
    )
    ssm_runtime._install_kernel(
        import_name = "causal_conv1d",
        display_name = "causal-conv1d",
        pypi_name = "causal-conv1d",
        package_version = "1.6.1",
        release_tag = "v1.6.1.post4",
        release_base_url = "x",
        status_cb = None,
        run = lambda *a, **k: pytest.fail("no subprocess in this test"),
    )
    return bool(reached)


def test_an_unreachable_hub_still_allows_kernel_installs_inside_the_window(
    monkeypatch, window_branch
):
    from core.export import worker
    from utils import transformers_version
    from utils.utils import hf_env_offline, package_installs_offline
    from utils.wheel_utils import package_installs_offline_env

    _isolated_offline_env(monkeypatch)
    monkeypatch.setattr(transformers_version, "hf_endpoint_unreachable", lambda: True)
    with worker._offline_window_if_unreachable(step = "activating transformers"):
        assert hf_env_offline()  # the Hub side is offline
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
        assert package_installs_offline() is False
        assert package_installs_offline_env() is False
        assert _install_kernel_reaches_the_lock(monkeypatch)
    # Marker and env restored on exit.
    assert "_UNSLOTH_HF_OFFLINE_AUTO" not in os.environ
    assert "HF_HUB_OFFLINE" not in os.environ


def test_uv_offline_still_blocks_installs_inside_the_window(monkeypatch, window_branch):
    from core.export import worker
    from utils import transformers_version
    from utils.utils import package_installs_offline
    from utils.wheel_utils import package_installs_offline_env

    _isolated_offline_env(monkeypatch, UV_OFFLINE = "1")
    monkeypatch.setattr(transformers_version, "hf_endpoint_unreachable", lambda: True)
    with worker._offline_window_if_unreachable():
        assert package_installs_offline() is True
        assert package_installs_offline_env() is True
        assert not _install_kernel_reaches_the_lock(monkeypatch)


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_an_operator_set_hf_offline_still_blocks_installs(monkeypatch, variable):
    from core.export import worker
    from utils import transformers_version
    from utils.utils import package_installs_offline
    from utils.wheel_utils import package_installs_offline_env

    _isolated_offline_env(monkeypatch, **{variable: "1"})
    monkeypatch.setattr(transformers_version, "hf_endpoint_unreachable", lambda: True)
    with worker._offline_window_if_unreachable():
        assert "_UNSLOTH_HF_OFFLINE_AUTO" not in os.environ
        assert package_installs_offline() is True
        assert package_installs_offline_env() is True
        assert not _install_kernel_reaches_the_lock(monkeypatch)
    assert package_installs_offline() is True


def test_a_reachable_hub_sets_no_marker(monkeypatch):
    from core.export import worker
    from utils import transformers_version

    _isolated_offline_env(monkeypatch)
    monkeypatch.setattr(transformers_version, "hf_endpoint_unreachable", lambda: False)
    with worker._offline_window_if_unreachable():
        assert "_UNSLOTH_HF_OFFLINE_AUTO" not in os.environ
        assert "HF_HUB_OFFLINE" not in os.environ
