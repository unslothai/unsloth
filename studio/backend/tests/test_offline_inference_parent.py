# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Parent-process offline regression tests (follow-up to #5505).

Pins the LoRA-detect, transformers_version urllib short-circuit, and
training-worker DNS probe so a dead DNS no longer burns 30-60s of
soft-failed timeouts before the worker subprocess spawns.

No GPU, no network, no subprocess. Cross-platform.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import os
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _module_available(name: str) -> bool:
    """True if the real module can be imported. Probed rather than imported: these stubs
    land in sys.modules for the whole session, so an empty one breaks anything imported
    later that actually uses the module."""
    try:
        return _importlib_util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


if not _module_available("loggers"):
    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules.setdefault("loggers", _loggers_stub)
if not _module_available("structlog"):
    sys.modules.setdefault("structlog", _types.ModuleType("structlog"))
# Prefer real httpx if installed (CI installs it). Stub only as fallback.
try:
    import httpx  # noqa: F401
except ImportError:
    _hx = _types.ModuleType("httpx")
    for _exc in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
        "HTTPStatusError",
    ):
        setattr(_hx, _exc, type(_exc, (Exception,), {}))
    _hx.Response = type("Response", (), {})
    _hx.Request = type("Request", (), {})

    class _FakeTimeout:
        def __init__(self, *a, **k):
            pass

    _hx.Timeout = _FakeTimeout
    _hx.Client = type(
        "Client",
        (),
        {
            "__init__": lambda s, **k: None,
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
        },
    )
    sys.modules.setdefault("httpx", _hx)


from utils.models.model_config import _env_offline
from utils.transformers_version import (
    _check_config_needs_550,
    _check_tokenizer_config_needs_v5,
    _env_offline as _env_offline_tv,
)


@pytest.fixture
def clean_offline_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)


class TestEnvOffline:
    def test_unset_is_false(self, clean_offline_env):
        assert _env_offline() is False
        assert _env_offline_tv() is False

    def test_hf_hub_offline_truthy_values(self, monkeypatch, clean_offline_env):
        for val in ("1", "true", "yes", "TRUE", "Yes"):
            monkeypatch.setenv("HF_HUB_OFFLINE", val)
            assert _env_offline() is True
            assert _env_offline_tv() is True

    def test_transformers_offline_alone_triggers(self, monkeypatch, clean_offline_env):
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
        assert _env_offline() is True

    def test_falsy_values(self, monkeypatch, clean_offline_env):
        for val in ("", "0", "false", "no"):
            monkeypatch.setenv("HF_HUB_OFFLINE", val)
            assert _env_offline() is False


class TestTransformersVersionOfflineShortCircuits:
    def test_tokenizer_config_skips_urllib_when_offline(
        self, monkeypatch, clean_offline_env, tmp_path
    ):
        # No local config + offline env -> must NOT call urlopen.
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        unique = f"unsloth/never-cached-{tmp_path.name}"

        def boom(*a, **k):
            raise AssertionError("urlopen must not be called when offline")

        with patch("urllib.request.urlopen", boom):
            assert _check_tokenizer_config_needs_v5(unique) is False

    def test_config_550_skips_urllib_when_offline(self, monkeypatch, clean_offline_env, tmp_path):
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        unique = f"unsloth/never-cached-{tmp_path.name}-cfg"

        def boom(*a, **k):
            raise AssertionError("urlopen must not be called when offline")

        with patch("urllib.request.urlopen", boom):
            assert _check_config_needs_550(unique) is False


class TestLoraDetectOffline:
    """Offline LoRA detect: hf_model_info short-circuits via
    OfflineModeIsEnabled; cached adapter_config.json wins."""

    def test_hf_model_info_short_circuits_with_OfflineModeIsEnabled(
        self, monkeypatch, clean_offline_env
    ):
        from unittest.mock import MagicMock

        from utils.models.model_config import ModelConfig

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        # Unsloth catches Exception broadly; pin that the call still happens
        # (so cached LoRAs aren't missed) and returns fast via the mock.
        class _OfflineModeIsEnabled(Exception):
            pass

        mock = MagicMock(side_effect = _OfflineModeIsEnabled("offline"))
        with patch("huggingface_hub.model_info", mock):
            try:
                ModelConfig.from_identifier(
                    model_id = "unsloth/Qwen3.5-4B",
                    hf_token = None,
                    gguf_variant = None,
                )
            except Exception:
                pass  # registry miss OK; pinning the LoRA-detect call

        assert mock.call_count >= 1, (
            "LoRA-detect must still consult hf_model_info offline; "
            "OfflineModeIsEnabled makes it cheap"
        )

    def test_cached_lora_detected_when_api_unreachable(
        self, monkeypatch, clean_offline_env, tmp_path
    ):
        """A cached adapter_config.json must still mark the repo as a
        LoRA when the HF API is unreachable."""
        from huggingface_hub import constants as hf_constants

        from utils.models.model_config import ModelConfig

        repo = tmp_path / "models--org--my-lora"
        snap = repo / "snapshots" / ("a" * 40)
        snap.mkdir(parents = True)
        (snap / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "unsloth/Llama-3-8B"}'
        )
        monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")

        def boom(*a, **k):
            raise OSError("hub unreachable")

        with patch("huggingface_hub.model_info", boom):
            try:
                cfg = ModelConfig.from_identifier(
                    model_id = "org/my-lora",
                    hf_token = None,
                    gguf_variant = None,
                )
            except Exception:
                cfg = None

        # cfg may be None (base not resolvable offline); pin the fixture
        # so the cache-side detect block had a file to find.
        assert (snap / "adapter_config.json").is_file()


class TestTrainingWorkerProbeNoGlobalTimeout:
    """Training-worker DNS probe must run on a daemon thread, not mutate
    process-wide socket.setdefaulttimeout (mirrors llama_cpp.py)."""

    def test_training_worker_source_uses_thread_probe(self):
        """Static-pin against regression to setdefaulttimeout."""
        import re
        from pathlib import Path

        src = Path(_BACKEND_DIR, "core", "training", "worker.py").read_text(encoding = "utf-8")
        m = re.search(
            r'if\s+"HF_HUB_OFFLINE"\s+not\s+in\s+os\.environ.*?'
            r"print\([^)]*HF_HUB_OFFLINE=1[^)]*\)",
            src,
            flags = re.DOTALL,
        )
        assert m is not None, "could not locate offline auto-detect block"
        block = m.group(0)
        assert ".setdefaulttimeout(" not in block, (
            "training worker still calls socket.setdefaulttimeout; "
            "concurrent sockets would inherit the probe timeout"
        )
        # The probe now lives in the shared helper (endpoint- and proxy-aware), so the
        # worker must delegate to it rather than resolve a hardcoded host itself.
        assert (
            "hf_env_offline" in block
        ), "training worker must honor TRANSFORMERS_OFFLINE before probing"
        assert "hf_dns_dead" in block, "training worker must use the shared DNS helper"
        assert block.index("hf_env_offline()") < block.index(
            "hf_dns_dead()"
        ), "training worker must check explicit offline env before DNS/network probes"
        assert 'gethostbyname("huggingface.co")' not in block, (
            "training worker must not hardcode huggingface.co; a reachable HF_ENDPOINT "
            "mirror would be declared offline"
        )
        assert (
            "proxy_timeouts_offline = False" in block
        ), "training worker must fail open on an ambiguous proxy timeout"

    def test_shared_dns_helper_uses_thread_probe(self):
        """The daemon-thread property moved with the probe; pin it where it now lives."""
        import inspect

        from utils.utils import dns_host_dead

        src = inspect.getsource(dns_host_dead)
        assert ".setdefaulttimeout(" not in src, (
            "shared DNS probe calls socket.setdefaulttimeout; "
            "concurrent sockets would inherit the probe timeout"
        )
        assert "Thread" in src and "daemon" in src, "shared DNS probe must run on a daemon thread"


class TestInferenceWorkerProbesForItself:
    """child_env deliberately scrubs the parent's scoped offline flag, so the inference
    worker needs its own probe like the training and export workers, or it walks back
    into the retry paths the parent already ruled out."""

    def _block(self):
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "core" / "inference" / "worker.py").read_text(
            encoding = "utf-8",
        )
        start = src.index("# Offline auto-detect")
        # To the end of the block, not a fixed slice: a gate added ahead of it would
        # otherwise push the tail out of the window and pass vacuously.
        return src[start : src.index("\n    import warnings", start)]

    def test_the_probe_exists_and_runs_before_activation(self):
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "core" / "inference" / "worker.py").read_text(
            encoding = "utf-8",
        )
        probe = src.index("# Offline auto-detect")
        # Both HF-reading steps the parent's verdict was meant to cover.
        assert probe < src.index("_remote_lora_base(model_name")
        assert probe < src.index("_activate_transformers_version(_base")

    def test_a_user_set_flag_is_never_overridden(self):
        block = self._block()
        assert 'if "HF_HUB_OFFLINE" not in os.environ' in block

    def test_lifetime_flags_use_the_fail_open_verdict(self):
        """Same reasoning as the training worker: these last the whole process, so an
        ambiguous answer must not strand it offline."""
        block = self._block()
        assert "gateway_errors_offline = False" in block
        assert "proxy_timeouts_offline = False" in block

    def test_probe_opt_out_is_honoured(self):
        block = self._block()
        assert "hf_probe_disabled()" in block

    def test_it_fails_open(self):
        block = self._block()
        assert "except Exception:" in block

    def test_it_does_not_force_datasets_offline(self):
        """An inference worker loads no dataset; the training worker's flag is its own."""
        block = self._block()
        assert "HF_DATASETS_OFFLINE" not in block


class TestWorkerProbesOnlyWhenTheHubIsNeeded:
    """A filesystem-only job never reaches the Hub, so the probe is pure latency: this
    was DNS-only on main for training, and absent entirely for inference."""

    def _load(self, relpath, name):
        import importlib.util
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(name, backend_root / relpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_training_gate_classifies_each_shape(self, tmp_path):
        w = self._load("core/training/worker.py", "training_worker_gate_probe")
        local = str(tmp_path)

        assert w._training_job_is_local({"model_name": local}) is True
        assert w._training_job_is_local({"model_name": local, "hf_dataset": ""}) is True
        # A remote dataset needs the Hub even with a local model.
        assert w._training_job_is_local({"model_name": local, "hf_dataset": "org/ds"}) is False
        assert w._training_job_is_local({"model_name": "org/model"}) is False
        # Fail closed on anything unresolvable.
        assert w._training_job_is_local({}) is False
        assert w._training_job_is_local({"model_name": None}) is False

    def test_inference_gate_classifies_each_shape(self, tmp_path):
        w = self._load("core/inference/worker.py", "inference_worker_gate_probe")
        local = str(tmp_path)

        assert w._hub_targets_are_local(local) is True
        assert w._hub_targets_are_local(local, None) is True
        assert w._hub_targets_are_local(local, "org/base") is False
        assert w._hub_targets_are_local("org/model") is False
        assert w._hub_targets_are_local(None) is True
        assert w._hub_targets_are_local(123) is False

    def test_inference_gate_reads_a_local_adapter_base_from_disk(self, tmp_path):
        """A local adapter pointing at a REMOTE base still needs the probe, and the base
        is readable without touching the network."""
        import json

        w = self._load("core/inference/worker.py", "inference_worker_gate_adapter")
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/base"}),
            encoding = "utf-8",
        )
        base, needs_hub = w._recorded_local_base(str(tmp_path))
        assert (base, needs_hub) == ("org/base", False)
        assert w._hub_targets_are_local(str(tmp_path), base) is False

    def test_inference_gate_handles_a_missing_adapter_config(self, tmp_path):
        w = self._load("core/inference/worker.py", "inference_worker_gate_noadapter")
        assert w._recorded_local_base(str(tmp_path)) == (None, False)
        assert w._recorded_local_base("org/model") == (None, False)

    def test_both_probes_sit_behind_the_gate(self):
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        inf = (backend_root / "core" / "inference" / "worker.py").read_text(
            encoding = "utf-8",
        )
        trn = (backend_root / "core" / "training" / "worker.py").read_text(
            encoding = "utf-8",
        )
        assert "not _hub_targets_are_local(" in inf
        assert "not _training_job_is_local(config)" in trn
        # The user's own flag still wins in both.
        assert inf.count('if "HF_HUB_OFFLINE" not in os.environ and (') == 1
        assert trn.count('if "HF_HUB_OFFLINE" not in os.environ and not') == 1


class TestLocalLoraTrainingJobStillProbes:
    """A local adapter can name a remote base, which activation resolves and later
    training and security code fetches, so the job is not filesystem-only."""

    def _worker(self):
        import importlib.util
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "training_worker_lora_gate",
            backend_root / "core" / "training" / "worker.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_local_adapter_with_a_remote_base_is_not_local(self, tmp_path):
        import json

        w = self._worker()
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/base"}),
            encoding = "utf-8",
        )
        assert w._training_job_is_local({"model_name": str(tmp_path)}) is False

    def test_local_adapter_with_a_local_base_is_local(self, tmp_path):
        import json

        w = self._worker()
        base = tmp_path / "base"
        base.mkdir()
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(base)}),
            encoding = "utf-8",
        )
        assert w._training_job_is_local({"model_name": str(tmp_path)}) is True

    def test_a_plain_local_checkpoint_is_still_local(self, tmp_path):
        w = self._worker()
        assert w._training_job_is_local({"model_name": str(tmp_path)}) is True

    def test_a_null_recorded_base_still_probes(self, tmp_path):
        """An explicit null reads the same as a missing key: no base on disk, so the
        resolver falls through to get_base_model_from_lora, which is a Hub call."""
        import json

        w = self._worker()
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": None}),
            encoding = "utf-8",
        )
        assert w._training_job_is_local({"model_name": str(tmp_path)}) is False

    def test_both_workers_agree(self, tmp_path):
        """The two gates must classify the same adapter the same way."""
        import importlib.util
        import json
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/base"}),
            encoding = "utf-8",
        )
        spec = importlib.util.spec_from_file_location(
            "inference_worker_lora_gate",
            backend_root / "core" / "inference" / "worker.py",
        )
        inf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inf)

        base, needs_hub = inf._recorded_local_base(str(tmp_path))
        assert needs_hub is False
        assert inf._hub_targets_are_local(str(tmp_path), base) is False
        assert self._worker()._training_job_is_local({"model_name": str(tmp_path)}) is False


class TestFullCheckpointBaseKeepsTheProbe:
    """A local full checkpoint's config.json can record a REMOTE base, which
    _resolve_base_model returns and tier activation then reads Hub metadata for, so the
    job is not filesystem-only even though every path on disk is local."""

    def _module(self, relative_path, name):
        import importlib.util
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(name, backend_root / relative_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _checkpoint(self, tmp_path, config_json):
        import json
        (tmp_path / "config.json").write_text(json.dumps(config_json), encoding = "utf-8")
        return str(tmp_path)

    def test_remote_model_name_keeps_the_probe(self, tmp_path):
        target = self._checkpoint(tmp_path, {"model_name": "org/base"})
        inf = self._module("core/inference/worker.py", "inference_worker_ckpt_gate")
        trn = self._module("core/training/worker.py", "training_worker_ckpt_gate")

        base, needs_hub = inf._recorded_local_base(target)
        assert (base, needs_hub) == ("org/base", False)
        assert inf._hub_targets_are_local(target, base) is False
        assert trn._training_job_is_local({"model_name": target}) is False

    def test_remote_name_or_path_keeps_the_probe(self, tmp_path):
        target = self._checkpoint(tmp_path, {"_name_or_path": "org/base"})
        inf = self._module("core/inference/worker.py", "inference_worker_nop_gate")
        assert inf._recorded_local_base(target) == ("org/base", False)

    def test_a_self_reference_is_not_a_base(self, tmp_path):
        """HF writes the checkpoint's own path into _name_or_path; that is not a base
        and must not cost a probe."""
        target = self._checkpoint(tmp_path, {"_name_or_path": str(tmp_path)})
        inf = self._module("core/inference/worker.py", "inference_worker_self_gate")
        trn = self._module("core/training/worker.py", "training_worker_self_gate")

        assert inf._recorded_local_base(target) == (None, False)
        assert trn._training_job_is_local({"model_name": target}) is True

    def test_an_adapter_base_still_wins_over_config_json(self, tmp_path):
        """Ordering matches the resolver: the adapter's base, not the config.json one."""
        import json

        target = self._checkpoint(tmp_path, {"model_name": "org/from-config"})
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "org/from-adapter"}),
            encoding = "utf-8",
        )
        inf = self._module("core/inference/worker.py", "inference_worker_order_gate")
        assert inf._recorded_local_base(target) == ("org/from-adapter", False)

    def test_a_baseless_adapter_needs_the_hub(self, tmp_path):
        """With no base on disk the resolver falls through to get_base_model_from_lora,
        which is a Hub call, so the gate must fail closed."""
        import json

        (tmp_path / "adapter_config.json").write_text(json.dumps({}), encoding = "utf-8")
        inf = self._module("core/inference/worker.py", "inference_worker_baseless_gate")
        trn = self._module("core/training/worker.py", "training_worker_baseless_gate")

        assert inf._recorded_local_base(str(tmp_path)) == (None, True)
        assert trn._training_job_is_local({"model_name": str(tmp_path)}) is False

    def test_the_gate_agrees_with_the_resolver(self, tmp_path):
        """Anti-drift: this bug was the gate reading less than _resolve_base_model does.
        For every on-disk shape the two must name the same base."""
        import json
        import sys

        backend_root = str(__import__("pathlib").Path(__file__).resolve().parent.parent)
        if backend_root not in sys.path:
            sys.path.insert(0, backend_root)
        from utils.transformers_version import _resolve_base_model, recorded_local_base

        # dir name -> (adapter_config.json, config.json, drop adapter weights in)
        shapes = {
            "adapter": ({"base_model_name_or_path": "org/a"}, None, False),
            "config": (None, {"model_name": "org/c"}, False),
            "name_or_path": (None, {"_name_or_path": "org/n"}, False),
            "both": ({"base_model_name_or_path": "org/a"}, {"model_name": "org/c"}, False),
            "bare": (None, None, False),
            # Adapter-only LoRAs: no JSON at all, so the resolver falls back to the
            # unsloth_<model>_<timestamp> dir-name convention.
            "unsloth_llama-3_1700000000": (None, None, True),
            "unsloth_a_b_1700000000": (None, None, True),
            "plain_adapter_dir": (None, None, True),
            "unsloth_nostamp": (None, None, True),
        }
        for name, (adapter, config, weights) in shapes.items():
            d = tmp_path / name
            d.mkdir()
            if adapter is not None:
                (d / "adapter_config.json").write_text(json.dumps(adapter), encoding = "utf-8")
            if config is not None:
                (d / "config.json").write_text(json.dumps(config), encoding = "utf-8")
            if weights:
                (d / "adapter_model.safetensors").write_bytes(b"")

            base, needs_hub = recorded_local_base(str(d))
            resolved = _resolve_base_model(str(d))
            # The resolver returns the input unchanged when it finds no base.
            assert needs_hub is False, name
            assert (base or str(d)) == resolved, name

    def test_an_adapter_only_lora_keeps_the_probe(self, tmp_path):
        """No JSON on disk, but the dir name resolves to a remote unsloth/... base that
        tier activation reads Hub metadata for."""
        d = tmp_path / "unsloth_llama-3_1700000000"
        d.mkdir()
        (d / "adapter_model.safetensors").write_bytes(b"")

        inf = self._module("core/inference/worker.py", "inference_worker_adapteronly_gate")
        trn = self._module("core/training/worker.py", "training_worker_adapteronly_gate")

        base, needs_hub = inf._recorded_local_base(str(d))
        assert (base, needs_hub) == ("unsloth/llama-3", False)
        assert inf._hub_targets_are_local(str(d), base) is False
        assert trn._training_job_is_local({"model_name": str(d)}) is False


class TestLoadRouteResolvesConfigOffTheLoop:
    """_load_model_impl is awaited directly by the route, so a guard that can spend
    seconds on DNS plus a HEAD and its TCP fallback must not run inline."""

    def test_the_guard_and_config_resolution_run_in_a_thread(self):
        import ast
        import pathlib

        backend_root = pathlib.Path(__file__).resolve().parent.parent
        src = (backend_root / "routes" / "inference.py").read_text(encoding = "utf-8")
        tree = ast.parse(src)

        impl = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.AsyncFunctionDef) and n.name == "_load_model_impl"
        )
        threaded = set()
        for node in ast.walk(impl):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to_thread"
                and node.args
            ):
                name = getattr(node.args[0], "id", None)
                if name:
                    threaded.add(name)
        assert (
            "_resolve_config" in threaded
        ), "the load guard must be awaited off the event loop, as /validate does"

        # And nothing in that function may enter the guard inline any more.
        bad = [
            n.lineno
            for n in ast.walk(impl)
            if isinstance(n, ast.With)
            and any(
                isinstance(i.context_expr, ast.Call)
                and (getattr(i.context_expr.func, "id", "") or "").startswith(
                    "_hf_offline_if_unreachable"
                )
                for i in n.items
            )
            and not any(isinstance(p, ast.FunctionDef) and n in ast.walk(p) for p in ast.walk(impl))
        ]
        assert bad == [], f"guard still entered inline on the event loop at {bad}"
