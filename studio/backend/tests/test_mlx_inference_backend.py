# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import contextlib
import copy
import json
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


class _DummyMetal:
    @staticmethod
    def is_available():
        return False


class _DummyMX:
    metal = _DummyMetal()

    @staticmethod
    def set_wired_limit(_limit):
        return None

    @staticmethod
    def device_info():
        return {"max_recommended_working_set_size": 1024}


class _DummyTokenizer:
    pass


class _DummyProcessor:
    tokenizer = _DummyTokenizer()


class _DummyModel:
    pass


def _install_fake_mlx(monkeypatch):
    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_utils = types.ModuleType("mlx.utils")
    mlx_core.metal = _DummyMetal()
    mlx_core.set_wired_limit = _DummyMX.set_wired_limit
    mlx_core.device_info = _DummyMX.device_info
    mlx_utils.tree_unflatten = dict
    mlx_pkg.core = mlx_core
    mlx_pkg.utils = mlx_utils
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx.utils", mlx_utils)


def _install_fake_fast_mlx(monkeypatch, calls):
    class _FastMLXModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            if kwargs["text_only"] is False:
                return _DummyModel(), _DummyProcessor()
            return _DummyModel(), _DummyTokenizer()

    unsloth_zoo_pkg = types.ModuleType("unsloth_zoo")
    mlx_pkg = types.ModuleType("unsloth_zoo.mlx")
    mlx_loader = types.ModuleType("unsloth_zoo.mlx.loader")
    mlx_loader.FastMLXModel = _FastMLXModel
    unsloth_zoo_pkg.mlx = mlx_pkg
    mlx_pkg.loader = mlx_loader
    monkeypatch.setitem(sys.modules, "unsloth_zoo", unsloth_zoo_pkg)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", mlx_loader)


class _AdapterTree:
    def __init__(self, modules):
        self.modules = dict(modules)

    def named_modules(self):
        return list(self.modules.items())

    def update_modules(self, modules):
        self.modules.update(modules)


def test_temporary_mlx_adapter_state_bypasses_and_restores_wrappers(monkeypatch):
    _install_fake_mlx(monkeypatch)
    from core.inference.mlx_inference import _temporary_mlx_adapter_state

    base = object()
    wrapper = SimpleNamespace(lora_a = object(), lora_b = object(), linear = base, m = object())
    model = _AdapterTree({"model.layers.0.proj": wrapper})

    with pytest.raises(RuntimeError, match = "generation failed"):
        with _temporary_mlx_adapter_state(model, False):
            assert model.modules["model.layers.0.proj"] is base
            raise RuntimeError("generation failed")
    assert model.modules["model.layers.0.proj"] is wrapper


def test_temporary_mlx_adapter_state_validates_requests():
    from core.inference.mlx_inference import _temporary_mlx_adapter_state

    wrapper = SimpleNamespace(lora_a = object(), lora_b = object(), embedding = object())
    model = _AdapterTree({"embed_tokens": wrapper})
    with _temporary_mlx_adapter_state(model, True):
        assert model.modules["embed_tokens"] is wrapper
    with pytest.raises(NotImplementedError, match = "named adapter"):
        with _temporary_mlx_adapter_state(model, "other"):
            pass

    base_model = _AdapterTree({"proj": object()})
    with _temporary_mlx_adapter_state(base_model, None):
        pass
    with _temporary_mlx_adapter_state(base_model, True):
        pass

    unsupported = _AdapterTree({"proj": SimpleNamespace(lora_a = object(), lora_b = object())})
    with _temporary_mlx_adapter_state(unsupported, True):
        pass
    with pytest.raises(RuntimeError, match = "without their base modules"):
        with _temporary_mlx_adapter_state(unsupported, False):
            pass


def test_temporary_mlx_adapter_state_uses_real_mlx_module_tree():
    nn = pytest.importorskip("mlx.nn")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_lm.tuner.dora import DoRALinear
    from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

    from core.inference.mlx_inference import _temporary_mlx_adapter_state

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            quantized = nn.QuantizedLinear.from_linear(nn.Linear(32, 32), group_size = 32, bits = 4)
            self.quantized_proj = LoRALinear.from_base(quantized)
            self.dora_proj = DoRALinear.from_base(nn.Linear(4, 4))

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_Layer()]
            self.embed_tokens = LoRAEmbedding.from_base(nn.Embedding(16, 4))
            self.experts = LoRASwitchLinear.from_base(SwitchLinear(4, 4, 2))

    model = _Model()
    wrappers = {
        path: module
        for path, module in model.named_modules()
        if hasattr(module, "lora_a") and hasattr(module, "lora_b")
    }
    bases = {
        path: getattr(module, "linear", getattr(module, "embedding", None))
        for path, module in wrappers.items()
    }

    with _temporary_mlx_adapter_state(model, False):
        live = dict(model.named_modules())
        assert all(live[path] is base for path, base in bases.items())

    restored = dict(model.named_modules())
    assert all(restored[path] is wrapper for path, wrapper in wrappers.items())


def test_mlx_inference_text_load_forwards_studio_settings(monkeypatch):
    _install_fake_mlx(monkeypatch)
    calls = []
    _install_fake_fast_mlx(monkeypatch, calls)

    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    config = SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = False)

    assert backend.load_model(
        config,
        max_seq_length = 4096,
        load_in_4bit = False,
        hf_token = "hf-token",
        trust_remote_code = True,
        dtype = "float16",
    )

    assert calls == [
        (
            ("fake/text",),
            {
                "max_seq_length": 4096,
                "dtype": "float16",
                "load_in_4bit": False,
                "token": "hf-token",
                "trust_remote_code": True,
                "text_only": True,
            },
        )
    ]
    assert backend._is_vlm is False
    assert isinstance(backend._tokenizer, _DummyTokenizer)
    # Non-LoRA text model: no base_model on the record.
    assert backend.models["fake/text"]["base_model"] is None


def test_mlx_text_lora_record_keeps_base_model_for_native_template(monkeypatch):
    # A LoRA adapter's own tokenizer often ships no chat template; the native tool-calling template
    # lives on the base model.
    _install_fake_mlx(monkeypatch)
    calls = []
    _install_fake_fast_mlx(monkeypatch, calls)

    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    config = SimpleNamespace(
        identifier = "fake/text-adapter",
        is_vision = False,
        is_lora = True,
        base_model = "fake/text-base",
    )

    assert backend.load_model(config, max_seq_length = 4096, hf_token = "hf-token")

    record = backend.models["fake/text-adapter"]
    assert record["is_lora"] is True
    assert record["base_model"] == "fake/text-base"


def test_mlx_inference_vlm_lora_uses_unsloth_loader_without_native_adapter_rewrite(
    monkeypatch, tmp_path
):
    _install_fake_mlx(monkeypatch)
    calls = []
    _install_fake_fast_mlx(monkeypatch, calls)

    def _native_vlm_load(*_args, **_kwargs):
        raise AssertionError("Unsloth MLX VLM inference must use FastMLXModel")

    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.load = _native_vlm_load
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    cfg_path = adapter_dir / "adapter_config.json"
    original_cfg = '{"base_model_name_or_path": "fake/base", "rank": 8}\n'
    cfg_path.write_text(original_cfg)

    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    config = SimpleNamespace(
        identifier = str(adapter_dir),
        is_vision = True,
        is_lora = True,
        base_model = "fake/base",
    )

    assert backend.load_model(
        config,
        max_seq_length = 8192,
        load_in_4bit = True,
        hf_token = "hf-token",
        trust_remote_code = True,
    )

    assert calls == [
        (
            (str(adapter_dir),),
            {
                "max_seq_length": 8192,
                "dtype": None,
                "load_in_4bit": True,
                "token": "hf-token",
                "trust_remote_code": True,
                "text_only": False,
            },
        )
    ]
    assert cfg_path.read_text() == original_cfg
    assert backend._is_vlm is True
    assert isinstance(backend._processor, _DummyProcessor)
    assert isinstance(backend._tokenizer, _DummyTokenizer)


def test_mlx_inference_distributed_vlm_forwards_group_to_fast_mlx(monkeypatch):
    _install_fake_mlx(monkeypatch)
    calls = []
    _install_fake_fast_mlx(monkeypatch, calls)
    from core.inference.mlx_inference import MLXInferenceBackend

    group = SimpleNamespace(size = lambda: 2, rank = lambda: 0)
    config = SimpleNamespace(identifier = "fake/vlm", is_vision = True, is_lora = False)
    for mode, group_key in (("tensor", "tensor_group"), ("pipeline", "pipeline_group")):
        calls.clear()
        assert MLXInferenceBackend().load_model(config, parallel_mode = mode, distributed_group = group)
        _, kwargs = calls.pop()
        assert kwargs["text_only"] is False and kwargs[group_key] is group

    calls.clear()
    singleton = SimpleNamespace(size = lambda: 1, rank = lambda: 0)
    assert MLXInferenceBackend().load_model(
        config, parallel_mode = "tensor", distributed_group = singleton
    )
    assert not {"tensor_group", "pipeline_group"} & set(calls.pop()[1])

    config = SimpleNamespace(identifier = "fake/adapter", is_vision = False, is_lora = True)
    with pytest.raises(ValueError, match = "LoRA adapter repos"):
        MLXInferenceBackend().load_model(config, parallel_mode = "tensor", distributed_group = group)


@pytest.mark.parametrize("accepts_backend", (True, False))
def test_mlx_distributed_init_selects_jaccl_backend(monkeypatch, accepts_backend):
    _install_fake_mlx(monkeypatch)
    from core.inference.mlx_inference import _init_mlx_distributed

    group = SimpleNamespace(rank = lambda: 1, size = lambda: 2)
    calls = []

    def _init(**kwargs):
        calls.append(kwargs)
        if kwargs and not accepts_backend:
            raise TypeError("backend keyword unsupported")
        return group

    sys.modules["mlx.core"].distributed = SimpleNamespace(init = _init)
    monkeypatch.setenv("MLX_JACCL_COORDINATOR", "127.0.0.1:12345")
    monkeypatch.setenv("MLX_IBV_DEVICES", "/tmp/devices.json")

    assert _init_mlx_distributed() == (group, 1, 2)
    assert calls == ([{"backend": "jaccl"}] if accepts_backend else [{"backend": "jaccl"}, {}])


def test_worker_share_object_receives_distributed_payload(monkeypatch):
    from core.inference import worker

    shared_obj = {"type": "turn", "text": "hi"}
    payload = worker._encode_share_object(shared_obj)

    def _array(value):
        val = value.item() if hasattr(value, "item") else value
        return SimpleNamespace(
            item = lambda: val,
            tolist = lambda: list(val) if hasattr(val, "__iter__") else [val],
        )

    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.uint8 = "uint8"
    mlx_core.array = _array
    mlx_core.zeros = lambda *_a, **_k: _array([])

    def _all_sum(value, group = None):
        value = value.item() if hasattr(value, "item") else value
        return _array(len(payload)) if value == 0 else _array(payload)

    mlx_core.distributed = SimpleNamespace(all_sum = _all_sum)
    mlx_pkg.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)

    responses = []
    worker._handle_share_object(
        SimpleNamespace(
            _distributed_group = object(),
            _distributed_rank = 1,
            _distributed_world_size = 2,
        ),
        {"type": "share_object", "request_id": "rid", "object": None},
        SimpleNamespace(put = responses.append),
    )

    response = responses[0]
    assert response["object"] == shared_obj


def test_worker_activates_mlx_sidecar_before_hardware_detection(tmp_path):
    backend_dir = Path(__file__).resolve().parent.parent
    fake_modules = tmp_path / "base"
    sidecar = tmp_path / ".venv_t5_530"
    packages = {
        fake_modules / "transformers" / "__init__.py": '__version__ = "4.57.6"\n',
        fake_modules / "mlx" / "__init__.py": "",
        fake_modules / "mlx" / "core.py": "",
        fake_modules / "mlx_lm" / "__init__.py": "import transformers\n",
        fake_modules / "mlx_lm" / "sample_utils.py": "",
        fake_modules / "mlx_vlm" / "__init__.py": "",
        sidecar / "transformers" / "__init__.py": '__version__ = "5.3.0"\n',
    }
    for path, contents in packages.items():
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(contents)

    script = r"""
import json
import os
import sys

sys.path.insert(0, os.environ["FAKE_MODULES"])
from core.inference import worker
from utils.hardware import hardware
import utils.mlx_repair as mlx_repair
import utils.transformers_version as transformers_version

bootstrap_roots = sorted(
    {
        name.split(".", 1)[0]
        for name in sys.modules
        if name.split(".", 1)[0]
        in {
            "huggingface_hub",
            "mlx",
            "mlx_lm",
            "mlx_vlm",
            "torch",
            "transformers",
            "unsloth",
            "unsloth_zoo",
        }
    }
)
assert not bootstrap_roots, f"worker bootstrap imported ML modules: {bootstrap_roots}"

worker.is_apple_silicon = lambda: True
hardware.is_apple_silicon = lambda: True
hardware._has_torch = lambda: False
mlx_repair._mlx_versions_satisfy_minimums = lambda: True
# The fake mlx packages have no dist-info, so the version check has to be stood down at
# the level the gate reads: it now asks for the blocker LIST, since the same measurement
# both decides the verdict and explains it.
mlx_repair._mlx_version_blockers = lambda: []
transformers_version._VENV_T5_530_DIR = os.environ["SIDECAR"]
transformers_version._ensure_venv_t5_530_exists = lambda: True

observed = {"bootstrap_roots": bootstrap_roots}

def capture_active_version(_backend, _config, _responses):
    module = sys.modules["transformers"]
    observed["active"] = module.__version__
    observed["file"] = module.__file__
    observed["device"] = hardware.DEVICE.value

class CommandQueue:
    def get(self, timeout):
        return {"type": "shutdown"}

class ResponseQueue:
    def put(self, _response):
        pass

worker._handle_load = capture_active_version
worker.run_inference_process(
    cmd_queue = CommandQueue(),
    resp_queue = ResponseQueue(),
    cancel_event = None,
    config = {
        "model_name": "Ministral-3-regression",
        "hf_token": "",
        "resolved_gpu_ids": None,
        "device_backend": "mlx",
    },
)
observed["tier"] = transformers_version.get_transformers_tier(
    "Ministral-3-regression"
)
print("RESULT " + json.dumps(observed, sort_keys = True))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd = backend_dir,
        env = {
            **__import__("os").environ,
            "FAKE_MODULES": str(fake_modules),
            "SIDECAR": str(sidecar),
            "UNSLOTH_STUDIO_HOME": str(tmp_path),
            "HF_HOME": str(tmp_path / "hf"),
            "HF_HUB_CACHE": str(tmp_path / "hf" / "hub"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
        capture_output = True,
        text = True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    result_line = next(
        (
            line.removeprefix("RESULT ")
            for line in result.stdout.splitlines()
            if line.startswith("RESULT ")
        ),
        None,
    )
    assert result_line is not None, result.stdout + result.stderr
    observed = json.loads(result_line)
    assert observed["bootstrap_roots"] == []
    assert observed["tier"] == "530"
    assert observed["device"] == "mlx"
    assert observed["active"] == "5.3.0"
    assert observed["file"] == str(sidecar / "transformers" / "__init__.py")


def test_worker_share_object_oversize_notifies_peers(monkeypatch):
    from core.inference import worker

    calls = []

    mlx_pkg = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx_core.array = lambda value, **_kwargs: SimpleNamespace(item = lambda: value)
    mlx_core.eval = lambda value: value
    mlx_core.distributed = SimpleNamespace(
        all_sum = lambda value, group = None: calls.append(value.item()) or value
    )
    mlx_pkg.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setattr(worker, "_SHARE_OBJECT_MAX_BYTES", 8)

    responses = []
    worker._handle_share_object(
        SimpleNamespace(
            _distributed_group = object(),
            _distributed_rank = 0,
            _distributed_world_size = 2,
        ),
        {"type": "share_object", "request_id": "rid", "object": {"text": "too long"}},
        SimpleNamespace(put = responses.append),
    )

    assert calls == [worker._SHARE_OBJECT_ERROR_SIZE]
    assert responses[0]["type"] == "share_error"


# Regression: generate_chat_response must accept the four template kwargs
# (tools / enable_thinking / reasoning_effort / preserve_thinking) so the route
# layer can forward UI toggles. The old signature raised
# "got an unexpected keyword argument 'tools'" on Mac.


def test_mlx_generate_chat_response_accepts_template_kwargs():
    import inspect
    from core.inference.mlx_inference import MLXInferenceBackend

    sig = inspect.signature(MLXInferenceBackend.generate_chat_response)
    params = sig.parameters
    for name in ("tools", "enable_thinking", "reasoning_effort", "preserve_thinking"):
        assert name in params, (
            f"MLX.generate_chat_response is missing the {name!r} kwarg; "
            "the route layer forwards this and a missing kwarg raises "
            "TypeError on Mac"
        )
        assert (
            params[name].default is None
        ), f"{name!r} must default to None so existing callers stay valid"


def test_mlx_vlm_reemits_think_prefill_inside_adapter_context(monkeypatch):
    """A prefilled <think> block must be re-emitted as the first VLM snapshot,
    inside the adapter context (so unsupported requests still raise first), so
    the UI renders the thinking block during prefill and a pre-first-token
    cancel does not drop it. Mirrors _generate_text."""
    from core.inference import mlx_inference

    MLXInferenceBackend = mlx_inference.MLXInferenceBackend

    order = []

    @contextmanager
    def _adapter_state(_model, state):
        assert backend._generation_lock.locked()
        order.append("adapter_enter")
        try:
            yield
        finally:
            order.append("adapter_exit")

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _adapter_state)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.detect_think_prefill",
        lambda *_a, **_k: "<think>\n",
    )

    prompt_utils = SimpleNamespace(
        MODEL_CONFIG = {"deepseek_vl_v2": object()},
        apply_chat_template = lambda *_a, **_k: "<image> model-aware",
    )
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.prompt_utils = prompt_utils

    def _vlm_stream(*_a, **_k):
        # The prefill must have been emitted before any generated token.
        assert order[-1] == "adapter_enter"
        yield SimpleNamespace(text = "ok", prompt_tokens = 3, generation_tokens = 1)

    mlx_vlm.stream_generate = _vlm_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _t, _m, **_k: "<image> model-aware",
    )

    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "deepseek_vl_v2"})
    backend._processor = SimpleNamespace(tokenizer = SimpleNamespace())
    args = ([{"role": "user", "content": [{"type": "image"}]}], object(), 0, 1, 0, 0, 1, 1, None)

    gen = backend._generate_vlm(*args, _adapter_state = False)
    # First snapshot is the prefill alone, emitted after entering the adapter context.
    assert next(gen) == "<think>\n"
    assert order == ["adapter_enter"]
    # Subsequent snapshots are cumulative (prefill + generated text).
    assert next(gen) == "<think>\nok"
    gen.close()
    assert order == ["adapter_enter", "adapter_exit"]


def test_mlx_vlm_generation_selects_renderer_by_capability(monkeypatch):
    from core.inference import mlx_inference

    MLXInferenceBackend = mlx_inference.MLXInferenceBackend

    calls = {"generic": [], "model": [], "stream": []}
    adapter_events = []
    adapter_active = {"value": False}

    @contextmanager
    def _adapter_state(_model, state):
        assert backend._generation_lock.locked()
        adapter_events.append(("enter", state))
        adapter_active["value"] = True
        try:
            yield
        finally:
            adapter_active["value"] = False
            adapter_events.append(("exit", state))

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _adapter_state)
    state = {"generic": "serialized", "model": "<image> model-aware"}
    prompt_utils = SimpleNamespace(
        MODEL_CONFIG = {"deepseek_vl_v2": object()},
        apply_chat_template = lambda *_args, **kwargs: (
            calls["model"].append(kwargs) or state["model"]
        ),
    )
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.prompt_utils = prompt_utils

    def _vlm_stream(*args, **kwargs):
        assert adapter_active["value"]
        calls["stream"].append((args, kwargs))
        yield SimpleNamespace(text = "ok", prompt_tokens = 3, generation_tokens = 1)

    mlx_vlm.stream_generate = _vlm_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)

    def generic(_target, _messages, **kwargs):
        calls["generic"].append(kwargs)
        if isinstance(state["generic"], Exception):
            raise state["generic"]
        if state["generic"] == "serialized":
            return f"User: {_messages[0]['content']}"
        return state["generic"]

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        generic,
    )
    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "deepseek_vl_v2"})
    backend._processor = SimpleNamespace(tokenizer = SimpleNamespace())
    args = ([{"role": "user", "content": [{"type": "image"}]}], object(), 0, 1, 0, 0, 1, 1, None)
    tools = [{"function": {"name": "search"}}]
    generator = backend._generate_vlm(*args, _adapter_state = False)
    assert next(generator) == "ok"
    assert adapter_active["value"] and backend._generation_lock.locked()
    generator.close()
    assert adapter_events == [("enter", False), ("exit", False)]
    assert calls["model"][0]["num_images"] == 1
    assert calls["stream"][0][0][2] == "<image> model-aware"
    with pytest.raises(RuntimeError, match = "dropping requested tools"):
        list(backend._generate_vlm(*args, tools = tools))
    with pytest.raises(RuntimeError, match = "dropping requested tools or reasoning"):
        list(backend._generate_vlm(*args, enable_thinking = False))
    backend._processor = SimpleNamespace(chat_template = "template")
    state["generic"] = "<image> healthy generic"
    assert list(backend._generate_vlm(*args, tools = tools, enable_thinking = False)) == ["ok"]
    assert calls["generic"][-1]["enable_thinking"] is False
    assert calls["stream"][-1][0][2] == "<image> healthy generic"
    state["generic"] = "generic prompt"
    text_messages = [{"role": "user", "content": "hello"}]
    assert list(backend._generate_vlm(*((text_messages, None) + args[2:]), tools = tools)) == ["ok"]
    assert calls["generic"][-1]["tools"] == tools
    assert calls["stream"][-1][0][2] == "generic prompt"
    two_images = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}]}]
    with pytest.raises(RuntimeError, match = "2 structured image item"):
        list(backend._generate_vlm(*((two_images,) + args[1:]), tools = tools))
    state["generic"] = "serialized"
    tool_history = args[0] + [{"role": "assistant", "tool_calls": [{"id": "call-1"}]}]
    with pytest.raises(RuntimeError, match = "tool-call history"):
        list(backend._generate_vlm(*((tool_history,) + args[1:]), tools = tools))
    state["generic"] = ValueError("generic rendering failed")
    state["model"] = f"User: {args[0][0]['content']}"
    with pytest.raises(ValueError, match = "generic rendering failed"):
        list(backend._generate_vlm(*args))


def test_mlx_vlm_image_injection_reuses_media_aliases(monkeypatch):
    from core.inference.mlx_inference import MLXInferenceBackend, _prompt_serializes_vlm_media

    media = [{"type": "image"}]
    quoted = [{"role": "user", "content": media}, {"role": "user", "content": f"Explain {media}"}]
    assert _prompt_serializes_vlm_media(f"<image>\n{media[0]}", quoted[:1])
    assert not _prompt_serializes_vlm_media(f"<image>\nExplain {media}", quoted)
    assert _prompt_serializes_vlm_media(f"User: {media}\nExplain {media}", quoted)
    quoted[1]["content"] = [{"type": "text", "text": f'Explain "this" {media}'}]
    assert not _prompt_serializes_vlm_media(f'<image>\nExplain "this" {media}', quoted)
    json_media = [{"type": "image_url"}]
    json_repr = '{"type": "image_url"}'
    assert _prompt_serializes_vlm_media(f"<image>\n{json_repr}", [{"content": json_media}])
    assert not _prompt_serializes_vlm_media(
        f"<image>\nExplain {json_repr}",
        [{"content": json_media}, {"content": f"Explain {json_repr}"}],
    )

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._is_vlm = True
    captured = []
    backend._generate_vlm = lambda messages, *_args, **_kwargs: (
        captured.append(messages) or iter(())
    )
    messages = [{"role": "user", "content": [{"type": "image_url"}]}]
    list(backend.generate_chat_response(messages, image = object()))
    assert captured[0][0]["content"] == [{"type": "image_url"}]


def test_mlx_vlm_model_config_prefers_config_with_model_type():
    from core.inference.mlx_inference import _mlx_vlm_model_config

    # config present but missing model_type must fall back to _config
    m = SimpleNamespace(config = {}, _config = {"model_type": "deepseek_vl_v2"})
    assert _mlx_vlm_model_config(m) == ({"model_type": "deepseek_vl_v2"}, "deepseek_vl_v2")
    # an object config whose model_type is None also falls back
    m = SimpleNamespace(config = SimpleNamespace(model_type = None), _config = {"model_type": "qwen2_vl"})
    assert _mlx_vlm_model_config(m)[1] == "qwen2_vl"
    # a config that already carries a model_type is preferred and returned unchanged
    assert _mlx_vlm_model_config(SimpleNamespace(config = {"model_type": "gemma3"})) == (
        {"model_type": "gemma3"},
        "gemma3",
    )


def test_mlx_generate_text_forwards_kwargs_into_template_helper(monkeypatch):
    """Mac text path must route through apply_chat_template_for_generation so
    reasoning / tool kwargs reach the tokenizer."""
    _install_fake_mlx(monkeypatch)
    from core.inference import mlx_inference

    MLXInferenceBackend = mlx_inference.MLXInferenceBackend
    real_adapter_state = mlx_inference._temporary_mlx_adapter_state

    # The text path renders once with tools, then the native-template fallback makes a second no-
    # tools probe call (tools=None) to detect whether the template dropped the schema.
    captured_calls = []

    def _fake_apply(tokenizer, messages, **kwargs):
        captured_calls.append({"tokenizer": tokenizer, "messages": messages, "kwargs": kwargs})
        return "<rendered prompt>"

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        _fake_apply,
        raising = True,
    )

    # mlx_lm.stream_generate yields response objects with .token; use a
    # one-token generator so _generate_text returns without the real stack.
    import types as _types

    mlx_lm_pkg = _types.ModuleType("mlx_lm")
    mlx_lm_sample = _types.ModuleType("mlx_lm.sample_utils")
    mlx_lm_sample.make_sampler = lambda **_kw: object()
    mlx_lm_sample.make_logits_processors = lambda **_kw: None

    adapter_events = []
    adapter_active = {"value": False}
    stream_state = {"fail": False}

    @contextmanager
    def _adapter_state(_model, state):
        assert backend._generation_lock.locked()
        adapter_events.append(("enter", state))
        adapter_active["value"] = True
        try:
            yield
        finally:
            adapter_active["value"] = False
            adapter_events.append(("exit", state))

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _adapter_state)

    class _Resp:
        def __init__(self, tok):
            self.token = tok

    def _stream_generate(_model, _tokenizer, **_kw):
        assert adapter_active["value"]
        if stream_state["fail"]:
            raise RuntimeError("generation failed")
        yield _Resp(1)

    mlx_lm_pkg.stream_generate = _stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    class _Tok:
        chat_template = "x"

        def decode(
            self,
            ids,
            skip_special_tokens = False,
        ):
            return "hi"

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = _Tok()
    backend._is_vlm = False

    generator = backend.generate_with_adapter_control(
        use_adapter = False,
        messages = [{"role": "user", "content": "ping"}],
        tools = [{"function": {"name": "web_search"}}],
        enable_thinking = True,
        reasoning_effort = "medium",
        preserve_thinking = True,
        max_new_tokens = 1,
    )
    assert next(generator) == "hi"
    assert adapter_active["value"] and backend._generation_lock.locked()
    generator.close()
    assert adapter_events == [("enter", False), ("exit", False)]
    stream_state["fail"] = True
    with pytest.raises(RuntimeError, match = "generation failed"):
        list(
            backend.generate_with_adapter_control(
                use_adapter = False,
                messages = [{"role": "user", "content": "ping"}],
                max_new_tokens = 1,
            )
        )
    assert adapter_events[-2:] == [("enter", False), ("exit", False)]
    assert not backend._generation_lock.locked()

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", real_adapter_state)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.detect_think_prefill",
        lambda *_args, **_kwargs: "<think>",
    )
    stream_state["fail"] = False
    named = backend.generate_with_adapter_control(
        use_adapter = "named",
        messages = [{"role": "user", "content": "ping"}],
        max_new_tokens = 1,
    )
    with pytest.raises(NotImplementedError, match = "named adapter"):
        next(named)
    assert not adapter_active["value"] and not backend._generation_lock.locked()
    # The toggled kwargs must reach the chat-template helper on the real render
    # (one of the calls carries the tools; the fallback probe passes tools=None).
    tool_renders = [
        c
        for c in captured_calls
        if c["kwargs"].get("tools") == [{"function": {"name": "web_search"}}]
    ]
    assert tool_renders, captured_calls
    render = tool_renders[0]
    assert render["kwargs"]["enable_thinking"] is True
    assert render["kwargs"]["reasoning_effort"] == "medium"
    assert render["kwargs"]["preserve_thinking"] is True


def test_mlx_text_normalizes_native_reasoning_and_close_releases_lock(monkeypatch):
    _install_fake_mlx(monkeypatch)
    from core.inference.mlx_inference import MLXInferenceBackend

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: "prompt",
        raising = True,
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.render_with_native_template_fallback",
        lambda formatted_prompt, **_kwargs: SimpleNamespace(
            prompt = formatted_prompt,
            reasoning_channel_markers = ("<|channel>thought\n", "<channel|>"),
        ),
        raising = True,
    )

    mlx_lm_pkg = types.ModuleType("mlx_lm")
    mlx_lm_sample = types.ModuleType("mlx_lm.sample_utils")
    mlx_lm_sample.make_sampler = lambda **_kw: object()
    mlx_lm_sample.make_logits_processors = lambda **_kw: None

    class _Resp:
        def __init__(self, text, tok):
            self.text = text
            self.token = tok

    def _stream_generate(_model, _tokenizer, **_kw):
        yield _Resp("<|channel>thought\n", 10)
        yield _Resp("r", 11)
        yield _Resp("<channel|>", 12)
        yield _Resp("a", 13)

    mlx_lm_pkg.stream_generate = _stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = SimpleNamespace(all_special_tokens = [])
    backend._is_vlm = False

    assert list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "ping"}],
            max_new_tokens = 4,
        )
    ) == ["<think>", "<think>r", "<think>r</think>", "<think>r</think>a"]

    gen = backend.generate_chat_response(
        messages = [{"role": "user", "content": "ping"}],
        max_new_tokens = 4,
    )
    assert next(gen) == "<think>"
    assert backend._generation_lock.locked()
    gen.close()
    assert not backend._generation_lock.locked()


def test_mlx_text_post_tool_prompt_opens_reasoning_channel(monkeypatch):
    """Gemma-style templates open the thought channel in the POST-TOOL prompt.

    Generation then emits only the closing marker, so a parser assuming it starts
    outside reasoning would leak the post-tool reasoning and a raw ``<channel|>``
    into the visible answer.
    """
    _install_fake_mlx(monkeypatch)
    from core.inference.mlx_inference import MLXInferenceBackend

    post_tool_prompt = (
        "<|turn>model\n<|tool_response>response:web_search{}<tool_response|><|channel>thought\n"
    )
    # Give the pre-fallback render a non-opening tail, so state read from the wrong
    # prompt fails here.
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: "<|turn>model\n",
        raising = True,
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.render_with_native_template_fallback",
        lambda formatted_prompt, **_kwargs: SimpleNamespace(
            prompt = post_tool_prompt,
            reasoning_channel_markers = ("<|channel>thought", "<channel|>"),
        ),
        raising = True,
    )

    mlx_lm_pkg = types.ModuleType("mlx_lm")
    mlx_lm_sample = types.ModuleType("mlx_lm.sample_utils")
    mlx_lm_sample.make_sampler = lambda **_kw: object()
    mlx_lm_sample.make_logits_processors = lambda **_kw: None

    class _Resp:
        def __init__(self, text, tok):
            self.text = text
            self.token = tok

    def _stream_generate(_model, _tokenizer, **_kw):
        yield _Resp("The search says 18C.", 10)
        yield _Resp("<channel|>", 11)
        yield _Resp("It is 18C.", 12)

    mlx_lm_pkg.stream_generate = _stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = SimpleNamespace(all_special_tokens = [])
    backend._is_vlm = False

    snapshots = list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "weather?"}],
            max_new_tokens = 3,
        )
    )
    assert snapshots[-1] == "<think>The search says 18C.</think>It is 18C."
    assert "<channel|>" not in snapshots[-1]
    assert all(later.startswith(earlier) for earlier, later in zip(snapshots, snapshots[1:]))

    # The flag survives into the next pass, but the tool result is the trailing turn,
    # so nothing was resumed and the prompt must still be read.
    resumed_then_tool = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "partial",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "18C"},
    ]
    assert (
        list(
            backend.generate_chat_response(
                messages = resumed_then_tool,
                max_new_tokens = 3,
                continue_final_message = True,
            )
        )[-1]
        == "<think>The search says 18C.</think>It is 18C."
    )


def test_mlx_text_native_metadata_preserves_prefilled_think_snapshots(monkeypatch):
    _install_fake_mlx(monkeypatch)
    from core.inference.mlx_inference import MLXInferenceBackend

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: "prompt<think>\n",
        raising = True,
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.render_with_native_template_fallback",
        lambda formatted_prompt, **_kwargs: SimpleNamespace(
            prompt = formatted_prompt,
            reasoning_channel_markers = ("<|channel>thought", "<channel|>"),
        ),
        raising = True,
    )

    mlx_lm_pkg = types.ModuleType("mlx_lm")
    mlx_lm_sample = types.ModuleType("mlx_lm.sample_utils")
    mlx_lm_sample.make_sampler = lambda **_kw: object()
    mlx_lm_sample.make_logits_processors = lambda **_kw: None

    class _Resp:
        def __init__(self, text, tok):
            self.text = text
            self.token = tok

    def _stream_generate(_model, _tokenizer, **_kw):
        yield _Resp("reason", 10)
        yield _Resp("</think>", 11)
        yield _Resp("answer", 12)

    mlx_lm_pkg.stream_generate = _stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = SimpleNamespace(all_special_tokens = [])
    backend._is_vlm = False

    snapshots = list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "ping"}],
            max_new_tokens = 3,
        )
    )
    assert snapshots == [
        "<think>\n",
        "<think>\nreason",
        "<think>\nreason</think>",
        "<think>\nreason</think>answer",
    ]
    assert all(current.startswith(previous) for previous, current in zip(snapshots, snapshots[1:]))


def test_mlx_vlm_normalizes_native_reasoning_channels(monkeypatch):
    _install_fake_mlx(monkeypatch)
    from core.inference.mlx_inference import MLXInferenceBackend

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: "prompt",
        raising = True,
    )

    mlx_vlm_pkg = types.ModuleType("mlx_vlm")

    class _Resp:
        def __init__(self, text, tok):
            self.text = text
            self.token = tok

    def _stream_generate(_model, _processor, _prompt, _images, **_kw):
        yield _Resp("<|channel>thought\n", 10)
        yield _Resp("vision", 11)
        yield _Resp("<channel|>", 12)
        yield _Resp(" answer", 13)

    mlx_vlm_pkg.stream_generate = _stream_generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm_pkg)

    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = SimpleNamespace())
    backend._processor = SimpleNamespace(
        chat_template = "<|channel>thought\n...<channel|>",
        all_special_tokens = [],
        apply_chat_template = lambda *_args, **_kwargs: "prompt",
    )
    backend._is_vlm = True

    assert list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "describe"}],
            image = object(),
            max_new_tokens = 4,
        )
    ) == [
        "<think>",
        "<think>vision",
        "<think>vision</think>",
        "<think>vision</think> answer",
    ]


def test_mlx_vlm_post_tool_prompt_opens_reasoning_channel(monkeypatch):
    """The VLM snapshot path must derive channel state from its rendered prompt too."""
    _install_fake_mlx(monkeypatch)
    from core.inference.mlx_inference import MLXInferenceBackend

    post_tool_prompt = "<|tool_response>response:web_search{}<tool_response|><|channel>thought\n"
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: post_tool_prompt,
        raising = True,
    )

    mlx_vlm_pkg = types.ModuleType("mlx_vlm")

    class _Resp:
        def __init__(self, text, tok):
            self.text = text
            self.token = tok

    def _stream_generate(_model, _processor, _prompt, _images, **_kw):
        yield _Resp("looking at it", 10)
        yield _Resp("<channel|>", 11)
        yield _Resp("A cat.", 12)

    mlx_vlm_pkg.stream_generate = _stream_generate
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm_pkg)

    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = SimpleNamespace())
    backend._processor = SimpleNamespace(
        chat_template = "<|channel>thought\n...<channel|>",
        all_special_tokens = [],
        apply_chat_template = lambda *_args, **_kwargs: post_tool_prompt,
    )
    backend._is_vlm = True

    snapshots = list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "describe"}],
            image = object(),
            max_new_tokens = 3,
        )
    )
    assert snapshots[-1] == "<think>looking at it</think>A cat."
    assert "<channel|>" not in snapshots[-1]


class _FakeLRUPromptCache:
    def __init__(
        self,
        max_size = 10,
        max_bytes = 1 << 63,
    ):
        self.max_size = max_size
        self.max_bytes = max_bytes
        self.entries = {}

    def fetch_nearest_cache(self, key, tokens):
        import copy

        stored = self.entries.get(key, {})
        exact = stored.get(tuple(tokens))
        if exact is not None:
            return copy.deepcopy(exact), []
        best = None
        for candidate, cache in stored.items():
            if len(candidate) < len(tokens) and tuple(tokens[: len(candidate)]) == candidate:
                if best is None or len(candidate) > len(best[0]):
                    best = (candidate, cache)
        if best is not None:
            return copy.deepcopy(best[1]), list(tokens[len(best[0]) :])
        return None, list(tokens)

    def insert_cache(
        self,
        key,
        tokens,
        prompt_cache,
        *,
        cache_type = "assistant",
    ):
        import copy
        self.entries.setdefault(key, {})[tuple(tokens)] = copy.deepcopy(prompt_cache)


class _FakeCacheEntry:
    def __init__(
        self,
        offset = 0,
        nbytes = 1,
    ):
        self.offset = offset
        self.nbytes = nbytes


def _install_fake_prompt_cache_api(monkeypatch, trimmable = True):
    from core.inference import mlx_inference

    def _make_prompt_cache(_model):
        return [_FakeCacheEntry()]

    def _can_trim_prompt_cache(_cache):
        return trimmable

    def _trim_prompt_cache(cache, num):
        cache[0].offset = max(cache[0].offset - num, 0)
        return num

    monkeypatch.setattr(
        mlx_inference,
        "_mlx_prompt_cache_api",
        lambda: (
            _FakeLRUPromptCache,
            _make_prompt_cache,
            _can_trim_prompt_cache,
            _trim_prompt_cache,
        ),
    )


def test_mlx_prompt_cache_max_bytes_budget(monkeypatch):
    from core.inference.mlx_inference import (
        PROMPT_CACHE_FALLBACK_BYTES,
        PROMPT_CACHE_MEMORY_FRACTION,
        _prompt_cache_max_bytes,
    )

    monkeypatch.delenv("UNSLOTH_MLX_PROMPT_CACHE_BYTES", raising = False)
    assert _prompt_cache_max_bytes(None) == PROMPT_CACHE_FALLBACK_BYTES
    assert _prompt_cache_max_bytes(20.0) == int(20.0 * 1e9 * PROMPT_CACHE_MEMORY_FRACTION)

    monkeypatch.setenv("UNSLOTH_MLX_PROMPT_CACHE_BYTES", "4096")
    assert _prompt_cache_max_bytes(20.0) == 4096
    monkeypatch.setenv("UNSLOTH_MLX_PROMPT_CACHE_BYTES", "0")
    assert _prompt_cache_max_bytes(20.0) == 0
    monkeypatch.setenv("UNSLOTH_MLX_PROMPT_CACHE_BYTES", "not-a-number")
    assert _prompt_cache_max_bytes(20.0) == int(20.0 * 1e9 * PROMPT_CACHE_MEMORY_FRACTION)


def test_mlx_prompt_cache_never_returns_empty_remainder(monkeypatch):
    _install_fake_prompt_cache_api(monkeypatch)
    from core.inference.mlx_inference import _MLXPromptCacheHistory

    history = _MLXPromptCacheHistory(6, 1 << 30)
    tokens = list(range(10))
    cache, rest = history.fetch(object(), "key", tokens)
    assert len(rest) == 10
    cache[0].offset = len(tokens)
    history.insert("key", tokens, cache)

    _cache, rest = history.fetch(object(), "key", tokens)
    assert rest == tokens[-1:]

    longer = tokens + [99, 100]
    _cache, rest = history.fetch(object(), "key", longer)
    assert rest == [99, 100]

    _install_fake_prompt_cache_api(monkeypatch, trimmable = False)
    history = _MLXPromptCacheHistory(6, 1 << 30)
    cache, _rest = history.fetch(object(), "key", tokens)
    cache[0].offset = len(tokens)
    history.insert("key", tokens, cache)
    _cache, rest = history.fetch(object(), "key", tokens)
    assert rest == tokens, "untrimmable entry must not be reused"


def test_mlx_prompt_cache_key_isolates_adapter_state(monkeypatch):
    _install_fake_prompt_cache_api(monkeypatch)
    _install_fake_mlx(monkeypatch)
    from core.inference.mlx_inference import MLXInferenceBackend

    class _Tok:
        bos_token = None

        def encode(
            self,
            text,
            add_special_tokens = True,
        ):
            return [ord(c) for c in text]

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = _Tok()
    backend.active_model_name = "model-a"

    prompt = "shared prefix"
    _rest, cache, key, tokens, cached = backend._prepare_prompt_cache(prompt, True)
    assert cached == 0
    cache[0].offset = len(tokens)
    backend._prompt_cache_history.insert(key, tokens, cache)

    _rest, _cache, _key, _tokens, cached_same = backend._prepare_prompt_cache(prompt, True)
    assert cached_same > 0
    _rest, _cache, _key, _tokens, cached_flipped = backend._prepare_prompt_cache(prompt, False)
    assert cached_flipped == 0


def _install_fake_text_stack(
    monkeypatch,
    token_map,
    captured,
    markers = None,
):
    import types as _types

    from core.inference import mlx_inference

    _install_fake_mlx(monkeypatch)
    monkeypatch.setattr(
        mlx_inference,
        "_temporary_mlx_adapter_state",
        lambda _model, _state: __import__("contextlib").nullcontext(),
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _tok, messages, **_kw: messages[-1]["content"],
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.render_with_native_template_fallback",
        lambda formatted_prompt, **_kw: SimpleNamespace(
            prompt = formatted_prompt,
            reasoning_channel_markers = markers,
        ),
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.detect_think_prefill",
        lambda *_a, **_kw: "",
    )

    class _Resp:
        def __init__(self, token, processed):
            self.token = token
            self.text = f"<{token}>"
            self.prompt_tokens = processed
            self.prompt_tps = 10.0
            self.generation_tokens = 1
            self.generation_tps = 5.0

    def _stream_generate(_model, _tokenizer, **kwargs):
        captured.append(kwargs)
        processed = len(kwargs["prompt"])
        cache = kwargs.get("prompt_cache")
        if cache is not None:
            cache[0].offset += processed
        for token in token_map["generated"]:
            if cache is not None:
                cache[0].offset += 1
            yield _Resp(token, processed)

    mlx_lm_pkg = _types.ModuleType("mlx_lm")
    mlx_lm_pkg.stream_generate = _stream_generate
    mlx_lm_sample = _types.ModuleType("mlx_lm.sample_utils")
    mlx_lm_sample.make_sampler = lambda **_kw: object()
    mlx_lm_sample.make_logits_processors = lambda **_kw: []
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    class _Tok:
        bos_token = None
        chat_template = "x"

        def encode(
            self,
            text,
            add_special_tokens = True,
        ):
            return list(token_map[text])

        def decode(
            self,
            ids,
            skip_special_tokens = False,
        ):
            return "".join(str(i) for i in ids)

    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = _Tok()
    backend._is_vlm = False
    backend.active_model_name = "model-a"
    return backend


def _run_turn(backend, prompt):
    list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": prompt}],
            max_new_tokens = 4,
        )
    )


def test_mlx_text_reuses_prompt_cache_on_the_next_turn(monkeypatch):
    _install_fake_prompt_cache_api(monkeypatch)
    captured = []
    token_map = {
        "P1": [1, 2, 3],
        "P2": [1, 2, 3, 7, 8, 9, 10],
        "generated": [7, 8],
    }
    backend = _install_fake_text_stack(monkeypatch, token_map, captured)

    _run_turn(backend, "P1")
    assert captured[0]["prompt"] == [1, 2, 3]
    assert "prompt_cache" in captured[0]
    assert backend.last_generation_stats["timings"]["cache_n"] == 0

    _run_turn(backend, "P2")
    assert captured[1]["prompt"] == [9, 10], "turn two should prefill only the new tail"

    stats = backend.last_generation_stats
    assert stats["timings"]["cache_n"] == 5
    assert stats["timings"]["prompt_n"] == 2
    assert stats["usage"]["prompt_tokens"] == 7


def test_mlx_text_without_lru_prompt_cache_prefills_the_full_prompt(monkeypatch):
    from core.inference import mlx_inference

    monkeypatch.setattr(mlx_inference, "_mlx_prompt_cache_api", lambda: None)
    captured = []
    token_map = {"P1": [1, 2, 3], "generated": [7]}
    backend = _install_fake_text_stack(monkeypatch, token_map, captured)

    _run_turn(backend, "P1")
    assert captured[0]["prompt"] == "P1"
    assert "prompt_cache" not in captured[0]
    assert backend.last_generation_stats["timings"]["cache_n"] == 0


def test_mlx_text_tracks_tokens_on_the_native_reasoning_path(monkeypatch):
    _install_fake_prompt_cache_api(monkeypatch)
    captured = []
    token_map = {"P1": [1, 2, 3], "P2": [1, 2, 3, 7, 8, 9], "generated": [7, 8]}
    backend = _install_fake_text_stack(monkeypatch, token_map, captured, markers = ("<a>", "</a>"))

    _run_turn(backend, "P1")
    _run_turn(backend, "P2")
    assert captured[1]["prompt"] == [9]


def test_mlx_presence_penalty_latches_the_first_decode_step():
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    from core.inference.mlx_inference import _make_mlx_presence_penalty_processor

    processor = _make_mlx_presence_penalty_processor(2.0)
    logits = mx.zeros((1, 5))
    out = processor(mx.array([3]), logits)
    assert np.array_equal(np.array(out), np.zeros((1, 5))), "prompt must not be penalized"
    out = processor(mx.array([3, 1]), mx.zeros((1, 5)))
    penalized = np.array(out)[0]
    assert penalized[1] == -2.0
    assert penalized[3] == 0.0


def test_mlx_prompt_cache_survives_reset_but_not_unload(monkeypatch):
    _install_fake_prompt_cache_api(monkeypatch)
    _install_fake_mlx(monkeypatch)
    sys.modules["mlx.core"].clear_cache = lambda: None
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    backend.active_model_name = "model-a"
    history = backend._prompt_cache()
    assert history is not None

    backend.reset_generation_state()
    assert backend._prompt_cache_history is history

    backend.unload_model("model-a")
    assert backend._prompt_cache_history is None


def test_mlx_prompt_cache_skips_entries_over_budget(monkeypatch):
    _install_fake_prompt_cache_api(monkeypatch)
    from core.inference.mlx_inference import _MLXPromptCacheHistory

    history = _MLXPromptCacheHistory(6, 1000)
    history.insert("key", [1, 2, 3], [_FakeCacheEntry(offset = 3, nbytes = 400)])
    assert len(history._lru.entries.get("key", {})) == 1

    history.insert("key", list(range(50)), [_FakeCacheEntry(offset = 50, nbytes = 5000)])
    stored = history._lru.entries.get("key", {})
    assert tuple([1, 2, 3]) in stored
    assert tuple(range(50)) not in stored


def test_mlx_prompt_cache_keys_on_what_the_kv_covers(monkeypatch):
    _install_fake_prompt_cache_api(monkeypatch)
    from core.inference.mlx_inference import _MLXPromptCacheHistory

    class _Entry:
        def __init__(
            self,
            offset,
            nbytes = 1,
        ):
            self.offset = offset
            self.nbytes = nbytes

    history = _MLXPromptCacheHistory(6, 1 << 30)

    history.insert("key", list(range(10)), [_Entry(offset = 8)])
    assert tuple(range(8)) in history._lru.entries["key"]
    assert tuple(range(10)) not in history._lru.entries["key"]

    history.insert("other", list(range(4)), [_Entry(offset = 9)])
    assert "other" not in history._lru.entries


def test_mlx_prompt_cache_only_stores_verifiable_prefix_coverage(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import CacheList, ChunkedKVCache, KVCache, RotatingKVCache

    _install_fake_prompt_cache_api(monkeypatch)
    from core.inference.mlx_inference import _kv_prefix_coverage, _MLXPromptCacheHistory

    def feed(entry, n):
        for _ in range(n):
            block = mx.zeros((1, 2, 1, 4), dtype = mx.float16)
            entry.update_and_fetch(block, block)
        mx.eval(entry.state)
        return entry

    plain = feed(KVCache(), 30)
    unwrapped = feed(RotatingKVCache(max_size = 100, keep = 2), 30)
    wrapped = feed(RotatingKVCache(max_size = 10, keep = 2), 30)
    chunked = feed(ChunkedKVCache(chunk_size = 8), 30)
    slid = feed(ChunkedKVCache(chunk_size = 8), 30)
    slid.maybe_trim_front()

    assert _kv_prefix_coverage([plain]) == 30
    assert _kv_prefix_coverage([unwrapped]) == 30
    assert _kv_prefix_coverage([chunked]) == 30
    assert wrapped.offset == 30 and wrapped.state[0].shape[2] == 10
    assert _kv_prefix_coverage([wrapped]) is None
    assert slid.start_position > 0
    assert _kv_prefix_coverage([slid]) is None
    assert _kv_prefix_coverage([CacheList(feed(KVCache(), 30), feed(KVCache(), 30))]) == 30
    assert _kv_prefix_coverage([CacheList(feed(KVCache(), 30), wrapped)]) is None
    assert _kv_prefix_coverage([feed(KVCache(), 30), feed(KVCache(), 29)]) is None
    assert _kv_prefix_coverage([]) is None

    history = _MLXPromptCacheHistory(6, 1 << 40)
    for unsafe in (wrapped, slid):
        history.insert("key", list(range(30)), [unsafe])
    assert "key" not in history._lru.entries

    history.insert("key", list(range(30)), [plain])
    assert tuple(range(30)) in history._lru.entries["key"]


def test_mlx_prompt_cache_covers_hybrid_recurrent_layouts(monkeypatch):
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.models.cache import (
        ArraysCache,
        CacheList,
        KVCache,
        RotatingKVCache,
        can_trim_prompt_cache,
    )

    _install_fake_prompt_cache_api(monkeypatch)
    from core.inference.mlx_inference import _kv_prefix_coverage, _MLXPromptCacheHistory

    def attention(entry, n):
        for _ in range(n):
            block = mx.zeros((1, 2, 1, 4), dtype = mx.float16)
            entry.update_and_fetch(block, block)
        mx.eval(entry.state)
        return entry

    def recurrent():
        # A GatedDeltaNet layer: conv + recurrent state, neither grows with the prompt.
        entry = ArraysCache(size = 2)
        entry[0] = mx.zeros((1, 4, 4), dtype = mx.float16)
        entry[1] = mx.zeros((1, 2, 4, 4), dtype = mx.float16)
        mx.eval(entry.state)
        return entry

    assert getattr(recurrent(), "offset", None) is None

    # qwen3_5/qwen3_next: a full-attention layer every fourth layer.
    hybrid = [recurrent(), recurrent(), recurrent(), attention(KVCache(), 30)]
    assert _kv_prefix_coverage(hybrid) == 30
    # falcon_h1: both halves inside one CacheList.
    nested = [CacheList(recurrent(), attention(KVCache(), 30))]
    assert _kv_prefix_coverage(nested) == 30

    # Pin it: an mlx-lm that made these trimmable must fail here, not reuse stale state.
    assert can_trim_prompt_cache(hybrid) is False
    assert can_trim_prompt_cache(nested) is False

    class _TrimmableOpaqueState:
        def is_trimmable(self):
            return True

    assert _kv_prefix_coverage([_TrimmableOpaqueState(), attention(KVCache(), 30)]) is None

    # mamba/rwkv: nothing attests to a token count.
    assert _kv_prefix_coverage([recurrent(), recurrent()]) is None

    # A recurrent entry does not excuse an attention sibling that cannot attest.
    windowed = attention(RotatingKVCache(max_size = 10, keep = 2), 30)
    assert _kv_prefix_coverage([recurrent(), windowed]) is None
    assert (
        _kv_prefix_coverage([recurrent(), attention(KVCache(), 30), attention(KVCache(), 29)])
        is None
    )

    history = _MLXPromptCacheHistory(6, 1 << 40)
    history.insert("recurrent", list(range(40)), [recurrent(), recurrent()])
    assert "recurrent" not in history._lru.entries

    history.insert("hybrid", list(range(40)), hybrid)
    assert tuple(range(30)) in history._lru.entries["hybrid"]


# ── Tests: audio-input capability + generation ───────────────────────


def _audio_model(module_paths = ("audio_tower",), model_type = "gemma3n"):
    """Model stub shaped like a loaded mlx-vlm tree (families nest differently)."""
    return SimpleNamespace(
        config = {"model_type": model_type},
        named_modules = lambda: [(p, object()) for p in module_paths],
    )


def _audio_processor(sr = 16000, extractor = True):
    fe = {"feature_extractor": SimpleNamespace(sampling_rate = sr)} if extractor else {}
    return SimpleNamespace(tokenizer = _DummyTokenizer(), **fe)


@pytest.mark.parametrize(
    ("processor", "renders", "capable", "expected"),
    [
        (_audio_processor(), True, True, "audio_vlm"),
        (_audio_processor(), False, True, None),  # our renderer places no marker
        (_audio_processor(), True, False, None),  # zoo refuses the checkpoint
        (_audio_processor(extractor = False), True, True, None),  # no rate to read
        (_audio_processor(sr = 24000), True, True, None),  # route decodes 16 kHz
        (None, True, True, None),  # text-only load
    ],
)
def test_mlx_audio_classification(monkeypatch, processor, renders, capable, expected):
    """Unsloth keeps the rate and rendering gates; the checkpoint answer is zoo's."""
    from core.inference import mlx_inference

    seen = {}

    def _render(
        _proc,
        _model,
        _messages,
        _num_images,
        num_audios = 0,
    ):
        return "P<audio>" if (num_audios and renders) else "P"

    def _capability(
        model,
        proc,
        texts = None,
    ):
        seen["texts"] = texts
        return SimpleNamespace(capable = capable, reason = "stub")

    fake_utils = types.ModuleType("unsloth_zoo.mlx.utils")
    fake_utils.audio_input_capability = _capability
    fake_utils.audio_extractor_sampling_rate = lambda proc: getattr(
        getattr(proc, "feature_extractor", None), "sampling_rate", None
    )
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", fake_utils)
    monkeypatch.setattr(mlx_inference, "_render_registered_vlm_prompt", _render)

    audio_type = mlx_inference._classify_mlx_audio_type(
        _audio_model(), processor, processor is not None
    )
    assert audio_type == expected
    # audio_vlm keeps is_audio (the TTS flag) False → TTS redirect can't fire.
    assert (audio_type is not None and audio_type != "audio_vlm") is False
    # The capability call is probed with OUR rendered prompt.
    if expected == "audio_vlm":
        assert seen["texts"] == "P<audio>"


@pytest.mark.parametrize(
    "capability",
    [
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("zoo blew up")),
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        lambda *a, **k: SimpleNamespace(),  # older/newer API: no .capable
        lambda *a, **k: None,
    ],
)
def test_mlx_audio_classification_survives_a_broken_dependency(monkeypatch, capability):
    """A probe must never fail a model load, whatever the dependency does.

    unsloth_zoo is pinned by a floor, so a version whose capability call raises
    or returns another shape has to degrade to "no audio", not abort the load.
    """
    from core.inference import mlx_inference

    fake_utils = types.ModuleType("unsloth_zoo.mlx.utils")
    fake_utils.audio_input_capability = capability
    fake_utils.audio_extractor_sampling_rate = lambda proc: 16000
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", fake_utils)
    monkeypatch.setattr(
        mlx_inference,
        "_render_registered_vlm_prompt",
        lambda *a, num_audios = 0: "P<audio>" if num_audios else "P",
    )

    assert mlx_inference._classify_mlx_audio_type(_audio_model(), _audio_processor(), True) is None


def test_mlx_audio_classification_survives_an_absent_dependency(monkeypatch):
    """The import itself is inside the guard: no zoo, no audio, still a load."""
    from core.inference import mlx_inference

    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", None)
    assert mlx_inference._classify_mlx_audio_type(_audio_model(), _audio_processor(), True) is None


@pytest.mark.parametrize("codec", ["snac", "dac", "bicodec", "csm", "whisper"])
def test_mlx_audio_classification_keeps_a_classification_it_cannot_judge(monkeypatch, codec):
    """A TTS codec or Whisper is never a vision model, so the probe never runs.

    The worker mirrors this entry over the pre-load config, so returning a bare
    None here would strip is_audio from a codec checkpoint mlx-lm loads happily
    (Orpheus/Llasa/Spark-TTS are plain llama/qwen2), and the chat route's TTS
    redirect would stop firing.
    """
    from core.inference import mlx_inference

    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", None)
    assert (
        mlx_inference._classify_mlx_audio_type(
            _audio_model(),
            None,
            False,
            config_audio_type = codec,
        )
        == codec
    )


@pytest.mark.parametrize(
    "capability",
    [
        None,  # dependency absent entirely
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("zoo blew up")),
        lambda *a, **k: SimpleNamespace(),  # older/newer API: no .capable
    ],
)
def test_mlx_audio_capability_survives_a_probe_that_could_not_run(monkeypatch, capability):
    """An unjudgeable probe defers; it does not retract audio_vlm.

    Every currently released unsloth_zoo lands here, so treating "could not look"
    as a verified negative would hide the upload control the pre-load detection
    had already earned.
    """
    from core.inference import mlx_inference

    if capability is None:
        monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", None)
    else:
        fake_utils = types.ModuleType("unsloth_zoo.mlx.utils")
        fake_utils.audio_input_capability = capability
        fake_utils.audio_extractor_sampling_rate = lambda proc: 16000
        monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", fake_utils)
    monkeypatch.setattr(
        mlx_inference,
        "_render_registered_vlm_prompt",
        lambda *a, num_audios = 0: "P<audio>" if num_audios else "P",
    )

    assert (
        mlx_inference._classify_mlx_audio_type(
            _audio_model(),
            _audio_processor(),
            True,
            config_audio_type = "audio_vlm",
        )
        == "audio_vlm"
    )


def test_mlx_audio_probe_that_answered_no_still_retracts_audio_vlm(monkeypatch):
    """The corrective direction survives: a real negative downgrades."""
    from core.inference import mlx_inference

    fake_utils = types.ModuleType("unsloth_zoo.mlx.utils")
    fake_utils.audio_input_capability = lambda *a, **k: SimpleNamespace(
        capable = False, reason = "processor drops audio"
    )
    fake_utils.audio_extractor_sampling_rate = lambda proc: 16000
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.utils", fake_utils)
    monkeypatch.setattr(
        mlx_inference,
        "_render_registered_vlm_prompt",
        lambda *a, num_audios = 0: "P<audio>" if num_audios else "P",
    )

    assert (
        mlx_inference._classify_mlx_audio_type(
            _audio_model(),
            _audio_processor(),
            True,
            config_audio_type = "audio_vlm",
        )
        is None
    )


@pytest.mark.parametrize(
    ("model_type", "places_marker"),
    [
        ("gemma3n", True),
        ("gemma4", True),
        ("phi4mm", True),
        ("minicpmo", True),
        ("qwen3_omni_moe", True),
    ],
)
def test_mlx_vlm_renderer_audio_marker_contract(model_type, places_marker):
    """Pins which mlx-vlm message builders honour num_audios.

    This is the message-construction layer, not the final template render:
    Every currently registered native-audio formatter preserves the requested
    audio count under mlx-vlm 0.6.10.
    """
    pu = pytest.importorskip("mlx_vlm.prompt_utils")
    render = lambda n: pu.apply_chat_template(
        None,
        {"model_type": model_type},
        "hi",
        num_images = 0,
        num_audios = n,
        return_messages = True,
    )
    assert (render(1) != render(0)) is places_marker


def test_mlx_registered_renderer_accepts_published_nemotron_model_type_case():
    """The official checkpoint capitalizes its model type while mlx-vlm's
    registry uses lowercase. Unsloth must reach the registered renderer rather
    than rejecting the checkpoint before the loader's normalization can run."""
    # Real mlx-vlm and Zoo, like the renderer contract test above. Bare
    # backend CI ships neither, so skip rather than error.
    pytest.importorskip("mlx_vlm.prompt_utils")
    loader = pytest.importorskip("unsloth_zoo.mlx.loader")
    from core.inference.mlx_inference import _render_registered_vlm_prompt

    loader._ensure_vlm_prompt_utils_patched()
    published = "NemotronH_Nano_Omni_Reasoning_V3"
    config = {"model_type": published}
    model = SimpleNamespace(config = config)
    messages = [{"role": "user", "content": "Transcribe this audio."}]

    plain = _render_registered_vlm_prompt(
        None,
        model,
        messages,
        num_images = 0,
        num_audios = 0,
    )
    marked = _render_registered_vlm_prompt(
        None,
        model,
        messages,
        num_images = 0,
        num_audios = 1,
    )

    assert plain and marked and plain != marked
    # Later capability and export logic must not see a value the config
    # never carried.
    assert config["model_type"] == published
    assert model.config is config


@pytest.mark.parametrize(
    ("published", "resolves"),
    [
        ("NemotronH_Nano_Omni_Reasoning_V3", True),  # the real published case
        ("SMOLVLM", True),  # ordinary ASCII case shift
        ("smolvlm", True),  # already canonical
        ("ſmolvlm", False),  # casefold("ſ") == "s"
        ("smolvlẞ", False),  # casefold("ẞ") == "ss"
        ("no_such_renderer_anywhere", False),
    ],
)
def test_mlx_renderer_canonicalization_is_ascii_only(monkeypatch, published, resolves):
    """casefold maps U+017F onto "s" and U+1E9E onto "ss", so a checkpoint
    publishing those would reach a renderer it never named."""
    from core.inference.mlx_inference import _render_registered_vlm_prompt

    rendered = []
    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.MODEL_CONFIG = {"smolvlm": object(), "nemotronh_nano_omni_reasoning_v3": object()}

    def _apply_chat_template(_processor, config, _messages, **kwargs):
        rendered.append(config["model_type"])
        return f"prompt<{config['model_type']}>"

    prompt_utils.apply_chat_template = _apply_chat_template
    fake = types.ModuleType("mlx_vlm")
    fake.prompt_utils = prompt_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    model = SimpleNamespace(config = {"model_type": published})
    out = _render_registered_vlm_prompt(
        None,
        model,
        [{"role": "user", "content": "hi"}],
        num_images = 0,
    )

    if resolves:
        assert out and rendered and rendered[0].isascii() and rendered[0].islower()
    else:
        assert out is None and rendered == []
    # Never mutate what the checkpoint published.
    assert model.config["model_type"] == published


def test_mlx_renderer_refuses_ambiguous_case_insensitive_registry(monkeypatch):
    """Two keys folding to the same value is not evidence for either one."""
    from core.inference.mlx_inference import _render_registered_vlm_prompt

    prompt_utils = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.MODEL_CONFIG = {"dupe_model": object(), "Dupe_Model": object()}
    prompt_utils.apply_chat_template = lambda *a, **k: "should not be reached"
    fake = types.ModuleType("mlx_vlm")
    fake.prompt_utils = prompt_utils
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_utils)

    model = SimpleNamespace(config = {"model_type": "DUPE_MODEL"})
    assert (
        _render_registered_vlm_prompt(
            None,
            model,
            [{"role": "user", "content": "hi"}],
            num_images = 0,
        )
        is None
    )


def test_mlx_generate_audio_input_deltas_and_reject(monkeypatch):
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    calls = {}

    stats = dict(prompt_tokens = 3, prompt_tps = 1.0, generation_tokens = 3, generation_tps = 1.0)

    def _fake_stream(model, processor, prompt, **kwargs):
        calls["kwargs"] = kwargs
        for text in ("H", "e", "l"):
            yield SimpleNamespace(text = text, **stats)

    def _fake_render(
        processor,
        model,
        messages,
        num_images = 0,
        num_audios = 0,
    ):
        calls["audios"], calls["messages"] = num_audios, messages
        return "P<audio>"

    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.stream_generate = _fake_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setattr(mlx_inference, "_render_registered_vlm_prompt", _fake_render)

    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._generation_lock = __import__("threading").Lock()
    backend._model, backend._processor = _audio_model(), _audio_processor()
    backend.active_model_name = "m"
    backend.last_generation_stats = None
    backend.models, audio = {"m": {"audio_type": "audio_vlm"}}, [0.0, 0.1, -0.1]
    turn = [{"role": "user", "content": "what is said?"}]
    args = dict(messages = turn, system_prompt = "", audio_array = audio, max_new_tokens = 64)

    # Deltas (never cumulative "HHeHel"); waveform passthrough; greedy parity.
    assert list(backend.generate_audio_input_response(**args)) == ["H", "e", "l"]
    assert calls["kwargs"]["audio"] == [audio] and calls["kwargs"]["temperature"] == 0.0
    assert calls["audios"] == 1 and backend.last_generation_stats is not None

    # Audio-only current turn → transcribe default, never older-turn text.
    args["messages"] = [
        {"role": "user", "content": "old unrelated question"},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    list(backend.generate_audio_input_response(**args))
    assert "Please transcribe this audio." in str(calls["messages"])
    assert "old unrelated question" not in str(calls["messages"])

    backend.models["m"]["audio_type"] = None
    with pytest.raises(RuntimeError, match = "not supported .* MLX"):
        next(backend.generate_audio_input_response(**args))


def test_mlx_audio_input_normalizes_split_native_reasoning_channels(monkeypatch):
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    stats = dict(prompt_tokens = 3, prompt_tps = 1.0, generation_tokens = 6, generation_tps = 1.0)

    def _fake_stream(*_args, **_kwargs):
        for text in ("<|chan", "nel>thought\n", "transcript", "<chan", "nel|>", " answer"):
            yield SimpleNamespace(text = text, **stats)

    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.stream_generate = _fake_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setattr(
        mlx_inference,
        "_render_registered_vlm_prompt",
        lambda *a, num_images = 0, num_audios = 0: "P<audio>",
    )

    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._generation_lock = __import__("threading").Lock()
    # Protocol selection follows the template, never a Gemma model-name branch.
    backend._model = _audio_model(model_type = "template_declared_audio")
    backend._processor = _audio_processor()
    backend._processor.chat_template = "...<|channel>thought\n...<channel|>"
    backend.active_model_name = "m"
    backend.last_generation_stats = None
    backend.models = {"m": {"audio_type": "audio_vlm"}}

    assert list(
        backend.generate_audio_input_response(
            messages = [{"role": "user", "content": "transcribe"}],
            system_prompt = "",
            audio_array = [0.0],
            max_new_tokens = 8,
        )
    ) == ["<think>", "transcript", "</think>", " answer"]

    def _open_stream(*_args, **_kwargs):
        for text in ("<|channel>thought\n", "unfinished"):
            yield SimpleNamespace(text = text, **stats)

    fake_vlm.stream_generate = _open_stream
    assert list(
        backend.generate_audio_input_response(
            messages = [{"role": "user", "content": "transcribe"}],
            system_prompt = "",
            audio_array = [0.0],
        )
    ) == ["<think>", "unfinished", "</think>"]

    cancelled = __import__("threading").Event()
    cancelled.set()
    assert list(
        backend.generate_audio_input_response(
            messages = [{"role": "user", "content": "transcribe"}],
            system_prompt = "",
            audio_array = [0.0],
            cancel_event = cancelled,
        )
    ) == ["<think>"]


def test_mlx_audio_input_honors_adapter_selection(monkeypatch):
    """Base-vs-LoRA compare sends audio_base64 and use_adapter in one body.

    The audio stream has to enter _temporary_mlx_adapter_state like the text and
    vision paths, or the base side silently runs the loaded adapter.
    """
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    seen = {}

    @contextlib.contextmanager
    def _fake_adapter_state(model, use_adapter):
        seen["use_adapter"] = use_adapter
        seen["entered"] = True
        yield
        seen["exited"] = True

    def _fake_stream(model, processor, prompt, **kwargs):
        seen["inside"] = seen.get("entered") and not seen.get("exited")
        yield SimpleNamespace(
            text = "x",
            prompt_tokens = 1,
            prompt_tps = 1.0,
            generation_tokens = 1,
            generation_tps = 1.0,
        )

    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.stream_generate = _fake_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setattr(
        mlx_inference,
        "_render_registered_vlm_prompt",
        lambda *a, num_images = 0, num_audios = 0: "P<audio>",
    )
    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _fake_adapter_state)

    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._generation_lock = __import__("threading").Lock()
    backend._model, backend._processor = _audio_model(), _audio_processor()
    backend.active_model_name = "m"
    backend.last_generation_stats = None
    backend.models = {"m": {"audio_type": "audio_vlm"}}

    list(
        backend.generate_audio_input_response(
            messages = [{"role": "user", "content": "hi"}],
            system_prompt = "",
            audio_array = [0.0],
            max_new_tokens = 8,
            use_adapter = False,
        )
    )
    assert seen["use_adapter"] is False
    # Held for the whole stream, and restored afterwards.
    assert seen["inside"] is True and seen["exited"] is True


def test_worker_forwards_use_adapter_on_the_audio_command():
    """The wire carries the selection only when the caller set it."""
    import inspect

    from core.inference import worker

    src = inspect.getsource(worker._handle_generate_audio_input)
    assert 'cmd.get("use_adapter")' in src
    assert '"use_adapter"' in src


def test_kv_quant_status_applies_only_when_eligible_and_notes_vlm_cost(monkeypatch):
    from core.inference import mlx_inference

    monkeypatch.setattr(
        mlx_inference, "_kv_quant_eligibility", lambda m, v, b = None: ("full", "", True)
    )
    # Unset stays unset: no kwarg, so generation is byte-identical to today.
    assert mlx_inference._kv_quant_status(None, object(), False)["kv_bits"] is None
    # Out of domain degrades rather than raising.
    assert mlx_inference._normalize_mlx_kv_bits(7) is None
    assert mlx_inference._normalize_mlx_kv_bits("eight") is None
    assert mlx_inference._normalize_mlx_kv_bits(8) == 8
    # Every width the runtime accepts is offered; mlx-lm adds no domain of its own.
    assert [mlx_inference._normalize_mlx_kv_bits(b) for b in (2, 3, 5, 6)] == [2, 3, 5, 6]

    text = mlx_inference._kv_quant_status(8, object(), False)
    vlm = mlx_inference._kv_quant_status(8, object(), True)
    assert text["kv_bits"] == 8 and not text["note"]
    # The threshold caveat rides with the resolved value, so an API client sees it.
    assert vlm["kv_bits"] == 8 and "vision models" in vlm["note"]
    assert str(mlx_inference._vlm_quantized_kv_start()) in vlm["note"]

    monkeypatch.setattr(
        mlx_inference,
        "_kv_quant_eligibility",
        lambda m, v, b = None: ("refused", "rotating cache", True),
    )
    refused = mlx_inference._kv_quant_status(8, object(), False)
    assert refused["kv_bits"] is None and refused["eligibility"] == "refused"
    assert refused["requested_kv_bits"] == 8  # what the reload decision compares


def _tiny_lm(cache_factory, dim = 128):
    """Minimal model whose forward populates whatever cache it is given."""
    import mlx.core as mx

    class _LM:
        def make_cache(self):
            return cache_factory()

        def __call__(
            self,
            inputs,
            cache = None,
        ):
            for entry in cache or ():
                target = getattr(entry, "update_and_fetch", None)
                if target is not None:
                    k = mx.zeros((1, 2, inputs.shape[1], dim))
                    target(k, k)
            return mx.zeros((1, inputs.shape[1], 8))

    return _LM()


def test_reload_comparison_and_response_carry_the_resolved_setting():
    """A load-time knob must force a reload and reach the client.

    Both were silently broken: the comparator tripped on a mirror key CUDA
    never stored, and the response fields sat on a request model.
    """
    from routes.inference import _mlx_runtime_settings_match
    from models.inference import LoadRequest, LoadResponse

    # The real request model, not a stand-in, so it stays in step with real loads.
    def req(**knobs):
        return LoadRequest(model = "m", model_path = "m", **knobs)

    be = SimpleNamespace(active_model_name = "m", models = {"m": {"mlx_kv_bits_requested": 8}})
    assert _mlx_runtime_settings_match(be, req(mlx_kv_bits = 8))
    assert not _mlx_runtime_settings_match(be, req(mlx_kv_bits = 4))
    be.models["m"] = {"mlx_kv_bits_requested": None}
    assert _mlx_runtime_settings_match(be, req(mlx_kv_bits = 7))  # both normalize away
    assert _mlx_runtime_settings_match(
        SimpleNamespace(active_model_name = "m", models = {"m": {}}),
        req(mlx_kv_bits = 8),
    )

    # Compared against the REQUESTED value, or a refused template reloads every request.
    be.models["m"] = {
        "mlx_kv_bits_requested": None,
        "chat_template_override_requested": "{{ custom }}",
    }
    assert _mlx_runtime_settings_match(be, req(chat_template_override = "{{ custom }}"))
    assert not _mlx_runtime_settings_match(be, req(chat_template_override = "{{ other }}"))
    assert not _mlx_runtime_settings_match(be, req())
    # A refusal records the request, so the same request still matches.
    be.models["m"] = {
        "mlx_kv_bits_requested": None,
        "chat_template_override_requested": "{{ refused }}",
        "chat_template_override_reason": "it could not render a conversation",
    }
    assert _mlx_runtime_settings_match(be, req(chat_template_override = "{{ refused }}"))

    resp = LoadResponse(
        status = "loaded",
        model = "m",
        display_name = "m",
        inference = {},
        mlx_kv_bits = 8,
        mlx_kv_quant_eligibility = "full",
        mlx_kv_quant_note = "n",
    )
    assert resp.mlx_kv_bits == 8 and resp.mlx_kv_quant_note == "n"


def test_kv_quant_probe_reports_what_the_runtime_would_really_do(monkeypatch):
    """Attempt the conversion instead of predicting it from config or names.

    Static proxies were wrong both ways: a declared head_dim a model ignores,
    and windows spelled differently from `max_size`.
    """
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import cache as lm_cache
    from core.inference import mlx_inference

    def elig(factory, dim = 128):
        lm = _tiny_lm(factory, dim)
        monkeypatch.setattr(lm_cache, "make_prompt_cache", lambda m, **_: lm.make_cache())
        return mlx_inference._kv_quant_eligibility(lm, False, 8)[0]

    assert elig(lambda: [lm_cache.KVCache(), lm_cache.KVCache()]) == "full"
    # A width mx.quantize rejects is caught by attempting it, not by naming it.
    assert elig(lambda: [lm_cache.KVCache()], dim = 80) == "refused"
    # A container the quantizer never descends into is skipped, not fatal.
    assert elig(lambda: [lm_cache.CacheList(lm_cache.KVCache())]) == "none"
    # Mixed quantizable/non-quantizable is a real success, reported as partial.
    assert elig(lambda: [lm_cache.KVCache(), lm_cache.CacheList(lm_cache.KVCache())]) == "partial"


def test_parent_mirror_omits_runtime_fields_a_backend_never_reported():
    """A CUDA load must not gain a None the reload comparison then trips on."""
    from core.inference.orchestrator import _mlx_runtime_mirror_fields

    assert _mlx_runtime_mirror_fields({"is_vision": False}) == {}
    assert _mlx_runtime_mirror_fields(
        {"mlx_kv_bits": 8, "mlx_kv_bits_requested": 8, "other": 1}
    ) == {"mlx_kv_bits": 8, "mlx_kv_bits_requested": 8}


def test_chat_template_override_installs_only_where_one_already_exists():
    """Replace a template; never create one.

    Both render selectors choose their target by whether a template is
    present, so creating one where there was none moves the render to a
    different object -- on the vision side, to one that never selects the
    tool_use variant.
    """
    from core.inference import mlx_inference

    tokenizer = SimpleNamespace(chat_template = "native")
    # A processor with no template of its own must be left alone, or
    # chat_render_target would start selecting it.
    processor = SimpleNamespace(chat_template = None, tokenizer = tokenizer)
    status = mlx_inference._install_template_override(
        "custom", tokenizer, processor, lambda: "rendered"
    )
    assert status["applied"] == "custom"
    assert tokenizer.chat_template == "custom"
    assert processor.chat_template is None

    # A processor that already renders from its own template receives it too.
    own = SimpleNamespace(chat_template = "native")
    owner = SimpleNamespace(chat_template = "native", tokenizer = own)
    mlx_inference._install_template_override("custom", own, owner, lambda: "ok")
    assert own.chat_template == "custom" and owner.chat_template == "custom"

    # Unset must not touch anything.
    untouched = SimpleNamespace(chat_template = "native")
    assert (
        mlx_inference._install_template_override(None, untouched, None, lambda: "")["applied"]
        is None
    )
    assert untouched.chat_template == "native"


def test_chat_template_override_reports_each_way_it_cannot_apply():
    from core.inference import mlx_inference

    def reason(
        tok,
        proc = None,
        probe = lambda: "ok",
    ):
        return mlx_inference._install_template_override("custom", tok, proc, probe)["reason"]

    # mlx-lm holds the template as code and prefers it over the attribute.
    callable_tpl = SimpleNamespace(chat_template = "native", _chat_template = lambda: "x")
    assert reason(callable_tpl) == mlx_inference.MLX_TEMPLATE_CALLABLE

    # A named set would lose its tool_use variant if flattened to one string.
    assert reason(SimpleNamespace(chat_template = {"default": "a", "tool_use": "b"})) == (
        mlx_inference.MLX_TEMPLATE_NAMED_SET
    )

    # ...but only on the object that RENDERS. Real models (aya-vision) keep a named set
    # on a nested tokenizer nothing reads. Without apply_chat_template the processor
    # cannot render, so the nested tokenizer's set does veto.
    nested_set = SimpleNamespace(chat_template = {"default": "a"})
    renders_string = SimpleNamespace(
        chat_template = "native", apply_chat_template = lambda *a, **k: "", tokenizer = nested_set
    )
    targets, status = mlx_inference._template_override_status("custom", nested_set, renders_string)
    assert status["reason"] is None
    assert renders_string in targets and nested_set not in targets

    cannot_render = SimpleNamespace(chat_template = "native", tokenizer = nested_set)
    assert (
        mlx_inference._template_override_status("custom", nested_set, cannot_render)[1]["reason"]
        == mlx_inference.MLX_TEMPLATE_NAMED_SET
    )

    # The same for a callable held by an object that does not render.
    nested_callable = SimpleNamespace(chat_template = "t", _chat_template = lambda: "x")
    renders_own = SimpleNamespace(
        chat_template = "native",
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested_callable,
    )
    assert (
        mlx_inference._template_override_status("custom", nested_callable, renders_own)[1]["reason"]
        is None
    )

    # Nothing to replace, processor present: creating one takes the render from
    # mlx-vlm's fallback, which places the markers.
    bare = SimpleNamespace(chat_template = None)
    assert reason(bare, SimpleNamespace(chat_template = None, tokenizer = bare)) == (
        mlx_inference.MLX_TEMPLATE_NO_TARGET
    )

    # A text model has no selector to move and cannot chat without a template.
    for empty in (None, "", "   "):
        blank = SimpleNamespace(chat_template = empty)
        targets, status = mlx_inference._template_override_status("custom", blank, None)
        assert status["reason"] is None, empty
        assert targets == [blank]

    # The probe turns invalid Jinja into one load-time reason, and restores the original.
    broken = SimpleNamespace(chat_template = "native")
    got = reason(broken, probe = lambda: (_ for _ in ()).throw(ValueError("bad tag")))
    assert "could not render" in got and "bad tag" in got
    assert broken.chat_template == "native"


def test_chat_template_override_crosses_both_ipc_hops():
    """The backend runs in a subprocess; the parent rebuilds its entry from an
    enumerated key set at each hop, and /status and the reload check read the
    parent's copy."""
    import inspect

    from core.inference import orchestrator, worker
    from models.inference import LoadRequest

    assert (
        "chat_template_override"
        in inspect.signature(orchestrator.InferenceOrchestrator.load_model).parameters
    )
    orch = inspect.getsource(orchestrator.InferenceOrchestrator.load_model)
    assert '"chat_template_override": chat_template_override' in orch

    # The applied value is not carried: /status and the reload decision key on requested.
    reported = (
        "chat_template_override_requested",
        "chat_template_override_reason",
    )
    worker_source = inspect.getsource(worker)
    # Whole statements, not name presence: each name is a prefix of another here.
    assert (
        'load_kwargs["chat_template_override"] = config.get(\n'
        '                    "chat_template_override"\n'
        "                )"
        in worker_source
        or 'load_kwargs["chat_template_override"] = config.get("chat_template_override")'
        in worker_source
    )
    for name in reported:
        assert f'"{name}",' in worker_source
    assert [n for n in reported if n not in orchestrator._MLX_RUNTIME_MIRROR_FIELDS] == []
    assert "chat_template_override" in LoadRequest.model_fields


def test_template_probe_renders_through_the_path_generation_uses(monkeypatch):
    """The probe must exercise the real renderer, and reject an empty prompt.

    Rendering through the VLM recovery renderer instead would pass for a model
    outside mlx-vlm's family list, because that helper returns None rather than
    raising -- so an unrenderable template would be recorded as applied and
    then throw on every generation.
    """
    from core.inference import mlx_inference

    seen = {}

    def fake_render(target, messages, **kwargs):
        seen["target"] = target
        tpl = getattr(target, "chat_template", None)
        if tpl == "empty":
            return ""
        # Whitespace only, which the vision path also treats as an empty prompt.
        return "  \n " if tpl == "blank" else "rendered"

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        fake_render,
    )
    backend = mlx_inference.MLXInferenceBackend.__new__(mlx_inference.MLXInferenceBackend)

    # Text: the tokenizer is the render target.
    backend._tokenizer = SimpleNamespace(chat_template = "t")
    backend._processor = None
    assert backend._render_template_probe(False) == "rendered"
    assert seen["target"] is backend._tokenizer

    # Vision: whatever chat_render_target selects, not the recovery renderer.
    nested = SimpleNamespace(chat_template = "t")
    processor = SimpleNamespace(
        chat_template = "own", tokenizer = nested, apply_chat_template = lambda *a, **k: ""
    )
    backend._tokenizer = nested
    backend._processor = processor
    backend._render_template_probe(True)
    assert seen["target"] is processor

    # A template that renders nothing is a failure, not a pass.
    for empty in ("empty", "blank"):
        backend._tokenizer = SimpleNamespace(chat_template = empty)
        backend._processor = None
        with pytest.raises(ValueError):
            backend._render_template_probe(False)


def test_the_audio_refusal_puts_the_native_template_back(monkeypatch):
    """The refusal itself, not just the marker check.

    Capability was classified against the native template, so an override that
    stops marking audio has to be undone -- otherwise the model keeps
    advertising an input it can no longer place.
    """
    from core.inference import mlx_inference

    marks = {"audio": False}
    monkeypatch.setattr(
        mlx_inference,
        "_render_registered_vlm_prompt",
        lambda p, m, msgs, imgs, num_audios = 0: (
            "<audio> hi" if num_audios and marks["audio"] else "hi"
        ),
    )
    tok = SimpleNamespace(chat_template = "custom")
    status = {
        "requested": "custom",
        "applied": "custom",
        "reason": None,
        "restore": [(tok, "native")],
    }
    mlx_inference._revoke_override_that_drops_audio(status, object(), object())
    assert status["applied"] is None
    assert status["reason"] == mlx_inference.MLX_TEMPLATE_DROPS_AUDIO
    assert tok.chat_template == "native", "the model must be left as if unset"
    assert status["restore"] == []

    # An override that keeps the marker is left alone.
    marks["audio"] = True
    kept = SimpleNamespace(chat_template = "custom")
    keep = {
        "requested": "custom",
        "applied": "custom",
        "reason": None,
        "restore": [(kept, "native")],
    }
    mlx_inference._revoke_override_that_drops_audio(keep, object(), object())
    assert keep["applied"] == "custom" and kept.chat_template == "custom"

    # Nothing installed: no reason invented even when the marker is absent.
    marks["audio"] = False
    unset = {"requested": None, "applied": None, "reason": None, "restore": []}
    mlx_inference._revoke_override_that_drops_audio(unset, object(), object())
    assert unset["reason"] is None


def test_a_created_template_is_not_reported_as_the_model_default():
    """A model that shipped no template must keep reporting none.

    The override is installed on the tokenizer for text models, so reading the
    live object back would report the user's own template as the model's
    default -- which makes the editor treat it as the default and clear it on
    the next save, leaving the model unchattable again.
    """
    from core.inference import mlx_inference

    backend = mlx_inference.MLXInferenceBackend.__new__(mlx_inference.MLXInferenceBackend)
    tokenizer = SimpleNamespace(chat_template = None)
    backend.models = {"m": {"tokenizer": tokenizer, "processor": None}}

    # What load_model does: capture first, install, then record.
    native = getattr(tokenizer, "chat_template", None)
    tokenizer.chat_template = "{{ user_supplied }}"
    backend._populate_chat_template_info("m", native)

    info = backend.models["m"]["chat_template_info"]
    assert info["template"] is None, "the override is not the model's template"
    assert info["has_template"] is False

    # A model that did ship one still reports its own, not the override.
    shipped = SimpleNamespace(chat_template = "{{ native }}")
    backend.models["n"] = {"tokenizer": shipped, "processor": None}
    native = getattr(shipped, "chat_template", None)
    shipped.chat_template = "{{ user_supplied }}"
    backend._populate_chat_template_info("n", native)
    assert backend.models["n"]["chat_template_info"]["template"] == "{{ native }}"

    # Called without a capture at all, it still reads the live object.
    plain = SimpleNamespace(chat_template = "{{ live }}")
    backend.models["o"] = {"tokenizer": plain, "processor": None}
    backend._populate_chat_template_info("o")
    assert backend.models["o"]["chat_template_info"]["template"] == "{{ live }}"


def _marker_renderer(monkeypatch, marks_image):
    """Stand in for the real render: only marks_image=True reacts to an image part."""

    def render(target, messages, **kwargs):
        parts = messages[0]["content"]
        text = "".join(p["text"] for p in parts if p["type"] == "text")
        images = sum(1 for p in parts if p["type"] == "image")
        return f"<img>{text}" if (marks_image and images) else text

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation", render
    )


def test_an_override_that_stops_marking_images_is_revoked(monkeypatch):
    """A text-only probe renders fine while the image placeholder is gone.

    The failure would otherwise land on the first image request, inside a
    processor that counts markers against image features.
    """
    from core.inference import mlx_inference

    processor = SimpleNamespace(
        chat_template = "{{ native }}",
        apply_chat_template = lambda *a, **k: "",
        tokenizer = SimpleNamespace(chat_template = "{{ nested }}"),
    )
    tokenizer = processor.tokenizer

    _marker_renderer(monkeypatch, marks_image = True)
    assert mlx_inference._image_marker_survives(tokenizer, processor) is True

    status = mlx_inference._install_template_override(
        "{{ blind }}", tokenizer, processor, lambda: None
    )
    assert status["applied"] == "{{ blind }}"

    # The override renders text but never places the image.
    _marker_renderer(monkeypatch, marks_image = False)
    mlx_inference._revoke_override_that_drops_image(status, tokenizer, processor)
    assert status["applied"] is None
    assert status["reason"] == mlx_inference.MLX_TEMPLATE_DROPS_IMAGE
    assert processor.chat_template == "{{ native }}"
    assert tokenizer.chat_template == "{{ nested }}"


def test_an_override_that_keeps_the_image_marker_is_left_alone(monkeypatch):
    from core.inference import mlx_inference

    processor = SimpleNamespace(
        chat_template = "{{ native }}",
        apply_chat_template = lambda *a, **k: "",
        tokenizer = SimpleNamespace(chat_template = "{{ nested }}"),
    )
    _marker_renderer(monkeypatch, marks_image = True)
    status = mlx_inference._install_template_override(
        "{{ good }}", processor.tokenizer, processor, lambda: None
    )
    mlx_inference._revoke_override_that_drops_image(status, processor.tokenizer, processor)
    assert status["applied"] == "{{ good }}"
    assert processor.chat_template == "{{ good }}"


def test_the_model_default_comes_from_the_object_that_renders():
    """A processor owning its own template is what generation reads.

    Reporting the nested tokenizer's instead lets "reset to default" install the
    tokenizer's template over the processor, losing its media and tool variants.
    """
    from core.inference import mlx_inference

    nested = SimpleNamespace(chat_template = "{{ nested }}")
    processor = SimpleNamespace(
        chat_template = "{{ processor }}",
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested,
    )
    assert mlx_inference._native_template_source(nested, processor) is processor

    # No processor template: the nested tokenizer is both render target and default.
    bare = SimpleNamespace(chat_template = None, tokenizer = nested)
    assert mlx_inference._native_template_source(nested, bare) is nested

    # A named set renders but is not an editable default, so report the nested string.
    named = SimpleNamespace(
        chat_template = {"default": "{{ a }}"},
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested,
    )
    assert mlx_inference._native_template_source(nested, named) is nested

    # Text model: no processor at all.
    assert mlx_inference._native_template_source(nested, None) is nested


def test_template_targets_follow_the_object_that_can_actually_render():
    """A processor holding a template it cannot render with is not the target.

    chat_render_target falls back to the nested tokenizer there, so judging
    eligibility against the processor would install onto an object nothing
    reads and then pass a probe rendered from the untouched native template.
    """
    from core.inference import mlx_inference

    nested = SimpleNamespace(chat_template = "{{ nested }}")
    # chat_template but no apply_chat_template: it cannot render.
    unrenderable = SimpleNamespace(chat_template = "{{ processor }}", tokenizer = nested)
    assert mlx_inference._template_render_targets(nested, unrenderable) == [nested]

    renderable = SimpleNamespace(
        chat_template = "{{ processor }}",
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested,
    )
    assert mlx_inference._template_render_targets(nested, renderable) == [renderable]
    assert mlx_inference._template_render_targets(nested, None) == [nested]

    # A named set on the unusable target still reports the nested string.
    named = SimpleNamespace(
        chat_template = {"default": "{{ a }}"},
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested,
    )
    assert mlx_inference._native_template_source(nested, named) is nested


def test_an_entry_the_upstream_lru_cannot_size_is_not_retainable(monkeypatch):
    """Upstream LRUPromptCache.insert_cache sums entry.nbytes itself.

    So an mlx-lm whose QuantizedKVCache.nbytes raises cannot admit the entry
    however else it could be measured, and measuring it another way here would
    drop the no-reuse caveat while every quantized turn still failed to insert.
    """
    from core.inference import mlx_inference

    class _Broken:
        state = ()
        keys = SimpleNamespace(nbytes = 64)
        values = SimpleNamespace(nbytes = 64)

        @property
        def nbytes(self):
            raise NameError("tree_reduce")

    assert mlx_inference._kv_entry_nbytes(_Broken()) is None
    assert mlx_inference._kv_entry_nbytes(SimpleNamespace(nbytes = 128)) == 128

    _install_fake_mlx(monkeypatch)
    mx = sys.modules["mlx.core"]
    mx.random = SimpleNamespace(state = [0])
    mx.array = lambda v: v
    mx.eval = lambda *a: None
    mx.clear_cache = lambda: None

    def probe(quantized):
        entry = SimpleNamespace(
            to_quantized = lambda group_size, bits: quantized,
            max_size = None,
            window_size = None,
            state = (),
        )
        return mlx_inference._kv_quant_probe(lambda *a, **k: None, [entry], 8)

    converted, skipped, failure, retainable = probe(_Broken())
    assert (converted, skipped, failure) == (1, 0, None)
    assert retainable is False

    # A working property is retainable, so the caveat is not blanket.
    assert probe(SimpleNamespace(state = (), nbytes = 128))[3] is True


class _SentinelRandomState:
    """``mx.random.state`` as mlx >= 0.32.1 exposes it: readable, not writable."""

    def __init__(self, words):
        self._words = list(words)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index not in (0, -1):
            raise IndexError("random state index out of range")
        return SimpleNamespace(tolist = lambda: list(self._words))

    def __iter__(self):
        return iter([self[0]])


def test_kv_quant_probe_rewinds_the_rng_without_assigning_to_the_state(monkeypatch):
    """mlx 0.32.1 made mx.random.state a sentinel with no __setitem__, so the
    assignment this used to do raised out of the probe's finally and failed the
    whole model load.
    """
    from core.inference import mlx_inference

    _install_fake_mlx(monkeypatch)
    mx = sys.modules["mlx.core"]
    # High word's top bit set, so a rewind that packs the seed as signed is visible.
    words = (0xFEDCBA98, 0x76543210)
    events = []
    mx.random = SimpleNamespace(
        state = _SentinelRandomState(words),
        seed = lambda value: events.append(("seed", value)),
    )
    mx.array = lambda v: v
    mx.eval = lambda *a: None
    mx.clear_cache = lambda: None

    def language_model(*args, **kwargs):
        events.append(("forward", None))

    def to_quantized(group_size, bits):
        events.append(("convert", None))
        return SimpleNamespace(state = (), nbytes = 128)

    entry = SimpleNamespace(
        to_quantized = to_quantized,
        max_size = None,
        window_size = None,
        state = (),
    )
    outcome = mlx_inference._kv_quant_probe(language_model, [entry], 8)

    assert outcome == (1, 0, None, True)
    # A rewind that ran before the forward pass would leave the probe's own
    # draws in the stream the caller goes on to sample.
    assert events == [
        ("forward", None),
        ("convert", None),
        ("seed", (words[0] << 32) | words[1]),
    ]


def test_a_successful_override_does_not_pin_the_tokenizer_past_load(monkeypatch):
    """The restore pairs hold the tokenizer, so unload_model cannot free it.

    Nothing reads them once the audio and image checks have run, and the worker
    outlives the model, so holding them defeats part of what unload releases.
    """
    _install_fake_mlx(monkeypatch)
    _install_fake_fast_mlx(monkeypatch, [])
    # A text model with no template of its own is the one case an override may create.
    monkeypatch.setattr(_DummyTokenizer, "chat_template", None, raising = False)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda target, messages, **kwargs: "rendered",
    )
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    config = SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = False)
    assert backend.load_model(config, chat_template_override = "{{ custom }}")

    status = backend._template_override
    assert status["applied"] == "{{ custom }}"
    assert status["restore"] == []
    assert backend._tokenizer.chat_template == "{{ custom }}"


def test_an_override_that_renders_the_image_as_prose_is_not_a_marker(monkeypatch):
    """A difference between the two probes is not proof of a placeholder.

    A template emitting "Image attached", or the content dict itself, renders
    differently from the text-only probe while placing nothing the processor
    can bind the image to.
    """
    from core.inference import mlx_inference

    nested = SimpleNamespace(chat_template = "{{ nested }}")
    processor = SimpleNamespace(
        chat_template = "{{ native }}",
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested,
        image_token = "<|image_pad|>",
    )
    assert mlx_inference._image_placeholder(nested, processor) == "<|image_pad|>"

    def render_as(text_for_image):
        def render(target, messages, **kwargs):
            parts = messages[0]["content"]
            text = "".join(p["text"] for p in parts if p["type"] == "text")
            images = sum(1 for p in parts if p["type"] == "image")
            return (text_for_image * images) + text

        monkeypatch.setattr(
            "core.inference.chat_template_helpers.apply_chat_template_for_generation", render
        )

    survives = lambda: mlx_inference._image_marker_survives(nested, processor, "<|image_pad|>")

    render_as("<|image_pad|>")
    assert survives() is True

    # Differs from the text-only render, but places no placeholder.
    render_as("Image attached. ")
    assert survives() is False

    # Ignores the image entirely.
    render_as("")
    assert survives() is False

    # Emits the structured content object, which _vlm_prompt_issue already names.
    render_as(str({"type": "image"}))
    assert survives() is False

    # A model naming no placeholder still gets the weaker difference test.
    del processor.image_token
    assert mlx_inference._image_placeholder(nested, processor) is None
    render_as("Image attached. ")
    assert mlx_inference._image_marker_survives(nested, processor, None) is True


def test_the_load_response_declares_every_runtime_field_status_reports():
    """A field only /status declares is silently dropped from a load response.

    Pydantic ignores an extra keyword, so a route can construct the echo and
    the client still sees nothing, with no error anywhere to notice it by.
    """
    from models.inference import (
        InferenceStatusResponse,
        LoadResponse,
        _InferenceRuntimeFields,
    )

    shared = set(_InferenceRuntimeFields.model_fields)
    runtime = {
        name
        for name in InferenceStatusResponse.model_fields
        if name.startswith(("mlx_", "chat_template_override")) or name == "is_mlx"
    }
    assert runtime <= shared, f"declared on status only: {sorted(runtime - shared)}"
    assert runtime <= set(LoadResponse.model_fields)

    loaded = LoadResponse(
        success = True,
        status = "loaded",
        model = "m",
        display_name = "m",
        inference = {},
        chat_template_override = "{{ custom }}",
        chat_template_override_reason = "why",
    )
    assert loaded.model_dump()["chat_template_override"] == "{{ custom }}"


def test_a_vision_override_is_checked_even_when_the_native_render_needs_recovery(monkeypatch):
    """Gating the check on the native template skipped the models that need it.

    A native template that places nothing still renders images, because
    _generate_vlm falls back to the registered renderer once _vlm_prompt_issue
    fires. An override rendering plain text fires nothing, so the image goes
    unplaced with no error, and recovery is unavailable anyway once tools or
    reasoning controls are set.
    """
    _install_fake_mlx(monkeypatch)
    _install_fake_fast_mlx(monkeypatch, [])
    monkeypatch.setattr(_DummyProcessor, "chat_template", "{{ native }}", raising = False)
    monkeypatch.setattr(_DummyProcessor, "apply_chat_template", lambda *a, **k: "", raising = False)
    monkeypatch.setattr(_DummyTokenizer, "chat_template", "{{ nested }}", raising = False)

    def render(target, messages, **kwargs):
        # The native template serializes the content dict, so it only works via recovery.
        content = messages[0]["content"]
        if isinstance(content, str):
            return content
        text = "".join(p["text"] for p in content if p["type"] == "text")
        images = [p for p in content if p["type"] == "image"]
        if getattr(target, "chat_template", None) == "{{ blind }}":
            return text
        return "".join(str(p) for p in images) + text

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation", render
    )
    from core.inference import mlx_inference

    # The native template fails the check, which used to skip the override's.
    assert mlx_inference._image_marker_survives(_DummyTokenizer(), _DummyProcessor(), None) is False

    backend = mlx_inference.MLXInferenceBackend()
    config = SimpleNamespace(identifier = "fake/vlm", is_vision = True, is_lora = False)
    assert backend.load_model(config, chat_template_override = "{{ blind }}")

    assert backend._template_override["applied"] is None
    assert backend._template_override["reason"] == mlx_inference.MLX_TEMPLATE_DROPS_IMAGE
    assert backend._processor.chat_template == "{{ native }}"


# ── Per-request seed ────────────────────────────────────────────────


def test_vlm_seed_rides_on_the_sampler_not_a_seed_kwarg(monkeypatch):
    """The pinned mlx-vlm has no seed parameter, and a newer one ignores it at
    Unsloth's default min_p/top_k -- so the seed must ride on the sampler, built
    with the whole filtering chain the runtime would otherwise have built."""
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    seen, built = [], []
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.prompt_utils = SimpleNamespace(
        MODEL_CONFIG = {}, apply_chat_template = lambda *_a, **_k: "<image> prompt"
    )

    def _vlm_stream(*_args, **kwargs):
        seen.append(kwargs)
        yield SimpleNamespace(text = "ok", prompt_tokens = 1, generation_tokens = 1)

    mlx_vlm.stream_generate = _vlm_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_a, **_k: "<image> prompt",
    )
    monkeypatch.setattr(
        mlx_inference, "_temporary_mlx_adapter_state", lambda *_a, **_k: contextlib.nullcontext()
    )
    # Stubbed so the wiring is checked without an mlx wheel: what matters here is
    # which seed and stages reach the builder, not the array maths inside it.
    monkeypatch.setattr(
        mlx_inference,
        "_make_seeded_mlx_sampler",
        lambda seed, **kw: built.append((seed, kw)) or (lambda _l: _l),
    )
    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "generic_vlm"})
    backend._processor = SimpleNamespace(chat_template = "template")
    args = (
        [{"role": "user", "content": [{"type": "image"}]}],
        object(),
        0.7,
        0.9,
        40,
        0.01,
        4,
        1.0,
        None,
    )

    assert list(backend._generate_vlm(*args, seed = 4242)) == ["ok"]
    assert callable(seen[-1]["sampler"]) and "seed" not in seen[-1]
    assert built == [(4242, {"temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.01})]
    assert list(backend._generate_vlm(*args)) == ["ok"]
    assert "sampler" not in seen[-1] and len(built) == 1


@pytest.mark.parametrize(
    "factory_name, factory_args, vocab, sequence, expected",
    [
        # Frequency counts multiplicity where presence charges once.
        (
            "_make_mlx_frequency_penalty_processor",
            (0.5,),
            20,
            [10, 11, 5, 5, 5, 6, 99, -1],
            {5: -1.5, 6: -0.5, 10: 0.0, 19: 0.0, 0: 0.0},
        ),
        # Bias is history-free, so it also applies on the prompt-only first call.
        (
            "_make_mlx_logit_bias_processor",
            ({1: 4.0, 3: -2.5, 99: 100.0, -1: 100.0},),
            8,
            [10, 11],
            {1: 4.0, 3: -2.5, 0: 0.0, 7: 0.0},
        ),
    ],
)
def test_mlx_processors_penalize_in_range_ids_and_route_strays_away(
    factory_name, factory_args, vocab, sequence, expected
):
    # MLX does no bounds checking, so a stray id is undefined behaviour.
    mx = pytest.importorskip("mlx.core")
    from core.inference import mlx_inference

    proc = getattr(mlx_inference, factory_name)(*factory_args)
    proc(mx.array([10, 11]), mx.zeros((1, vocab)))  # first call latches prompt_len
    out = proc(mx.array(sequence), mx.zeros((1, vocab)))
    for token, value in expected.items():
        assert float(out[0, token]) == pytest.approx(value), token


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(config = {"model_type": "generic_vlm", "eos_token_id": [1, 2]}),
        SimpleNamespace(config = SimpleNamespace(eos_token_id = [1, 2])),
        SimpleNamespace(_config = {"eos_token_id": [1, 2]}),
        SimpleNamespace(_config = SimpleNamespace(eos_token_id = [1, 2])),
        SimpleNamespace(config = {"model_type": "g"}, _config = {"eos_token_id": [1, 2]}),
    ],
    ids = ["dict-config", "object-config", "dict-_config", "object-_config", "config-then-_config"],
)
def test_eos_ids_are_read_from_every_shape_a_config_arrives_in(model):
    """A checkpoint's config is a dict on some models and an object on others,
    under config or _config -- the spread _mlx_vlm_model_config already walks. A
    getattr-only read silently fell through to the tokenizer, which is the source
    the priority order exists to outrank, so an EOS sampled on the last allowed
    token was reported as truncation."""
    from core.inference.mlx_inference import _mlx_finish_reason, _mlx_stop_token_ids

    tokenizer = SimpleNamespace(eos_token_id = 9)  # disagrees with the config
    stop_ids = _mlx_stop_token_ids(tokenizer, model)
    assert stop_ids == (1, 2)
    at_cap = SimpleNamespace(finish_reason = None, token = 2)
    assert _mlx_finish_reason(at_cap, stop_ids, 10, 10) == "stop"


def test_the_tokenizer_is_still_the_fallback_when_no_config_carries_eos():
    from core.inference.mlx_inference import _mlx_stop_token_ids

    tokenizer = SimpleNamespace(eos_token_id = 9)
    assert _mlx_stop_token_ids(tokenizer, SimpleNamespace(config = {"model_type": "g"})) == (9,)
    assert _mlx_stop_token_ids(tokenizer, None) == (9,)


def test_finish_reason_separates_truncation_from_natural_end():
    """At the limit the count alone is ambiguous -- a stop token sampled as the
    final allowed token looks identical to exhaustion -- so the last token
    decides, against the ids read from the source the runtime stops on. Those
    sources disagree on real repos: Kimi-VL lists two config ids and a different
    tokenizer id, and each may be a bare int (Qwen2-VL) rather than a list."""
    from core.inference.mlx_inference import _mlx_finish_reason, _mlx_stop_token_ids

    model = SimpleNamespace(config = SimpleNamespace(eos_token_id = [163584, 163586]))
    ids = _mlx_stop_token_ids(SimpleNamespace(eos_token_ids = 163594), model)
    assert ids == (163584, 163586)
    assert _mlx_stop_token_ids(SimpleNamespace(eos_token_ids = 151645)) == (151645,)
    assert _mlx_stop_token_ids(SimpleNamespace()) == ()
    assert _mlx_finish_reason(SimpleNamespace(token = 5), ids, 3, 8) == "stop"
    assert _mlx_finish_reason(SimpleNamespace(token = 5), ids, 8, 8) == "length"
    assert _mlx_finish_reason(SimpleNamespace(token = 163584), ids, 8, 8) == "stop"


# ── Stop sequences ──────────────────────────────────────────────────


def test_stop_sequences_cut_the_reply_and_never_show_a_partial_match():
    """The matcher decides how much of a reply may be shown: text that could still
    grow into a sequence is held back, since a client cannot unsee a fragment the
    next token completes."""
    from core.inference.mlx_inference import _mlx_stop_cut

    # Held back while it could still grow into "ab"; released once it cannot.
    assert [_mlx_stop_cut(t, ["ab"]) for t in ("xa", "xac", "xab")] == [
        (1, False),
        (3, False),
        (1, True),
    ]
    # A sequence at position 0 ends the turn with nothing shown.
    assert _mlx_stop_cut("abc", ["a"]) == (0, True)
    # The longest partial across all sequences wins; a shorter one cannot release it.
    assert _mlx_stop_cut("aaAB", ["ABC", "BX"]) == (2, False)
    # The earliest match ends the turn, whether one sequence matches twice or two
    # sequences match in a different order than they were declared.
    assert _mlx_stop_cut("a then a", ["a"]) == (0, True)
    assert _mlx_stop_cut("early late", ["late", "early"]) == (0, True)
    # An unresolved character is not a character yet: it can neither be shown nor
    # complete a sequence, and dropping it can uncover the start of one. Only the
    # trailing run is unresolved -- one the reply has already written past is text.
    assert _mlx_stop_cut("hi\ufffd", ["END"]) == (2, False)
    assert _mlx_stop_cut("a\ufffd", ["a"]) == (0, True)
    assert _mlx_stop_cut("a\ufffd\ufffd", ["\ufffd"]) == (1, False)
    assert _mlx_stop_cut("a\ufffdb", ["\ufffd"]) == (1, True)


def _fake_rng_state(monkeypatch, words):
    from core.inference import mlx_inference

    _install_fake_mlx(monkeypatch)
    mx = sys.modules["mlx.core"]
    seeded = []
    mx.random = SimpleNamespace(
        state = _SentinelRandomState(words),
        seed = lambda value: seeded.append(value),
    )
    return mlx_inference, seeded


@pytest.mark.parametrize(
    "words,expected",
    [
        ((-1, -2), (0xFFFFFFFF, 0xFFFFFFFE)),
        ((-2147483648, 5), (0x80000000, 5)),
        ((0, -1), (0, 0xFFFFFFFF)),
        ((0, 0), (0, 0)),
        ((0xFFFFFFFF, 0xFFFFFFFF), (0xFFFFFFFF, 0xFFFFFFFF)),
    ],
)
def test_rng_capture_reinterprets_signed_words(monkeypatch, words, expected):
    """A negative word is the two's complement of the uint32 mlx stores.

    Reinterpreting it loses nothing, and it is what keeps the seed inside the
    uint64 domain. The rewind is deliberately unguarded, which only holds if the
    words cannot put it out of range; capture does not type-check the state, so
    this conversion is what makes that true. A raise would land in the probe's
    finally and replace the probe's own outcome, the failure shape #9478 set out
    to remove.
    """
    mlx_inference, seeded = _fake_rng_state(monkeypatch, words)

    captured = mlx_inference._mlx_rng_key_words()
    assert captured == expected

    mlx_inference._restore_mlx_rng_key(captured)
    assert seeded == [(expected[0] << 32) | expected[1]]
    assert 0 <= seeded[0] < 2**64


@pytest.mark.parametrize(
    "words", [(2**32, 0), (0, 2**32), (2**63, 1), (-(2**31) - 1, 0), (0, -(2**40))]
)
def test_rng_capture_declines_words_that_are_not_32_bit(monkeypatch, words):
    """Masking these would be worse than declining them.

    (2**32, 0) masks to (0, 0): a key we cannot represent becomes a plausible
    wrong one, the probe reports success, and sampling silently diverges from an
    unprobed run. Declining is the outcome the caller already handles, and it is
    the only one that says so out loud.
    """
    mlx_inference, seeded = _fake_rng_state(monkeypatch, words)
    warnings = _capture_rng_warnings(monkeypatch, mlx_inference)

    assert mlx_inference._mlx_rng_key_words() is None
    assert any("32-bit word" in w for w in warnings), warnings

    # The rewind must stay a no-op on the value capture actually returns, and
    # total besides: handed these words directly it declines rather than raising
    # into the probe's finally, and never seeds a wrong key.
    mlx_inference._restore_mlx_rng_key(None)
    mlx_inference._restore_mlx_rng_key(words)
    assert seeded == []


def _capture_rng_warnings(monkeypatch, mlx_inference):
    """Collect this module's warnings. It logs through structlog, which caplog
    does not see."""
    warnings = []
    monkeypatch.setattr(
        mlx_inference.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(msg % args if args else msg),
    )
    return warnings


@pytest.mark.parametrize("n", [1, 3, 4])
def test_rng_capture_reports_a_key_that_is_not_two_words(monkeypatch, n):
    """Returning a bare None would leave the probe silently not restoring, which
    is the same shape of silent divergence the item assignment used to cause,
    just moved from the write to the read."""
    from core.inference import mlx_inference

    _install_fake_mlx(monkeypatch)
    mx = sys.modules["mlx.core"]
    mx.random = SimpleNamespace(
        state = _SentinelRandomState(tuple(range(n))),
        seed = lambda value: None,
    )
    warnings = _capture_rng_warnings(monkeypatch, mlx_inference)

    assert mlx_inference._mlx_rng_key_words() is None
    assert any("random key" in w for w in warnings), warnings


def test_rng_capture_stays_quiet_when_the_state_cannot_be_read(monkeypatch):
    """An unreadable state is an intentional no-op, not a surprise. Warning on it
    every call would train operators to ignore the warning that matters."""
    from core.inference import mlx_inference

    _install_fake_mlx(monkeypatch)
    mx = sys.modules["mlx.core"]
    mx.random = SimpleNamespace(state = lambda: {"counter": 0}, seed = lambda value: None)
    warnings = _capture_rng_warnings(monkeypatch, mlx_inference)

    assert mlx_inference._mlx_rng_key_words() is None
    assert warnings == []


# Llama 3.1 carries the second beside its scaled window, mlx-lm's Kimi Linear the third.
@pytest.mark.parametrize("name", ["n_ctx", "original_max_position_embeddings", "model_max_length"])
def test_mlx_native_context_length_reads_every_spelling_and_every_config(name):
    from core.inference.mlx_inference import mlx_native_context_length

    # One term per family, matched by shape rather than by name.
    args = SimpleNamespace(**{name: 40960})
    assert mlx_native_context_length(SimpleNamespace(args = args)) == 40960
    # Pre-load metadata answers by the same rule.
    from routes.models import _get_max_position_embeddings

    assert _get_max_position_embeddings(SimpleNamespace(**{name: 40960})) == 40960
    # The wrapper counts too: mlx-vlm's Phi-3-V keeps a 4096 text config beside its real 131072.
    stub = SimpleNamespace(args = SimpleNamespace(**{name: 4096}))
    text = SimpleNamespace(**{name: 4096})
    cfg = SimpleNamespace(model_type = "x", text_config = text, **{name: 131072})
    vlm = SimpleNamespace(language_model = stub, config = cfg)
    assert mlx_native_context_length(vlm) == 131072


# Real enough for the capability classifier to read. The named one is the shape that made
# classifying without tools wrong: its tool markup lives only in the tool_use branch.
_PLAIN_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
_TOOL_TEMPLATE = (
    "{% if tools %}{% for t in tools %}{{ t.function.name }}{% endfor %}{% endif %}"
    "{% for m in messages %}{{ m['content'] }}"
    "{% if m.tool_calls %}<tool_call>{{ m.tool_calls[0].function.name }}</tool_call>{% endif %}"
    "{% endfor %}"
)
_NAMED_TOOL_TEMPLATE = {"default": _PLAIN_TEMPLATE, "tool_use": _TOOL_TEMPLATE}


def _mirror(template = None):
    """The parent's view of a worker-held MLX model, template included."""
    return {
        "mlx/model": {
            "is_mlx": True,
            "is_vision": False,
            "is_audio": False,
            "chat_template_info": {"template": template},
        }
    }


class _RenderRecordingBackend:
    active_model_name = "mlx/model"

    def __init__(self):
        self.messages = self.system = self.tools = None

    def count_chat_tokens(
        self,
        messages,
        system_prompt = "",
        **kwargs,
    ):
        self.messages, self.system = messages, system_prompt
        self.tools = kwargs.get("tools")
        return 11, "mlx/model"


def _count_route(
    monkeypatch,
    backend,
    template = _TOOL_TEMPLATE,
    models = None,
    request = None,
    generations = None,
    **fields,
):
    """Drive the endpoint against `backend`, classifying from a real template."""
    backend_dir = str(Path(__file__).resolve().parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from routes import inference as route

    backend.models = models or _mirror(template)
    monkeypatch.setattr(route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False))
    monkeypatch.setattr(route.active_generations, "count", generations or (lambda: 0))
    return asyncio.run(
        route.chat_count_tokens(
            route.ChatCountTokensRequest(**fields),
            request = request,
            current_subject = "tester",
        )
    )


def test_an_mlx_count_is_served_where_llama_cpp_would_have_refused(monkeypatch):
    """llama.cpp not being loaded used to be the whole answer, for vision models too."""
    from fastapi import HTTPException

    hello = [{"role": "user", "content": "hello"}]
    served = _count_route(monkeypatch, _RenderRecordingBackend(), messages = hello)
    assert json.loads(served.body) == {"input_tokens": 11, "model": "mlx/model"}
    served = _count_route(
        monkeypatch,
        _RenderRecordingBackend(),
        messages = hello,
        models = {"mlx/model": {"is_mlx": True, "is_vision": True, "is_audio": False}},
    )
    assert json.loads(served.body)["input_tokens"] == 11

    with pytest.raises(HTTPException) as refused:
        _count_route(monkeypatch, _RenderRecordingBackend(), messages = [])
    assert refused.value.status_code == 503


@pytest.mark.parametrize(
    "template, expected_tools",
    [
        (_PLAIN_TEMPLATE, None),
        (_TOOL_TEMPLATE, ["web_search"]),
        # Tool markup only in tool_use: classifying with no tools reads `default`.
        (_NAMED_TOOL_TEMPLATE, ["web_search"]),
    ],
    ids = ["renders-none", "renders-tools", "renders-tools-in-a-named-branch"],
)
def test_an_mlx_count_prices_the_tools_the_completion_would_render(
    monkeypatch, template, expected_tools
):
    """`unsloth studio run` defaults tools on, so a count naming none inherits that: mostly
    the nudge's length, but not stale markup the completion strips."""
    from state import tool_policy

    monkeypatch.setattr(tool_policy, "_tool_policy_default", True)
    backend = _RenderRecordingBackend()
    _count_route(
        monkeypatch,
        backend,
        template = template,
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "sure <tool_call>{}</tool_call> done"},
        ],
        enabled_tools = ["web_search"],
    )
    names = None if backend.tools is None else [t["function"]["name"] for t in backend.tools]
    assert names == expected_tools, "the count must render what the completion would"
    if expected_tools is None:
        assert not backend.system, "no tools rendered, so no nudge either"
        return

    from routes.inference import _apply_rag_nudge, _build_tool_action_nudge

    plain = _build_tool_action_nudge(tools = backend.tools, model_name = "mlx/model", full_access = False)
    nudge = asyncio.run(_apply_rag_nudge(plain, backend.tools, rag_scope = None))
    assert backend.system == nudge
    assert backend.messages[-1]["content"] == "sure  done", "stale markup the completion removes"


def test_an_mlx_count_prices_the_relay_the_tool_loop_did_not_claim(monkeypatch):
    """A declined request carrying tool history goes to the relay, which keeps the
    structured tool_calls the extraction flattens away."""
    fn = {"name": "web_search", "arguments": '{"q": "x"}'}
    call = {"id": "c1", "type": "function", "function": fn}
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "a result"},
    ]

    backend = _RenderRecordingBackend()
    _count_route(monkeypatch, backend, messages = history, enable_tools = False)
    assert backend.messages[1].get("tool_calls"), "the relay renders structured calls, not prose"
    assert backend.messages[1]["tool_calls"][0]["function"]["arguments"] == {"q": "x"}
    assert backend.tools is None, "the loop declined it, so no server catalog is rendered"

    backend = _RenderRecordingBackend()
    _count_route(
        monkeypatch,
        backend,
        messages = [{"role": "developer", "content": ""}, *history],
        enable_tools = False,
    )
    assert backend.messages[0]["role"] == "system"

    backend = _RenderRecordingBackend()
    _count_route(
        monkeypatch,
        backend,
        messages = history,
        enable_tools = False,
        tools = [{"function": {"name": "mine", "parameters": {}}}],
    )
    assert backend.tools[0]["type"] == "function"

    # Outside the MLX branch it returns before normalizing anything, yet the request
    # still arrives normalized.
    route = sys.modules["routes.inference"]
    monkeypatch.setattr(route, "get_inference_backend", lambda: SimpleNamespace(models = {}))
    payload = route.ChatCountTokensRequest(
        messages = [{"role": "developer", "content": ""}],
        tools = [{"function": {"name": "mine", "parameters": {}}}],
    )
    with contextlib.suppress(Exception):
        asyncio.run(route.chat_count_tokens(payload, current_subject = "tester"))
    assert payload.messages[0].role == "system"
    assert payload.tools[0]["type"] == "function"


def test_a_load_serves_the_request_or_the_window_the_model_was_trained_for():
    """A request wins verbatim; asking for nothing takes the trained window; unreadable
    dims resolve to nothing rather than to a number of their own."""
    from core.inference.mlx_inference import MLXInferenceBackend

    resolve = MLXInferenceBackend._resolve_context_lengths
    wide = SimpleNamespace(args = SimpleNamespace(max_position_embeddings = 262144))
    blank = SimpleNamespace(args = SimpleNamespace())
    assert resolve(None, wide, 0) == (262144, 262144, 262144)
    assert resolve(None, wide, 8192) == (8192, 262144, 262144)
    assert resolve(None, wide, 1048576) == (1048576, 262144, 262144)  # above native, still verbatim
    held = SimpleNamespace(args = wide.args, max_seq_length = 4096)
    assert resolve(None, held, 0) == (4096, 262144, 262144)  # what the load attached wins
    assert resolve(None, blank, 0) == (None, None, None)
    assert resolve(None, blank, 8192) == (8192, None, None)


def test_the_resident_context_gate_compares_requests_not_served_windows():
    """Reuse on model identity alone leaves a context change with no reload to carry it.
    Served windows cannot answer it: auto-sized to 32768 and pinned at 32768 are equal."""
    backend_dir = str(Path(__file__).resolve().parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from routes.inference import _resident_context_satisfies as gate

    # Absent or null: the mirror always writes the key, so reading presence alone would
    # reload every transformers model on its own reuse path.
    assert gate({}, 8192) is True
    assert gate({"requested_context_length": None}, 8192) is True

    # Same request, either spelling of "size it yourself".
    assert gate({"requested_context_length": 8192}, 8192) is True
    assert gate({"requested_context_length": 0}, 0) is True
    assert gate({"requested_context_length": 0}, None) is True

    # Different request: pinning and unpinning both reload.
    assert gate({"requested_context_length": 0}, 8192) is False
    assert gate({"requested_context_length": 8192}, 0) is False

    # The served window is not the request and cannot stand in for it.
    assert gate({"requested_context_length": 0, "context_length": 32768}, 32768) is False


def test_the_load_policy_bounds_a_pin_only_where_the_bound_can_be_enforced(monkeypatch):
    """mlx-lm cannot quantize a rotating cache (to_quantized raises, from the first token),
    so exactly one applies. A pin is an explicit memory instruction; a self-chosen window
    is not. An unenforceable pin buys nothing, so it does not spend the quantization."""
    pytest.importorskip("mlx_lm.models.cache")
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend
    from mlx_lm.models.cache import KVCache

    # About the pin/enforceability combination, not the eligibility probe.
    monkeypatch.setattr(
        mlx_inference, "_kv_quant_eligibility", lambda *_a, **_k: ("full", "", True)
    )
    honours = SimpleNamespace(layers = [object(), object()])
    # llama without sliding layers: quantizable, never bounded.
    owns = SimpleNamespace(layers = [object(), object()], make_cache = lambda: [KVCache(), KVCache()])

    unset = object()

    def policy(
        model,
        kv_bits,
        requested,
        served = unset,
    ):
        backend = MLXInferenceBackend()
        backend._model = model
        backend._is_vlm = False
        quant, window, enforced = backend._resolve_kv_policy(
            False, kv_bits, requested, requested if served is unset else served
        )
        return quant["kv_bits"], window, enforced

    assert policy(honours, 4, 8192) == (None, 8192, True)  # enforceable pin outranks quant
    # Auto yields to quantization, and a window that bounds nothing says so.
    assert policy(honours, 4, 0, 262144) == (4, None, False)
    assert policy(honours, None, 8192) == (None, 8192, True)
    assert policy(honours, None, 0, 262144) == (None, 262144, True)  # nothing to yield to

    # An unenforceable pin spends nothing, and the verdict still travels to the API.
    assert policy(owns, 4, 8192) == (4, None, False)
    assert policy(owns, None, 8192) == (None, None, False)

    # A window nobody could read bounds nothing rather than bounding at zero.
    assert policy(honours, None, 0, None) == (None, None, False)  # unreadable
    assert policy(honours, None, 0, 0) == (None, None, False)  # non-positive

    # A shape the probe cannot judge is null, not a confirmed "not enforced".
    unreadable = SimpleNamespace(layers = [object()], make_cache = lambda: None)
    assert policy(unreadable, None, 8192) == (None, None, None)


def test_quantization_is_refused_for_a_pinned_context_rather_than_raising_mid_stream():
    """Left to mlx-lm this surfaces as NotImplementedError on the first generated token."""
    from core.inference.mlx_inference import _kv_quant_status

    status = _kv_quant_status(4, None, False, context_pinned = True)

    assert status["kv_bits"] is None
    assert status["eligibility"] == "refused"
    assert "quantize a limited cache" in status["reason"]


def test_the_bound_is_checked_on_a_real_cache_at_the_size_that_was_asked_for():
    """make_cache ignores max_kv_size, and those caches range from constant-state to
    unbounded, so the argument does not say whether it applied. Nor does an
    architecture-chosen cap serve a narrower request."""
    pytest.importorskip("mlx_lm.models.cache")
    from core.inference.mlx_inference import _kv_window_enforced
    from mlx_lm.models.cache import ArraysCache, ChunkedKVCache, KVCache, RotatingKVCache

    honours = SimpleNamespace(layers = [object(), object()])
    hybrid = SimpleNamespace(
        layers = [object(), object()],
        make_cache = lambda: [RotatingKVCache(1024), KVCache()],
    )
    undeclared = SimpleNamespace(layers = [object()], make_cache = lambda: [ArraysCache(2)])
    chunked = SimpleNamespace(layers = [object()], make_cache = lambda: [ChunkedKVCache(512)])
    native = SimpleNamespace(layers = [object()], make_cache = lambda: [RotatingKVCache(2048)])
    # mlx-vlm's Florence2: tuples of a class that concatenates forever.
    nested = SimpleNamespace(
        layers = [object()],
        make_cache = lambda: [(SimpleNamespace(keys = None), SimpleNamespace(keys = None))],
    )

    assert _kv_window_enforced(honours, False, 4096) is True
    assert _kv_window_enforced(hybrid, False, 4096) is False  # a full-attention layer survives
    assert _kv_window_enforced(chunked, False, 4096) is True
    # Nothing that declares no cap is taken on trust, however it is packaged.
    assert _kv_window_enforced(undeclared, False, 4096) is False
    assert _kv_window_enforced(nested, False, 4096) is False
    # Recurrent Gemma: its own window outlives a narrower pin.
    assert _kv_window_enforced(native, False, 128) is False
    assert _kv_window_enforced(native, False, 4096) is True
    # A window a rotating cache cannot rotate within retains everything instead.
    assert _kv_window_enforced(honours, False, 4) is False
    assert _kv_window_enforced(SimpleNamespace(), False, 4096) is None  # unreadable
    # The VLM branch reaches mlx-vlm's factory and unwraps the language tower.
    assert _kv_window_enforced(SimpleNamespace(language_model = honours), True, 4096) is True


def test_the_probe_reads_a_cache_shape_without_needing_the_mlx_wheels(monkeypatch):
    """Same branches as the real-cache test, through an injected factory. That one
    importorskips, so it proves nothing without mlx; this runs everywhere."""
    from core.inference.mlx_inference import _kv_window_enforced

    def factory(make):
        """Stand in for mlx_lm.models.cache, present or not; return a model."""
        cache_mod = types.ModuleType("mlx_lm.models.cache")
        cache_mod.make_prompt_cache = make
        models_mod = types.ModuleType("mlx_lm.models")
        models_mod.cache = cache_mod
        root = types.ModuleType("mlx_lm")
        root.models = models_mod
        for name, module in (
            ("mlx_lm", root),
            ("mlx_lm.models", models_mod),
            ("mlx_lm.models.cache", cache_mod),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return SimpleNamespace(layers = [object()])

    def caches(*entries):
        return factory(lambda _model, max_kv_size = None: list(entries))

    bounded = SimpleNamespace(max_size = 4096, keep = 4)
    wider = SimpleNamespace(max_size = 8192, keep = 4)
    undeclared = SimpleNamespace()
    unrotatable = SimpleNamespace(max_size = 4, keep = 4)
    chunked = SimpleNamespace(chunk_size = 512)

    assert _kv_window_enforced(caches(bounded, bounded), False, 4096) is True
    assert _kv_window_enforced(caches(bounded, undeclared), False, 4096) is False
    assert _kv_window_enforced(caches(wider), False, 4096) is False
    assert _kv_window_enforced(caches(unrotatable), False, 4096) is False
    assert _kv_window_enforced(caches(chunked), False, 4096) is True
    assert _kv_window_enforced(caches(), False, 4096) is False
    # A factory that answers with a shape this cannot walk is unknown, not a failed load.
    for broken in (None, object(), SimpleNamespace(caches = 7)):
        model = factory(lambda _m, max_kv_size = None, _b = broken: _b)
        assert _kv_window_enforced(model, False, 4096) is None


def test_the_window_reaches_the_runtime_on_every_generation_route(monkeypatch):
    """The runtimes read max_kv_size only when no prompt_cache is passed, so each route
    building its own cache carries the bound. The audio route once did not."""
    import types as _types

    from core.inference.mlx_inference import MLXInferenceBackend

    kwargs = MLXInferenceBackend._kv_window_generate_kwargs
    assert kwargs(SimpleNamespace(_kv_cache_window = None)) == {}
    assert kwargs(SimpleNamespace()) == {}
    assert kwargs(SimpleNamespace(_kv_cache_window = 8192)) == {"max_kv_size": 8192}

    # The prompt-cache history builds its own cache; it has to carry the window too.
    pytest.importorskip("mlx_lm.models.cache")
    from core.inference.mlx_inference import _MLXPromptCacheHistory

    model = SimpleNamespace(layers = [object(), object()])
    bounded = _MLXPromptCacheHistory(4, 1 << 20, 512).fetch(model, "k", [1, 2])[0]
    unbounded = _MLXPromptCacheHistory(4, 1 << 20, None).fetch(model, "k", [1, 2])[0]
    assert {getattr(e, "max_size", None) for e in bounded} == {512}
    assert {getattr(e, "max_size", None) for e in unbounded} == {None}

    captured: list = []
    backend = _install_fake_text_stack(monkeypatch, {"P1": [1], "generated": [1]}, captured)
    backend._kv_cache_window = 8192
    _run_turn(backend, "P1")
    assert captured[0]["max_kv_size"] == 8192

    seen: list = []

    def _vlm_stream(*_args, **kw):
        seen.append(kw)
        yield SimpleNamespace(text = "ok", prompt_tokens = 1, generation_tokens = 1)

    mlx_vlm = _types.ModuleType("mlx_vlm")
    mlx_vlm.stream_generate = _vlm_stream
    mlx_vlm.prompt_utils = SimpleNamespace(
        MODEL_CONFIG = {"m": object()},
        apply_chat_template = lambda *_a, **_k: "<image> p",
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.mlx_inference._temporary_mlx_adapter_state",
        lambda _m, _s: contextlib.nullcontext(),
    )
    vlm = MLXInferenceBackend()
    vlm._kv_cache_window = 4096
    vlm._model = SimpleNamespace(config = {"model_type": "m"})
    vlm._processor = SimpleNamespace(tokenizer = SimpleNamespace())
    next(
        vlm._generate_vlm(
            [{"role": "user", "content": [{"type": "image"}]}],
            object(),
            0,
            1,
            0,
            0,
            1,
            1,
            None,
            _adapter_state = False,
        )
    )
    assert seen[-1]["max_kv_size"] == 4096

    vlm.active_model_name = "m"
    vlm.models["m"] = {"audio_type": "audio_vlm"}
    next(vlm.generate_audio_input_response([{"role": "user", "content": "x"}], "", object()))
    assert seen[-1]["max_kv_size"] == 4096, "the audio route must carry the bound too"


class _BlockMLX:
    """Make `import mlx_lm...` fail the way it does on a box without the wheels."""

    def __init__(self, *names):
        self.names = names

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname in self.names or fullname.startswith(tuple(f"{n}." for n in self.names)):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name = fullname)
        return None


@contextmanager
def _without_mlx(*names):
    blocker = _BlockMLX(*names)
    dropped = {k: v for k, v in sys.modules.items() if k.split(".")[0] in names}
    for key in dropped:
        del sys.modules[key]
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(dropped)


def test_the_mlx_module_imports_on_a_machine_with_no_mlx_wheels():
    """Linux and Windows import this tree without mlx. The backend is constructed behind
    the Darwin gate, but an mlx import at module scope would fail every non-Mac boot."""
    import importlib

    with _without_mlx("mlx", "mlx_lm", "mlx_vlm"):
        module = importlib.reload(importlib.import_module("core.inference.mlx_inference"))
        assert (
            module.mlx_native_context_length(
                SimpleNamespace(config = SimpleNamespace(max_position_embeddings = 131072))
            )
            == 131072
        )
    importlib.reload(module)


def test_the_probe_answers_unknown_rather_than_raising_without_mlx():
    """Nothing can be built to judge, which is "unknown", not "unbounded"."""
    from core.inference.mlx_inference import _kv_window_enforced
    with _without_mlx("mlx", "mlx_lm", "mlx_vlm"):
        assert _kv_window_enforced(SimpleNamespace(layers = [object()]), False, 4096) is None


def test_an_mlx_count_prices_the_current_date_the_completion_prepends(monkeypatch):
    """The completion applies this once for both non-GGUF backends before it branches, so
    skipping it under-reports every interactive MLX prompt and the usage bar claims room."""
    from starlette.datastructures import Headers

    # No API key, which is what makes the prompt Studio's to compose.
    interactive = SimpleNamespace(headers = Headers({}), query_params = {}, cookies = {})
    backend = _RenderRecordingBackend()
    _count_route(
        monkeypatch,
        backend,
        template = _PLAIN_TEMPLATE,
        request = interactive,
        messages = [{"role": "user", "content": "hi"}],
    )
    from routes.inference import current_date_prompt_line

    line = current_date_prompt_line(request = interactive)
    assert line, "the harness must actually produce a date line"
    assert backend.system.startswith(
        line
    ), f"the count dropped the date line the completion prepends: {backend.system!r}"


def test_an_mlx_count_prices_the_archive_tool_and_its_compaction_nudge(monkeypatch):
    """An archive adds search_conversation and a compaction nudge, both gated on the
    thread id, so a count that drops it under-reports."""
    from routes import inference as route
    from state import tool_policy

    monkeypatch.setattr(tool_policy, "_tool_policy_default", True)
    monkeypatch.setattr(route, "_thread_has_conversation_archive", lambda tid: bool(tid))
    backend = _RenderRecordingBackend()
    _count_route(
        monkeypatch,
        backend,
        template = _TOOL_TEMPLATE,
        messages = [{"role": "user", "content": "hi"}],
        enabled_tools = ["web_search"],
        thread_id = "thread-with-an-archive",
    )
    names = [t["function"]["name"] for t in (backend.tools or [])]
    assert (
        "search_conversation" in names
    ), f"the archive tool the completion renders was not priced: {names}"
    from routes.inference import _apply_compaction_nudge

    assert _apply_compaction_nudge("", backend.tools), "the nudge must be non-empty here"
    assert (
        _apply_compaction_nudge("", backend.tools) in backend.system
    ), "the count omitted the compaction nudge the completion appends"


def test_an_mlx_count_prices_the_date_an_api_key_tool_loop_still_gets(monkeypatch):
    """`_wants_current_date` is false for an API-key request, but the tool-loop completion
    reapplies it with include_api_key, so a count that did not would be short that line."""
    from starlette.datastructures import Headers
    from state import tool_policy

    monkeypatch.setattr(tool_policy, "_tool_policy_default", True)
    keyed = SimpleNamespace(
        headers = Headers({"authorization": "Bearer sk-unsloth-test"}),
        query_params = {},
        cookies = {},
    )
    backend = _RenderRecordingBackend()
    _count_route(
        monkeypatch,
        backend,
        template = _TOOL_TEMPLATE,
        request = keyed,
        enabled_tools = ["web_search"],
        messages = [{"role": "user", "content": "hi"}],
    )
    from routes.inference import current_date_prompt_line

    line = current_date_prompt_line(request = keyed)
    assert line, "the harness must produce a date line"
    assert (
        line in backend.system
    ), f"the count dropped the date the tool-loop completion adds: {backend.system!r}"


def test_an_mlx_count_reports_the_advertised_model_id(monkeypatch):
    """The caller drops a count naming a different model, so one loaded from a resolved
    path must still answer as the repo id."""
    from routes import inference as route

    backend = _RenderRecordingBackend()
    monkeypatch.setattr(route, "_orchestrator_public_model_id", lambda _b: "org/advertised-repo-id")
    served = _count_route(monkeypatch, backend, messages = [{"role": "user", "content": "hello"}])
    assert json.loads(served.body)["model"] == "org/advertised-repo-id"


def test_an_mlx_count_is_dropped_when_a_same_model_reload_lands_under_it(monkeypatch):
    """The active name cannot see a same-ID reload, which is how a template override lands.
    A count routed from the old entry must not be published."""
    from fastapi import HTTPException
    from routes import inference as route

    backend = _RenderRecordingBackend()
    backend.load_generation = 7

    real_count = backend.count_chat_tokens

    def _reload_midway(*args, **kwargs):
        # Lands while the tokenizer runs: same name, new generation.
        backend.load_generation = 8
        return real_count(*args, **kwargs)

    backend.count_chat_tokens = _reload_midway
    with pytest.raises(HTTPException) as excinfo:
        _count_route(monkeypatch, backend, messages = [{"role": "user", "content": "hi"}])
    assert excinfo.value.status_code == 503
    assert "changed while counting" in str(excinfo.value.detail)


def test_an_mlx_count_yields_to_a_generation_that_started_while_it_prepared(monkeypatch):
    """Everything between admission and the tokenizer awaits, so a chat starting in the gap
    would wait behind this count for the orchestrator lock. GGUF re-checks; this must too."""
    from fastapi import HTTPException
    from routes import inference as route

    backend = _RenderRecordingBackend()
    counts = iter([0, 1])  # admitted at the entry check, busy by the last checkpoint
    with pytest.raises(HTTPException) as excinfo:
        _count_route(
            monkeypatch,
            backend,
            generations = lambda: next(counts, 1),
            messages = [{"role": "user", "content": "hi"}],
        )
    assert excinfo.value.status_code == 503
    assert "generation is in progress" in str(excinfo.value.detail)


def test_the_mlx_mcp_snapshot_is_taken_under_the_same_guard_the_gguf_count_uses():
    """The MCP handlers hold this guard across the row change and the schema-cache
    invalidation, so a snapshot outside it can pair a new row with a stale schema."""
    import inspect

    from routes import inference as route

    body = inspect.getsource(route._mlx_count_chat_tokens)
    assert "mcp_server_snapshot_guard" in body, "the MLX snapshot is unguarded"
    guard = body.index("async with mcp_server_snapshot_guard():")
    snapshot = body.index("asyncio.to_thread(cached_mcp_tools)")
    assert guard < snapshot, "the guard must be held across the snapshot, not after it"
