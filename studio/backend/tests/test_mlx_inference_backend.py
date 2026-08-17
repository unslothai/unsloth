# SPDX-License-Identifier: AGPL-3.0-only

import copy
import contextlib
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

    def model_render(_processor, _config, messages, **kwargs):
        calls["model"].append(kwargs)
        return state["model"]

    prompt_utils = SimpleNamespace(
        MODEL_CONFIG = {"deepseek_vl_v2": object()},
        apply_chat_template = model_render,
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
        if state["generic"] == "generic text":
            return f"generic prompt {_messages[0]['content']}"
        if state["generic"].startswith("serialized"):
            index = -1 if state["generic"] == "serialized_last" else 0
            return f"User: {_messages[index]['content']}"
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
    state["model"] = "<image> model-aware"
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
    padding_attempts = []

    def padding_stream(*stream_args, **stream_kwargs):
        padding_attempts.append((stream_args, stream_kwargs))
        if len(padding_attempts) == 1:
            raise ValueError(
                "Failed to process inputs with error: ImagesKwargs.__init__() got an unexpected keyword argument 'padding'"
            )
        yield SimpleNamespace(text = "ok", prompt_tokens = 3, generation_tokens = 1)

    mlx_vlm.stream_generate = padding_stream
    state["generic"] = "<image> healthy generic"
    processor_calls, exact_commits = [], []
    backend._processor = lambda **kwargs: processor_calls.append(kwargs)
    backend._vlm_prompt_cache_history = SimpleNamespace(
        insert = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("inserted abandoned block state")
        )
    )
    backend._prepare_vlm_prompt_cache = lambda *_args: (
        SimpleNamespace(),
        ("scope", "prompt", None),
        [],
        SimpleNamespace(commit = lambda: exact_commits.append(True)),
    )
    assert list(backend._generate_vlm(*args)) == ["ok"]
    assert (
        len(padding_attempts) == 2
        and {"prompt_cache_state", "apc_manager"} <= padding_attempts[0][1].keys()
    )
    assert {"prompt_cache_state", "apc_manager", "prefill_step_size"}.isdisjoint(
        padding_attempts[1][1]
    )
    padding_attempts[1][0][1].process("prompt", images = ["image"], padding = True)
    assert processor_calls == [{"text": "prompt", "images": ["image"], "return_tensors": "mlx"}]
    assert (
        isinstance(padding_attempts[1][0][1], mlx_inference._VLMProcessorWithoutImagePadding)
        and not exact_commits
    )
    assert backend._vlm_prompt_cache_unavailable and backend._vlm_prompt_cache_history is None
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


# fmt: off
def test_mlx_vlm_normalizes_native_reasoning_channels(monkeypatch):
    _install_fake_mlx(monkeypatch)
    from core.inference.mlx_inference import MLXInferenceBackend, content_to_text

    def render(_target, messages, **kwargs):
        prompt = " ".join(content_to_text(message.get("content")) for message in messages)
        return prompt.replace("<think>private chain</think>", "") if kwargs.get("preserve_thinking") is False else prompt

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        render,
        raising = True,
    )

    mlx_vlm_pkg = types.ModuleType("mlx_vlm")

    class _Resp:
        def __init__(self, text, tok):
            self.text = text
            self.token = tok

    def _stream_generate(_model, _processor, _prompt, _images, **_kw):
        assert "private chain" not in _prompt and "visible answer" in _prompt
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
            messages = [{"role": "assistant", "content": "<think>private chain</think>visible answer"}, {"role": "user", "content": "again"}],
            image = object(),
            max_new_tokens = 4,
            preserve_thinking = False,
        )
    ) == [
        "<think>",
        "<think>vision",
        "<think>vision</think>",
        "<think>vision</think> answer",
    ]
# fmt: on


# Turn 2 literally extends turn 1 as an append-only chat render does; the fork
# shares only the opening turn and the unrelated prompt shares no prefix.
_VLM_TURN1, _VLM_TURN2, _VLM_FORK, _VLM_OTHER = "P11", "P11 A1 P2", "P11 A1 P3", "Q11"


class _FakeKV:
    def __init__(self, length):
        self.shape = (1, 1, length, 8)

    def __getitem__(self, item):
        return _FakeKV(item[2].stop)


def _fake_entries(length, offset = 0):
    kv = dict(nbytes = 64, keys = _FakeKV(length), values = _FakeKV(length + 4))
    ring = dict(max_size = 4096, keep = 0, _idx = offset)
    return [SimpleNamespace(offset = offset, **kv), SimpleNamespace(offset = offset, **kv, **ring)]


def _fake_vlm_cache_backend(monkeypatch):
    from core.inference import mlx_inference

    control = {"mode": "ok"}

    def stream(_model, _processor, prompt, _images, **kwargs):
        state = kwargs.get("prompt_cache_state")
        control["state"] = state
        control["seeded"] = list(getattr(state, "token_ids", None) or ())
        ids = [ord(character) for character in prompt]
        manager = kwargs.get("apc_manager")
        control["manager"] = manager
        if manager is not None:
            cache, prefix = manager.lookup_exact_cache(ids)
            control["returned"] = prefix
            if cache is None:
                checkpoint = len(ids) - manager.exact_cache_guard_tokens
                if 0 < checkpoint < len(ids):
                    manager.store_exact_cache(
                        ids[:checkpoint], _fake_entries(checkpoint, checkpoint)
                    )
                manager.store_exact_cache(ids, _fake_entries(len(ids) + 1, len(ids) + 1))
            if control["mode"] == "error":
                raise RuntimeError("vlm failed")
            yield SimpleNamespace(
                text = "x", prompt_tokens = len(ids), generation_tokens = 1, cached_tokens = prefix
            )
            if control["mode"] == "cancel_after_update":
                control["cancel"].set()
            return
        cache = state.cache if state and state.cache else _fake_entries(64)
        if state is not None:
            control["returned"] = state.find_prefix_length(ids)
            control["cut"] = [(e.keys.shape[2], e.values.shape[2], e.offset) for e in cache]
            control["idx"] = [getattr(e, "_idx", 0) for e in cache]
        for entry in cache:
            entry.offset = len(ids) + 1
            entry.keys = entry.values = _FakeKV(entry.offset)
            if hasattr(entry, "_idx"):
                entry._idx = entry.offset
        if control["mode"] == "error":
            raise RuntimeError("vlm failed")
        yield SimpleNamespace(text = "x", prompt_tokens = len(ids), generation_tokens = 1)
        if state is not None:
            state.token_ids, state.cache = ids + [9], cache
        if control["mode"] == "cancel_after_update":
            control["cancel"].set()

    package = types.ModuleType("mlx_vlm")

    class _State:
        def __init__(self):
            self.cache, self.token_ids = None, None

        def find_prefix_length(self, new_ids):
            stored = self.token_ids or []
            common = 0
            for old, new in zip(stored, new_ids):
                if old != new:
                    break
                common += 1
            return common

    package.PromptCacheState = _State
    package.stream_generate = stream
    package.__version__ = "0.6.8"
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    apc = types.ModuleType("mlx_vlm.apc")
    apc.model_apc_mode = lambda _model: "block"
    apc._clone_prompt_cache_for_apc = lambda cache, **_kwargs: copy.deepcopy(cache)
    monkeypatch.setitem(sys.modules, "mlx_vlm.apc", apc)
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_logits_processors = lambda **_kwargs: []
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _target, messages, **_kwargs: messages[-1]["content"][-1]["text"],
    )
    backend = mlx_inference.MLXInferenceBackend()
    monkeypatch.setattr(mlx_inference, "MLX_VLM_PREFILL_STEP_SIZE", 1)
    backend._model = SimpleNamespace(
        config = SimpleNamespace(image_token_id = 99),
        language_model = SimpleNamespace(),
        named_modules = lambda: (),
    )
    backend._processor = SimpleNamespace(chat_template = "x", apply_chat_template = lambda: None)
    backend._is_vlm, backend.active_model_name = True, "model-a"
    return backend, control


def _cached_vlm_turn(backend, prompt, image, **overrides):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    kwargs = {"max_new_tokens": 1, **overrides}
    consume = kwargs.pop("_consume", True)
    stream = backend.generate_chat_response(messages, image = image, **kwargs)
    return list(stream) if consume else stream


def test_mlx_vlm_cache_reuses_only_compatible_state_and_reports_timing(monkeypatch):
    from PIL import Image as PILImage

    backend, control = _fake_vlm_cache_backend(monkeypatch)
    image, recolored = PILImage.new("RGB", (1, 1), (255, 0, 0)), PILImage.new("RGB", (1, 1))
    _cached_vlm_turn(backend, _VLM_TURN1, image)
    _cached_vlm_turn(backend, _VLM_TURN2, image)
    timings = backend.last_generation_stats["timings"]
    assert timings["cache_n"] == 2 and timings["prompt_n"] == 7 and control["seeded"]
    entries = backend._vlm_prompt_cache_history._entries
    # Continuing supersedes the entry it grew from; a branch is retained beside it.
    assert len(entries) == 1
    assert (
        control["cut"] == [(2, 2, 2)] * 2 and control["idx"] == [0, 2] and control["returned"] == 2
    )
    _cached_vlm_turn(backend, _VLM_TURN2, image)
    assert control["seeded"] and backend.last_generation_stats["timings"]["cache_n"] == 7
    _cached_vlm_turn(backend, _VLM_FORK, image)
    assert {_VLM_TURN2, _VLM_FORK} <= {entry[3] for entry in entries.values()}
    for kwargs in (
        {"prompt": _VLM_OTHER},
        {"_adapter_state": False},
        {"image": recolored},
        {"model": "model-b"},
        {"repetition_penalty": 1.1},
        {"presence_penalty": 1.1},
    ):
        media, prompt = kwargs.pop("image", image), kwargs.pop("prompt", _VLM_TURN2)
        backend.active_model_name = kwargs.pop("model", backend.active_model_name)
        _cached_vlm_turn(backend, prompt, media, **kwargs)
        assert backend.last_generation_stats["timings"]["cache_n"] == 0


def test_mlx_vlm_cache_rolls_back_failures_cancellation_and_cleans_up(monkeypatch):
    backend, control = _fake_vlm_cache_backend(monkeypatch)
    image = bytearray(b"a")
    _cached_vlm_turn(backend, _VLM_TURN1, image)
    history = backend._vlm_prompt_cache_history
    retained = next(iter(history._entries.values()))[0]
    control["mode"] = "error"
    with pytest.raises(RuntimeError, match = "vlm failed"):
        _cached_vlm_turn(backend, _VLM_TURN2, image)
    cancel = __import__("threading").Event()
    control.update(mode = "cancel_after_update", cancel = cancel)
    _cached_vlm_turn(backend, _VLM_TURN2, image, cancel_event = cancel)
    control["mode"] = "ok"
    closing = _cached_vlm_turn(backend, _VLM_TURN2, image, _consume = False)
    next(closing), closing.close()
    monkeypatch.setattr(history, "insert", lambda *_: (_ for _ in ()).throw(ValueError("insert")))
    _cached_vlm_turn(backend, _VLM_TURN2, image)
    assert next(iter(history._entries.values()))[0] is retained and retained.cache[0].offset == 2
    history.fetch = lambda *_: (_ for _ in ()).throw(ValueError("lookup"))
    assert _cached_vlm_turn(backend, _VLM_TURN2, image) == ["x"]
    assert next(iter(history._entries.values()))[0] is retained
    scope = ("model-a", "None", "fingerprint")

    def kv(nbytes, tokens = (1,)):
        keys = SimpleNamespace(shape = (1, 1, len(tokens), 8))
        cache = [SimpleNamespace(offset = len(tokens), nbytes = nbytes, keys = keys, values = keys)]
        return SimpleNamespace(token_ids = list(tokens), cache = cache)

    bounded, tiny, counted = type(history)(2, 100), type(history)(1, 10), type(history)(1, 1000)
    assert all(h.insert((scope, p, None), kv(64)) for h in (bounded, counted) for p in "ab")
    assert len(bounded._entries) == 1 and len(counted._entries) == 1
    assert not tiny.insert((scope, "a", None), kv(20)) and not tiny._entries
    for insertion_order in ((("a", (1,)), ("ab", (1, 2))), (("ab", (1, 2)), ("a", (1,)))):
        picker = type(history)(4, 1000)
        for prompt, tokens in insertion_order:
            picker.insert((scope, prompt, None), kv(64, tokens))
        assert picker.fetch(SimpleNamespace, scope, "abc")[1] == [1, 2]
        assert picker.fetch(SimpleNamespace, scope, "ab rewritten suffix")[1]
    for unsupported in ({"keys": None}, {"window_size": 8}, {"offset": 9}, {"values": _FakeKV(0)}):
        entry = SimpleNamespace(**{**vars(kv(64).cache[0]), **unsupported})
        state = SimpleNamespace(token_ids = [1] * 9, cache = [entry], observed_prefix = 0)
        assert not picker.insert((scope, "x", None), state)
    _, _, matched = picker.fetch(SimpleNamespace, scope, "abc")
    assert picker.insert((scope, "abc", matched), kv(64, (1, 2, 3)))
    retained = list(picker._entries.values())
    assert {entry[3] for entry in retained} == {"a", "abc"}
    assert picker._nbytes == sum(entry[1] for entry in retained)
    fingerprint = sys.modules["core.inference.mlx_inference"]._vlm_media_fingerprint
    assert fingerprint(__file__) is None and fingerprint(b"x.png") is None
    backend._clear_prompt_cache()
    backend._model.language_model = SimpleNamespace(_image_cache = None)
    assert (
        sys.modules["core.inference.mlx_inference"]._mlx_vlm_prompt_cache_api(backend._model)
        is None
    )
    module = sys.modules["core.inference.mlx_inference"]
    ring = _fake_entries(64, offset = 8)[1]
    ring._idx = 3
    assert module._vlm_cache_entry_shape(ring) is None
    backend._model.language_model = SimpleNamespace(_rope_deltas = None)
    assert module._mlx_vlm_prompt_cache_api(backend._model) is None
    backend._model.language_model = SimpleNamespace()
    package = sys.modules["mlx_vlm"]
    package.__version__ = "0.6.7"
    assert module._mlx_vlm_prompt_cache_api(backend._model) is None
    _cached_vlm_turn(backend, _VLM_TURN1, image)
    assert control["state"] is control["manager"] is None
    assert backend.last_generation_stats["timings"]["cache_n"] == 0
    package.__version__ = "0.6.8"
    assert module._mlx_vlm_prompt_cache_api(backend._model) is not None
    backend._model.config = SimpleNamespace(
        model_type = "supported",
        text_config = SimpleNamespace(cross_attention_layers = [3, 8]),
    )
    assert module._mlx_vlm_prompt_cache_api(backend._model) is None
    backend._model.config.text_config.cross_attention_layers = []
    assert module._mlx_vlm_prompt_cache_api(backend._model) is not None
    backend._model._config = {"text_config": {"cross_attention_layers": [3, 8]}}
    assert module._mlx_vlm_prompt_cache_api(backend._model) is None
    del backend._model._config
    for config in (
        SimpleNamespace(model_type = "idefics2"),
        {"model_type": "kimi_vl"},
        SimpleNamespace(model_type = "llava"),
        {"model_type": "llava_next"},
    ):
        backend._model.config = config
        assert module._mlx_vlm_prompt_cache_api(backend._model) is None
    backend._model.config, backend._model._config = SimpleNamespace(), {"model_type": "kimi_vl"}
    assert module._mlx_vlm_prompt_cache_api(backend._model) is None
    del backend._model._config
    del sys.modules["mlx_vlm"].PromptCacheState
    _cached_vlm_turn(backend, _VLM_TURN1, image)
    assert control["state"] is None
    monkeypatch.setattr(
        module,
        "_vlm_media_fingerprint",
        lambda _media: (_ for _ in ()).throw(AssertionError("fingerprinted after disable")),
    )
    _cached_vlm_turn(backend, _VLM_TURN1, image)


def test_mlx_vlm_exact_manager_commits_only_aligned_completed_snapshots():
    from core.inference import mlx_inference as module

    history = module._MLXVLMPromptCacheHistory(2, 1000, step_size = 4)
    scope = ("model", "adapter", "media")
    clone = lambda cache, **_kwargs: copy.deepcopy(cache)
    manager = module._StudioVLMExactCacheManager(history, scope, "first", clone, 4)
    tokens = list(range(13))
    assert (manager.lookup_exact_cache(tokens), manager.exact_cache_guard_tokens) == ((None, 0), 5)
    cache = [SimpleNamespace(nbytes = 32, state = [1], meta_state = ()) for _ in range(2)]
    cache[0].offset = 8
    full_cache = copy.deepcopy(cache)
    full_cache[0].offset = len(tokens) + 1
    assert manager.store_exact_cache(tokens[:8], cache)
    assert not manager.store_exact_cache(tokens, full_cache)
    assert manager.commit() and len(history._entries) == 1
    limited = module._StudioVLMExactCacheManager(history, scope, "limited", clone, 4)
    assert limited.lookup_exact_cache(tokens, max_prefix_tokens = 7) == (None, 0)
    allowed = module._StudioVLMExactCacheManager(history, scope, "allowed", clone, 4)
    assert allowed.lookup_exact_cache(tokens, max_prefix_tokens = 8)[1] == 8

    mismatched = module._StudioVLMExactCacheManager(
        type(history)(2, 1000, step_size = 4), scope, "expanded", clone, 4
    )
    assert mismatched.lookup_exact_cache(tokens) == (None, 0)
    assert mismatched.store_exact_cache(tokens[:8], cache)
    expanded_cache = copy.deepcopy(full_cache)
    expanded_cache[0].offset = len(tokens) + 2
    assert not mismatched.store_exact_cache(tokens, expanded_cache)
    assert not mismatched.commit()
    short = module._StudioVLMExactCacheManager(history, scope, "short", clone, 4)
    assert (short.lookup_exact_cache(list(range(7))), short.exact_cache_guard_tokens) == (
        (None, 0),
        0,
    )


def test_mlx_vlm_exact_cache_refreshes_after_completed_warm_turn(monkeypatch):
    backend, control = _fake_vlm_cache_backend(monkeypatch)
    sys.modules["mlx_vlm.apc"].model_apc_mode = lambda _model: "exact"
    image = bytearray(b"a")
    turn3, turn4 = _VLM_TURN2 + " A2 P3", _VLM_TURN2 + " A2 P3 A3 P4"
    _cached_vlm_turn(backend, _VLM_TURN1, image)
    _cached_vlm_turn(backend, _VLM_TURN2, image)
    history = backend._vlm_prompt_cache_history
    retained = next(iter(history._entries.values()))[0]
    assert backend.last_generation_stats["timings"]["cache_n"] == 1
    cancel = __import__("threading").Event()
    control.update(mode = "cancel_after_update", cancel = cancel)
    _cached_vlm_turn(backend, turn3, image, cancel_event = cancel)
    assert next(iter(history._entries.values()))[0] is retained
    control["mode"] = "ok"
    _cached_vlm_turn(backend, turn3, image)
    assert backend.last_generation_stats["timings"]["cache_n"] == 0
    assert len(next(iter(history._entries.values()))[0].token_ids) == len(turn3) - 2
    _cached_vlm_turn(backend, turn4, image)
    assert backend.last_generation_stats["timings"]["cache_n"] == len(turn3) - 2


def _mrope_model(arch = "qwen2_vl", **lm):
    """Qwen-style mRoPE model; pass an attribute as None to omit it."""
    fields = {
        "_rope_deltas": object(),
        "_position_ids": object(),
        "get_rope_index": lambda *a, **k: None,
        **lm,
    }
    kept = {k: v for k, v in fields.items() if v is not None}
    return SimpleNamespace(
        config = SimpleNamespace(model_type = arch), language_model = SimpleNamespace(**kept)
    )


# fmt: off
def test_mlx_vlm_mrope_admission_and_rope_suppression(monkeypatch):
    from core.inference import mlx_inference as module

    monkeypatch.setattr(module, "_runtime_primes_rope", lambda: True)
    for arch in ("qwen2_vl", "qwen2_5_vl", "qwen3_5"):
        assert module._vlm_mrope_reuse_arch(_mrope_model(arch)) == arch
    dict_config = SimpleNamespace(
        config = {"model_type": "qwen2_vl"}, language_model = _mrope_model().language_model
    )
    assert module._vlm_mrope_reuse_arch(dict_config) == "qwen2_vl"
    # Refused: missing Qwen-style state or Falcon spatial state.
    assert module._vlm_mrope_reuse_arch(_mrope_model(_rope_deltas = None)) is None
    assert module._vlm_mrope_reuse_arch(_mrope_model(_position_ids = None)) is None
    assert module._vlm_mrope_reuse_arch(_mrope_model(get_rope_index = None)) is None
    assert module._vlm_mrope_reuse_arch(_mrope_model(_pos_hw = object())) is None

    ids = lambda n: SimpleNamespace(shape = (1, n))

    class _Model:
        config = SimpleNamespace(model_type = "qwen2_vl")
        language_model = SimpleNamespace(
            _rope_deltas = object(), _position_ids = None, get_rope_index = lambda *a, **k: None
        )

        def get_input_embeddings(
            self,
            input_ids = None,
            pixel_values = None,
            **kwargs,
        ):
            return SimpleNamespace(rope_deltas = object(), position_ids = object())

        def _set_position_state(self, input_ids):
            self.language_model._position_ids = SimpleNamespace(shape = (3, 1, input_ids.shape[-1]))
            self.language_model._rope_deltas = "wrapper delta"

    model = _Model()
    original_rope_index = model.language_model.get_rope_index
    # Reuse: the runtime primes a 12-token map and trims the ids to the 7-token suffix, so
    # the map is exactly the cut length longer and the suffix-derived fields are dropped.
    with module._temporary_mlx_vlm_rope_suppression(model, SimpleNamespace(observed_prefix = 5)):
        assert "get_input_embeddings" in vars(model)
        position_ids, rope_deltas = model.language_model.get_rope_index(ids(12))
        assert position_ids.shape == (3, 1, 12) and rope_deltas == "wrapper delta"
        reused = model.get_input_embeddings(ids(7), pixel_values = None)
        assert reused.rope_deltas is None and reused.position_ids is None
    assert "get_input_embeddings" not in vars(model)
    assert model.language_model.get_rope_index is original_rope_index
    # A cold request keeps the state it just computed.
    with module._temporary_mlx_vlm_rope_suppression(model, SimpleNamespace(observed_prefix = 0)):
        cold = model.get_input_embeddings(ids(12), pixel_values = None)
        assert cold.rope_deltas is not None and cold.position_ids is not None

    _Language = type("_Language", (), {"_rope_deltas": object(), "_position_ids": None, "get_rope_index": lambda *_args, **_kwargs: None})
    class _RejectingModel(_Model):
        language_model = _Language()
        get_input_embeddings = property(lambda self: _Model.get_input_embeddings.__get__(self))
    rejecting = _RejectingModel()
    with pytest.raises(AttributeError): module._temporary_mlx_vlm_rope_suppression(rejecting, SimpleNamespace(observed_prefix = 5)).__enter__()
    assert "get_rope_index" not in vars(rejecting.language_model)
    assert rejecting.language_model.get_rope_index.__func__ is _Language.get_rope_index
    cleanup_rejecting = _Model()
    monkeypatch.setattr(_Model, "__delattr__", lambda *_args: (_ for _ in ()).throw(RuntimeError("cleanup refused")))
    cleanup_rope_index = cleanup_rejecting.language_model.get_rope_index
    with pytest.raises(RuntimeError, match = "cleanup refused"), module._temporary_mlx_vlm_rope_suppression(cleanup_rejecting, SimpleNamespace(observed_prefix = 5)): pass
    assert "get_input_embeddings" in vars(cleanup_rejecting)
    assert cleanup_rejecting.language_model.get_rope_index is cleanup_rope_index
    # A measured architecture is refused on a runtime that cannot prime.
    monkeypatch.setattr(module, "_runtime_primes_rope", lambda: False)
    assert module._vlm_mrope_reuse_arch(_mrope_model("qwen2_vl")) is None
# fmt: on


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
    backend._vlm_prompt_cache_history = object()

    backend.reset_generation_state()
    assert backend._prompt_cache_history is history

    backend.unload_model("model-a")
    assert backend._prompt_cache_history is None
    assert backend._vlm_prompt_cache_history is None


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
    """Studio keeps the rate and rendering gates; the checkpoint answer is zoo's."""
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
    registry uses lowercase. Studio must reach the registered renderer rather
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
