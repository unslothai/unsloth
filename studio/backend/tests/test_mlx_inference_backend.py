# SPDX-License-Identifier: AGPL-3.0-only

import sys
import types
from contextlib import contextmanager
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
