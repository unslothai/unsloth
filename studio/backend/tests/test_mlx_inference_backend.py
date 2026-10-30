# SPDX-License-Identifier: AGPL-3.0-only

import ast
import asyncio
import importlib
import builtins
import contextlib
import copy
import json
import os
import subprocess
import sys
import threading
import types
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def native_vlm_generation_context():
    from core.inference import mlx_inference
    return mlx_inference._vlm_generation_context


FUSIONS = (
    "moe_gate_up",
    "decode_conv_silu",
    "residual_norm",
    "moe_router",
    "moe_routed_experts",
    "residual_norm_handoff",
    "packed_gated_delta",
)
# Stub scopes kept out of the FUSIONS parametrisations: unsloth_zoo #1757 added fused_verify_attention
# around speculative verify. The stub still has to answer them.
ZOO_ONLY_FUSIONS = ("verify_attention",)


def _zoo_registry():
    """The stub's INFERENCE_SCOPES in zoo's order: (scope name, call traits it serves[, opt-in switch])."""
    return [
        *((f"fused_{name}", ()) for name in FUSIONS[:5]),
        ("nax_quantized_linear", ()),
        ("dense_prefill_linear", ()),
        *((f"fused_{name}", ()) for name in FUSIONS[5:]),
        ("exact_speculative_verify", ("speculative",)),
        ("fused_verify_attention", ("speculative",)),
    ]


def _stub_inference_scopes(module):
    """zoo's inference_scopes over the stub: each registered scope is looked up by name when entered,
    so a test that replaces a stub scope sees it entered through the registry."""

    @contextmanager
    def inference_scopes(
        model = None,
        call = None,
        *,
        int8_prefill = None,
        ane_prefill = None,
        on_error = None,
        _modules = None,
    ):
        # The call-shape replay below passes placeholders; anything but a tuple serves every scope.
        traits = set(call) if isinstance(call, tuple) else None
        with contextlib.ExitStack() as stack:
            for name, serves, *opt_in in module.INFERENCE_SCOPES:
                if traits is not None and serves and traits.isdisjoint(serves):
                    continue
                if opt_in and opt_in[0] and os.environ.get(opt_in[0]) != "1":
                    continue
                args = (
                    (int8_prefill,)
                    if name == "nax_quantized_linear"
                    else (ane_prefill,)
                    if name == "ane_prefill_mlp"
                    else ()
                )
                try:
                    stack.enter_context(getattr(module, name)(model, *args))
                except Exception as error:
                    if on_error is None:
                        raise
                    on_error(name, error)
            yield model

    return inference_scopes


@pytest.fixture(autouse = True)
def mlx_inference_patches(monkeypatch, native_vlm_generation_context):
    from core.inference import mlx_inference

    monkeypatch.setattr(mlx_inference, "_vlm_generation_context", contextlib.nullcontext)
    module = types.ModuleType("unsloth_zoo.mlx.inference")
    for name in FUSIONS + ZOO_ONLY_FUSIONS:
        setattr(module, f"fused_{name}", _neutral_scope)
    for name, helper in {**PLAIN_HELPERS, **NEUTRAL_PLAIN_HELPERS}.items():
        setattr(module, name, helper)
    module.INFERENCE_SCOPES = _zoo_registry()
    module.inference_scopes = _stub_inference_scopes(module)
    module.__getattr__ = _neutral_zoo_helper
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.inference", module)
    return module


def _neutral_scope(
    model = None,
    *_args,
    **_kwargs,
):
    """Zoo passes extra arguments to some scopes, which a bare nullcontext rejects with TypeError."""
    return contextlib.nullcontext(model)


def _fusion_modules(model, modules):
    """zoo's own body (unsloth_zoo.mlx.inference): the modules a fusion scope walks."""
    return (
        modules
        if modules is not None
        else (model.named_modules() if hasattr(model, "named_modules") else ())
    )


# Plain-function helpers zoo's generate.py imports; the neutral scope cannot stand in for them.
PLAIN_HELPERS = {"_fusion_modules": _fusion_modules}


def _fused_drafter_ops(model, *_args, **_kwargs):
    """zoo's fused_drafter_ops (unsloth_zoo #2031) is a plain call returning how many drafter
    modules it fused, not a scope. Its body needs Metal kernels, so the stub fuses none."""
    return 0


# Plain calls whose zoo body cannot run here, so the stub returns their do-nothing answer instead.
NEUTRAL_PLAIN_HELPERS = {"fused_drafter_ops": _fused_drafter_ops}


def _neutral_zoo_helper(name):
    """Every other public helper zoo's generate.py enters around a model is neutral too.

    unsloth_zoo #1541 added nax_quantized_linear to that import list, and a stub holding only the
    fused_* names failed every real-model test on macOS with an ImportError. fused_*,
    inference_scopes and warm_inference stay explicit, so a test that deletes one still sees it missing.
    """
    if (
        name.startswith("_")
        or name.startswith("fused_")
        or name in ("inference_scopes", "warm_inference")
    ):
        raise AttributeError(name)
    return _neutral_scope


@pytest.fixture
def mlx_moe(mlx_inference_patches):
    return mlx_inference_patches


@pytest.fixture
def mlx_decode(mlx_inference_patches):
    return mlx_inference_patches


@pytest.mark.parametrize("feature", FUSIONS)
@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("injected", name = "unsloth_zoo.mlx.inference"),
        ModuleNotFoundError("injected", name = "mlx"),
        ModuleNotFoundError("injected", name = "unsloth_zoo"),
        # A skewed or partial install raises ImportError, not ModuleNotFoundError.
        ImportError("cannot import name 'gather_qmm'"),
        RuntimeError("zoo fusion module failed at import"),
    ],
    ids = ["feature", "mlx", "unsloth_zoo", "import-error", "raising-module"],
)
def test_mlx_fusion_import_never_fails_the_request(monkeypatch, error, feature):
    """No Zoo state may fail a load or a generation: raising would take down a path that
    worked before the optimization existed, so a transitive failure is logged instead."""
    from core.inference import mlx_inference

    module_name = "unsloth_zoo.mlx.inference"
    helper = lambda model: mlx_inference._mlx_optional_fusion(f"fused_{feature}", model)

    original_import = mlx_inference.importlib.import_module

    def fail(name, *args, **kwargs):
        if name == module_name:
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(mlx_inference.importlib, "import_module", fail)
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    model = object()
    with helper(model) as active:
        assert active is model


@pytest.mark.parametrize("feature", FUSIONS)
def test_mlx_fusion_that_cannot_be_entered_keeps_native(
    monkeypatch, mlx_inference_patches, feature
):
    """Packing can fail on the model in hand (headroom, an unsupported layout) after the
    module imported cleanly. That must degrade to native, not fail the request."""
    from core.inference import mlx_inference

    @contextmanager
    def refuse(_model):
        raise RuntimeError("cannot pack this model")
        yield  # pragma: no cover

    monkeypatch.setattr(mlx_inference_patches, f"fused_{feature}", refuse)
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    model = object()
    with mlx_inference._mlx_optional_fusion(f"fused_{feature}", model) as active:
        assert active is model


@pytest.mark.parametrize("feature", FUSIONS)
@pytest.mark.parametrize(
    "message",
    [
        "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
        "(0000000b:kIOGPUCommandBufferCallbackErrorTimeout)",
        "[METAL] Command buffer execution failed: Ignored (for causing prior/excessive GPU "
        "errors) (0000000e:kIOGPUCommandBufferCallbackErrorSubmissionsIgnored)",
    ],
    ids = ["watchdog", "ignored"],
)
def test_mlx_dead_gpu_queue_is_not_reported_as_a_working_fallback(
    monkeypatch, mlx_inference_patches, feature, message
):
    from core.inference import mlx_inference

    @contextmanager
    def fault(_model):
        raise RuntimeError(message)
        yield  # pragma: no cover

    monkeypatch.setattr(mlx_inference_patches, f"fused_{feature}", fault)
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    with pytest.raises(RuntimeError, match = "METAL"):
        with mlx_inference._mlx_optional_fusion(f"fused_{feature}", object()):
            pass  # pragma: no cover
    assert mlx_inference._MLX_FUSION_UNAVAILABLE == set()


@pytest.mark.parametrize("feature", FUSIONS)
def test_mlx_packing_that_runs_out_of_memory_still_keeps_native(
    monkeypatch, mlx_inference_patches, feature
):
    from core.inference import mlx_inference

    @contextmanager
    def refuse(_model):
        raise RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory.")
        yield  # pragma: no cover

    monkeypatch.setattr(mlx_inference_patches, f"fused_{feature}", refuse)
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    model = object()
    with mlx_inference._mlx_optional_fusion(f"fused_{feature}", model) as active:
        assert active is model


def test_mlx_dead_gpu_queue_at_load_fails_the_load(monkeypatch, mlx_moe):
    from core.inference import mlx_inference

    backend = _install_fake_text_stack(monkeypatch, {"p": [1, 2], "generated": [7, 8]}, [])
    _install_fake_fast_mlx(monkeypatch, [])
    sys.modules["mlx.core"].clear_cache = lambda: None

    @contextmanager
    def fault(_model):
        raise RuntimeError("[METAL] Command buffer execution failed: Caused GPU Timeout Error.")
        yield  # pragma: no cover

    monkeypatch.setattr(mlx_moe, "fused_moe_gate_up", fault)
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    config = SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = False)
    with pytest.raises(RuntimeError, match = "METAL"):
        backend.load_model(config)


def test_mlx_fusion_failure_at_load_still_loads_the_model(monkeypatch, mlx_moe):
    """The base-model scope is entered inside load_model, so a fusion that refuses there
    must not turn a good load into a failed one."""
    from core.inference import mlx_inference

    backend = _install_fake_text_stack(monkeypatch, {"p": [1, 2], "generated": [7, 8]}, [])
    _install_fake_fast_mlx(monkeypatch, [])
    sys.modules["mlx.core"].clear_cache = lambda: None

    @contextmanager
    def refuse(_model):
        raise RuntimeError("no headroom to pack")
        yield  # pragma: no cover

    monkeypatch.setattr(mlx_moe, "fused_moe_gate_up", refuse)
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    config = SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = False)
    assert backend.load_model(config) is True
    assert backend.active_model_name == "fake/text"
    assert backend._model is not None
    assert backend.unload_model("fake/text")


@pytest.mark.parametrize(
    "outcome", ["warmed", "int8", "ane", "failed", "dead_queue", "distributed"]
)
def test_a_load_warms_zoo_scopes_under_its_int8_decision(monkeypatch, mlx_moe, outcome):
    from core.inference import mlx_inference

    backend = _install_fake_text_stack(monkeypatch, {"p": [1, 2], "generated": [7, 8]}, [])
    _install_fake_fast_mlx(monkeypatch, [])
    sys.modules["mlx.core"].clear_cache = lambda: None
    warnings = _capture_rng_warnings(monkeypatch, mlx_inference)
    events = []

    @contextmanager
    def rng():
        events.append("rng")
        try:
            yield
        finally:
            events.append("/rng")

    @contextmanager
    def gate_up(model, **_):
        events.append("gate_up")
        yield model

    @contextmanager
    def int8(
        model,
        int8_prefill = None,
        **_,
    ):
        events.append(("int8", int8_prefill))
        try:
            yield model
        finally:
            events.append("/int8")

    def warm(model, int8_prefill):
        events.append(("warm", model is backend._model, int8_prefill))
        if outcome == "failed":
            raise RuntimeError("no headroom")
        if outcome == "dead_queue":
            raise RuntimeError("[METAL] Command buffer execution failed: Caused GPU Timeout Error.")

    mlx_moe.fused_moe_gate_up, mlx_moe.nax_quantized_linear, mlx_moe.warm_inference = (
        gate_up,
        int8,
        warm,
    )
    monkeypatch.setattr(mlx_inference, "mlx_rng_preserved", rng)
    if outcome == "int8":
        status = {"requested": True, "active": True, "reason": "", "scope": int8}
        monkeypatch.setattr(mlx_inference, "_int8_prefill_status", lambda *_: status)
    if outcome == "ane":
        # Calibration in the warm-up's first prepare rejects the split the static check allowed.
        ane = lambda model, on: (events.append(("ane", on)), contextlib.nullcontext())[1]
        reads = iter(
            [
                {"requested": True, "active": True, "reason": "", "scope": ane},
                {"requested": True, "active": False, "reason": "no_gain", "scope": None},
            ]
        )
        monkeypatch.setattr(mlx_inference, "_ane_prefill_status", lambda *_: next(reads))
    config = SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = False)
    group = SimpleNamespace(size = lambda: 2, rank = lambda: 0)
    kwargs = (
        {"parallel_mode": "tensor", "distributed_group": group} if outcome == "distributed" else {}
    )
    with (
        pytest.raises(RuntimeError, match = "METAL")
        if outcome == "dead_queue"
        else contextlib.nullcontext()
    ):
        assert backend.load_model(config, **kwargs) is True
    active = outcome == "int8"
    warmed = (
        []
        if outcome == "distributed"
        else [
            "rng",
            ("int8", active),
            *[("ane", True)] * (outcome == "ane"),
            ("warm", True, active),
            "/int8",
            "/rng",
        ]
    )
    assert events == ["gate_up", *warmed]
    if outcome == "ane":
        entry = backend.models["fake/text"]
        assert (entry["mlx_ane_prefill"], entry["mlx_ane_prefill_reason"]) == (False, "no_gain")
        assert not backend._ane_prefill["active"]
    assert any("MLX warm-up failed" in warning for warning in warnings) == (outcome == "failed")


def test_mlx_fusion_that_refuses_everywhere_still_generates(monkeypatch, mlx_moe):
    from core.inference import mlx_inference

    backend = _install_fake_text_stack(monkeypatch, {"p": [1, 2], "generated": [7, 8]}, [])
    tokenizer = backend._tokenizer
    _install_fake_fast_mlx(monkeypatch, [])
    sys.modules["mlx.core"].clear_cache = lambda: None
    refused = []

    def _refusing(name):
        @contextmanager
        def refuse(_model):
            refused.append(name)
            raise RuntimeError("no headroom to pack")
            yield  # pragma: no cover

        return refuse

    for name in FUSIONS:
        monkeypatch.setattr(mlx_moe, f"fused_{name}", _refusing(name))
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    config = SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = False)
    assert backend.load_model(config) is True
    backend._tokenizer = tokenizer

    stream = backend.generate_chat_response(messages = [{"role": "user", "content": "p"}])
    assert next(stream) == "7"
    stream.close()
    assert Counter(refused) == Counter({**dict.fromkeys(FUSIONS, 1), "moe_gate_up": 2})


def _all_fusions_refuse(monkeypatch, patches):
    from core.inference import mlx_inference

    refused = []

    def _refusing(name):
        @contextmanager
        def refuse(_model):
            refused.append(name)
            raise RuntimeError("no headroom to pack")
            yield  # pragma: no cover

        return refuse

    for name in FUSIONS:
        monkeypatch.setattr(patches, f"fused_{name}", _refusing(name))
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    return refused


def test_mlx_vlm_generation_survives_every_fusion_refusing(monkeypatch, mlx_inference_patches):
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.prompt_utils = SimpleNamespace(
        MODEL_CONFIG = {}, apply_chat_template = lambda *_a, **_k: "<image> prompt"
    )
    mlx_vlm.stream_generate = lambda *_a, **_k: iter(
        [SimpleNamespace(text = "ok", prompt_tokens = 1, generation_tokens = 1)]
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_a, **_k: "<image> prompt",
    )
    monkeypatch.setattr(
        mlx_inference, "_temporary_mlx_adapter_state", lambda *_a, **_k: contextlib.nullcontext()
    )
    refused = _all_fusions_refuse(monkeypatch, mlx_inference_patches)

    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "generic_vlm"})
    backend._processor = SimpleNamespace(chat_template = "template")
    backend._is_vlm = True
    backend._reads_vision = True
    args = (
        [{"role": "user", "content": [{"type": "image"}]}],
        [object()],
        0.7,
        0.9,
        40,
        0.01,
        4,
        1.0,
        None,
    )
    assert list(backend._generate_vlm(*args)) == ["ok"]
    assert Counter(refused) == Counter(dict.fromkeys(FUSIONS, 1))


def test_mlx_audio_input_generation_survives_every_fusion_refusing(
    monkeypatch, mlx_inference_patches
):
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    stats = dict(prompt_tokens = 3, prompt_tps = 1.0, generation_tokens = 3, generation_tps = 1.0)
    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.stream_generate = lambda *_a, **_k: (
        SimpleNamespace(text = text, **stats) for text in ("H", "e", "l")
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
    monkeypatch.setattr(
        mlx_inference,
        "_render_registered_vlm_prompt",
        lambda *_a, **_k: "P<audio>",
    )
    refused = _all_fusions_refuse(monkeypatch, mlx_inference_patches)

    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._generation_lock = __import__("threading").Lock()
    backend._model, backend._processor = _audio_model(), _audio_processor()
    backend.active_model_name = "m"
    backend.last_generation_stats = None
    backend.models = {"m": {"audio_type": "audio_vlm"}}
    assert list(
        backend.generate_audio_input_response(
            messages = [{"role": "user", "content": "what is said?"}],
            system_prompt = "",
            audio_array = [0.0, 0.1, -0.1],
            max_new_tokens = 64,
        )
    ) == ["H", "e", "l"]
    assert Counter(refused) == Counter(dict.fromkeys(FUSIONS, 1))


class _Ids(list):
    def tolist(self):
        return list(self)


def _run_studio_site(site, monkeypatch):
    """Drive one of Studio's own MLX generation paths once, on fakes."""
    from core.inference import mlx_inference

    backend = mlx_inference.MLXInferenceBackend()
    if site == "text":
        backend = _install_fake_text_stack(monkeypatch, {"p": [1, 2], "generated": [7]}, [])
        tokenizer = backend._tokenizer
        _install_fake_fast_mlx(monkeypatch, [])
        sys.modules["mlx.core"].clear_cache = lambda: None
        config = SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = False)
        assert backend.load_model(config)
        backend._tokenizer = tokenizer
        assert list(backend.generate_chat_response(messages = [{"role": "user", "content": "p"}]))
    elif site.startswith("vision"):
        if site == "vision-drafted":
            pytest.importorskip("mlx_lm")
            utils = types.ModuleType("mlx_vlm.utils")
            utils.prepare_inputs = lambda *_a, **_k: {"input_ids": [_Ids([7])]}
            monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
            backend._speculative_draft = SimpleNamespace(
                draft_kind = "mtp", draft_n = 1, draft_n_accepted = 1, prepare = lambda *_a: None
            )
        assert "draft_model" in _drive_vlm_generation(backend, monkeypatch) or site == "vision"
    elif site == "audio":
        fake_vlm = types.ModuleType("mlx_vlm")
        fake_vlm.stream_generate = lambda *_a, **_k: iter(
            [SimpleNamespace(text = "H", prompt_tokens = 3, generation_tokens = 1)]
        )
        monkeypatch.setitem(sys.modules, "mlx_vlm", fake_vlm)
        monkeypatch.setattr(mlx_inference, "_render_registered_vlm_prompt", lambda *_a, **_k: "P")
        backend._model, backend._processor = _audio_model(), _audio_processor()
        backend.active_model_name, backend.models = "m", {"m": {"audio_type": "audio_vlm"}}
        response = backend.generate_audio_input_response(
            messages = [{"role": "user", "content": "?"}],
            system_prompt = "",
            audio_array = [0.0, 0.1],
            max_new_tokens = 1,
        )
        assert list(response) == ["H"]
    else:
        generate = types.ModuleType("mlx_lm.generate")
        generate.BatchGenerator = lambda *_a, **_k: SimpleNamespace(close = lambda: None)
        backend._model, backend._tokenizer = SimpleNamespace(), SimpleNamespace(eos_token_ids = [2])
        with (
            monkeypatch.context() as patch
        ):  # only while it is built: mlx_lm itself imports from it
            patch.setitem(sys.modules, "mlx_lm.generate", generate)
            mlx_inference._TextBatchSession(backend, width = 2).close()


def test_experts_on_disk_prefill_every_text_and_batch_path_in_the_planned_step(
    monkeypatch, mlx_inference_patches
):
    from dataclasses import dataclass

    from core.inference import mlx_inference

    engine = _batch_engine(
        monkeypatch
    )  # before the fake MLX, which unsloth_zoo cannot import under
    _install_fake_mlx(monkeypatch)
    backend = _byte_level_mlx_backend(monkeypatch, [b"a", b"b"])
    stream, asked = sys.modules["mlx_lm"].stream_generate, []
    sys.modules["mlx_lm"].stream_generate = lambda m, t, **kw: asked.append(kw) or stream(
        m, t, **kw
    )
    chat = lambda: list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "?"}], max_new_tokens = 2
        )
    )
    chat()
    assert "prefill_step_size" not in asked[-1]
    backend._prefill_step = 8192
    chat()
    assert asked[-1]["prefill_step_size"] == 8192

    built = []
    generate = types.ModuleType("mlx_lm.generate")
    generate.BatchGenerator = lambda *_a, **k: built.append(k) or SimpleNamespace(
        close = lambda: None
    )
    backend._tokenizer.eos_token_ids = [2]
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "mlx_lm.generate", generate)
        mlx_inference._TextBatchSession(backend, width = 2).close()
        generate.BatchGenerator = (
            lambda model, stop_tokens, completion_batch_size, prefill_batch_size: (
                built.append({}) or SimpleNamespace(close = lambda: None)
            )
        )
        mlx_inference._TextBatchSession(backend, width = 2).close()
    assert built[-2]["prefill_step_size"] == 8192 and built[-1] == {}

    @dataclass
    class Defaults:
        prefill_batch_size: int = 1
        completion_batch_size: int = 1
        prefill_step_size: int = None

    @dataclass
    class Older:
        prefill_batch_size: int = 1
        completion_batch_size: int = 1

    opened = []
    monkeypatch.setattr(
        engine,
        "BatchStream",
        lambda *a, defaults, **k: opened.append(defaults) or SimpleNamespace(close = lambda: None),
        raising = False,
    )
    backend._processor = object()
    for kind, step in ((Defaults, 2048), (Older, None)):
        monkeypatch.setattr(engine, "GenerationDefaults", kind, raising = False)
        mlx_inference._VisionBatchSession(backend, width = 2).close()
        assert getattr(opened[-1], "prefill_step_size", None) == step


@pytest.fixture(params = ["stub", "zoo"])
def zoo_registry(request, monkeypatch, mlx_inference_patches):
    """Register scopes in zoo's registry only, entered through the stub's composer or zoo's own."""
    if request.param == "stub":
        register = lambda name, serves = (), opt_in = None: (
            mlx_inference_patches.INFERENCE_SCOPES.append((name, serves, opt_in))
        )
        return register
    monkeypatch.delitem(sys.modules, "unsloth_zoo.mlx.inference")
    try:
        zoo = importlib.import_module("unsloth_zoo.mlx.inference")
    except Exception as error:
        pytest.skip(f"unsloth_zoo.mlx.inference is not importable here: {error}")
    finally:
        monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.inference", mlx_inference_patches)
    if not hasattr(zoo, "INFERENCE_SCOPES"):
        pytest.skip("this unsloth_zoo has no inference scope registry")
    monkeypatch.setattr(zoo, "INFERENCE_SCOPES", [])
    monkeypatch.setattr(mlx_inference_patches, "inference_scopes", zoo.inference_scopes)

    def register(
        name,
        serves = (),
        opt_in = None,
    ):
        scope = getattr(mlx_inference_patches, name)
        enter = lambda model, *args, _modules: scope(model, *args)
        enter.__name__ = name
        zoo.INFERENCE_SCOPES.append(zoo.InferenceScope(enter, frozenset(serves), opt_in))

    return register


@pytest.mark.parametrize("site", ["text", "vision", "vision-drafted", "audio", "text-batch"])
def test_a_scope_zoo_registers_reaches_each_studio_site_with_no_studio_change(
    monkeypatch, mlx_inference_patches, zoo_registry, site
):
    from core.inference import mlx_inference

    entered = []

    def scope(name):
        @contextmanager
        def enter(model, *_args):
            entered.append(name)
            yield model

        return enter

    traits = ("speculative", "vision", "audio", "batch")
    for name, serves, opt_in in (
        ("every", (), None),
        *((trait, (trait,), None) for trait in traits),
        ("switched_off", (), "UNSLOTH_MLX_STUDIO_TEST_SCOPE"),
    ):
        setattr(mlx_inference_patches, name, scope(name))
        zoo_registry(name, serves, opt_in)
    monkeypatch.delenv("UNSLOTH_MLX_STUDIO_TEST_SCOPE", raising = False)
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    _run_studio_site(site, monkeypatch)
    assert mlx_inference._MLX_FUSION_UNAVAILABLE == set()  # the registry and every scope entered
    call = {
        "text": (),
        "vision": ("vision",),
        "vision-drafted": ("speculative", "vision"),
        "audio": ("vision", "audio"),
        "text-batch": ("batch",),
    }[site]
    assert entered == ["every", *call]


def test_an_older_zoo_without_the_registry_keeps_the_scopes_studio_listed(
    monkeypatch, mlx_inference_patches
):
    from core.inference import mlx_inference

    pytest.importorskip("mlx_lm")  # the drafted vision site imports zoo's generate
    entered = []
    for name in (*(f"fused_{name}" for name in FUSIONS), "exact_speculative_verify"):
        monkeypatch.setattr(
            mlx_inference_patches,
            name,
            lambda model, name = name: entered.append(name) or contextlib.nullcontext(model),
        )
    monkeypatch.delattr(mlx_inference_patches, "inference_scopes")
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    backend = _install_fake_text_stack(monkeypatch, {"p": [1, 2], "generated": [7, 8]}, [])
    _install_fake_fast_mlx(monkeypatch, [])
    sys.modules["mlx.core"].clear_cache = lambda: None
    assert backend.load_model(
        SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = False)
    )
    legacy = [f"fused_{name}" for name in FUSIONS]
    # The load-time gate/up pack, then each site as Studio entered it before the registry.
    want = {
        "load": ["fused_moe_gate_up"],
        "text": ["fused_moe_gate_up", *legacy],
        "vision": legacy,
        "vision-drafted": [*legacy, "exact_speculative_verify"],
        "audio": legacy,
        "text-batch": [],
    }
    assert entered == want["load"]
    for site in ("text", "vision", "vision-drafted", "audio", "text-batch"):
        entered.clear()
        _run_studio_site(site, monkeypatch)
        assert entered == want[site], site
    assert mlx_inference._MLX_FUSION_UNAVAILABLE == set()


@pytest.mark.parametrize("dead", [False, True], ids = ["refused", "dead-queue"])
@pytest.mark.parametrize("failing", ["inference_scopes", "fused_moe_router"])
def test_a_registry_or_scope_that_fails_to_enter_keeps_the_request_native(
    monkeypatch, mlx_inference_patches, failing, dead
):
    """The whole registry is guarded like each scope in it: logged once, unless the Metal queue died."""
    from core.inference import mlx_inference

    message = "[METAL] Command buffer execution failed: Caused GPU Timeout Error" if dead else "no"

    @contextmanager
    def broken(*_a, **_k):
        raise RuntimeError(message)
        yield  # pragma: no cover

    monkeypatch.setattr(mlx_inference_patches, failing, broken)
    monkeypatch.setattr(mlx_inference, "_MLX_FUSION_UNAVAILABLE", set())
    if dead:
        with pytest.raises(RuntimeError, match = "METAL"):
            _run_studio_site("audio", monkeypatch)
        assert mlx_inference._MLX_FUSION_UNAVAILABLE == set()
    else:
        _run_studio_site("audio", monkeypatch)
        assert [key[0] for key in mlx_inference._MLX_FUSION_UNAVAILABLE] == [failing]


def _zoo_inference_imports():
    """Names zoo's own modules import from unsloth_zoo.mlx.inference, read from source without importing zoo."""
    import importlib.util

    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("unsloth_zoo source is not available")
    root = Path(next(iter(spec.submodule_search_locations))) / "mlx"
    names = set()
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
            if isinstance(node, ast.ImportFrom) and (
                (node.level == 1 and node.module == "inference")
                or node.module == "unsloth_zoo.mlx.inference"
            ):
                names.update(alias.name for alias in node.names)
    if not names:
        pytest.skip("this unsloth_zoo imports nothing from mlx.inference")
    return names


def test_the_stub_answers_every_helper_zoo_imports_from_inference(mlx_inference_patches):
    import importlib

    stub = importlib.import_module("unsloth_zoo.mlx.inference")
    assert stub is mlx_inference_patches
    missing = sorted(name for name in _zoo_inference_imports() if not hasattr(stub, name))
    assert not missing, f"zoo imports {missing} from mlx.inference and the stub cannot answer them"


def test_an_unknown_zoo_helper_is_neutral_but_a_deleted_fusion_stays_missing(
    monkeypatch, mlx_inference_patches
):
    from unsloth_zoo.mlx.inference import nax_quantized_linear  # noqa: F401

    model = object()
    with mlx_inference_patches.some_future_helper(model) as active:
        assert active is model
    monkeypatch.delattr(mlx_inference_patches, "fused_moe_router")
    with pytest.raises(ImportError):
        from unsloth_zoo.mlx.inference import fused_moe_router  # noqa: F401
    assert not hasattr(mlx_inference_patches, "_private")


@pytest.mark.parametrize(
    "args, kwargs",
    [((), {}), ((object(),), {}), ((object(), True), {}), ((object(),), {"int8_prefill": False})],
)
def test_every_stub_scope_takes_any_arguments_and_yields_the_model(
    mlx_inference_patches, args, kwargs
):
    for name in [*(f"fused_{f}" for f in FUSIONS), "nax_quantized_linear", "some_future_helper"]:
        with getattr(mlx_inference_patches, name)(*args, **kwargs) as active:
            assert active is (args[0] if args else None), name


def test_the_stub_accepts_every_call_zoo_makes_to_an_inference_helper(mlx_inference_patches):
    """Read zoo's calls to the helpers it imports from mlx.inference and replay each one's shape
    against the stub, so a new argument fails here on Linux, not only in the macOS job."""
    import importlib.util

    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("unsloth_zoo source is not available")
    imported = _zoo_inference_imports()
    root = Path(next(iter(spec.submodule_search_locations))) / "mlx"
    calls = []
    for path in sorted(root.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in imported
            ):
                calls.append((path.name, node))
    if not calls:
        pytest.skip("this unsloth_zoo calls no mlx.inference helper by name")
    for filename, call in calls:
        args = [object() for _ in call.args]
        kwargs = {kw.arg: object() for kw in call.keywords if kw.arg}
        helper = getattr(mlx_inference_patches, call.func.id)
        if call.func.id in PLAIN_HELPERS or call.func.id in NEUTRAL_PLAIN_HELPERS:
            helper(*args, **kwargs)
            continue
        with helper(*args, **kwargs):
            pass


def _without_docstring(body):
    """The body minus a leading docstring only: any other bare expression (a call, a log line) is
    behaviour, and dropping it would let the copy miss it."""
    first = body[0] if body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return body[1:]
    return body


def test_each_plain_helper_stub_matches_zoos_body():
    """A plain helper is not neutral, so the stub copies zoo's body; it must stay a copy."""
    import importlib.util
    import inspect
    import textwrap

    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("unsloth_zoo source is not available")
    source = (Path(next(iter(spec.submodule_search_locations))) / "mlx" / "inference.py").read_text(
        encoding = "utf-8"
    )
    zoo = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    for name, helper in PLAIN_HELPERS.items():
        if name not in zoo:
            continue
        ours = ast.parse(textwrap.dedent(inspect.getsource(helper))).body[0]
        ours_body, zoo_body = (
            [ast.dump(n) for n in _without_docstring(f.body)] for f in (ours, zoo[name])
        )
        assert ours_body == zoo_body, f"{name} no longer matches unsloth_zoo.mlx.inference"


@pytest.mark.parametrize("feature", FUSIONS)
def test_mlx_missing_inference_export_keeps_native(monkeypatch, mlx_inference_patches, feature):
    from core.inference import mlx_inference

    name = f"fused_{feature}"
    monkeypatch.delattr(mlx_inference_patches, name)
    model = object()
    with mlx_inference._mlx_optional_fusion(name, model) as active:
        assert active is model


class _Resp:
    def __init__(self, text, tok):
        self.text = text
        self.token = tok


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

    @staticmethod
    def synchronize(_stream = None):
        return None


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
    mlx_core.synchronize = _DummyMX.synchronize
    mlx_core.clear_cache = lambda: None
    mlx_utils.tree_unflatten = dict
    mlx_nn = types.ModuleType("mlx.nn")
    mlx_nn.Module = type("Module", (), {"__init__": lambda self: None})
    mlx_pkg.core = mlx_core
    mlx_pkg.utils = mlx_utils
    mlx_pkg.nn = mlx_nn
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    monkeypatch.setitem(sys.modules, "mlx.utils", mlx_utils)
    monkeypatch.setitem(sys.modules, "mlx.nn", mlx_nn)


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
    from core.inference.mlx_inference import _temporary_mlx_adapter_state

    _install_fake_mlx(monkeypatch)

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
    from core.inference.mlx_inference import _temporary_mlx_adapter_state

    nn = pytest.importorskip("mlx.nn")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_lm.tuner.dora import DoRALinear
    from mlx_lm.tuner.lora import LoRAEmbedding, LoRALinear, LoRASwitchLinear

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
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)
    calls = []
    _install_fake_fast_mlx(monkeypatch, calls)

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
                "quantize_embeddings": True,
            },
        )
    ]
    assert backend._is_vlm is False
    assert isinstance(backend._tokenizer, _DummyTokenizer)
    assert backend.models["fake/text"]["base_model"] is None


def test_mlx_load_offloads_what_unsloth_zoo_applies_and_says_why_not(monkeypatch):
    from core.inference import mlx_inference, mlx_memory
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)
    calls, fits, chunks, steps = [], [], [], []
    _install_fake_fast_mlx(monkeypatch, calls)
    loader = sys.modules["unsloth_zoo.mlx.loader"].FastMLXModel
    stock, refusal, has_experts = loader.from_pretrained, {"on": False}, {"on": True}

    def from_pretrained(
        *args,
        offload = None,
        **kwargs,
    ):
        refused = refusal["on"] and isinstance(offload, dict)
        if callable(offload):
            offload = None if refusal["on"] else offload("/snapshot")
        kwargs.update(
            {"offload": {k: v for k, v in offload.items() if v is not None}} if offload else {}
        )
        model, tokenizer = stock(*args, **kwargs)
        if refused:
            raise ValueError("Unsloth: offload reads the checkpoint's tensors as stored")
        if (offload or {}).get("experts") and has_experts["on"]:
            model._unsloth_offload = {"experts": {"on_disk_bytes": 7}}
        return model, tokenizer

    monkeypatch.setattr(loader, "from_pretrained", staticmethod(from_pretrained))
    monkeypatch.setattr(mlx_memory, "mlx_offload_planner", lambda: object())
    monkeypatch.setattr(
        mlx_inference,
        "_fitted_context",
        lambda *a, **kw: fits.append(kw.get("offload"))
        or chunks.append(kw.get("prefill_chunk"))
        or (None, None),
    )
    planned, history = [], []  # here auto takes 0.6 of the experts
    budget = lambda retains_history: history.append(retains_history) or 1 << 40
    monkeypatch.setattr(mlx_inference, "mlx_memory_budget", budget)
    monkeypatch.setattr(mlx_inference, "mlx_checkpoint_window", lambda model_dir: 32768)
    monkeypatch.setattr(
        mlx_memory,
        "mlx_offload_plan",
        lambda model_dir, **kw: planned.append((model_dir, kw))
        or (None if kw["ple"] == "auto" else kw["ple"], 0.6, 8192),
    )
    config = SimpleNamespace(identifier = "fake/moe", is_vision = False, is_lora = False)

    base = {"max_seq_length": 0, "load_in_4bit": False}

    def load(ple, experts, *more):
        calls.clear()
        backend = MLXInferenceBackend()
        backend.load_model(config, offload = (ple, experts), **{**base, **dict(*more)})
        steps.append(backend._prefill_step)
        record = backend.models["fake/moe"]["mlx_offload"]
        return calls[-1][1].get("offload", {}), record

    sent, record = load(0.5, 0.75)
    assert sent == {"ple": 0.5, "experts": 0.75} and fits[-1] == (None, 0.75)
    assert steps[-1] is None and chunks[-1] is None
    assert record == {
        "ple": {
            "applied": None,
            "requested": 0.5,
            "reason": "the model has no PLE tables unsloth_zoo can read from disk",
        },
        "experts": {"applied": 0.75, "requested": 0.75, "reason": None},
        "disk_bytes": 7,
    }
    sent, record = load("off", "off")
    assert sent == {} and fits[-1] is None and record["ple"]["reason"] is None
    assert not planned
    sent, record = load(None, None)
    assert sent == {"experts": 0.6} and fits[-1] == (None, 0.6) and history == [True]
    assert steps[-1] == chunks[-1] == 8192
    has_experts["on"] = False
    load(None, None)
    assert steps.pop() is None and chunks.pop() is None and planned.pop()
    has_experts["on"] = True
    pick = lambda *keys: tuple(planned[-1][1][key] for key in keys)
    assert planned[-1][0] == "/snapshot" and pick("ple", "experts") == ("auto", "auto")
    assert pick("context", "max_ctx", "kv_bits", "load_in_4bit", "vision") == (
        None,
        32768,
        None,
        False,
        None,
    )
    assert (record["experts"]["applied"], record["experts"]["requested"]) == (0.6, None)
    with monkeypatch.context() as patch:  # a speculative text load is priced on mlx-vlm
        from core.inference import mlx_speculative

        patch.setattr(mlx_speculative, "speculates_on_route", lambda *a: True)
        patch.setattr(mlx_speculative, "speculation_refusal", lambda **k: None)
        patch.setattr(
            mlx_speculative,
            "resolve_speculation",
            lambda mode, *a, **k: mlx_speculative.SpecResolution(mode),
        )
        patch.setattr(mlx_speculative, "build_draft", lambda *a, **k: (None, None, None, None))
        load(None, None)
    assert planned[-1][1]["vision"] is True
    load("off", None, {"max_seq_length": 8192})  # an explicit off is planned as off
    assert pick("ple", "context", "max_ctx") == (None, 8192, 8192)
    sent, record = load(None, None, {"dtype": "float16"})  # not priced, so not planned: off
    assert sent == {} and len(planned) == 3 and record["experts"]["reason"] is None
    sent, record = load(0.5, None)  # decided: the hand-set part the model lacks says so
    assert sent == {"ple": 0.5, "experts": 0.6} and record["ple"]["reason"].startswith(
        "the model has"
    )
    # Refused by unsloth_zoo for this load: loaded again without it, saying why; auto turns off.
    refusal["on"] = True
    sent, record = load("off", 1.0)
    assert sent == {} and len(calls) == 2 and record["experts"]["applied"] is None
    assert record["experts"]["reason"].startswith("Unsloth: offload reads")
    sent, record = load(None, None)
    assert sent == {} and not (record["experts"]["applied"] or record["experts"]["reason"])
    assert steps[-1] is None
    # A hand-set part beside "auto" goes off with the planner unsloth_zoo never called, saying why.
    sent, record = load(1.0, None)
    assert (
        sent == {}
        and len(calls) == 1
        and record["ple"]["reason"].startswith("unsloth_zoo cannot offload this load")
    )
    refusal["on"] = False
    monkeypatch.setattr(mlx_memory, "mlx_offload_planner", lambda: None)
    sent, record = load("off", 1.0)
    assert (
        sent == {}
        and record["experts"]["reason"] == "the installed unsloth_zoo cannot read weights from disk"
    )


def test_auto_offload_is_planned_as_the_load_is_priced(monkeypatch):
    from core.inference import mlx_inference as mi
    from core.inference import mlx_memory as mm

    plans = []
    monkeypatch.setattr(mm, "mlx_offload_planner", lambda: object())
    monkeypatch.setattr(
        mm, "mlx_offload_plan", lambda d, **kw: plans.append(kw) or (1.0, 0.5, 8192)
    )
    monkeypatch.setattr(mi, "mlx_memory_budget", lambda retains_history: 1 << 40)
    monkeypatch.setattr(mi, "mlx_checkpoint_window", lambda d: 32768)
    monkeypatch.setattr(mi, "mlx_kv_quant_is_refused", lambda d, bits: True)
    priced = {"pinned": None, "load_in_4bit": False, "retains_history": True, "kv_bits": 2}
    auto = lambda at, *knobs: mi.mlx_resolve_offload(at, knobs, **priced)
    assert auto("/snap", "auto", "auto") == (1.0, 0.5, 8192) and plans[-1]["vision"] is None
    assert (plans[-1]["budget_bytes"], plans[-1]["kv_bits"]) == (1 << 40, None)
    mi.mlx_resolve_offload("/snap", ("auto", None), vision = True, **priced)
    assert plans[-1]["vision"] is True
    plans.pop()
    assert (
        auto("/snap", 0.5, None) == (0.5, None, None) and auto(None, "auto", "auto") == (None,) * 3
    )
    monkeypatch.setattr(mi, "mlx_memory_budget", lambda retains_history: None)
    assert auto("/snap", "auto", "auto") == (None,) * 3 and len(plans) == 1
    monkeypatch.setattr(mm, "mlx_offload_planner", lambda: None)
    monkeypatch.setattr(mi, "mlx_memory_budget", lambda retains_history: 1 / 0)
    assert auto("/snap", "auto", 0.5) == (None, 0.5, None) and len(plans) == 1
    monkeypatch.setattr(mm, "mlx_offload_planner", lambda: object())
    plan = lambda d, knobs: (None, 0.6, 8192) if "auto" in knobs else (*knobs, None)
    request = mi._mlx_offload_request(("auto", "auto"), plan)
    decide = request["kwargs"]["offload"]
    assert decide("/snap") == decide("/snap") == {"ple": None, "experts": 0.6}
    assert request["prefill_step"] == 8192


def test_an_auto_drafter_gets_offload_room_only_within_the_cap_where_it_then_fits(monkeypatch):
    from core.inference import mlx_inference as mi

    plans, fits, asked = {}, {"on": True}, []
    monkeypatch.setattr(
        mi, "mlx_resolve_offload", lambda d, knobs, drafter = None, **_: plans[bool(drafter)]
    )
    monkeypatch.setattr(mi, "mlx_checkpoint_window", lambda d: 32768)
    monkeypatch.setattr(
        mi,
        "mlx_drafter_fit",
        lambda d, ceiling, pinned, source, **kw: asked.append(
            (ceiling, pinned, source.path, source.builtin, kw)
        )
        or (fits["on"], None),
    )
    resolve = lambda auto, drafter = ("/head", True): mi.mlx_resolve_drafter_offload(
        "/snap", ("auto", "auto"), drafter, auto = auto, pinned = None, load_in_4bit = False
    )
    cases = [  # (with the drafter, without it, the auto plan)
        ((0.5, 0.53, 4096), (0.5, 0.5, 8192), "room"),
        ((None, 0.04, 4096), (None, None, 8192), "room"),
        ((0.5, 0.55, 4096), (0.5, 0.5, 8192), "alone"),
        ((0.55, 0.5, 4096), (0.5, 0.5, 8192), "alone"),
        ((None, 0.05, 4096), (None, None, 8192), "alone"),
    ]
    for room, alone, chosen in cases:
        plans.update({True: room, False: alone})
        assert resolve(True) == ({"room": room, "alone": alone}[chosen], alone)
        assert resolve(False) == (room, None)  # an explicit mode's drafter always gets its room
    assert resolve(True, None) == resolve(False, None) == (alone, None)
    # Room only the costless fit would then use: asked on the room plan against the target-only one.
    plans.update({True: (0.5, 0.53, 4096), False: (0.5, 0.5, 8192)})
    asked.clear(), fits.update(on = False)
    assert resolve(True) == ((0.5, 0.5, 8192),) * 2
    assert asked == [
        (
            32768,
            None,
            "/head",
            True,
            {
                "load_in_4bit": False,
                "costless": True,
                "baseline": {"offload": (0.5, 0.5), "prefill_chunk": 8192},
                "offload": (0.5, 0.53),
                "prefill_chunk": 4096,
            },
        )
    ]
    plans.update({True: (0.5, 0.5, 8192)})
    assert resolve(True) == ((0.5, 0.5, 8192),) * 2 and len(asked) == 1
    # An explicit drafter priced not to fit its room plan, even under MLX's own cap, keeps the
    # target-only plan, not all-on-disk.
    plans.update({True: (None, 1.0, None), False: (None, 0.78, 8192)})
    asked.clear()
    assert resolve(False) == ((None, 0.78, 8192), None)
    assert asked[-1][4] == {
        "load_in_4bit": False,
        "offload": (None, 1.0),
        "prefill_chunk": None,
        "reserve": False,
    }
    fits.update(on = None)  # unpriceable: the load keeps it, so does the plan
    assert resolve(False) == ((None, 1.0, None), None)


def test_an_auto_load_fits_its_drafter_against_the_target_only_plan(monkeypatch, tmp_path):
    pytest.importorskip("mlx.core")
    from core.inference import mlx_inference as mi
    from core.inference import mlx_memory as mm
    from core.inference import mlx_speculative

    fits = []

    class _Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            kwargs["offload"](str(tmp_path)), kwargs["offload"](str(tmp_path))  # as unsloth_zoo may
            model = SimpleNamespace(
                _src_path = str(tmp_path), _unsloth_offload = {"experts": {"on_disk_bytes": 1}}
            )
            return model, SimpleNamespace()

    loader = types.ModuleType("unsloth_zoo.mlx.loader")
    loader.FastMLXModel = _Loader
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", loader)
    monkeypatch.setattr(mm, "mlx_offload_planner", lambda: object())
    monkeypatch.setattr(
        mi,
        "mlx_resolve_offload",
        lambda d, knobs, drafter = None, **_: (
            ((None, 0.52, 4096) if drafter else (None, 0.5, 8192))
            if "auto" in knobs
            else (*knobs, None)
        ),
    )
    monkeypatch.setattr(mi, "_classify_mlx_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(
        mi.MLXInferenceBackend, "_resolve_context_lengths", lambda *a: (8192, 8192, 8192)
    )
    source = mlx_speculative.DrafterSource("mtp", str(tmp_path), True)
    monkeypatch.setattr(mlx_speculative, "speculation_refusal", lambda **_k: None)
    monkeypatch.setattr(
        mlx_speculative,
        "resolve_speculation",
        lambda mode, *_a, **_k: mlx_speculative.SpecResolution(mode, (source,)),
    )
    monkeypatch.setattr(mi, "mlx_drafter_fit", lambda *a, **kw: fits.append(kw) or (True, None))
    config = SimpleNamespace(identifier = "org/moe", is_vision = True, is_lora = False)

    mi.MLXInferenceBackend().load_model(config, max_seq_length = 0)
    assert (fits[-1]["offload"], fits[-1]["prefill_chunk"]) == ((None, 0.52), 4096)
    assert fits[-1]["baseline"] == {"offload": (None, 0.5), "prefill_chunk": 8192}
    mi.MLXInferenceBackend().load_model(config, max_seq_length = 0, speculative_type = "mtp")
    assert fits[-1]["offload"] == (None, 0.52) and fits[-1]["baseline"] is None


@pytest.mark.parametrize("decided_on_mlx_vlm", [False, True])
def test_a_load_falling_back_to_mlx_lm_plans_auto_offload_again(monkeypatch, decided_on_mlx_vlm):
    pytest.importorskip("mlx.core")
    from core.inference import mlx_inference as mi
    from core.inference import mlx_memory as mm
    from core.inference import mlx_speculative

    plans, applied = [], []

    class _Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            offload = kwargs.get("offload")
            if not kwargs["text_only"]:
                if decided_on_mlx_vlm and callable(offload):
                    offload("/snap")
                raise ValueError("mlx-vlm has no such architecture")
            applied.append(offload("/snap") if callable(offload) else offload)
            return SimpleNamespace(config = {}), SimpleNamespace()

    def resolve(model_dir, knobs, **kw):
        plans.append((knobs, kw["vision"], kw.get("drafter")))
        # A plan for the mlx-vlm route differs from the one mlx-lm needs, so a reused plan shows.
        return (None, 0.25, 4096) if kw["vision"] else (None, 1.0, 8192)

    loader = types.ModuleType("unsloth_zoo.mlx.loader")
    loader.FastMLXModel = _Loader
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", loader)
    monkeypatch.setattr(mm, "mlx_offload_planner", lambda: object())
    monkeypatch.setattr(mi, "mlx_resolve_offload", resolve)
    monkeypatch.setattr(mi, "_classify_mlx_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(
        mi.MLXInferenceBackend, "_resolve_context_lengths", lambda *a: (8192, 8192, 8192)
    )
    source = mlx_speculative.DrafterSource("dflash", "/drafter", False)
    monkeypatch.setattr(mlx_speculative, "speculation_refusal", lambda **_k: None)
    monkeypatch.setattr(
        mlx_speculative,
        "resolve_speculation",
        lambda mode, *_a, **_k: mlx_speculative.SpecResolution(mode, (source,)),
    )
    config = SimpleNamespace(identifier = "org/moe", is_vision = False, is_lora = False)

    mi.MLXInferenceBackend().load_model(config, speculative_type = "dflash")
    # The speculative mlx-vlm plan makes room for the drafter; the mlx-lm one, which does not speculate, does not.
    assert plans == [(("auto", "auto"), True, ("/drafter", False))] * decided_on_mlx_vlm + [
        (("auto", "auto"), None, None)
    ]
    assert applied == [{"ple": None, "experts": 1.0}]


def test_offload_knobs_validate_and_reload_on_change():
    from pydantic import ValidationError

    from core.inference import orchestrator
    from models.inference import EstimateMemoryRequest, LoadRequest
    from routes.inference import _mlx_runtime_settings_match

    req = lambda **knobs: LoadRequest(model = "m", model_path = "m", **knobs)
    assert (
        req(mlx_ple_offload = 1).mlx_ple_offload,
        req(mlx_expert_offload = "off").mlx_expert_offload,
    ) == (1.0, "off")
    assert req(mlx_ple_offload = "auto").mlx_ple_offload == "auto"
    for bad in (0, 1.5, -0.5, True, "on", "0.5"):
        for model in (LoadRequest, EstimateMemoryRequest):
            with pytest.raises(ValidationError):
                model(
                    model_path = "m",
                    **({"model": "m"} if model is LoadRequest else {}),
                    mlx_expert_offload = bad,
                )
    be = SimpleNamespace(active_model_name = "m", models = {"m": {"mlx_kv_quant_requested": "auto"}})
    assert _mlx_runtime_settings_match(be, req(mlx_ple_offload = "auto", mlx_expert_offload = None))
    assert not _mlx_runtime_settings_match(be, req(mlx_ple_offload = "off"))
    assert not _mlx_runtime_settings_match(be, req(mlx_expert_offload = 0.5))
    be.models["m"]["mlx_offload"] = {"ple": {"requested": None}, "experts": {"requested": 0.5}}
    assert _mlx_runtime_settings_match(be, req(mlx_expert_offload = 0.5))
    assert not _mlx_runtime_settings_match(be, req(mlx_expert_offload = 0.75))
    assert not _mlx_runtime_settings_match(be, req(mlx_expert_offload = 0.5, mlx_ple_offload = 1.0))
    import inspect

    from routes import inference as routes

    part = lambda applied, requested: {"applied": applied, "requested": requested}
    info = {"mlx_offload": {"ple": part(1.0, 1.0), "experts": part(0.6, None)}}
    assert "reads 60% of its MoE experts from disk" in routes._mlx_offload_warning(info)
    info["mlx_offload"]["experts"]["requested"] = 0.6
    assert routes._mlx_offload_warning(info) is None
    assert inspect.getsource(routes).count("memory_warning = _mlx_offload_warning(") == 3
    load = inspect.getsource(orchestrator.InferenceOrchestrator.load_model)
    assert "mlx_offload" in orchestrator._MLX_RUNTIME_MIRROR_FIELDS
    assert '"mlx_offload": tuple(mlx_offload)' in load  # carried to the worker


def test_the_worker_carries_the_offload_knobs_into_the_load_and_back(monkeypatch):
    from core.inference import worker

    model = SimpleNamespace(
        identifier = "org/moe",
        display_name = "moe",
        is_vision = False,
        is_lora = False,
        base_model = None,
        is_audio = False,
        audio_type = None,
        has_audio_input = False,
    )
    asked, sent = {}, []
    entry = {"mlx_offload": {"experts": {"applied": 0.75, "requested": 0.75}, "disk_bytes": 7}}
    backend = SimpleNamespace(
        device = "mlx",
        active_model_name = model.identifier,
        models = {model.identifier: entry},
        load_model = lambda **kw: asked.update(kw) or True,
    )
    monkeypatch.setattr(worker, "_build_model_config", lambda config: model)
    monkeypatch.setattr(worker, "_run_security_gates", lambda *a, **k: True)
    monkeypatch.setattr(worker, "_resolve_lora_4bit", lambda *a: False)
    monkeypatch.setattr(worker, "_needs_nemotron_trust", lambda *a, **k: False)
    xet = types.ModuleType("utils.hf_xet_fallback")
    xet.start_watchdog = lambda **k: SimpleNamespace(set = lambda: None)
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", xet)
    queue = SimpleNamespace(put = lambda item, *a, **k: sent.append(item))

    worker._handle_load(
        backend,
        {"model_name": model.identifier, "mlx_offload": ("off", 0.75)},
        queue,
    )
    assert asked["offload"] == ("off", 0.75)
    (loaded,) = [item for item in sent if item.get("type") == "loaded"]
    assert {k: loaded["model_info"][k] for k in entry} == entry


@pytest.mark.parametrize("is_lora", [False, True])
def test_mlx_base_fusion_lifetime_and_lora_request_scope(monkeypatch, mlx_moe, is_lora):
    backend = _install_fake_text_stack(monkeypatch, {"p": [1, 2], "generated": [7, 8]}, [])
    tokenizer = backend._tokenizer
    _install_fake_fast_mlx(monkeypatch, [])
    sys.modules["mlx.core"].clear_cache = lambda: None
    events = []

    class FusedModel(_DummyModel):
        pass

    @contextmanager
    def fusion(model):
        if type(model) is FusedModel:
            yield model
            return
        original = type(model)
        model.__class__ = FusedModel
        events.append(("enter", model))
        try:
            yield model
        finally:
            assert backend._model is model
            model.__class__ = original
            events.append(("exit", model))

    monkeypatch.setattr(mlx_moe, "fused_moe_gate_up", fusion)
    config = SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = is_lora)
    assert backend.load_model(config)
    first = backend._model
    expected_class = _DummyModel if is_lora else FusedModel
    assert type(first) is expected_class
    backend._tokenizer = tokenizer
    stream = backend.generate_chat_response(messages = [{"role": "user", "content": "p"}])
    assert next(stream) == "7"
    assert type(first) is FusedModel
    stream.close()
    assert type(first) is expected_class
    assert events == ([("enter", first), ("exit", first)] if is_lora else [("enter", first)])
    backend.reset_generation_state()
    assert type(first) is expected_class

    assert backend.load_model(config)
    second = backend._model
    assert second is not first and type(first) is _DummyModel
    assert type(second) is expected_class
    backend.unload_model(config.identifier)
    assert type(second) is _DummyModel and backend._model is None

    assert backend.load_model(config)
    third = backend._model

    def fail(*args, **kwargs):
        raise RuntimeError("load failed")

    loader = sys.modules["unsloth_zoo.mlx.loader"].FastMLXModel
    monkeypatch.setattr(loader, "from_pretrained", fail)
    with pytest.raises(RuntimeError, match = "load failed"):
        backend.load_model(config)
    assert type(third) is _DummyModel
    backend.unload_model(config.identifier)
    assert sum(event == "enter" for event, _ in events) == sum(
        event == "exit" for event, _ in events
    )


def test_mlx_text_lora_record_keeps_base_model_for_native_template(monkeypatch):
    # A LoRA adapter's tokenizer often lacks a chat template; the tool template is on the base.
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)
    calls = []
    _install_fake_fast_mlx(monkeypatch, calls)

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
    from core.inference.mlx_inference import MLXInferenceBackend

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
                "quantize_embeddings": False,
            },
        )
    ]
    assert cfg_path.read_text() == original_cfg
    assert backend._is_vlm is True
    assert isinstance(backend._processor, _DummyProcessor)
    assert isinstance(backend._tokenizer, _DummyTokenizer)


def test_mlx_inference_distributed_vlm_forwards_group_to_fast_mlx(monkeypatch):
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)
    calls = []
    _install_fake_fast_mlx(monkeypatch, calls)

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


# The route layer forwards UI toggles as these four template kwargs.


def test_mlx_generate_chat_response_accepts_template_kwargs():
    from core.inference.mlx_inference import MLXInferenceBackend
    import inspect

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


@pytest.mark.parametrize("ending", ["exhaust", "close", "error"])
@pytest.mark.parametrize("reverse", [False, True])
def test_vlm_iterator_restores_each_callers_stream_and_closes_on_generation_stream(
    ending, reverse, monkeypatch, native_vlm_generation_context
):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm.generate")
    if not mx.metal.is_available():
        pytest.skip("Metal is required")
    from core.inference import mlx_inference

    monkeypatch.setattr(mlx_inference, "_vlm_generation_context", native_vlm_generation_context)
    from core.inference.mlx_inference import _iter_vlm_responses
    from mlx_vlm.generate import generation_stream

    original = mx.default_stream(mx.gpu)
    callers = [original, mx.new_stream(mx.gpu)]
    if reverse:
        callers.reverse()
    with mx.stream(generation_stream):
        generation = mx.default_stream(mx.gpu)
    assert all(caller != generation for caller in callers)
    seen, closed = [], []

    def responses():
        try:
            for index in range(3):
                seen.append(mx.default_stream(mx.gpu))
                if index == 2 and ending == "error":
                    raise RuntimeError("generation failed")
                yield index
        finally:
            closed.append(mx.default_stream(mx.gpu))

    iterator = _iter_vlm_responses(responses())
    for index, caller in enumerate(callers):
        with mx.stream(caller):
            assert next(iterator) == index
            assert mx.default_stream(mx.gpu) == caller
        assert mx.default_stream(mx.gpu) == original
    with mx.stream(callers[0]):
        if ending == "close":
            iterator.close()
        elif ending == "error":
            with pytest.raises(RuntimeError, match = "generation failed"):
                next(iterator)
        else:
            assert next(iterator) == 2
            with pytest.raises(StopIteration):
                next(iterator)
        assert mx.default_stream(mx.gpu) == callers[0]
    assert mx.default_stream(mx.gpu) == original
    assert seen == [generation] * (2 if ending == "close" else 3)
    assert closed == [generation]


@pytest.mark.parametrize("feature", FUSIONS)
def test_mlx_vlm_reemits_think_prefill_inside_adapter_context(
    monkeypatch, mlx_moe, mlx_decode, feature
):
    """Prefilled <think> is re-emitted as the first VLM snapshot, so a pre-token cancel does not drop it."""
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    MLXInferenceBackend = mlx_inference.MLXInferenceBackend

    order = []
    stream_state = {"fail": False, "in_generation": False}

    @contextmanager
    def _generation_context():
        assert not stream_state["in_generation"]
        stream_state["in_generation"] = True
        try:
            yield
        finally:
            stream_state["in_generation"] = False

    monkeypatch.setattr(mlx_inference, "_vlm_generation_context", _generation_context)

    @contextmanager
    def _adapter_state(_model, state):
        assert backend._generation_lock.locked()
        order.append("adapter_enter")
        try:
            yield
        finally:
            order.append("adapter_exit")

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _adapter_state)

    @contextmanager
    def _fusion(model):
        assert model is backend._model and backend._generation_lock.locked()
        assert order[-1] == "snapshots_released"
        order.append("fusion_enter")
        try:
            yield
        finally:
            assert backend._generation_lock.locked()
            order.append("fusion_exit")

    monkeypatch.setattr(
        mlx_moe if feature == "moe_gate_up" else mlx_decode, f"fused_{feature}", _fusion
    )
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
        assert order[-1] == "fusion_enter"
        assert stream_state["in_generation"]
        try:
            if stream_state["fail"]:
                raise RuntimeError("generation failed")
            yield SimpleNamespace(text = "ok", prompt_tokens = 3, generation_tokens = 1)
        finally:
            assert stream_state["in_generation"]
            order.append("producer_closed")

    mlx_vlm.stream_generate = _vlm_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _t, _m, **_k: "<image> model-aware",
    )

    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "deepseek_vl_v2"})
    backend._processor = SimpleNamespace(tokenizer = SimpleNamespace())
    backend._reads_vision = True
    monkeypatch.setattr(
        backend, "_release_vlm_snapshots", lambda: order.append("snapshots_released")
    )
    args = ([{"role": "user", "content": [{"type": "image"}]}], [object()], 0, 1, 0, 0, 1, 1, None)

    gen = backend._generate_vlm(*args, _adapter_state = False)
    assert next(gen) == "<think>\n"
    entered = ["adapter_enter", "snapshots_released", "fusion_enter"]
    assert order == entered
    assert next(gen) == "<think>\nok"
    assert not stream_state["in_generation"]
    gen.close()
    completed = entered + ["producer_closed", "fusion_exit", "adapter_exit"]
    assert order == completed
    assert not backend._generation_lock.locked()
    stream_state["fail"] = True
    with pytest.raises(RuntimeError, match = "generation failed"):
        list(backend._generate_vlm(*args, _adapter_state = False))
    assert order == completed * 2
    assert not backend._generation_lock.locked()


def test_mlx_vlm_generation_selects_renderer_by_capability(monkeypatch):
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

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
    backend._reads_vision = True
    backend._processor = SimpleNamespace(tokenizer = SimpleNamespace())
    args = ([{"role": "user", "content": [{"type": "image"}]}], [object()], 0, 1, 0, 0, 1, 1, None)
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
    assert list(backend._generate_vlm(*((text_messages, []) + args[2:]), tools = tools)) == ["ok"]
    assert calls["generic"][-1]["tools"] == tools
    assert calls["stream"][-1][0][2] == "generic prompt"
    two_images = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}]}]
    with pytest.raises(RuntimeError, match = "2 structured image item"):
        list(backend._generate_vlm(*((two_images,) + args[1:]), tools = tools))
    state["generic"] = "serialized"
    tool_history = args[0] + [{"role": "assistant", "tool_calls": [{"id": "call-1"}]}]
    with pytest.raises(RuntimeError, match = "tool-call history"):
        list(backend._generate_vlm(*((tool_history,) + args[1:]), tools = tools))
    state["generic"] = "generic prompt"
    backend._kv_quant = {"kv_bits": 3.5}
    backend._turboquant = True
    list(backend._generate_vlm(*((text_messages, None) + args[2:]), tools = tools))
    assert calls["stream"][-1][1]["kv_bits"] == 3.5
    assert calls["stream"][-1][1]["kv_quant_scheme"] == "turboquant"
    assert "prefill_step_size" not in calls["stream"][-1][1]
    backend._prefill_step = 8192
    list(backend._generate_vlm(*((text_messages, None) + args[2:]), tools = tools))
    assert calls["stream"][-1][1]["prefill_step_size"] == 8192

    state["generic"] = ValueError("generic rendering failed")
    state["model"] = f"User: {args[0][0]['content']}"
    with pytest.raises(ValueError, match = "generic rendering failed"):
        list(backend._generate_vlm(*args))


def test_mlx_vlm_image_injection_reuses_media_aliases(monkeypatch):
    from core.inference.mlx_inference import MLXInferenceBackend

    from core.inference.chat_template_helpers import prompt_serializes_structured_media

    media = [{"type": "image"}]
    quoted = [{"role": "user", "content": media}, {"role": "user", "content": f"Explain {media}"}]
    assert prompt_serializes_structured_media(f"<image>\n{media[0]}", quoted[:1])
    assert not prompt_serializes_structured_media(f"<image>\nExplain {media}", quoted)
    assert prompt_serializes_structured_media(f"User: {media}\nExplain {media}", quoted)
    quoted[1]["content"] = [{"type": "text", "text": f'Explain "this" {media}'}]
    assert not prompt_serializes_structured_media(f'<image>\nExplain "this" {media}', quoted)
    json_media = [{"type": "image_url"}]
    json_repr = '{"type": "image_url"}'
    assert prompt_serializes_structured_media(f"<image>\n{json_repr}", [{"content": json_media}])
    assert not prompt_serializes_structured_media(
        f"<image>\nExplain {json_repr}",
        [{"content": json_media}, {"content": f"Explain {json_repr}"}],
    )

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._is_vlm = True
    backend._reads_vision = True
    captured = []
    backend._generate_vlm = lambda messages, *_args, **_kwargs: (
        captured.append(messages) or iter(())
    )
    messages = [{"role": "user", "content": [{"type": "image_url"}]}]
    list(backend.generate_chat_response(messages, image = object()))
    assert captured[0][0]["content"] == [{"type": "image_url"}]


def test_mlx_vlm_model_config_prefers_config_with_model_type():
    from core.inference.mlx_inference import _mlx_vlm_model_config

    m = SimpleNamespace(config = {}, _config = {"model_type": "deepseek_vl_v2"})
    assert _mlx_vlm_model_config(m) == ({"model_type": "deepseek_vl_v2"}, "deepseek_vl_v2")
    m = SimpleNamespace(config = SimpleNamespace(model_type = None), _config = {"model_type": "qwen2_vl"})
    assert _mlx_vlm_model_config(m)[1] == "qwen2_vl"
    assert _mlx_vlm_model_config(SimpleNamespace(config = {"model_type": "gemma3"})) == (
        {"model_type": "gemma3"},
        "gemma3",
    )


@pytest.mark.parametrize("feature", FUSIONS)
def test_mlx_generate_text_forwards_kwargs_into_template_helper(
    monkeypatch, mlx_moe, mlx_decode, feature
):
    """Mac text path must route through apply_chat_template_for_generation so
    reasoning / tool kwargs reach the tokenizer."""
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend
    import types as _types

    _install_fake_mlx(monkeypatch)

    MLXInferenceBackend = mlx_inference.MLXInferenceBackend
    real_adapter_state = mlx_inference._temporary_mlx_adapter_state

    # The text path renders twice: once with tools, then a tools=None probe.
    captured_calls = []

    def _fake_apply(tokenizer, messages, **kwargs):
        captured_calls.append({"tokenizer": tokenizer, "messages": messages, "kwargs": kwargs})
        return "<rendered prompt>"

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        _fake_apply,
        raising = True,
    )

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

    @contextmanager
    def _fusion(model):
        assert model is backend._model
        assert adapter_active["value"] and backend._generation_lock.locked()
        adapter_events.append(("fusion_enter", False))
        try:
            yield
        finally:
            assert adapter_active["value"] and backend._generation_lock.locked()
            adapter_events.append(("fusion_exit", False))

    monkeypatch.setattr(
        mlx_moe if feature == "moe_gate_up" else mlx_decode, f"fused_{feature}", _fusion
    )

    class _Resp:
        def __init__(self, tok):
            self.token = tok

    def _stream_generate(_model, _tokenizer, **_kw):
        assert adapter_active["value"]
        assert adapter_events[-1] == ("fusion_enter", False)
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
    completed = [(name, False) for name in ("enter", "fusion_enter", "fusion_exit", "exit")]
    assert adapter_events == completed
    stream_state["fail"] = True
    with pytest.raises(RuntimeError, match = "generation failed"):
        list(
            backend.generate_with_adapter_control(
                use_adapter = False,
                messages = [{"role": "user", "content": "ping"}],
                max_new_tokens = 1,
            )
        )
    assert adapter_events == completed * 2
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


def _byte_level_mlx_backend(monkeypatch, pieces):
    from core.inference import mlx_inference

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: "prompt",
        raising = True,
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.render_with_native_template_fallback",
        lambda formatted_prompt, **_kwargs: SimpleNamespace(
            prompt = formatted_prompt,
            reasoning_channel_markers = None,
        ),
        raising = True,
    )
    mlx_lm_pkg = types.ModuleType("mlx_lm")
    mlx_lm_sample = types.ModuleType("mlx_lm.sample_utils")
    mlx_lm_sample.make_sampler = lambda **_kwargs: object()
    mlx_lm_sample.make_logits_processors = lambda **_kwargs: None

    def _stream_generate(_model, _tokenizer, **_kwargs):
        for token_id in range(len(pieces)):
            yield SimpleNamespace(token = token_id)

    mlx_lm_pkg.stream_generate = _stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    class _ByteLevelTokenizer:
        chat_template = "x"
        all_special_ids = []

        def decode(self, ids, **_kwargs):
            return b"".join(pieces[int(i)] for i in ids).decode("utf-8", "replace")

    backend = mlx_inference.MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = _ByteLevelTokenizer()
    backend._is_vlm = False
    return backend


def _delivered(snapshots):
    # What the chat routes stream: each snapshot's growth past the previous one's length.
    text = prev = ""
    for snapshot in snapshots:
        text += snapshot[len(prev) :]
        prev = snapshot
    return text


@pytest.mark.parametrize(
    "pieces, expected",
    [
        # 🎉 and ✨ are byte-fallback pairs in Qwen3's vocab (unslothai/unsloth#8594).
        (
            [b"Party ", b"\xf0\x9f\x8e", b"\x89", b" sparkle ", b"\xe2\x9c", b"\xa8", b" done"],
            "Party \U0001f389 sparkle \u2728 done",
        ),
        ([b"\xf0\x9f", b"\x8e", b"\x89"], "\U0001f389"),
        # A turn cut mid-character still delivers what it decoded to.
        ([b"ok ", b"\xf0\x9f\x8e"], "ok \ufffd"),
    ],
)
def test_mlx_plain_stream_keeps_characters_split_across_tokens(monkeypatch, pieces, expected):
    _install_fake_mlx(monkeypatch)
    backend = _byte_level_mlx_backend(monkeypatch, pieces)
    snapshots = list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "emoji please"}],
            max_new_tokens = len(pieces),
        )
    )
    assert _delivered(snapshots) == expected


def test_mlx_stop_during_prefill_ends_the_reply_before_the_prompt_is_read(monkeypatch):
    _install_fake_mlx(monkeypatch)
    backend = _byte_level_mlx_backend(monkeypatch, [b"a", b"b"])
    cancel, chunks = threading.Event(), []

    def _stream_generate(
        _model,
        _tokenizer,
        prompt_progress_callback = None,
        **_kwargs,
    ):
        progress = prompt_progress_callback or (lambda *_: None)
        progress(0, 8)
        for done in range(2, 8, 2):
            chunks.append(done)
            cancel.set()
            progress(done, 8)
        for token_id in range(2):
            yield SimpleNamespace(token = token_id)

    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _stream_generate)
    snapshots = list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "long prompt"}],
            max_new_tokens = 2,
            cancel_event = cancel,
        )
    )
    assert chunks == [2]
    assert _delivered(snapshots) == ""


@pytest.mark.parametrize(
    "pieces, expected",
    [
        ([b"Party ", b"\xf0\x9f\x8e", b"\x89", b" done"], "Party \U0001f389 done"),
        ([b"ok ", b"\xf0\x9f\x8e"], "ok \ufffd"),
    ],
)
def test_mlx_batched_row_keeps_characters_split_across_tokens(monkeypatch, pieces, expected):
    _install_fake_mlx(monkeypatch)
    from core.inference import mlx_inference

    stream = mlx_inference._RowStream(
        decode = lambda ids: b"".join(pieces[i] for i in ids).decode("utf-8", "replace"),
    )
    snapshots = [snap for i in range(len(pieces)) for snap in stream.feed(token = i)]
    snapshots += stream.settle()
    assert _delivered(snapshots) == expected


def test_mlx_tool_turn_preserves_native_gemma_special_token_ids(monkeypatch):
    _install_fake_mlx(monkeypatch)
    from core.inference import mlx_inference
    from core.inference.tool_call_parser import parse_tool_calls_from_text

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: "prompt",
        raising = True,
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.render_with_native_template_fallback",
        lambda formatted_prompt, **_kwargs: SimpleNamespace(
            prompt = formatted_prompt,
            reasoning_channel_markers = None,
        ),
        raising = True,
    )

    mlx_lm_pkg = types.ModuleType("mlx_lm")
    mlx_lm_sample = types.ModuleType("mlx_lm.sample_utils")
    mlx_lm_sample.make_sampler = lambda **_kwargs: object()
    mlx_lm_sample.make_logits_processors = lambda **_kwargs: None

    token_ids = (1, 2, 3, 4, 3, 5, 6, 7)

    def _stream_generate(_model, _tokenizer, **_kwargs):
        for token_id in token_ids:
            yield SimpleNamespace(token = token_id)

    mlx_lm_pkg.stream_generate = _stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    class _Tokenizer:
        chat_template = "x"
        all_special_ids = [1, 3, 6, 7]
        all_special_tokens = ["<|tool_call>", '<|"|>', "<tool_call|>", "<eos>"]
        pieces = {
            1: "<|tool_call>",
            2: "call:terminal{command:",
            3: '<|"|>',
            4: "id",
            5: "}",
            6: "<tool_call|>",
            7: "<eos>",
        }

        def convert_ids_to_tokens(self, token_id):
            return self.pieces[token_id]

        def decode(
            self,
            ids,
            *,
            skip_special_tokens = False,
            **_kwargs,
        ):
            special = set(self.all_special_ids) if skip_special_tokens else set()
            return "".join(
                self.pieces[int(token_id)] for token_id in ids if token_id not in special
            )

    backend = mlx_inference.MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = _Tokenizer()
    backend._is_vlm = False

    snapshots = list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "run it"}],
            tools = [{"type": "function", "function": {"name": "terminal"}}],
            max_new_tokens = len(token_ids),
        )
    )
    assert snapshots[-1] == '<|tool_call>call:terminal{command:<|"|>id<|"|>}<tool_call|>'
    calls = parse_tool_calls_from_text(snapshots[-1], enabled_tool_names = {"terminal"})
    assert [call["function"]["name"] for call in calls] == ["terminal"]


def test_mlx_reasoning_tool_turn_keeps_provenance_across_buffered_segments(monkeypatch):
    """A kept control has to reach the parser once, where the model wrote it, and a suppressed one
    must not ride along inside the segment that carries it."""
    # Built before the fake mlx modules land, so the real detokenizer still imports.
    turn = _SpmTurn(
        (
            "<|channel>",
            "thought",
            "\n",
            "weighing",
            "<pad>",
            "▁it",
            "<channel|>",
            "<|tool_call>",
            "call:terminal{command:id}",
            "<tool_call|>",
            "<eos>",
        ),
        specials = (
            "<|channel>",
            "<channel|>",
            "<|tool_call>",
            "<tool_call|>",
            "<pad>",
            "<eos>",
        ),
        eos = "<eos>",
        chat_template = "x",
        ends = "stop",
    )
    _install_fake_mlx(monkeypatch)
    from core.inference import mlx_inference
    from core.inference.tool_call_parser import parse_tool_calls_from_text

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: "prompt",
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
    mlx_lm_sample.make_sampler = lambda **_kwargs: object()
    mlx_lm_sample.make_logits_processors = lambda **_kwargs: None
    mlx_lm_pkg.stream_generate = lambda _model, _tokenizer, **_kwargs: turn.stream()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    backend = mlx_inference.MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = turn
    backend._is_vlm = False

    snapshots = list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "run it"}],
            tools = [{"type": "function", "function": {"name": "terminal"}}],
            max_new_tokens = len(turn.ids),
        )
    )
    assert snapshots[-1] == (
        "<think>weighing it</think><|tool_call>call:terminal{command:id}<tool_call|>"
    )
    assert "<pad>" not in snapshots[-1] and "<eos>" not in snapshots[-1]
    assert snapshots[-1].count("<|tool_call>") == 1
    calls = parse_tool_calls_from_text(snapshots[-1], enabled_tool_names = {"terminal"})
    assert [call["function"]["name"] for call in calls] == ["terminal"]


def _decode_alone_and_batched(
    monkeypatch,
    turn,
    markers,
    cancel_after = None,
    prompt = "prompt",
    **request,
):
    """*turn* answered by the single path and by a text batch row: every snapshot each emits."""
    import threading

    _install_fake_mlx(monkeypatch)
    from core.inference import mlx_inference

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: prompt,
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.render_with_native_template_fallback",
        lambda formatted_prompt, **_kwargs: SimpleNamespace(
            prompt = formatted_prompt, reasoning_channel_markers = markers
        ),
    )
    rows, last = {}, "stop" if turn._ends == "stop" else "length"

    def step():
        events = [(uid, ids.pop(0)) for uid, ids in rows.items() if ids]
        return [], [
            SimpleNamespace(uid = uid, token = token, finish_reason = None if rows[uid] else last)
            for uid, token in events
        ]

    generator = SimpleNamespace(
        insert = lambda *_a, **_k: [rows.setdefault(len(rows), list(turn.ids)) and len(rows) - 1],
        next = step,
        remove = lambda uids, **_k: [rows[uid].clear() for uid in uids] and {},
        close = lambda: None,
    )
    for name, attrs in (
        ("mlx_lm", {"stream_generate": lambda *_a, **_k: turn.stream()}),
        (
            "mlx_lm.sample_utils",
            {"make_sampler": lambda **_k: None, "make_logits_processors": None},
        ),
        ("mlx_lm.generate", {"BatchGenerator": lambda *_a, **_k: generator}),
    ):
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    backend = mlx_inference.MLXInferenceBackend()
    backend._model, backend._tokenizer, backend._is_vlm = object(), turn, False
    request = {"messages": [{"role": "user", "content": "run it"}], "max_new_tokens": 99, **request}
    cancel, alone = threading.Event(), []
    for snapshot in backend.generate_chat_response(**request, cancel_event = cancel):
        alone.append(snapshot)
        if len(alone) == cancel_after:
            cancel.set()

    backend._batch_prompt_row = lambda _prompt, _state: ([], [0], None, None, [0])
    session = mlx_inference._TextBatchSession(backend, width = 2)
    # Two rows decoding together, so no state a row keeps can be shared or misrouted.
    batched = {row: [prefix] if (prefix := session.admit(request, row)) else [] for row in (0, 1)}
    while session.rows_in_flight:
        cut = cancel_after is not None and len(batched[0]) >= cancel_after
        events = session.withdraw([0, 1]) if cut else session.step()
        for row, snapshot in events:
            if snapshot is not None:
                batched[row].append(snapshot)
    session.close()
    return alone, batched


_TOOL = [{"type": "function", "function": {"name": "terminal"}}]
_CHANNELS = ("<|channel>thought", "<channel|>")
_GEMMA_SPECIALS = ("<|channel>", "<channel|>", "<|tool_call>", "<tool_call|>", "<pad>", "<eos>")
_GEMMA_TOOL_TURN = (
    *("<|channel>", "thought", "\n", "weighing", "<pad>", "▁it", "<channel|>"),
    *("<|tool_call>", "call:terminal{command:id}", "<tool_call|>", "<eos>"),
)


@pytest.mark.parametrize(
    "markers, request_, last, ends, cancel_after",
    [
        (_CHANNELS, {"tools": _TOOL}, "<eos>", "stop", None),
        (None, {"tools": _TOOL}, "<eos>", "stop", None),
        (None, {"tools": [], "tool_protocol_active": True}, "<eos>", "stop", None),
        # Stopping on the call's closer keeps it; stopping on an opener closes nothing.
        (None, {"tools": _TOOL}, "<tool_call|>", "stop", None),
        (None, {"tools": _TOOL}, "<|tool_call>", "stop", None),
        (_CHANNELS, {"tools": _TOOL}, "call:terminal{command:id}", "length", None),
        (_CHANNELS, {"tools": _TOOL, "stop": ["{command"]}, "<eos>", "stop", None),
        # Text held to the end, its tail a stop sequence's first characters.
        (_CHANNELS, {"tools": _TOOL, "stop": ["<tool_call|>!"]}, "<eos>", "stop", None),
        (_CHANNELS, {"tools": _TOOL}, "<eos>", "stop", 3),
        (_CHANNELS, {}, "<eos>", "stop", None),
    ],
)
def test_a_batched_tool_row_decodes_what_the_reply_alone_does(
    monkeypatch, markers, request_, last, ends, cancel_after
):
    pieces = _GEMMA_TOOL_TURN[: _GEMMA_TOOL_TURN.index(last) + 1]
    specials = tuple(special for special in _GEMMA_SPECIALS if special in pieces)
    eos = last if ends == "stop" else None
    # Built before the fake mlx modules land, so the real detokenizer still imports.
    turn = _SpmTurn(pieces, specials = specials, eos = eos, chat_template = "x", ends = ends)
    alone, batched = _decode_alone_and_batched(monkeypatch, turn, markers, cancel_after, **request_)
    assert alone and batched == {0: alone, 1: alone}


@pytest.mark.parametrize(
    "markers, request_, allows_reasoning",
    [
        (_CHANNELS, {"tools": _TOOL}, False),
        (_CHANNELS, {"tools": _TOOL, "reasoning_is_extracted": True}, True),
        (None, {"tools": _TOOL}, True),
        (None, {}, False),
    ],
)
def test_a_batched_constrained_row_is_planned_as_the_reply_alone_is(
    monkeypatch, markers, request_, allows_reasoning
):
    from core.inference import mlx_inference

    turn = _SpmTurn(
        _GEMMA_TOOL_TURN, specials = _GEMMA_SPECIALS, eos = "<eos>", chat_template = "x", ends = "stop"
    )
    seen, decodes, sampling = [], [], mlx_inference._mlx_sampling_processors
    decode = turn.decode
    monkeypatch.setattr(turn, "decode", lambda ids, **kw: decodes.append(kw) or decode(ids, **kw))
    # A kept </think> is what tells the grammar a closer survives the decode.
    monkeypatch.setattr(mlx_inference, "decoder_preserves_token", lambda *_a: True)
    monkeypatch.setattr(
        mlx_inference,
        "_build_grammar_constraint",
        lambda response_format, *_a, **kw: response_format
        and SimpleNamespace(
            allows_reasoning = allows_reasoning,
            decoded_dropping = lambda ids: seen.append((_a, sorted(kw.items()), ids)),
        ),
    )
    monkeypatch.setattr(
        mlx_inference,
        "_mlx_sampling_processors",
        lambda **kw: seen.append(kw["grammar_constraint"] is not None) or sampling(**kw),
    )
    request_ = {**request_, "response_format": {"type": "json_object"}}
    # The template opens <think>, which a document-only reply must not re-emit.
    alone, batched = _decode_alone_and_batched(
        monkeypatch, turn, markers, prompt = "P<think>", **request_
    )
    assert alone and batched == {0: alone, 1: alone}
    third = len(seen) // 3
    assert seen and seen == seen[:third] * 3 and True in seen
    plain = [kw for kw in decodes if kw.get("skip_special_tokens")]
    assert all(kw["clean_up_tokenization_spaces"] is False for kw in plain)


def test_mlx_text_normalizes_native_reasoning_and_close_releases_lock(monkeypatch):
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)

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
    """Gemma-style post-tool prompts open the thought channel, so output shows only the closing marker."""
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)

    post_tool_prompt = (
        "<|turn>model\n<|tool_response>response:web_search{}<tool_response|><|channel>thought\n"
    )
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
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)

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
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: "prompt",
        raising = True,
    )

    mlx_vlm_pkg = types.ModuleType("mlx_vlm")

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
    backend._reads_vision = True

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
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)

    post_tool_prompt = "<|tool_response>response:web_search{}<tool_response|><|channel>thought\n"
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: post_tool_prompt,
        raising = True,
    )

    mlx_vlm_pkg = types.ModuleType("mlx_vlm")

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
    backend._reads_vision = True

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
    from core.inference.mlx_inference import _MLXPromptCacheHistory

    _install_fake_prompt_cache_api(monkeypatch)

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
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_prompt_cache_api(monkeypatch)
    _install_fake_mlx(monkeypatch)

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
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend
    import types as _types

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
            clean_up_tokenization_spaces = True,
        ):
            return "".join(str(i) for i in ids)

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


def _zoo_image_prefill(monkeypatch, **helpers):
    zoo = types.ModuleType("unsloth_zoo.mlx.generate")
    zoo.__dict__.update(helpers)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.generate", zoo)


def test_an_image_turn_through_mlx_lm_feeds_its_embeddings_outside_the_prompt_cache(monkeypatch):
    """Two images of one size expand to the same ids, so an image turn neither reuses nor banks a cache."""
    _install_fake_prompt_cache_api(monkeypatch)
    captured, seen = [], []
    token_map = {"P2": [1, 2, 3, 7, 8, 9, 10], "generated": [7, 8]}
    backend = _install_fake_text_stack(monkeypatch, token_map, captured)
    backend._reads_vision = True
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _tok, messages, **_kw: messages[-1]["content"]
        if isinstance(messages[-1]["content"], str)
        else "P1 with an image",
    )
    _zoo_image_prefill(
        monkeypatch,
        image_prompt_embeddings = lambda _m, _t, prompt, images: (
            seen.append((prompt, images)) or ([1, 2, 3], "embedded")
        ),
    )
    image = object()
    messages = [{"role": "user", "content": "P1"}]
    list(backend.generate_chat_response(messages = messages, image = image, max_new_tokens = 4))
    _run_turn(backend, "P2")
    assert seen == [("P1 with an image", [image])]
    assert captured[0]["prompt"] == [1, 2, 3] and captured[0]["input_embeddings"] == "embedded"
    assert "prompt_cache" not in captured[0] and captured[1]["prompt"] == token_map["P2"]


def test_an_image_row_through_mlx_lm_joins_the_batch_at_its_last_token(monkeypatch):
    from core.inference import mlx_inference

    inserted, rendered = [], []
    generator = SimpleNamespace(
        insert = lambda prompts, **kw: inserted.append((prompts, kw)) or [len(inserted)],
        remove = lambda *_a, **_k: {},
        close = lambda: None,
    )
    generate = types.ModuleType("mlx_lm.generate")
    generate.BatchGenerator = lambda *_a, **_k: generator
    _zoo_image_prefill(monkeypatch, prefill_image_prompt = lambda *_a: [1, 5, 5, 2])
    backend = mlx_inference.MLXInferenceBackend()
    backend._model, backend._tokenizer = SimpleNamespace(), SimpleNamespace(eos_token_ids = [2])
    backend._is_vlm, backend._reads_vision = False, True
    backend._make_row_cache = lambda: ["cache"]
    backend._batch_prompt_row = lambda prompt, _state: ([], [9], ["text cache"], "key", [9])
    plan = SimpleNamespace(
        prompt = "P", max_tokens = 4, sampler = None, processors = (), streamed = None, think_prefix = ""
    )
    backend._plan_text_row = lambda messages, **_kw: rendered.append(messages) or plan
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "mlx_lm.generate", generate)
        session = mlx_inference._TextBatchSession(backend, width = 2)
    image, messages = object(), [{"role": "user", "content": "hi"}]
    backend._unset_generation_budget = lambda prompt, tokens: 100 - len(tokens)
    session.admit({"messages": messages}, 0)
    session.admit({"messages": messages, "image": image, "max_new_tokens": None}, 1)
    assert rendered[0] == messages and {"type": "image"} in rendered[1][-1]["content"]
    assert [kw["max_tokens"] for _prompts, kw in inserted] == [[4], [96]]
    assert [(prompts, kw["caches"], kw["all_tokens"]) for prompts, kw in inserted] == [
        ([[9]], [["text cache"]], [[]]),
        ([[2]], [["cache"]], [[1, 5, 5]]),
    ]
    assert session._rows[1].cache_key is None and session._rows[1].prompt_len == 4
    session.close()


@pytest.mark.parametrize("ends", [None, "length", "stop"])
def test_a_row_whose_own_processor_fails_leaves_and_its_neighbour_decodes_on(monkeypatch, ends):
    import numpy

    from core.inference import mlx_inference

    rows, removed, steps = {}, [], []

    def insert(_prompts, logits_processors, **_kwargs):
        rows[len(rows)] = logits_processors[0]
        return [len(rows) - 1]

    def step():
        steps.append(None)
        for processors in rows.values():
            for processor in processors:
                processor(numpy.array([1]), "logits")
        # The healthy row runs out of budget every third step, so a swallowed failure ends too.
        healthy = "length" if len(steps) % 3 == 0 else None
        return [], [
            SimpleNamespace(uid = uid, token = 5, finish_reason = healthy if uid else ends)
            for uid in rows
        ]

    generator = SimpleNamespace(
        insert = insert,
        next = step,
        remove = lambda uids, **_k: [removed.append(uid) or rows.pop(uid) for uid in uids] and {},
        close = lambda: None,
    )
    generate = types.ModuleType("mlx_lm.generate")
    generate.BatchGenerator = lambda *_a, **_k: generator
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate)
    backend = mlx_inference.MLXInferenceBackend()
    backend._model, backend._tokenizer = SimpleNamespace(), SimpleNamespace(eos_token_ids = [2])
    backend._is_vlm = False
    backend._batch_prompt_row = lambda _prompt, _state: ([], [9], None, None, [9])

    def failing(_tokens, _logits):
        raise ValueError("schema")

    plans = iter([failing, None])
    backend._plan_text_row = lambda _messages, **_kw: SimpleNamespace(
        prompt = "P",
        think_prefix = "",
        stream = mlx_inference._RowStream(decode = lambda ids: "x" * len(ids)),
        streamed = None,
        sampler = None,
        processors = [processor] if (processor := next(plans)) else [],
        max_tokens = 8,
    )
    session = mlx_inference._TextBatchSession(backend, width = 2)
    session.admit({"messages": []}, "bad")
    session.admit({"messages": []}, "good")
    (handle, failure), *events = session.step()
    assert handle == "bad" and isinstance(failure, ValueError) and str(failure) == "schema"
    # Its token was sampled unconstrained, and the single path raises before yielding it.
    assert removed == [0] and session.handles == ["good"]
    assert events == [("good", "x")]
    assert list(session.step()) == [("good", "xx")]
    session.close()
    # A drafted batch drafts only for rows whose processors all read just the history.
    penalty = mlx_inference._mlx_sampling_processors(presence_penalty = 1.5)
    guarded = mlx_inference._failing_alone(None, [*penalty, lambda _tokens, logits: logits])
    assert [p.history_only for p in guarded] == [True, False]
    # The fixed multi-reply command fails as a single reply would.
    plans = iter([failing, None])
    backend.last_batch_generation_stats = [None, None]
    with pytest.raises(ValueError, match = "schema"):
        list(
            backend._generate_session_batch([{"messages": []}] * 2, mlx_inference._TextBatchSession)
        )


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


def test_mlx_completion_prompt_skips_the_template_and_reuses_the_prompt_cache(monkeypatch):
    _install_fake_prompt_cache_api(monkeypatch)
    captured = []
    token_map = {"P1": [1, 2, 3], "generated": [7, 8]}
    backend = _install_fake_text_stack(monkeypatch, token_map, captured)
    _run_turn(backend, "P1")

    monkeypatch.setattr(backend, "_render_text_prompt", lambda *_a, **_k: pytest.fail("templated"))
    seen = {}
    monkeypatch.setattr(
        "core.inference.mlx_inference._mlx_sampling_processors", lambda **kw: seen.update(kw)
    )
    monkeypatch.setattr("core.inference.grammar_constraint.GrammarConstraint", _RawConstraint)
    ids = [1, 2, 3, 7, 8, 9]
    out = list(backend.generate_completion(ids, max_new_tokens = 4, penalty_window = 4, grammar = "G"))

    assert captured[1]["prompt"] == [9]
    assert out[-1] == "78" and backend.last_generation_stats["timings"]["cache_n"] == 5
    assert seen["penalty_prompt"] == [1, 2, 3, 7, 8, 9] and seen["penalty_window"] == 4
    assert seen["grammar_constraint"].built == ("G", backend._tokenizer)
    # The reply decodes with skip_special_tokens, so the grammar must not spell text with them.
    assert seen["grammar_constraint"].options == {"strips_special_tokens": True}
    # Unconstrained too: llama-server returns the pieces without whitespace cleanup.
    cleanups, decode = [], type(backend._tokenizer).decode
    monkeypatch.setattr(
        type(backend._tokenizer),
        "decode",
        lambda tok, ids, **kw: cleanups.append(kw.get("clean_up_tokenization_spaces"))
        or decode(tok, ids, **kw),
    )
    list(backend.generate_completion(ids, max_new_tokens = 4))
    assert cleanups and set(cleanups) == {False}


def _RawConstraint(grammar, tokenizer, **kw):
    noop = lambda *_a: None
    return SimpleNamespace(
        built = (grammar, tokenizer),
        options = kw,
        allows_reasoning = False,
        decoded_dropping = noop,
        stops_on = noop,
    )


def test_mlx_vlm_completion_prompt_runs_untemplated_from_its_ids(monkeypatch):
    from core.inference import mlx_inference

    seen = {}
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.stream_generate = lambda *a, **kw: seen.update(args = a, kwargs = kw) or iter(
        [SimpleNamespace(text = "ok", prompt_tokens = 3, generation_tokens = 1)]
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    _install_fake_mlx(monkeypatch)
    sys.modules["mlx.core"].array = lambda rows: rows
    monkeypatch.setattr(
        mlx_inference, "_temporary_mlx_adapter_state", lambda *_a: contextlib.nullcontext()
    )
    backend = mlx_inference.MLXInferenceBackend()
    backend._model, backend._is_vlm = SimpleNamespace(), True
    backend._tokenizer = backend._processor = SimpleNamespace()
    monkeypatch.setattr(backend, "_render_vlm_prompt", lambda *_a, **_k: pytest.fail("templated"))
    monkeypatch.setattr(backend, "_vlm_prompt_cache_session", lambda *a, **_k: seen.update(cache = a))
    monkeypatch.setattr(mlx_inference, "_mlx_sampling_processors", lambda **kw: seen.update(kw))
    monkeypatch.setattr("core.inference.grammar_constraint.GrammarConstraint", _RawConstraint)

    assert list(backend.generate_completion([5, 6], max_new_tokens = 2, grammar = "G")) == ["ok"]
    assert seen["args"][2:] == ("", None) and seen["kwargs"]["input_ids"] == [[5, 6]]
    assert seen["kwargs"]["skip_special_tokens"] is True
    assert seen["penalty_prompt"] == [5, 6]
    assert seen["grammar_constraint"].built == ("G", backend._tokenizer)
    assert seen["cache"] == (None, None, "")

    # With sampled ids the reply is decoded from them: the runtime's text trims the first
    # piece's leading space and keeps control tokens, which a raw continuation must not.
    pieces = {7: " Paris", 8: "<|im_start|>", 9: "!"}
    backend._tokenizer = SimpleNamespace(
        decode = lambda ids, skip_special_tokens, clean_up_tokenization_spaces: "".join(
            pieces[i] for i in ids if not (skip_special_tokens and i == 8)
        )
    )
    responses = [
        SimpleNamespace(text = text, token = token, generation_tokens = n, prompt_tokens = 2)
        for n, (token, text) in enumerate([(7, "Paris"), (8, "<|im_start|>"), (9, "!")], 1)
    ]
    mlx_vlm.stream_generate = lambda *a, **kw: iter(responses)
    assert list(backend.generate_completion([5, 6], max_new_tokens = 3))[-1] == " Paris!"


def _numpy_mx(monkeypatch):
    np = pytest.importorskip("numpy")

    class _At:
        def __init__(self, array):
            self.array = array

        def __getitem__(self, idx):
            def add(values):
                out = np.array(self.array)
                np.add.at(out, np.asarray(idx), np.asarray(values))
                return out

            return SimpleNamespace(add = add)

    class _Arr(np.ndarray):
        @property
        def at(self):
            return _At(self)

    def _logsumexp(
        a,
        axis = -1,
        keepdims = False,
    ):
        peak = a.max(axis = axis, keepdims = True)
        out = np.log(np.exp(a - peak).sum(axis = axis, keepdims = True)) + peak
        return out if keepdims else out.squeeze(axis)

    core = sys.modules["mlx.core"]
    monkeypatch.setattr(core, "float32", np.float32, raising = False)
    monkeypatch.setattr(core, "int32", np.int32, raising = False)
    monkeypatch.setattr(
        core, "array", lambda v, dtype = None: np.array(v, dtype = dtype), raising = False
    )
    monkeypatch.setattr(
        core,
        "zeros",
        lambda shape, dtype = None: np.zeros(shape, dtype = dtype).view(_Arr),
        raising = False,
    )
    monkeypatch.setattr(core, "logsumexp", _logsumexp, raising = False)
    monkeypatch.setattr(core, "argpartition", lambda a, kth: np.argpartition(a, kth), raising = False)
    monkeypatch.setattr(core, "argsort", np.argsort, raising = False)
    return np


# Probabilities the fake model gives every step: token 0 is the argmax, token 3 unlikely.
_STEP_PROBS = [0.6, 0.3, 0.05, 0.04, 0.01]


def _processed_steps(np, processors, prompt, steps):
    """(token, response) per step as mlx_lm / mlx_vlm run processors: once per step, never
    during prefill, step n + 1 computed before token n is yielded."""
    raw = np.log(np.array([_STEP_PROBS], dtype = np.float32))
    tokens = np.array(prompt)

    def step(tokens):
        logits = raw.copy()
        for processor in processors or ():
            logits = processor(tokens, logits)
        return int(np.argmax(logits))

    y = step(tokens)
    for n in range(steps):
        tokens = np.append(tokens, y)
        upcoming = step(tokens) if n + 1 < steps else None
        yield y, n + 1
        y = upcoming


def test_mlx_completion_logprobs_read_the_logits_before_logit_bias(monkeypatch):
    import math

    _install_fake_prompt_cache_api(monkeypatch)
    captured = []
    backend = _install_fake_text_stack(monkeypatch, {"generated": []}, captured)
    np = _numpy_mx(monkeypatch)

    def _stream_generate(_model, _tokenizer, **kw):
        captured.append(kw)
        steps = _processed_steps(np, kw.get("logits_processors"), kw["prompt"], kw["max_tokens"])
        for token, n in steps:
            yield SimpleNamespace(
                token = token,
                text = "",
                prompt_tokens = len(kw["prompt"]),
                prompt_tps = 1.0,
                generation_tokens = n,
                generation_tps = 1.0,
                finish_reason = None,
            )

    monkeypatch.setattr(sys.modules["mlx_lm"], "stream_generate", _stream_generate)
    sampling = dict(max_new_tokens = 2, temperature = 0.0, logit_bias = {3: 100.0})
    out = list(backend.generate_completion([1, 2], n_probs = 2, **sampling))
    entries = backend.last_generation_stats["token_logprobs"]
    assert out[-1] == "33" and [e["id"] for e in entries] == [3, 3]
    first = entries[0]
    assert first["logprob"] == pytest.approx(math.log(0.04), abs = 1e-5)
    assert first["token"] == "3" and first["bytes"] == [51]
    assert [(t["id"], t["token"], t["bytes"]) for t in first["top_logprobs"]] == [
        (0, "0", [48]),
        (1, "1", [49]),
    ]
    assert first["top_logprobs"][0]["logprob"] == pytest.approx(math.log(0.6), abs = 1e-5)
    list(backend.generate_completion([1, 2], max_new_tokens = 2))
    assert "logits_processors" not in captured[-1]
    assert "token_logprobs" not in backend.last_generation_stats


def test_mlx_logprob_tokens_keep_a_sentencepiece_word_space(monkeypatch):
    from core.inference import mlx_inference

    _install_fake_mlx(monkeypatch)
    pieces = {7: "\u2581the", 8: "Hand"}
    tokenizer = SimpleNamespace(
        decode = lambda ids, **_kw: "".join(pieces[i] for i in ids)
        .replace("\u2581", " ")
        .removeprefix(" ")
    )
    assert mlx_inference._token_bytes_text(tokenizer, 7) == (" the", [32, 116, 104, 101])
    assert mlx_inference._token_bytes_text(tokenizer, 8) == ("Hand", [72, 97, 110, 100])


def test_mlx_vlm_completion_logprobs_skip_the_draft_and_repeated_responses(monkeypatch):
    from core.inference import mlx_inference

    _install_fake_mlx(monkeypatch)
    np = _numpy_mx(monkeypatch)
    seen = {}

    def _vlm_stream(*_a, **kw):
        seen.update(kw)
        steps = list(_processed_steps(np, kw["logits_processors"], kw["input_ids"][0], 2))
        for token, n in steps:
            yield SimpleNamespace(text = "", token = token, generation_tokens = n, prompt_tokens = 2)
        # mlx-vlm repeats the last token in its closing response.
        yield SimpleNamespace(text = "", token = token, generation_tokens = n, prompt_tokens = 2)

    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.stream_generate = _vlm_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        mlx_inference, "_temporary_mlx_adapter_state", lambda *_a: contextlib.nullcontext()
    )
    backend = mlx_inference.MLXInferenceBackend()
    backend._model, backend._is_vlm = SimpleNamespace(), True
    backend._processor = SimpleNamespace()
    backend._tokenizer = SimpleNamespace(
        decode = lambda ids, skip_special_tokens, clean_up_tokenization_spaces: "".join(
            str(i) for i in ids
        )
    )
    monkeypatch.setattr(backend, "_vlm_prompt_cache_session", lambda *_a, **_k: None)
    backend._speculative_draft = SimpleNamespace(drafter = None, draft_kind = "mtp")

    out = list(backend.generate_completion([5, 6], max_new_tokens = 2, n_probs = 1))
    entries = backend.last_generation_stats["token_logprobs"]
    assert out[-1] == "00" and [e["id"] for e in entries] == [0, 0]
    assert [t["id"] for t in entries[0]["top_logprobs"]] == [0]
    assert "draft_model" not in seen
    assert mlx_inference._request_batch_gap({"n_probs": 2})
    assert mlx_inference._request_batch_gap({"n_probs": 0}) is None


def test_a_raw_vision_prompt_decodes_on_its_own_even_when_zoo_could_batch_it(monkeypatch):
    from core.inference import mlx_inference

    engine = types.ModuleType("unsloth_zoo.mlx.generate")
    engine.VLM_TOKEN_ID_PROMPTS = True
    package = types.ModuleType("unsloth_zoo.mlx")
    package.generate = engine
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx", package)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.generate", engine)
    sys.modules["unsloth_zoo"].mlx = package
    monkeypatch.setattr(mlx_inference, "_batched_vision_engine_gap", lambda _e: None)
    monkeypatch.setattr(mlx_inference, "_row_processor_gap", lambda _e, _r: None)
    backend = mlx_inference.MLXInferenceBackend()
    backend._model = SimpleNamespace(_is_vlm_model = True)
    assert backend._vlm_batch_unavailable_reason([{}, {}]) is None
    assert backend._vlm_batch_unavailable_reason([{"prompt_ids": [1]}, {"prompt_ids": [2]}])


def test_mlx_batch_rows_plan_a_raw_completion_prompt_untemplated(monkeypatch):
    from core.inference import mlx_inference

    backend = _install_fake_text_stack(monkeypatch, {"generated": [7]}, [])
    monkeypatch.setattr(backend, "_render_text_prompt", lambda *_a, **_k: pytest.fail("templated"))
    seen = {}
    monkeypatch.setattr(mlx_inference, "_mlx_sampling_processors", lambda **kw: seen.update(kw))
    monkeypatch.setattr(mlx_inference, "_raw_prompt_constraint", _RawConstraint)
    row = mlx_inference._row_kwargs({"prompt_ids": [1, 2, 3], "penalty_window": 2})
    plan = backend._plan_text_row([], grammar = "G", **row)
    assert (plan.prompt, plan.think_prefix, seen["penalty_prompt"], seen["penalty_window"]) == (
        [1, 2, 3],
        "",
        [1, 2, 3],
        2,
    )
    assert seen["grammar_constraint"].built == ("G", backend._tokenizer)
    backend._make_row_cache = lambda: None
    assert backend._batch_prompt_row(plan.prompt, None)[1] == [1, 2, 3]
    kwargs = []
    monkeypatch.setattr(type(backend._tokenizer), "decode", lambda _t, _i, **kw: kwargs.append(kw))
    plan.stream._decode([7])
    assert kwargs[-1]["clean_up_tokenization_spaces"] is False
    monkeypatch.setattr(backend, "_render_vlm_prompt", lambda *_a, **_k: pytest.fail("templated"))
    monkeypatch.setattr(
        backend, "_generation_limit", lambda prompt, limit, ids, **_k: (prompt, ids)
    )
    vision = backend._plan_vlm_request({"prompt_ids": [4, 5], "messages": []})
    assert (vision.prompt, vision.max_tokens) == ([4, 5], ([4, 5], [4, 5]))
    assert backend._vlm_batch_prompt([4, 5]) == {"prompt_token_ids": [4, 5]}


_COMPLETION_TOK = SimpleNamespace(
    encode = lambda text, add_special_tokens: [100] * add_special_tokens + [ord(c) for c in text],
    decode = lambda ids, skip_special_tokens, clean_up_tokenization_spaces = True: "".join(
        chr(i % 256) for i in ids
    ).replace(" .", "." if clean_up_tokenization_spaces else " ."),
)


@pytest.mark.parametrize(
    "prompt, ids, text",
    [
        ("ab", [100, 97, 98], "ab"),
        # Only a leading string gets the tokenizer's special tokens.
        ([5, "ab"], [5, 97, 98], "\x05ab"),
        (["ab", 5, 6, "c"], [100, 97, 98, 5, 6, 99], "ab\x05\x06c"),
        ({"prompt_string": [5, "a"]}, [5, 97], "\x05a"),
        ([199], [199], "\xc7"),
        ([97, 32, 46], [97, 32, 46], "a ."),
        ([200], ValueError, None),
        ([-1], ValueError, None),
        ([], ValueError, None),
        ([True], ValueError, None),
    ],
)
def test_mlx_completion_prompt_resolves_as_llama_server_tokenizes_it(prompt, ids, text):
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(args = SimpleNamespace(vocab_size = 200))
    backend._tokenizer = _COMPLETION_TOK
    if ids is ValueError:
        with pytest.raises(ValueError):
            backend.resolve_completion_prompt(prompt)
    else:
        assert backend.resolve_completion_prompt(prompt) == {
            "ids": ids,
            "text": text,
            "n_vocab": 200,
        }


def test_mlx_penalty_window_scores_the_last_tokens_of_prompt_and_completion():
    mx = pytest.importorskip("mlx.core")
    from core.inference import mlx_inference

    presence = mlx_inference._make_mlx_presence_penalty_processor(1.0, window = 2)
    frequency = mlx_inference._make_mlx_frequency_penalty_processor(1.0, window = 3)
    z = mx.zeros((1, 5))
    # The prompt's own tail counts from the first step; tokens past the window do not.
    assert mx.array_equal(presence(mx.array([1, 2, 3]), z)[0], mx.array([0, 0, -1, -1, 0]))
    assert mx.array_equal(frequency(mx.array([1, 3, 3, 3]), z)[0], mx.array([0, 0, 0, -3, 0]))
    off = dict(repetition_penalty = 1.3, presence_penalty = 1.0, frequency_penalty = 1.0)
    assert mlx_inference._mlx_sampling_processors(penalty_window = 0, **off) is None
    # Prefill and prompt-cache reuse hand processors only the prompt's last token at first.
    make = lambda: mlx_inference._mlx_sampling_processors(
        presence_penalty = 1.0, penalty_window = 3, penalty_prompt = [0, 1, 2, 3]
    )[0]
    tail, whole = make(), make()
    assert tail.history_only
    assert mx.array_equal(tail(mx.array([3]), z)[0], mx.array([0, -1, -1, -1, 0]))
    assert mx.array_equal(tail(mx.array([3, 4]), z)[0], mx.array([0, 0, -1, -1, -1]))
    assert mx.array_equal(whole(mx.array([0, 1, 2, 3]), z)[0], mx.array([0, -1, -1, -1, 0]))


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
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_prompt_cache_api(monkeypatch)
    _install_fake_mlx(monkeypatch)
    sys.modules["mlx.core"].clear_cache = lambda: None

    backend = MLXInferenceBackend()
    backend.active_model_name = "model-a"
    history = backend._prompt_cache()
    assert history is not None

    backend.reset_generation_state()
    assert backend._prompt_cache_history is history

    backend.unload_model("model-a")
    assert backend._prompt_cache_history is None


def test_mlx_prompt_cache_keeps_entries_over_budget_out_of_the_lru(monkeypatch):
    from core.inference.mlx_inference import _MLXPromptCacheHistory

    _install_fake_prompt_cache_api(monkeypatch)

    history = _MLXPromptCacheHistory(6, 1000)
    history.insert("key", [1, 2, 3], [_FakeCacheEntry(offset = 3, nbytes = 400)])
    assert len(history._lru.entries.get("key", {})) == 1

    history.insert("key", list(range(50)), [_FakeCacheEntry(offset = 50, nbytes = 5000)])
    stored = history._lru.entries.get("key", {})
    assert tuple([1, 2, 3]) in stored
    assert tuple(range(50)) not in stored


def test_mlx_prompt_cache_keeps_reusing_a_conversation_over_budget(monkeypatch):
    from core.inference.mlx_inference import _MLXPromptCacheHistory

    _install_fake_prompt_cache_api(monkeypatch)

    history = _MLXPromptCacheHistory(6, 1000)
    turn = list(range(50))
    history.insert("key", turn[:3], [_FakeCacheEntry(offset = 3, nbytes = 400)])
    long_cache = [_FakeCacheEntry(offset = 50, nbytes = 5000)]
    history.insert("key", turn, long_cache)
    assert tuple(turn) not in history._lru.entries["key"]

    cache, rest = history.fetch(object(), "key", turn + [50, 51])
    assert cache is long_cache
    assert rest == [50, 51]

    cache[0].offset = 52
    history.insert("key", turn + [50, 51], cache)
    cache, rest = history.fetch(object(), "key", turn[:45] + [99])
    assert cache is long_cache
    assert cache[0].offset == 45
    assert rest == [99]

    history.insert("key", turn[:45], cache)
    cache, rest = history.fetch(object(), "key", turn[:5] + [99])
    assert cache is not long_cache
    assert rest == turn[3:5] + [99]
    cache, rest = history.fetch(object(), "key", turn[:45] + [99])
    assert cache is not long_cache
    assert rest == turn[3:45] + [99]

    history.insert("key", turn[:45], long_cache)
    cache, rest = history.fetch(object(), "key", turn[:45])
    assert cache is long_cache
    assert (cache[0].offset, rest) == (44, turn[44:45])

    history.insert("key", turn[:44], cache)
    cache, rest = history.fetch(object(), "other-adapter", turn[:44] + [99])
    assert cache is not long_cache
    assert rest == turn[:44] + [99]


def test_mlx_prompt_cache_never_serves_an_over_budget_cache_it_cannot_rewind(monkeypatch):
    from core.inference.mlx_inference import _MLXPromptCacheHistory

    _install_fake_prompt_cache_api(monkeypatch, trimmable = False)
    history = _MLXPromptCacheHistory(6, 1000)
    turn = list(range(50))
    long_cache = [_FakeCacheEntry(offset = 50, nbytes = 5000)]

    history.insert("key", turn, long_cache)
    cache, rest = history.fetch(object(), "key", turn + [50])
    assert (cache, rest) == (long_cache, [50])

    history.insert("key", turn, long_cache)
    cache, rest = history.fetch(object(), "key", turn[:45] + [99])
    assert cache is not long_cache
    assert rest == turn[:45] + [99]

    _install_fake_prompt_cache_api(monkeypatch)
    history = _MLXPromptCacheHistory(6, 1000)
    history._trim = lambda cache, num: num - 1
    history.insert("key", turn, long_cache)
    cache, rest = history.fetch(object(), "key", turn[:45] + [99])
    assert cache is not long_cache


def test_mlx_batch_releases_an_over_budget_cache_while_another_row_decodes(monkeypatch):
    from core.inference.mlx_inference import _MLXPromptCacheHistory, _TextBatchSession

    _install_fake_prompt_cache_api(monkeypatch)

    history = _MLXPromptCacheHistory(6, 1000)
    turn = list(range(50))
    long_cache = [_FakeCacheEntry(offset = 50, nbytes = 5000)]
    banked = {"long": (turn, long_cache), "short": ([7, 8, 9], [_FakeCacheEntry(offset = 3)])}
    session = object.__new__(_TextBatchSession)
    session.backend = SimpleNamespace(
        _prompt_cache_history = history,
        _bank_row_cache = lambda key, *_: history.insert("key", *banked[key]),
    )
    session._settled = {}
    stream = SimpleNamespace(settle = lambda: [], drain = lambda cancelled: [])

    def feed(row, event):
        if event.finish_reason is not None:
            session.backend._bank_row_cache(row.cache_key)
            yield from session._retire(row, cancelled = False)

    session._feed = feed

    def held_after(ended, withdraw = False):
        rows = {
            handle: SimpleNamespace(
                handle = handle,
                uid = handle,
                cache_key = handle,
                all_tokens = [],
                plan = SimpleNamespace(stream = stream, streamed = None),
                ready_at = None,
                processing_at = None,
                prompt_len = 1,
                generated = 1,
                cached = 0,
                reason = "stop",
                failure = None,
            )
            for handle in banked
        }
        session._rows, session._by_uid = rows, dict(rows)
        events = [
            SimpleNamespace(uid = handle, finish_reason = "stop" if handle in ended else None)
            for handle in banked
        ]
        session.generator = SimpleNamespace(
            next = lambda: ([], events),
            remove = lambda uids, return_prompt_caches: {},
        )
        list(session.withdraw(ended) if withdraw else session.step())
        return history.fetch(object(), "key", turn + [50])[0] is long_cache

    assert not held_after(["long"])
    assert held_after(["long", "short"])
    assert not held_after(["long"], withdraw = True)
    assert held_after(["long", "short"], withdraw = True)


def test_mlx_prompt_cache_keys_on_what_the_kv_covers(monkeypatch):
    from core.inference.mlx_inference import _MLXPromptCacheHistory

    _install_fake_prompt_cache_api(monkeypatch)

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
    from core.inference.mlx_inference import _MLXPromptCacheHistory

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
    from core.inference.mlx_inference import _MLXPromptCacheHistory

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
        entry = ArraysCache(size = 2)
        entry[0] = mx.zeros((1, 4, 4), dtype = mx.float16)
        entry[1] = mx.zeros((1, 2, 4, 4), dtype = mx.float16)
        mx.eval(entry.state)
        return entry

    assert getattr(recurrent(), "offset", None) is None

    hybrid = [recurrent(), recurrent(), recurrent(), attention(KVCache(), 30)]
    assert _kv_prefix_coverage(hybrid) == 30
    nested = [CacheList(recurrent(), attention(KVCache(), 30))]
    assert _kv_prefix_coverage(nested) == 30

    # Pin it: an mlx-lm that made these trimmable must fail here, not reuse stale state.
    assert can_trim_prompt_cache(hybrid) is False
    assert can_trim_prompt_cache(nested) is False

    class _TrimmableOpaqueState:
        def is_trimmable(self):
            return True

    assert _kv_prefix_coverage([_TrimmableOpaqueState(), attention(KVCache(), 30)]) is None

    assert _kv_prefix_coverage([recurrent(), recurrent()]) is None

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


def test_mlx_prompt_cache_resume_restores_the_newest_entries_under_the_same_load(tmp_path):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.cache import KVCache

    from core.inference.mlx_inference import (
        MLXInferenceBackend,
        VLMPromptSnapshotStore,
        _MLXPromptCacheHistory,
    )

    def kv(n, seed):
        entry = KVCache()
        rows = mx.random.normal((1, 2, n, 4), key = mx.random.key(seed)).astype(mx.float16)
        entry.update_and_fetch(rows, rows + 1)
        return [entry]

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")

    def backend(window = None):
        loaded = MLXInferenceBackend()
        loaded.active_model_name = str(model_dir)
        loaded._kv_cache_window = window
        loaded._prompt_cache_history = _MLXPromptCacheHistory(6, 1 << 30)
        loaded._vlm_snapshot_store = VLMPromptSnapshotStore(1 << 30)
        return loaded

    saved = backend()
    saved._prompt_cache_history.insert("k", list(range(8)), kv(8, 0))
    newest = kv(12, 1)
    saved._prompt_cache_history.insert("k|other", list(range(12)), newest)
    saved._vlm_snapshot_store.store("k|img", list(range(6)), kv(6, 2))
    saved._vlm_snapshot_store.store("k|img", [9] * 5, kv(5, 3), replay = True)
    out = tmp_path / "save"
    out.mkdir()
    manifest = saved.save_prompt_cache_for_resume(str(out))
    assert [slot["store"] for slot in manifest["slots"]] == ["text", "vlm", "vlm"]

    restored = backend()
    assert restored.restore_prompt_cache_for_resume(manifest) == 3
    [(key, tokens, cache, _)] = restored._prompt_cache_history.resume_entries()
    assert (key, tokens) == ("k|other", list(range(12)))
    assert mx.array_equal(cache[0].state[0], newest[0].state[0]).item()
    with pytest.raises(KeyError):
        restored._prompt_cache_history._lru._trie.get("k", list(range(8)))
    vlm = restored._vlm_snapshot_store.resume_entries()
    assert [(key, prefix, replay) for key, prefix, _, replay in vlm] == [
        ("k|img", list(range(6)), False),
        ("k|img", [9] * 5, True),
    ]

    assert backend(window = 64).restore_prompt_cache_for_resume(manifest) == 0
    (model_dir / "model.safetensors").write_bytes(b"rewritten weights")
    assert backend().restore_prompt_cache_for_resume(manifest) == 0

    # A turn over the LRU budget is held outside it, and still kept.
    over_budget = _MLXPromptCacheHistory(6, 1)
    over_budget.insert("k", list(range(8)), kv(8, 4))
    assert [tokens for _, tokens, _, _ in over_budget.resume_entries()] == [list(range(8))]


def test_mlx_vlm_resume_keeps_the_boundary_an_edited_turn_extends():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.cache import KVCache

    from core.inference.mlx_inference import VLMPromptSnapshotStore

    def kv(n):
        entry = KVCache()
        rows = mx.zeros((1, 2, n, 4), dtype = mx.float16)
        entry.update_and_fetch(rows, rows)
        return [entry]

    store = VLMPromptSnapshotStore(1 << 30)
    store.store("k", list(range(12)), kv(12))
    store.store("k", list(range(14)), kv(14), replay = True)
    # The edit keeps the first 4 tokens; the old, longer boundary is then touched last.
    edited = [*range(4), 90, 91]
    store.store("k", list(range(4)), kv(4))
    store.store("k", edited, kv(6), replay = True)
    store._touch(("k", tuple(range(12))))
    assert [(prefix, replay) for _, prefix, _, replay in store.resume_entries()] == [
        (list(range(4)), False),
        (edited, True),
    ]


def test_mlx_prompt_cache_resume_keys_on_the_weights_the_load_read(tmp_path, monkeypatch):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.cache import KVCache

    from core.inference.mlx_inference import MLXInferenceBackend, _MLXPromptCacheHistory

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"weights")

    def loaded():
        backend = MLXInferenceBackend()
        backend.active_model_name = str(model_dir)
        backend._resume_files = backend._resume_weight_files()  # what load_model records
        backend._prompt_cache_history = _MLXPromptCacheHistory(6, 1 << 30)
        entry = KVCache()
        rows = mx.zeros((1, 2, 8, 4), dtype = mx.float16)
        entry.update_and_fetch(rows, rows)
        backend._prompt_cache_history.insert("k", list(range(8)), [entry])
        return backend

    def save(backend):
        out = tmp_path / f"save-{len(list(tmp_path.glob('save-*')))}"
        out.mkdir()
        return backend.save_prompt_cache_for_resume(str(out))

    old = loaded()
    manifest = save(old)
    assert manifest is not None
    # Replaced while the old weights are resident: their KV must neither be saved nor restored.
    weights.write_bytes(b"replaced weights")
    assert save(old) is None
    assert loaded().restore_prompt_cache_for_resume(manifest) == 0

    # The same bounds as llama-server's slot save.
    monkeypatch.setenv("UNSLOTH_SLOT_SAVE_MAX_BYTES", "1")
    assert save(loaded()) is None


def test_mlx_resume_cache_loads_only_mlx_cache_classes(tmp_path):
    import collections
    import pickle

    mx = pytest.importorskip("mlx.core")
    from core.inference.mlx_inference import load_resume_cache, save_resume_cache

    from mlx_lm.models.cache import make_prompt_cache

    path = str(tmp_path / "cache.safetensors")
    # Outside the MLX modules, and inside them but not a class.
    for refused in (collections.OrderedDict(), make_prompt_cache):
        save_resume_cache(path, "k", [1], [mx.zeros(2), refused])
        with pytest.raises(pickle.UnpicklingError):
            load_resume_cache(path)


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
        (_audio_processor(), False, True, None),
        (_audio_processor(), True, False, None),
        (_audio_processor(extractor = False), True, True, None),
        (_audio_processor(sr = 24000), True, True, None),
        (None, True, True, None),
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
    if expected == "audio_vlm":
        assert seen["texts"] == "P<audio>"


@pytest.mark.parametrize(
    "capability",
    [
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("zoo blew up")),
        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()),
        lambda *a, **k: SimpleNamespace(),
        lambda *a, **k: None,
    ],
)
def test_mlx_audio_classification_survives_a_broken_dependency(monkeypatch, capability):
    """A broken or odd-shaped unsloth_zoo capability call must degrade to no audio, never abort the load."""
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
    """A bare None would strip is_audio from codec checkpoints mlx-lm loads, breaking the TTS redirect."""
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
        None,
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("zoo blew up")),
        lambda *a, **k: SimpleNamespace(),
    ],
)
def test_mlx_audio_capability_survives_a_probe_that_could_not_run(monkeypatch, capability):
    """A probe that could not run must defer, not count as a verified negative that retracts audio_vlm."""
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
    """Each registered native-audio formatter keeps the requested audio count under mlx-vlm 0.6.10."""
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
    from core.inference.mlx_inference import _render_registered_vlm_prompt

    # Needs real mlx-vlm and Zoo; bare backend CI ships neither.
    pytest.importorskip("mlx_vlm.prompt_utils")
    loader = pytest.importorskip("unsloth_zoo.mlx.loader")

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
    assert config["model_type"] == published
    assert model.config is config


@pytest.mark.parametrize(
    ("published", "resolves"),
    [
        ("NemotronH_Nano_Omni_Reasoning_V3", True),
        ("SMOLVLM", True),
        ("smolvlm", True),
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

    assert list(backend.generate_audio_input_response(**args)) == ["H", "e", "l"]
    assert calls["kwargs"]["audio"] == [audio] and calls["kwargs"]["temperature"] == 0.0
    assert calls["audios"] == 1 and backend.last_generation_stats is not None
    assert "prefill_step_size" not in calls["kwargs"]
    backend._prefill_step = 8192
    list(backend.generate_audio_input_response(**args))
    assert calls["kwargs"]["prefill_step_size"] == 8192
    backend._prefill_step = None
    assert calls["messages"][-1]["content"][1:] == [{"type": "text", "text": "what is said?"}]

    args["messages"] = [{"role": "user", "content": "  "}]
    list(backend.generate_audio_input_response(**args))
    assert calls["messages"][-1]["content"][1:] == [
        {"type": "text", "text": "Please transcribe this audio."}
    ]

    args["messages"] = [
        {"role": "user", "content": "old unrelated question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": ""},
    ]
    list(backend.generate_audio_input_response(**args))
    assert calls["messages"][1:] == [
        {"role": "user", "content": "old unrelated question"},
        {"role": "assistant", "content": "old answer"},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        },
    ]

    audio_turn = {
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio},
            {"type": "text", "text": "Please transcribe this audio."},
        ],
    }
    for history, expected in (
        ([{"role": "user", "content": "My name is Nilay"}], [audio_turn]),
        (
            [{"role": "user", "content": "My name is Nilay"}, {"role": "assistant", "content": ""}],
            [audio_turn],
        ),
        (
            [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                {"role": "assistant", "content": "It is sunny."},
            ],
            [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": "It is sunny."},
                audio_turn,
            ],
        ),
        (
            [
                {"role": "user", "content": "Write a haiku"},
                {"role": "assistant", "content": "Waves fold into foam"},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Please count from one to five."},
            ],
            [
                {"role": "user", "content": "Write a haiku"},
                {"role": "assistant", "content": "Waves fold into foam"},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "Please count from one to five."},
                audio_turn,
            ],
        ),
    ):
        args["messages"] = [*history, {"role": "user", "content": ""}]
        list(backend.generate_audio_input_response(**args))
        assert calls["messages"][1:] == expected

    backend.models["m"]["audio_type"] = None
    with pytest.raises(RuntimeError, match = "not supported .* MLX"):
        next(backend.generate_audio_input_response(**args))


@pytest.mark.parametrize(
    ("samples", "cancel_at", "reply"),
    [
        (4, None, ["Paris.", {"speech": "UklGRndhdg==", "sample_rate": 24000}]),
        (0, None, ["Paris."]),
        (4, 2, []),
    ],
)
def test_mlx_spoken_reply(monkeypatch, samples, cancel_at, reply):
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    cancel, sampled, seen = threading.Event(), [], {}

    class _SpeakingModel:
        def stream_slots(self):
            for slot in range(4):
                if slot == cancel_at:
                    cancel.set()
                yield slot

    def _generate_audio(model, processor, prompt, **kwargs):
        seen.update(kwargs, prompt = prompt)
        sampled.extend(model.stream_slots())

        class _Result:
            text, sample_rate, finish_reason = "Paris.", 24000, "stop"
            audio = SimpleNamespace(size = samples)
            prompt_tokens = prompt_tps = generation_tokens = generation_tps = 1

            def to_wav_bytes(self):
                return b"RIFFwav" if samples else pytest.fail("empty WAV")

        return _Result()

    render = lambda processor, model, messages, **kwargs: messages[0]["content"]
    monkeypatch.setattr(mlx_inference, "_render_registered_vlm_prompt", render)
    generate = SimpleNamespace(generate_audio = _generate_audio)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate)
    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._generation_lock = threading.Lock()
    backend._model, backend._processor = _SpeakingModel(), _audio_processor()
    backend.active_model_name, backend.last_generation_stats = "m", None
    backend.models = {"m": {"audio_type": "audio_vlm", "speaks": True}}
    args = dict(messages = [{"role": "user", "content": ""}], system_prompt = "Be brief.")
    args.update(audio_array = [0.1], cancel_event = cancel, speak = True)

    assert list(backend.generate_audio_input_response(**args)) == reply
    assert sampled == [0, 1, 2, 3][: cancel_at or 4]
    assert "stream_slots" not in vars(backend._model)
    assert (seen["generation_mode"], seen["audio"]) == ("interleaved", [[0.1]])
    assert seen["prompt"] == "Respond with interleaved text and audio.\n\nBe brief."


@pytest.mark.parametrize(
    ("config", "served", "zoo", "expected"),
    [
        ({"model_type": "lfm3_audio"}, False, True, True),
        ({"model_type": "oxr"}, False, False, False),
        # mlx-lm serves it, so it stays on the text path whatever mlx-vlm declares.
        (
            {"model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"]},
            True,
            True,
            False,
        ),
        (None, False, True, False),
        ({}, False, True, False),
    ],
)
def test_zoo_multimodal_detection_only_for_mlx_vlm_only_families(
    monkeypatch, config, served, zoo, expected
):
    from core.inference import mlx_inference
    from utils.models import model_config

    asked = []
    monkeypatch.setattr(model_config, "_read_raw_config_json", lambda name, token: config)
    monkeypatch.setattr(mlx_inference, "_mlx_lm_serves", lambda model_type: served)
    loader = SimpleNamespace(_is_vlm = lambda cfg: asked.append(cfg) or zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", loader)

    assert mlx_inference._zoo_declares_multimodal("repo") is expected
    assert asked == ([config] if config and not served else [])


def test_zoo_multimodal_detection_fails_open_to_text(monkeypatch):
    from core.inference import mlx_inference
    from utils.models import model_config

    def unreadable(name, token):
        raise OSError("offline")

    monkeypatch.setattr(model_config, "_read_raw_config_json", unreadable)
    assert mlx_inference._zoo_declares_multimodal("repo") is False


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


@pytest.mark.parametrize("feature", FUSIONS)
def test_mlx_audio_input_honors_adapter_selection(monkeypatch, mlx_moe, mlx_decode, feature):
    """Audio must enter _temporary_mlx_adapter_state like text/vision, or base runs the loaded adapter."""
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    seen = {}
    order = []
    stream_state = {"active": False}

    @contextmanager
    def _generation_context():
        assert not stream_state["active"]
        stream_state["active"] = True
        try:
            yield
        finally:
            stream_state["active"] = False

    monkeypatch.setattr(mlx_inference, "_vlm_generation_context", _generation_context)

    @contextlib.contextmanager
    def _fake_adapter_state(model, use_adapter):
        seen["use_adapter"] = use_adapter
        seen["entered"] = True
        seen["exited"] = False
        order.append("adapter_enter")
        try:
            yield
        finally:
            seen["exited"] = True
            order.append("adapter_exit")

    @contextmanager
    def _fusion(model):
        assert model is backend._model and backend._generation_lock.locked()
        assert seen["entered"] and not seen["exited"]
        assert order[-1] == "snapshots_released"
        order.append("fusion_enter")
        try:
            yield
        finally:
            assert not seen["exited"] and backend._generation_lock.locked()
            order.append("fusion_exit")

    monkeypatch.setattr(
        mlx_moe if feature == "moe_gate_up" else mlx_decode, f"fused_{feature}", _fusion
    )

    def _fake_stream(model, processor, prompt, **kwargs):
        seen["inside"] = seen.get("entered") and not seen.get("exited")
        assert order[-1] == "fusion_enter"
        assert stream_state["active"]
        try:
            if seen.get("fail"):
                raise RuntimeError("generation failed")
            yield SimpleNamespace(
                text = "x",
                prompt_tokens = 1,
                prompt_tps = 1.0,
                generation_tokens = 1,
                generation_tps = 1.0,
            )
        finally:
            assert stream_state["active"]
            order.append("producer_closed")

    retained = []

    def _retained_stream(*args, **kwargs):
        producer = _fake_stream(*args, **kwargs)
        retained.append(producer)
        return producer

    fake_vlm = types.ModuleType("mlx_vlm")
    fake_vlm.stream_generate = _retained_stream
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
    monkeypatch.setattr(
        backend, "_release_vlm_snapshots", lambda: order.append("snapshots_released")
    )

    args = dict(
        messages = [{"role": "user", "content": "hi"}],
        system_prompt = "",
        audio_array = [0.0],
        max_new_tokens = 8,
        use_adapter = False,
    )
    list(backend.generate_audio_input_response(**args))
    assert seen["use_adapter"] is False
    assert seen["inside"] is True and seen["exited"] is True
    completed = [
        "adapter_enter",
        "snapshots_released",
        "fusion_enter",
        "producer_closed",
        "fusion_exit",
        "adapter_exit",
    ]
    assert order == completed
    stream = backend.generate_audio_input_response(**args)
    assert next(stream) == "x"
    assert not stream_state["active"]
    stream.close()
    assert order == completed * 2
    assert not backend._generation_lock.locked()
    seen["fail"] = True
    with pytest.raises(RuntimeError, match = "generation failed"):
        list(backend.generate_audio_input_response(**args))
    assert order == completed * 3
    assert not backend._generation_lock.locked()

    seen["fail"] = False
    cancelled = __import__("threading").Event()
    for index, extra in enumerate(({"cancel_event": cancelled}, {"stop": ["x"]}), start = 4):
        cancelled.set()
        list(backend.generate_audio_input_response(**args, **extra))
        assert order == completed * index
        assert not stream_state["active"]
        assert not backend._generation_lock.locked()


def test_worker_forwards_use_adapter_on_the_audio_command():
    """The wire carries the selection only when the caller set it."""
    from core.inference import worker
    import inspect

    src = inspect.getsource(worker._handle_generate_audio_input)
    assert 'cmd.get("use_adapter")' in src
    assert '"use_adapter"' in src


def test_kv_quant_status_applies_only_when_eligible(monkeypatch):
    from core.inference import mlx_inference

    monkeypatch.setattr(
        mlx_inference, "_kv_quant_eligibility", lambda m, v, b = None: ("full", "", True)
    )
    assert mlx_inference._kv_quant_status(None, object(), False)["kv_bits"] is None
    assert mlx_inference._normalize_mlx_kv_bits(7) is None
    assert mlx_inference._normalize_mlx_kv_bits("eight") is None
    assert mlx_inference._normalize_mlx_kv_bits(8) == 8
    assert [mlx_inference._normalize_mlx_kv_bits(b) for b in (2, 3, 5, 6)] == [2, 3, 5, 6]
    assert [mlx_inference.parse_mlx_kv_quant(c) for c in ("auto", "4", "tq-4", "tq-3.5")] == [
        (None, False),
        (4, False),
        (4, True),
        (3.5, True),
    ]

    refuse = mlx_inference._turboquant_refusal
    assert refuse(is_vision = False, is_distributed = False, is_lora = True)
    assert refuse(is_vision = False, is_distributed = True, is_lora = False)
    assert not refuse(is_vision = False, is_distributed = False, is_lora = False)
    assert not refuse(is_vision = True, is_distributed = True, is_lora = True)

    turbo = mlx_inference.MLXInferenceBackend()
    turbo._kv_quant = {"kv_bits": 3.5}
    turbo._turboquant = True
    width = {"kv_bits": 3.5, "kv_quant_scheme": "turboquant", "quantized_kv_start": 0}
    assert turbo._kv_runtime_quant_kwargs() == width
    assert turbo._kv_quant_generate_kwargs() == {}

    text = mlx_inference._kv_quant_status(8, object(), False)
    vlm = mlx_inference._kv_quant_status(8, object(), True)
    assert text["kv_bits"] == 8 and not text["note"]
    assert vlm["kv_bits"] == 8 and not vlm["note"]

    monkeypatch.setattr(
        mlx_inference,
        "_kv_quant_eligibility",
        lambda m, v, b = None: ("refused", "rotating cache", True),
    )
    refused = mlx_inference._kv_quant_status(8, object(), False)
    assert refused["kv_bits"] is None and refused["eligibility"] == "refused"
    assert refused["requested_kv_bits"] == 8


def _tiny_lm(
    cache_factory,
    dim = 128,
    attends_quantized = True,
):
    """``attends_quantized = False`` rejects a quantized cache, as Gemma 4's KV-shared layers do."""
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
                    keys, _ = target(k, k)
                    if not attends_quantized and not isinstance(keys, mx.array):
                        raise TypeError("incompatible function arguments")
            return mx.zeros((1, inputs.shape[1], 8))

    return _LM()


def test_an_explicit_speculative_mode_loads_through_mlx_vlm_and_attaches_what_fits(monkeypatch):
    pytest.importorskip("mlx.core")
    drafters = pytest.importorskip("unsloth_zoo.mlx.speculative")

    from core.inference import mlx_inference, mlx_speculative

    attempts, fits = [], []
    failing = []

    class _Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            attempts.append(kwargs["text_only"])
            if not kwargs["text_only"] and failing:
                raise ValueError("mlx-vlm has no such architecture")
            return SimpleNamespace(config = {}), SimpleNamespace()

    loader = types.ModuleType("unsloth_zoo.mlx.loader")
    loader.FastMLXModel = _Loader
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", loader)
    monkeypatch.setattr(mlx_inference, "_classify_mlx_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(
        mlx_inference.MLXInferenceBackend, "_resolve_context_lengths", lambda *a: (8192, 8192, 8192)
    )
    source = mlx_speculative.DrafterSource("dflash", "/drafter", False)
    monkeypatch.setattr(mlx_speculative, "speculation_refusal", lambda **_k: None)
    monkeypatch.setattr(
        mlx_speculative,
        "resolve_speculation",
        lambda mode, *_a, **_k: mlx_speculative.SpecResolution(
            mode, () if mode == "ngram" else (source,)
        ),
    )
    monkeypatch.setattr(
        mlx_inference, "mlx_drafter_fit", lambda *a, **k: fits.append(a[1:4]) or (True, 4096)
    )
    monkeypatch.setattr(drafters, "companion_drafter", lambda path, target: SimpleNamespace())
    spent, rooms = [], {"/drafter": 123, "/cheap": 500, "/unused": 0}
    monkeypatch.setattr(
        mlx_inference,
        "mlx_prompt_cache_room",
        lambda model_dir, n_ctx, source, **_k: spent.append((n_ctx, source)) or rooms[source.path],
    )
    config = SimpleNamespace(identifier = "org/text", is_vision = False, is_lora = False)

    backend = mlx_inference.MLXInferenceBackend()
    backend.load_model(config, max_seq_length = 0, speculative_type = "dflash", spec_draft_n_max = 5)
    entry = backend.models["org/text"]
    assert attempts == [False] and fits == [(8192, None, source)]
    assert backend._prompt_cache_room is None and spent == []  # a drafter that fits spends no share
    assert (entry["spec_drafter_kind"], entry["spec_fallback_reason"]) == ("dflash", None)
    assert entry["context_length"] == 4096 and entry["spec_draft_n_max"] == 5
    assert backend._speculative_draft.controller.max_depth == 5

    attempts.clear(), failing.append(True)
    backend.load_model(config, speculative_type = "dflash")
    entry = backend.models["org/text"]
    assert attempts == [False, True] and backend._speculative_draft is None
    assert entry["spec_fallback_reason"] == mlx_speculative.RUNTIME_ERROR
    attempts.clear()
    backend.load_model(config)
    assert attempts == [True] and backend.models["org/text"]["speculative_type"] == "auto"
    failing.clear()
    fit = lambda *a, **kw: fits.append(kw.get("vision")) or (None, None)
    monkeypatch.setattr(mlx_inference, "_fitted_context", fit)
    room = [True]  # whether the drafter fits under MLX's own cap, spending the prompt cache's share
    monkeypatch.setattr(
        mlx_inference,
        "mlx_drafter_fit",
        lambda *a, reserve = True, **k: (not reserve and room[-1], None),
    )
    backend.load_model(config, max_seq_length = 0, speculative_type = "ngram")
    assert fits[-1] is True  # copies alone still load through mlx-vlm, so the fit prices that route
    fits.clear()
    backend.load_model(config, max_seq_length = 0, speculative_type = "dflash")
    assert backend._speculative_draft.drafter is not None and fits == [True]
    assert backend.models["org/text"]["spec_fallback_reason"] is None
    # That drafter spent the prompt cache's share, so both caches hold only what the cap still leaves.
    assert spent == [(8192, source)] and backend._prompt_cache_room == 123
    sized = []
    monkeypatch.setattr(
        mlx_inference,
        "_MLXPromptCacheHistory",
        lambda n, max_bytes, *a, **k: sized.append(max_bytes),
    )
    monkeypatch.setattr(
        mlx_inference, "VLMPromptSnapshotStore", lambda max_bytes: sized.append(max_bytes)
    )
    backend._prompt_cache(), backend._vlm_prompt_cache_store()
    assert sized in ([123, 123], [123])  # the second only where this mlx-vlm builds the store
    # The room is the attached drafter's: not one that failed to build, nor one never tried.
    cheap = mlx_speculative.DrafterSource("dflash", "/cheap", False)
    unused = mlx_speculative.DrafterSource("dflash", "/unused", False)
    resolve, build = mlx_speculative.resolve_speculation, drafters.companion_drafter
    monkeypatch.setattr(
        mlx_speculative,
        "resolve_speculation",
        lambda *_a, **_k: mlx_speculative.SpecResolution("dflash", (cheap, source, unused)),
    )
    monkeypatch.setattr(
        drafters,
        "companion_drafter",
        lambda path, target: (_ for _ in ()).throw(ValueError("no"))
        if path == "/cheap"
        else build(path, target),
    )
    spent.clear()
    backend.load_model(config, max_seq_length = 0, speculative_type = "dflash")
    assert backend._speculative_draft.drafter is not None and backend._prompt_cache_room == 123
    assert spent == [(8192, source)]
    monkeypatch.setattr(mlx_speculative, "resolve_speculation", resolve)
    monkeypatch.setattr(drafters, "companion_drafter", build)
    room.append(False)
    backend.load_model(config, max_seq_length = 0, speculative_type = "dflash")
    assert backend._speculative_draft is None and backend._prompt_cache_room is None
    assert backend.models["org/text"]["spec_fallback_reason"] == mlx_speculative.DRAFTER_NO_MEMORY
    backend.load_model(config, max_seq_length = 0, spec_draft_model = "org/d")
    assert fits[-2:] == [True, True]  # a named drafter enters the speculative route under Auto
    # ... and is a request, so a load that cannot honour it says why.
    failing.append(True)
    backend.load_model(config, spec_draft_model = "org/d")
    monkeypatch.setattr(
        mlx_speculative, "speculation_refusal", lambda **_k: mlx_speculative.KV_QUANT
    )
    reasons = [
        entry["spec_fallback_reason"]
        for entry in (
            dict(backend.models["org/text"]),
            backend.load_model(config, spec_draft_model = "org/d") and backend.models["org/text"],
        )
    ]
    assert reasons == [mlx_speculative.RUNTIME_ERROR, mlx_speculative.KV_QUANT]
    # A diffusion model gets no draft at load, so the resident batch cannot speculate on it either.
    failing.clear()
    monkeypatch.setattr(mlx_speculative, "speculation_refusal", lambda **_k: None)
    monkeypatch.setattr(mlx_inference, "_vlm_generation_is_diffusion", lambda model: True)
    backend.load_model(config, max_seq_length = 0, speculative_type = "ngram")
    entry = backend.models["org/text"]
    assert backend._speculative_draft is None and entry["spec_drafter_kind"] is None
    assert entry["spec_fallback_reason"] == mlx_speculative.RUNTIME_ERROR


def test_a_text_model_mlx_vlm_cannot_load_is_served_without_turboquant(monkeypatch):
    pytest.importorskip("mlx.core")
    from core.inference import mlx_inference

    attempts = []

    class _Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            # The retry must not run under the failed load's traceback, which pins its weights.
            attempts.append((kwargs["text_only"], sys.exc_info()[0]))
            if not kwargs["text_only"]:
                raise ValueError("Expected shape (262144, 640) but received shape (262144, 80)")
            return SimpleNamespace(config = {}), SimpleNamespace()

    loader = types.ModuleType("unsloth_zoo.mlx.loader")
    loader.FastMLXModel = _Loader
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", loader)
    monkeypatch.setattr(mlx_inference, "_classify_mlx_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(
        mlx_inference.MLXInferenceBackend, "_resolve_context_lengths", lambda *a: (2048, 2048, 2048)
    )

    backend = mlx_inference.MLXInferenceBackend()
    config = SimpleNamespace(identifier = "org/prequantized-text", is_vision = False, is_lora = False)
    backend.load_model(config, kv_quant = "tq-4")

    assert attempts == [(False, None), (True, None)]
    assert backend._turboquant_refusal == mlx_inference.MLX_TURBOQUANT_TEXT_LOAD
    assert backend._is_vlm is False
    assert backend._kv_quant["kv_bits"] is None

    attempts.clear()
    vision = SimpleNamespace(identifier = "org/vlm", is_vision = True, is_lora = False)
    with pytest.raises(ValueError, match = "Expected shape"):
        mlx_inference.MLXInferenceBackend().load_model(vision, kv_quant = "tq-4")
    assert attempts == [(False, None)]

    class _DeadQueue(_Loader):
        @staticmethod
        def from_pretrained(name, **kwargs):
            attempts.append((kwargs["text_only"], None))
            raise RuntimeError("[METAL] Command buffer execution failed: GPU Timeout Error")

    attempts.clear()
    loader.FastMLXModel = _DeadQueue
    with pytest.raises(RuntimeError, match = "GPU Timeout"):
        mlx_inference.MLXInferenceBackend().load_model(config, kv_quant = "tq-4")
    assert attempts == [(False, None)]


def test_a_vision_checkpoint_served_through_mlx_lm_reads_images_only_where_zoo_prefills_them(
    monkeypatch,
):
    """unsloth-zoo builds an image checkpoint mlx-vlm lacks (Muse Spark) through mlx-lm."""
    pytest.importorskip("mlx.core")
    from core.inference import mlx_inference

    attempts, built_by_mlx_vlm, reads_images = [], [], []

    class _Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            attempts.append(kwargs["text_only"])
            model = SimpleNamespace(config = {}, _is_vlm_model = bool(built_by_mlx_vlm))
            if reads_images:
                model._unsloth_supports_images = True
            return model, SimpleNamespace(tokenizer = "inner")

    loader = types.ModuleType("unsloth_zoo.mlx.loader")
    loader.FastMLXModel = _Loader
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", loader)
    monkeypatch.setattr(
        mlx_inference, "_classify_mlx_audio_type", lambda *a, config_audio_type: config_audio_type
    )
    monkeypatch.setattr(
        mlx_inference.MLXInferenceBackend, "_resolve_context_lengths", lambda *a: (2048, 2048, 2048)
    )
    vision = SimpleNamespace(
        identifier = "org/vision", is_vision = True, is_lora = False, audio_type = "audio_vlm"
    )

    backend = mlx_inference.MLXInferenceBackend()
    backend.load_model(vision, kv_quant = "tq-4")
    assert attempts == [False]
    entry = backend.models["org/vision"]
    assert entry["audio_type"] is None and entry["has_audio_input"] is False
    assert backend._is_vlm is False and backend._processor is None
    assert backend._reads_vision_input() is False
    assert backend.models["org/vision"]["is_vision"] is False
    assert backend._turboquant_refusal == mlx_inference.MLX_TURBOQUANT_TEXT_LOAD

    reads_images.append(True)
    generate = types.ModuleType("unsloth_zoo.mlx.generate")
    for zoo_prefills in (False, True):
        if zoo_prefills:
            generate.prefill_image_prompt = object()
        monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.generate", generate)
        backend = mlx_inference.MLXInferenceBackend()
        backend.load_model(vision, kv_quant = "tq-4")
        assert backend._is_vlm is False and backend._reads_vision_input() is zoo_prefills
        assert backend.models["org/vision"]["is_vision"] is zoo_prefills
        assert backend.models["org/vision"]["audio_type"] is None
        assert backend._turboquant_refusal == mlx_inference.MLX_TURBOQUANT_TEXT_LOAD

    reads_images.clear()
    built_by_mlx_vlm.append(True)
    backend = mlx_inference.MLXInferenceBackend()
    backend.load_model(vision)
    assert backend._is_vlm is True and backend._reads_vision_input() is True
    assert backend.models["org/vision"]["is_vision"] is True


def test_a_uniformly_quantized_text_load_goes_through_mlx_vlm_only_when_that_batches(monkeypatch):
    pytest.importorskip("mlx.core")
    from core.inference import mlx_inference

    engine = _batch_engine(monkeypatch)
    attempts, broken, probes = [], [], []

    class _Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            attempts.append(kwargs["text_only"])
            if broken and not kwargs["text_only"]:
                raise ValueError("Model type llama4_text not supported.")
            return SimpleNamespace(config = {}), SimpleNamespace()

    loader = types.ModuleType("unsloth_zoo.mlx.loader")
    loader.FastMLXModel = _Loader
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", loader)
    monkeypatch.setattr(mlx_inference, "_classify_mlx_audio_type", lambda *a, **k: None)
    monkeypatch.setattr(
        mlx_inference.MLXInferenceBackend, "_resolve_context_lengths", lambda *a: (2048, 2048, 2048)
    )
    text = SimpleNamespace(identifier = "org/text", is_vision = False, is_lora = False)

    def routes(
        kv_quant,
        gap = None,
        is_lora = False,
        verdict = "full",
        max_seq_length = 2048,
    ):
        attempts.clear()
        probes.clear()
        text.is_lora = is_lora
        monkeypatch.setattr(
            engine, "row_quantized_prompt_cache_unavailable_reason", lambda: gap, raising = False
        )
        monkeypatch.setattr(
            mlx_inference,
            "_kv_quant_eligibility",
            lambda *a, **k: probes.append(a) or (verdict, "no cache", True),
        )
        backend = mlx_inference.MLXInferenceBackend()
        backend.load_model(text, kv_quant = kv_quant, max_seq_length = max_seq_length)
        return list(attempts), backend._turboquant_refusal

    for max_seq_length in (2048, 0):
        assert routes("4", max_seq_length = max_seq_length) == ([False], "")
        assert len(probes) == 1
    assert routes("4", gap = "old mlx-vlm") == ([True], "")
    assert routes("auto") == ([True], "")
    assert routes("4", is_lora = True) == ([True], "")
    assert routes("4", verdict = "partial") == ([False], "")
    assert routes("4", verdict = "none") == ([False, True], "")
    assert routes("tq-4", verdict = "none") == ([False], "")
    broken.append(True)
    assert routes("4") == ([False, True], "")


def test_reload_comparison_and_response_carry_the_resolved_setting():
    """Load-time knobs must reach the response model and trigger reloads on the keys actually stored."""
    from routes.inference import _mlx_runtime_settings_match
    from models.inference import LoadRequest, LoadResponse

    def req(**knobs):
        return LoadRequest(model = "m", model_path = "m", **knobs)

    be = SimpleNamespace(active_model_name = "m", models = {"m": {"mlx_kv_quant_requested": "8"}})
    assert _mlx_runtime_settings_match(be, req(mlx_kv_quant = "8"))
    assert not _mlx_runtime_settings_match(be, req(mlx_kv_quant = "4"))
    assert _mlx_runtime_settings_match(be, req(mlx_kv_bits = 8))
    assert req(mlx_kv_quant = None, mlx_kv_bits = 8).mlx_kv_quant is None
    be.models["m"] = {"mlx_kv_quant_requested": "4"}
    assert not _mlx_runtime_settings_match(be, req(mlx_kv_quant = "tq-4"))
    be.models["m"] = {"mlx_kv_quant_requested": "auto"}
    assert _mlx_runtime_settings_match(be, req(mlx_kv_quant = "tq-6"))
    assert _mlx_runtime_settings_match(
        SimpleNamespace(active_model_name = "m", models = {"m": {}}),
        req(mlx_kv_quant = "8"),
    )

    # Compared against the REQUESTED value, or a refused template reloads every request.
    be.models["m"] = {
        "mlx_kv_quant_requested": "auto",
        "chat_template_override_requested": "{{ custom }}",
    }
    assert _mlx_runtime_settings_match(be, req(chat_template_override = "{{ custom }}"))
    assert not _mlx_runtime_settings_match(be, req(chat_template_override = "{{ other }}"))
    assert not _mlx_runtime_settings_match(be, req())
    be.models["m"] = {
        "mlx_kv_quant_requested": "auto",
        "chat_template_override_requested": "{{ refused }}",
        "chat_template_override_reason": "it could not render a conversation",
    }
    assert _mlx_runtime_settings_match(be, req(chat_template_override = "{{ refused }}"))
    # Speculation compares canonically: an absent mode is the resident auto.
    be.models["m"] = {"mlx_kv_quant_requested": "auto", "speculative_type": "auto"}
    assert _mlx_runtime_settings_match(be, req(speculative_type = "default"))
    assert not _mlx_runtime_settings_match(be, req(speculative_type = "mtp"))
    be.models["m"].update(speculative_type = "mtp", spec_draft_n_max = 4)
    assert _mlx_runtime_settings_match(be, req(speculative_type = "draft-mtp", spec_draft_n_max = 4))
    assert not _mlx_runtime_settings_match(be, req(speculative_type = "mtp"))
    assert not _mlx_runtime_settings_match(
        be, req(speculative_type = "mtp", spec_draft_n_max = 4, spec_draft_model = "org/drafter")
    )

    resp = LoadResponse(
        status = "loaded",
        model = "m",
        display_name = "m",
        inference = {},
        mlx_kv_quant = "8",
        mlx_kv_quant_eligibility = "full",
        mlx_kv_quant_note = "n",
    )
    assert resp.mlx_kv_quant == "8" and resp.mlx_kv_quant_note == "n"


def _int8_zoo(
    module,
    available = True,
    events = None,
):
    events = [] if events is None else events

    @contextmanager
    def nax_quantized_linear(model, int8_prefill = None):
        events.append(("int8", int8_prefill))
        yield
        events.append(("int8_exit", int8_prefill))

    module.nax_quantized_linear = nax_quantized_linear
    reason = "" if available else "no_eligible_projections"
    verdict = SimpleNamespace(available = available, reason = reason)
    module.int8_prefill_available = lambda m: verdict
    return events


def test_int8_prefill_reloads_on_change_and_reports_why_it_is_off(mlx_inference_patches):
    from core.inference import mlx_inference
    from models.inference import LoadRequest
    from routes.inference import _mlx_runtime_settings_match

    req = lambda **knobs: LoadRequest(model = "m", model_path = "m", **knobs)
    entry = {"mlx_kv_quant_requested": "auto"}
    be = SimpleNamespace(active_model_name = "m", models = {"m": entry})
    assert _mlx_runtime_settings_match(be, req())
    assert not _mlx_runtime_settings_match(be, req(mlx_int8_prefill = True))
    be.models["m"]["mlx_int8_prefill_requested"] = True
    assert _mlx_runtime_settings_match(be, req(mlx_int8_prefill = True))
    assert not _mlx_runtime_settings_match(be, req())

    from core.inference import orchestrator

    assert "mlx_int8_prefill_requested" in orchestrator._MLX_RUNTIME_MIRROR_FIELDS
    status = mlx_inference._int8_prefill_status
    _int8_zoo(mlx_inference_patches)
    mlx_inference_patches.nax_quantized_linear = lambda model: contextlib.nullcontext()
    assert status(True, object())["reason"] == "unsupported_zoo"
    _int8_zoo(mlx_inference_patches, available = False)
    assert status(True, object())["reason"] == "no_eligible_projections"
    _int8_zoo(mlx_inference_patches)
    assert status(True, object())["active"] and status(False, object())["scope"] is None


@pytest.mark.parametrize("registry", [True, False], ids = ["zoo-registry", "older-zoo"])
@pytest.mark.parametrize("requested,available", [(True, True), (True, False), (False, True)])
def test_int8_prefill_is_the_outermost_scope_of_generation_pinned_off_when_inactive(
    monkeypatch, mlx_inference_patches, requested, available, registry
):
    from core.inference import mlx_inference

    events = _int8_zoo(mlx_inference_patches, available = available)
    if not registry:
        monkeypatch.delattr(mlx_inference_patches, "inference_scopes")

    @contextmanager
    def fusion(_model):
        events.append(("fusion", None))
        yield

    for name in FUSIONS:
        monkeypatch.setattr(mlx_inference_patches, f"fused_{name}", fusion)
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.prompt_utils = SimpleNamespace(
        MODEL_CONFIG = {}, apply_chat_template = lambda *_a, **_k: "<image> prompt"
    )

    def stream(*_a, **_k):
        events.append(("stream", None))
        yield SimpleNamespace(text = "ok", prompt_tokens = 1, generation_tokens = 1)

    mlx_vlm.stream_generate = stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_a, **_k: "<image> prompt",
    )
    monkeypatch.setattr(
        mlx_inference, "_temporary_mlx_adapter_state", lambda *_a, **_k: contextlib.nullcontext()
    )
    backend = mlx_inference.MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "generic_vlm"})
    backend._processor = SimpleNamespace(chat_template = "template")
    backend._is_vlm = backend._reads_vision = True
    backend._int8_prefill = mlx_inference._int8_prefill_status(requested, backend._model)
    args = ([{"role": "user", "content": [{"type": "image"}]}], [object()], 0.7, 0.9, 40, 0.01, 4)
    assert list(backend._generate_vlm(*args, 1.0, None)) == ["ok"]

    active = requested and available
    # Zoo's registry enters the NAX scope again in its own place; the outer decision holds there too.
    fusions = [("fusion", None)] * len(FUSIONS)
    if registry:
        fusions[5:5] = [("int8", active)]
    assert events == [("int8", active), *fusions, ("stream", None)] + [("int8_exit", active)] * (
        1 + registry
    )

    def broken(model, int8_prefill = None):
        raise ImportError("skewed zoo")

    mlx_inference_patches.nax_quantized_linear = broken
    if not active:  # an inactive load never asked for the scope, so its failure costs no request
        with backend._prefill_decision_scopes():
            pass


@pytest.mark.parametrize("requested,available", [(True, True), (True, False), (False, True)])
def test_ane_prefill_is_pinned_inside_int8_released_on_unload_and_reloads_on_change(
    monkeypatch, mlx_inference_patches, requested, available
):
    from core.inference import mlx_inference
    from models.inference import LoadRequest
    from routes.inference import _mlx_runtime_settings_match

    _install_fake_mlx(monkeypatch)

    events = _int8_zoo(mlx_inference_patches)

    @contextmanager
    def ane_prefill_mlp(model, ane_prefill = None):
        events.append(("ane", ane_prefill))
        yield
        events.append(("ane_exit", ane_prefill))

    mlx_inference_patches.ane_prefill_mlp = ane_prefill_mlp
    verdict = SimpleNamespace(available = available, reason = "" if available else "no_gain")
    mlx_inference_patches.ane_prefill_available = lambda m: verdict
    mlx_inference_patches.release_ane_prefill = lambda m: events.append(("release", None))
    active = requested and available
    backend = mlx_inference.MLXInferenceBackend()
    backend._model, backend.models["m"] = object(), {}
    backend._ane_prefill = mlx_inference._ane_prefill_status(requested, backend._model)
    assert backend._ane_prefill["reason"] == ("no_gain" if requested and not available else "")
    with backend._prefill_decision_scopes():
        pass
    backend.unload_model("m")
    assert events == [
        ("int8", False),
        ("ane", active),
        ("ane_exit", active),
        ("int8_exit", False),
    ] + ([("release", None)] if active else [])
    if active:
        # A release that raises still finishes the unload.
        def broken_release(model):
            raise RuntimeError("ane")

        mlx_inference_patches.release_ane_prefill = broken_release
        backend._model, backend.models["m"] = object(), {}
        backend._ane_prefill = mlx_inference._ane_prefill_status(True, backend._model)
        backend.unload_model("m")
        assert backend._model is None and not backend._ane_prefill["active"]
    entry = {"mlx_kv_quant_requested": "auto", "mlx_ane_prefill_requested": requested}
    load = lambda on: LoadRequest(model = "m", model_path = "m", mlx_ane_prefill = on)
    be = SimpleNamespace(active_model_name = "m", models = {"m": entry})
    assert _mlx_runtime_settings_match(be, load(requested))
    assert not _mlx_runtime_settings_match(be, load(not requested))

    @contextmanager
    def int8(model, int8_prefill = None):
        events.append(("int8", int8_prefill))
        try:
            yield
        finally:
            events.append(("int8_exit", int8_prefill))

    @contextmanager
    def dead(model, on):
        raise RuntimeError("[METAL] Command buffer execution failed")
        yield

    events.clear()
    mlx_inference_patches.nax_quantized_linear = int8
    backend._ane_prefill = {"scope": dead}
    try:
        backend._prefill_decision_scopes()
    except RuntimeError:  # unwound before the error leaves, not when its traceback is collected
        assert events == [("int8", False), ("int8_exit", False)]
    else:
        pytest.fail("the Neural Engine scope's error was swallowed")


def test_a_batch_session_holds_int8_prefill_for_its_whole_life(monkeypatch, mlx_inference_patches):
    from core.inference import mlx_inference

    events = _int8_zoo(mlx_inference_patches)
    engine = _batch_engine(monkeypatch)
    monkeypatch.setattr(engine, "GenerationDefaults", SimpleNamespace, raising = False)
    stream = SimpleNamespace(close = lambda: events.append(("close", None)))
    built = lambda *a, **k: events.append(("stream", None)) or stream
    monkeypatch.setattr(engine, "BatchStream", built, raising = False)
    backend = mlx_inference.MLXInferenceBackend()
    backend._model = backend._processor = object()
    for requested in (True, False):
        events.clear()
        backend._int8_prefill = mlx_inference._int8_prefill_status(requested, backend._model)
        mlx_inference._VisionBatchSession(backend, width = 2).close()
        # Off is pinned too, or Zoo's own scopes would read UNSLOTH_MLX_INT8_PREFILL.
        want = [("int8", requested), ("stream", None), ("close", None), ("int8_exit", requested)]
        assert events == want


def test_kv_quant_probe_reports_what_the_runtime_would_really_do(monkeypatch):
    """Static config and name proxies were wrong both ways, so probe the real KV conversion instead."""
    from core.inference import mlx_inference

    pytest.importorskip("mlx_lm")
    from mlx_lm.models import cache as lm_cache

    def verdict(
        factory,
        dim = 128,
        **kwargs,
    ):
        lm = _tiny_lm(factory, dim, **kwargs)
        monkeypatch.setattr(lm_cache, "make_prompt_cache", lambda m, **_: lm.make_cache())
        return mlx_inference._kv_quant_eligibility(lm, False, 8)

    def elig(
        factory,
        dim = 128,
        **kwargs,
    ):
        return verdict(factory, dim, **kwargs)[0]

    assert elig(lambda: [lm_cache.KVCache(), lm_cache.KVCache()]) == "full"
    # A width mx.quantize rejects is caught by attempting it, not by naming it.
    assert elig(lambda: [lm_cache.KVCache()], dim = 80) == "refused"
    assert elig(lambda: [lm_cache.CacheList(lm_cache.KVCache())]) == "none"
    windowed = lambda: [lm_cache.KVCache(), lm_cache.RotatingKVCache(max_size = 8)]
    assert elig(windowed) == "partial"
    assert "sliding-window" in verdict(windowed)[1]
    mixed = lambda: [lm_cache.KVCache()] + windowed() + [lm_cache.CacheList(lm_cache.KVCache())]
    assert verdict(mixed)[0] == "partial" and "sliding-window" not in verdict(mixed)[1]
    assert elig(lambda: [lm_cache.RotatingKVCache(max_size = 8)]) == "none"
    assert verdict(lambda: [lm_cache.KVCache() for _ in range(3)]) == ("full", "", True)
    alone = verdict(lambda: [lm_cache.RotatingKVCache(max_size = 8)] * 2 + [lm_cache.KVCache()])
    assert alone[0] == "none" and "stays unquantized" in alone[1]
    assert elig(windowed, attends_quantized = False) == "refused"
    assert elig(lambda: [lm_cache.KVCache(), lm_cache.CacheList(lm_cache.KVCache())]) == "partial"


def test_parent_mirror_omits_runtime_fields_a_backend_never_reported():
    """A CUDA load must not gain a None the reload comparison then trips on."""
    from core.inference.orchestrator import _mlx_runtime_mirror_fields

    assert _mlx_runtime_mirror_fields({"is_vision": False}) == {}
    assert _mlx_runtime_mirror_fields(
        {"mlx_kv_quant": "8", "mlx_kv_quant_requested": "8", "other": 1}
    ) == {"mlx_kv_quant": "8", "mlx_kv_quant_requested": "8"}


def test_chat_template_override_installs_only_where_one_already_exists():
    """Overrides replace an existing chat template only; creating one moves the render to another object."""
    from core.inference import mlx_inference

    tokenizer = SimpleNamespace(chat_template = "native")
    processor = SimpleNamespace(chat_template = None, tokenizer = tokenizer)
    status = mlx_inference._install_template_override(
        "custom", tokenizer, processor, lambda: "rendered"
    )
    assert status["applied"] == "custom"
    assert tokenizer.chat_template == "custom"
    assert processor.chat_template is None

    own = SimpleNamespace(chat_template = "native")
    owner = SimpleNamespace(chat_template = "native", tokenizer = own)
    mlx_inference._install_template_override("custom", own, owner, lambda: "ok")
    assert own.chat_template == "custom" and owner.chat_template == "custom"

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

    callable_tpl = SimpleNamespace(chat_template = "native", _chat_template = lambda: "x")
    assert reason(callable_tpl) == mlx_inference.MLX_TEMPLATE_CALLABLE

    # A named set would lose its tool_use variant if flattened to one string.
    assert reason(SimpleNamespace(chat_template = {"default": "a", "tool_use": "b"})) == (
        mlx_inference.MLX_TEMPLATE_NAMED_SET
    )

    # Only the rendering object counts: a nested tokenizer's named set vetoes only when
    # the processor cannot render.
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

    bare = SimpleNamespace(chat_template = None)
    assert reason(bare, SimpleNamespace(chat_template = None, tokenizer = bare)) == (
        mlx_inference.MLX_TEMPLATE_NO_TARGET
    )

    for empty in (None, "", "   "):
        blank = SimpleNamespace(chat_template = empty)
        targets, status = mlx_inference._template_override_status("custom", blank, None)
        assert status["reason"] is None, empty
        assert targets == [blank]

    broken = SimpleNamespace(chat_template = "native")
    got = reason(broken, probe = lambda: (_ for _ in ()).throw(ValueError("bad tag")))
    assert "could not render" in got and "bad tag" in got
    assert broken.chat_template == "native"


def test_chat_template_override_crosses_both_ipc_hops():
    """The backend runs in a subprocess; the parent rebuilds its entry from an
    enumerated key set at each hop, and /status and the reload check read the
    parent's copy."""
    from core.inference import worker
    import inspect

    from core.inference import orchestrator, worker
    from models.inference import LoadRequest

    assert (
        "chat_template_override"
        in inspect.signature(orchestrator.InferenceOrchestrator.load_model).parameters
    )
    orch = inspect.getsource(orchestrator.InferenceOrchestrator.load_model)
    assert '"chat_template_override": chat_template_override' in orch

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
    """Probe through the generation path, since the VLM recovery helper returns None instead of raising."""
    from core.inference import mlx_inference

    seen = {}

    def fake_render(target, messages, **kwargs):
        seen["target"] = target
        tpl = getattr(target, "chat_template", None)
        if tpl == "empty":
            return ""
        return "  \n " if tpl == "blank" else "rendered"

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        fake_render,
    )
    backend = mlx_inference.MLXInferenceBackend.__new__(mlx_inference.MLXInferenceBackend)

    backend._tokenizer = SimpleNamespace(chat_template = "t")
    backend._processor = None
    assert backend._render_template_probe(False) == "rendered"
    assert seen["target"] is backend._tokenizer

    nested = SimpleNamespace(chat_template = "t")
    processor = SimpleNamespace(
        chat_template = "own", tokenizer = nested, apply_chat_template = lambda *a, **k: ""
    )
    backend._tokenizer = nested
    backend._processor = processor
    backend._render_template_probe(True)
    assert seen["target"] is processor

    for empty in ("empty", "blank"):
        backend._tokenizer = SimpleNamespace(chat_template = empty)
        backend._processor = None
        with pytest.raises(ValueError):
            backend._render_template_probe(False)


def _vlm_backend(
    monkeypatch,
    *,
    markers,
    prompt_tail = "",
):
    from core.inference import mlx_inference

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *a, **k: "<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n"
        + prompt_tail,
    )
    monkeypatch.setattr(
        mlx_inference, "detect_reasoning_channel_markers", lambda target, tools = None: markers
    )
    backend = mlx_inference.MLXInferenceBackend.__new__(mlx_inference.MLXInferenceBackend)
    backend._tokenizer = SimpleNamespace(chat_template = "t", all_special_tokens = [])
    backend._processor = SimpleNamespace(
        chat_template = "t", tokenizer = backend._tokenizer, apply_chat_template = lambda *a, **k: ""
    )
    backend._model = object()
    backend._is_vlm = True
    backend._reads_vision = True
    return backend


def _vlm_row(
    backend,
    *,
    stop = None,
    messages = None,
    continue_final_message = False,
):
    """One vision reply resolved by that backend."""
    return backend._plan_vlm_row(
        messages or [{"role": "user", "content": "hi"}],
        None,
        temperature = 0.7,
        top_p = 0.9,
        top_k = 0,
        min_p = 0.0,
        max_new_tokens = 16,
        repetition_penalty = 1.0,
        continue_final_message = continue_final_message,
        stop = stop,
    ).stream


def test_a_vision_reply_cuts_at_its_own_stop_sequence(monkeypatch):
    """The row carries the request's stop sequences, matched on the sampled text."""
    row = _vlm_row(_vlm_backend(monkeypatch, markers = None), stop = ["HALT"])
    assert row.feed(text = "some text HALT and more") == ["some text "]
    assert row.stopped


def test_the_audio_refusal_puts_the_native_template_back(monkeypatch):
    """Refusing audio must restore the native template, or the model keeps offering audio it can't place."""
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

    marks["audio"] = False
    unset = {"requested": None, "applied": None, "reason": None, "restore": []}
    mlx_inference._revoke_override_that_drops_audio(unset, object(), object())
    assert unset["reason"] is None


def test_a_created_template_is_not_reported_as_the_model_default():
    """A model with no shipped template must report none, or the editor clears the override on save."""
    from core.inference import mlx_inference

    backend = mlx_inference.MLXInferenceBackend.__new__(mlx_inference.MLXInferenceBackend)
    tokenizer = SimpleNamespace(chat_template = None)
    backend.models = {"m": {"tokenizer": tokenizer, "processor": None}}

    native = getattr(tokenizer, "chat_template", None)
    tokenizer.chat_template = "{{ user_supplied }}"
    backend._populate_chat_template_info("m", native)

    info = backend.models["m"]["chat_template_info"]
    assert info["template"] is None, "the override is not the model's template"
    assert info["has_template"] is False

    shipped = SimpleNamespace(chat_template = "{{ native }}")
    backend.models["n"] = {"tokenizer": shipped, "processor": None}
    native = getattr(shipped, "chat_template", None)
    shipped.chat_template = "{{ user_supplied }}"
    backend._populate_chat_template_info("n", native)
    assert backend.models["n"]["chat_template_info"]["template"] == "{{ native }}"

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
    """A text-only probe renders fine with the image placeholder gone; check the image marker too."""
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
    """Report the processor's own template, not the nested tokenizer's, or reset loses its variants."""
    from core.inference import mlx_inference

    nested = SimpleNamespace(chat_template = "{{ nested }}")
    processor = SimpleNamespace(
        chat_template = "{{ processor }}",
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested,
    )
    assert mlx_inference._native_template_source(nested, processor) is processor

    bare = SimpleNamespace(chat_template = None, tokenizer = nested)
    assert mlx_inference._native_template_source(nested, bare) is nested

    named = SimpleNamespace(
        chat_template = {"default": "{{ a }}"},
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested,
    )
    assert mlx_inference._native_template_source(nested, named) is nested

    assert mlx_inference._native_template_source(nested, None) is nested


def test_template_targets_follow_the_object_that_can_actually_render():
    """A processor template it cannot render is not the target; installing there would reach no reader."""
    from core.inference import mlx_inference

    nested = SimpleNamespace(chat_template = "{{ nested }}")
    unrenderable = SimpleNamespace(chat_template = "{{ processor }}", tokenizer = nested)
    assert mlx_inference._template_render_targets(nested, unrenderable) == [nested]

    renderable = SimpleNamespace(
        chat_template = "{{ processor }}",
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested,
    )
    assert mlx_inference._template_render_targets(nested, renderable) == [renderable]
    assert mlx_inference._template_render_targets(nested, None) == [nested]

    named = SimpleNamespace(
        chat_template = {"default": "{{ a }}"},
        apply_chat_template = lambda *a, **k: "",
        tokenizer = nested,
    )
    assert mlx_inference._native_template_source(nested, named) is nested


def test_an_entry_the_upstream_lru_cannot_size_is_not_retainable(monkeypatch):
    """Upstream sums entry.nbytes itself, so a raising QuantizedKVCache.nbytes blocks every insert."""
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

    assert probe(SimpleNamespace(state = (), nbytes = 128))[3] is True


def test_kv_quant_probe_raises_a_dead_metal_queue(monkeypatch):
    """The orchestrator retires a worker on this marker; a refusal would keep it serving on a dead queue."""
    from core.inference import mlx_inference

    _install_fake_mlx(monkeypatch)
    mx = sys.modules["mlx.core"]
    mx.random = SimpleNamespace(state = [0])
    mx.array = lambda v: v
    mx.eval = lambda *a: None
    calls = []

    def language_model(*args, **kwargs):
        calls.append(None)
        if len(calls) > 1:
            raise RuntimeError(f"[METAL] Command buffer execution failed: {message}")

    entry = SimpleNamespace(
        to_quantized = lambda group_size, bits: SimpleNamespace(state = (), nbytes = 1),
        max_size = None,
        window_size = None,
        state = (),
    )
    message = "GPU Timeout Error"
    with pytest.raises(RuntimeError, match = "GPU Timeout"):
        mlx_inference._kv_quant_probe(language_model, [entry], 8)

    calls.clear()
    message = "Insufficient Memory"
    failure = mlx_inference._kv_quant_probe(language_model, [entry], 8)[2]
    assert failure == "it cannot attend over a quantized cache (RuntimeError)"


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
    """mx.random.state has no __setitem__ since mlx 0.32.1, so the rewind must not assign to it."""
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
    # The rewind must run after the forward passes, or the probe's draws stay in the stream.
    assert events == [
        ("forward", None),
        ("convert", None),
        ("forward", None),
        ("seed", (words[0] << 32) | words[1]),
    ]


def test_generate_kwargs_and_history_carry_a_pre_quantized_cache_and_no_kv_bits():
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import cache as lm_cache
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    backend._model = _tiny_lm(
        lambda: [
            lm_cache.RotatingKVCache(max_size = 8),
            lm_cache.KVCache(),
            lm_cache.KVCache(),
            lm_cache.RotatingKVCache(max_size = 8),
        ]
    )
    backend._kv_quant = {"kv_bits": None}
    assert backend._kv_quant_generate_kwargs() == {}

    backend._kv_quant = {"kv_bits": 4}
    kwargs = backend._kv_quant_generate_kwargs()
    assert set(kwargs) == {"prompt_cache"}
    kinds = [type(entry).__name__ for entry in kwargs["prompt_cache"]]
    assert kinds == ["RotatingKVCache", "QuantizedKVCache", "KVCache", "RotatingKVCache"]
    assert kwargs["prompt_cache"][1].bits == 4
    assert [
        type(e).__name__
        for e in mlx_inference._quantize_kv_entries([lm_cache.KVCache(), lm_cache.KVCache()], 4)
    ] == ["QuantizedKVCache", "QuantizedKVCache"]

    cache, rest = backend._prompt_cache().fetch(backend._model, "key", [1, 2, 3])
    assert [type(entry).__name__ for entry in cache] == kinds and rest == [1, 2, 3]


def test_vlm_prompt_cache_session_starts_from_a_pre_quantized_cache(monkeypatch):
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import cache as lm_cache
    from core.inference.mlx_inference import MLXInferenceBackend, VLMPromptSnapshotStore

    names = ("mlx_vlm", "mlx_vlm.generate", "mlx_vlm.generate.common", "mlx_vlm.models")
    modules = {name: types.ModuleType(name) for name in names + ("mlx_vlm.models.cache",)}
    modules["mlx_vlm.models.cache"].make_prompt_cache = lambda _model, max_kv_size = None: [
        lm_cache.RotatingKVCache(max_size = 8),
        lm_cache.KVCache(),
    ]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._vlm_snapshot_store = VLMPromptSnapshotStore(10**6)
    backend._vlm_snapshot_store_unavailable = False
    backend._vlm_is_diffusion_model = lambda _model: False
    backend._model = SimpleNamespace(config = SimpleNamespace(), language_model = object())
    backend._kv_cache_window = None
    backend.active_model_name = "m"
    backend._kv_quant = {"kv_bits": 4}
    session = backend._vlm_prompt_cache_session("base")
    kinds = [type(entry).__name__ for entry in session.cache]
    assert kinds == ["RotatingKVCache", "QuantizedKVCache"] and session.cache[1].bits == 4

    import mlx.core as mx
    from core.inference.mlx_inference import (
        cache_entries_nbytes,
        cache_entries_offset,
        copy_cache_entries,
    )

    for entry in session.cache:
        entry.update_and_fetch(mx.ones((1, 1, 4, 64)), mx.ones((1, 1, 4, 64)))
    snapshot = copy_cache_entries(session.cache)
    assert type(snapshot[1]).__name__ == "QuantizedKVCache" and snapshot[1].bits == 4
    assert cache_entries_offset(snapshot) == 4
    assert cache_entries_nbytes(snapshot[1:]) > 0
    assert backend._vlm_snapshot_store.store("m", [1, 2, 3, 4], snapshot)


def test_a_successful_override_does_not_pin_the_tokenizer_past_load(monkeypatch):
    """Restore pairs must not outlive the load: they hold the tokenizer, so unload_model cannot free it."""
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)
    _install_fake_fast_mlx(monkeypatch, [])
    monkeypatch.setattr(_DummyTokenizer, "chat_template", None, raising = False)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda target, messages, **kwargs: "rendered",
    )

    backend = MLXInferenceBackend()
    config = SimpleNamespace(identifier = "fake/text", is_vision = False, is_lora = False)
    assert backend.load_model(config, chat_template_override = "{{ custom }}")

    status = backend._template_override
    assert status["applied"] == "{{ custom }}"
    assert status["restore"] == []
    assert backend._tokenizer.chat_template == "{{ custom }}"


def test_an_override_that_renders_the_image_as_prose_is_not_a_marker(monkeypatch):
    """A render difference is not a placeholder: prose such as 'Image attached' binds to nothing."""
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

    render_as("Image attached. ")
    assert survives() is False

    render_as("")
    assert survives() is False

    render_as(str({"type": "image"}))
    assert survives() is False

    del processor.image_token
    assert mlx_inference._image_placeholder(nested, processor) is None
    render_as("Image attached. ")
    assert mlx_inference._image_marker_survives(nested, processor, None) is True


def test_the_load_response_declares_every_runtime_field_status_reports():
    """Load responses must declare each /status runtime field, since Pydantic silently drops extras."""
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
    """Gating on the native template skipped models that need recovery; check vision overrides
    regardless."""
    from core.inference import mlx_inference

    _install_fake_mlx(monkeypatch)
    _install_fake_fast_mlx(monkeypatch, [])
    monkeypatch.setattr(_DummyProcessor, "chat_template", "{{ native }}", raising = False)
    monkeypatch.setattr(_DummyProcessor, "apply_chat_template", lambda *a, **k: "", raising = False)
    monkeypatch.setattr(_DummyTokenizer, "chat_template", "{{ nested }}", raising = False)

    def render(target, messages, **kwargs):
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

    assert mlx_inference._image_marker_survives(_DummyTokenizer(), _DummyProcessor(), None) is False

    backend = mlx_inference.MLXInferenceBackend()
    config = SimpleNamespace(identifier = "fake/vlm", is_vision = True, is_lora = False)
    assert backend.load_model(config, chat_template_override = "{{ blind }}")

    assert backend._template_override["applied"] is None
    assert backend._template_override["reason"] == mlx_inference.MLX_TEMPLATE_DROPS_IMAGE
    assert backend._processor.chat_template == "{{ native }}"


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
    monkeypatch.setattr(
        mlx_inference,
        "_make_seeded_mlx_sampler",
        lambda seed, **kw: built.append((seed, kw)) or (lambda _l: _l),
    )
    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "generic_vlm"})
    backend._processor = SimpleNamespace(chat_template = "template")
    backend._reads_vision = True
    args = (
        [{"role": "user", "content": [{"type": "image"}]}],
        [object()],
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
        (
            "_make_mlx_frequency_penalty_processor",
            (0.5,),
            20,
            [10, 11, 5, 5, 5, 6, 99, -1],
            {5: -1.5, 6: -0.5, 10: 0.0, 19: 0.0, 0: 0.0},
        ),
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
    from core.inference import mlx_inference

    mx = pytest.importorskip("mlx.core")

    proc = getattr(mlx_inference, factory_name)(*factory_args)
    proc(mx.array([10, 11]), mx.zeros((1, vocab)))
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
    """A getattr-only read of eos ids falls through to the tokenizer; config may be a dict or an object."""
    from core.inference.mlx_inference import _mlx_finish_reason, _mlx_stop_token_ids

    tokenizer = SimpleNamespace(eos_token_id = 9)
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
    """At the token limit the last token decides, since a stop token sampled last looks like exhaustion."""
    from core.inference.mlx_inference import _mlx_finish_reason, _mlx_stop_token_ids

    model = SimpleNamespace(config = SimpleNamespace(eos_token_id = [163584, 163586]))
    ids = _mlx_stop_token_ids(SimpleNamespace(eos_token_ids = 163594), model)
    assert ids == (163584, 163586)
    assert _mlx_stop_token_ids(SimpleNamespace(eos_token_ids = 151645)) == (151645,)
    assert _mlx_stop_token_ids(SimpleNamespace()) == ()
    assert _mlx_finish_reason(SimpleNamespace(token = 5), ids, 3, 8) == "stop"
    assert _mlx_finish_reason(SimpleNamespace(token = 5), ids, 8, 8) == "length"
    assert _mlx_finish_reason(SimpleNamespace(token = 163584), ids, 8, 8) == "stop"


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
    assert _mlx_stop_cut("abc", ["a"]) == (0, True)
    assert _mlx_stop_cut("aaAB", ["ABC", "BX"]) == (2, False)
    assert _mlx_stop_cut("a then a", ["a"]) == (0, True)
    assert _mlx_stop_cut("early late", ["late", "early"]) == (0, True)
    # A trailing unresolved character can neither show nor complete a sequence.
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
    """Negative words are reinterpreted as uint32 two's complement, so the seed stays in uint64 range."""
    from core.inference import mlx_inference

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
    """Words that are not 32-bit must be declined; masking turns them into a plausible wrong key."""
    from core.inference import mlx_inference

    mlx_inference, seeded = _fake_rng_state(monkeypatch, words)
    warnings = _capture_rng_warnings(monkeypatch, mlx_inference)

    assert mlx_inference._mlx_rng_key_words() is None
    assert any("32-bit word" in w for w in warnings), warnings

    mlx_inference._restore_mlx_rng_key(None)
    mlx_inference._restore_mlx_rng_key(words)
    assert seeded == []


def _capture_rng_warnings(monkeypatch, mlx_inference):
    """Collect this module's warnings. It logs through structlog, which caplog
    does not see."""
    from core.inference import mlx_inference

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


_BACKEND_SOURCE = Path(__file__).resolve().parents[1] / "core" / "inference" / "mlx_inference.py"
_BACKEND_TREE = ast.parse(_BACKEND_SOURCE.read_text(encoding = "utf-8"))


def _is_call(
    node,
    name,
    receiver = "mx",
):
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        return False
    func = node.value.func
    if isinstance(func, ast.Attribute):
        return func.attr == name and isinstance(func.value, ast.Name) and func.value.id == receiver
    return isinstance(func, ast.Name) and func.id == name


def _clear_cache_sites():
    for node in ast.walk(_BACKEND_TREE):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for index, statement in enumerate(block):
                if _is_call(statement, "clear_cache"):
                    yield statement.lineno, (block[index - 1] if index else None)


def test_backend_has_mlx_cache_clear_sites():
    assert list(_clear_cache_sites()), "no mx.clear_cache() found -- test is stale"


def test_every_cache_clear_drains_gpu_work_first():
    """MLX does not pin a dropped output array, so every clear must drain first."""
    unguarded = [
        line
        for line, previous in _clear_cache_sites()
        if not _is_call(previous, "_drain_generation_streams", receiver = None)
    ]
    assert not unguarded, (
        f"mx.clear_cache() at line(s) {unguarded} is not immediately preceded by "
        "_drain_generation_streams()"
    )


def test_the_drain_covers_both_generation_stream_modules():
    """Draining only the default stream would miss the stream generation actually uses."""
    assigned = [
        node
        for node in ast.walk(_BACKEND_TREE)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "_GENERATION_STREAM_MODULES" for t in node.targets)
    ]
    assert len(assigned) == 1
    names = {element.value for element in assigned[0].value.elts}
    assert names == {
        "mlx_lm.generate",
        "mlx_vlm.generate",
        "mlx_vlm.generate.dispatch",
        "mlx_vlm.generate.ar",
        "mlx_vlm.speculative.common",
    }


def _fake_stream_module(monkeypatch, name, stream):
    monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(generation_stream = stream))


def _recording_mx(monkeypatch, synchronize = None):
    from core.inference import mlx_inference

    events = []
    mx = types.SimpleNamespace(
        synchronize = synchronize
        or (
            lambda stream = None: events.append(
                f"synchronize:{'default' if stream is None else stream}"
            )
        ),
    )
    for name in mlx_inference._GENERATION_STREAM_MODULES:
        monkeypatch.delitem(sys.modules, name, raising = False)
    return mlx_inference, mx, events


def test_a_stream_that_cannot_be_drained_does_not_stop_the_caller(monkeypatch):
    """A plain mx.new_stream raises when synchronized off its creating thread."""
    from core.inference import mlx_inference

    def synchronize(stream = None):
        if stream == "foreign":
            raise RuntimeError("There is no Stream(gpu, 0) in current thread.")
        events.append(f"synchronize:{'default' if stream is None else stream}")

    mlx_inference, mx, events = _recording_mx(monkeypatch, synchronize)
    _fake_stream_module(monkeypatch, "mlx_lm.generate", "foreign")

    mlx_inference._drain_generation_streams(mx)

    assert events == ["synchronize:default"]


def test_a_module_getattr_that_raises_does_not_stop_the_caller(monkeypatch):
    from core.inference import mlx_inference

    mlx_inference, mx, events = _recording_mx(monkeypatch)
    module = types.ModuleType("mlx_vlm.generate")
    module.__getattr__ = lambda name: (_ for _ in ()).throw(RuntimeError(name))
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", module)

    mlx_inference._drain_generation_streams(mx)

    assert events == ["synchronize:default"]


def test_one_stream_shared_by_several_modules_is_drained_once(monkeypatch):
    """0.6.x defines the stream once and re-exports it from every candidate name."""
    from core.inference import mlx_inference

    mlx_inference, mx, events = _recording_mx(monkeypatch)
    for name in ("mlx_vlm.generate", "mlx_vlm.generate.dispatch", "mlx_vlm.generate.ar"):
        _fake_stream_module(monkeypatch, name, "shared")

    mlx_inference._drain_generation_streams(mx)

    assert events == ["synchronize:shared", "synchronize:default"]


def test_the_speculative_decoding_stream_is_drained_too(monkeypatch):
    from core.inference import mlx_inference

    mlx_inference, mx, events = _recording_mx(monkeypatch)
    _fake_stream_module(monkeypatch, "mlx_vlm.speculative.common", "speculative")

    mlx_inference._drain_generation_streams(mx)

    assert events == ["synchronize:speculative", "synchronize:default"]


def test_a_runtime_without_synchronize_is_not_an_error():
    from core.inference import mlx_inference
    mlx_inference._drain_generation_streams(types.SimpleNamespace())


@pytest.mark.parametrize("name", ["n_ctx", "original_max_position_embeddings", "model_max_length"])
def test_mlx_native_context_length_reads_every_spelling_and_every_config(name):
    from core.inference.mlx_inference import mlx_native_context_length

    args = SimpleNamespace(**{name: 40960})
    assert mlx_native_context_length(SimpleNamespace(args = args)) == 40960
    from routes.models import _get_max_position_embeddings

    assert _get_max_position_embeddings(SimpleNamespace(**{name: 40960})) == 40960
    # The wrapper counts too: mlx-vlm's Phi-3-V keeps a 4096 text config beside its real 131072.
    stub = SimpleNamespace(args = SimpleNamespace(**{name: 4096}))
    text = SimpleNamespace(**{name: 4096})
    cfg = SimpleNamespace(model_type = "x", text_config = text, **{name: 131072})
    vlm = SimpleNamespace(language_model = stub, config = cfg)
    assert mlx_native_context_length(vlm) == 131072


_PLAIN_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
_TOOL_TEMPLATE = (
    "{% if tools %}{% for t in tools %}{{ t.function.name }}{% endfor %}{% endif %}"
    "{% for m in messages %}{{ m['content'] }}"
    "{% if m.tool_calls %}<tool_call>{{ m.tool_calls[0].function.name }}</tool_call>{% endif %}"
    "{% endfor %}"
)
_DEFAULT_TOOL_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set system = messages[0]['content'] %}"
    "{% set rest = messages[1:] %}{% else %}{% set system = 'You are Qwen.' %}"
    "{% set rest = messages %}{% endif %}{{ system }}"
    "{% if tools %}{% for tool in tools %}{{ tool.function.name }}{% endfor %}{% endif %}"
    "{% for message in rest %}{{ message['content'] }}"
    "{% if message.tool_calls %}<tool_call>{{ message.tool_calls[0].function.name }}"
    "</tool_call>{% endif %}{% endfor %}"
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
        self.messages = self.system = self.tools = self.reasoning = None

    def count_chat_tokens(
        self,
        messages,
        system_prompt = "",
        **kwargs,
    ):
        self.messages, self.system = messages, system_prompt
        self.tools = kwargs.get("tools")
        self.reasoning = {
            key: kwargs.get(key)
            for key in ("enable_thinking", "reasoning_effort", "preserve_thinking")
        }
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
    from routes import inference as route

    backend_dir = str(Path(__file__).resolve().parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

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


def _count_hi(
    *args,
    messages = [{"role": "user", "content": "hi"}],
    **kwargs,
):
    """_count_route over the one-message chat every case counts."""
    return _count_route(*args, messages = messages, **kwargs)


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


def test_an_mlx_count_uses_the_shared_nested_reasoning_controls(monkeypatch):
    backend = _RenderRecordingBackend()
    _count_route(
        monkeypatch,
        backend,
        messages = [{"role": "user", "content": "hello"}],
        thinking = {"type": "disabled"},
        chat_template_kwargs = {
            "reasoning_effort": "high",
            "preserve_thinking": True,
        },
    )

    assert backend.reasoning == {
        "enable_thinking": True,
        "reasoning_effort": "high",
        "preserve_thinking": True,
    }


@pytest.mark.parametrize(
    "template, expected_tools",
    [
        (_PLAIN_TEMPLATE, None),
        (_TOOL_TEMPLATE, ["web_search"]),
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
    from routes import inference as route

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
    assert resolve(None, wide, 1048576) == (1048576, 262144, 262144)
    held = SimpleNamespace(args = wide.args, max_seq_length = 4096)
    assert resolve(None, held, 0) == (4096, 262144, 262144)
    assert resolve(None, blank, 0) == (None, None, None)
    assert resolve(None, blank, 8192) == (8192, None, None)


def test_the_resident_context_gate_compares_requests_not_served_windows():
    """Reuse on model identity alone leaves a context change with no reload to carry it.
    Served windows cannot answer it: auto-sized to 32768 and pinned at 32768 are equal."""
    backend_dir = str(Path(__file__).resolve().parent.parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    from routes.inference import _resident_context_satisfies as gate

    # The mirror always writes the key, so presence alone would reload every model.
    assert gate({}, 8192) is True
    assert gate({"requested_context_length": None}, 8192) is True

    assert gate({"requested_context_length": 8192}, 8192) is True
    assert gate({"requested_context_length": 0}, 0) is True
    assert gate({"requested_context_length": 0}, None) is True

    assert gate({"requested_context_length": 0}, 8192) is False
    assert gate({"requested_context_length": 8192}, 0) is False

    assert gate({"requested_context_length": 0, "context_length": 32768}, 32768) is False


def test_the_load_policy_bounds_a_pin_only_where_the_bound_can_be_enforced(monkeypatch):
    """mlx-lm cannot quantize a rotating cache (to_quantized raises, from the first token),
    so exactly one applies. A pin is an explicit memory instruction; a self-chosen window
    is not. An unenforceable pin buys nothing, so it does not spend the quantization."""
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    pytest.importorskip("mlx_lm.models.cache")
    from mlx_lm.models.cache import KVCache

    monkeypatch.setattr(
        mlx_inference, "_kv_quant_eligibility", lambda *_a, **_k: ("full", "", True)
    )
    honours = SimpleNamespace(layers = [object(), object()])
    owns = SimpleNamespace(layers = [object(), object()], make_cache = lambda: [KVCache(), KVCache()])

    unset = object()

    def policy(
        model,
        kv_bits,
        requested,
        served = unset,
        fitted = False,
    ):
        backend = MLXInferenceBackend()
        backend._model = model
        backend._is_vlm = False
        quant, window, enforced, budget = backend._resolve_kv_policy(
            False, kv_bits, requested, requested if served is unset else served, fitted = fitted
        )
        return quant["kv_bits"], window, enforced, budget

    assert policy(honours, 4, 8192) == (4, None, False, 8192)
    assert policy(honours, 4, 0, 262144) == (4, None, False, None)
    assert policy(honours, None, 8192) == (None, 8192, True, None)
    assert policy(honours, None, 0, 262144) == (None, 262144, True, None)

    assert policy(honours, 4, 0, 24576, fitted = True) == (4, None, False, 24576)
    assert policy(honours, None, 0, 24576, fitted = True) == (None, 24576, True, None)
    assert policy(owns, 4, 0, 24576, fitted = True) == (4, None, False, None)

    assert policy(owns, 4, 8192) == (4, None, False, None)
    assert policy(owns, None, 8192) == (None, None, False, None)

    # A window nobody could read bounds nothing rather than bounding at zero.
    assert policy(honours, None, 0, None) == (None, None, False, None)
    assert policy(honours, None, 0, 0) == (None, None, False, None)

    unreadable = SimpleNamespace(layers = [object()], make_cache = lambda: None)
    assert policy(unreadable, None, 8192) == (None, None, None, None)

    backend = MLXInferenceBackend()
    backend._model = honours
    backend._is_vlm = False
    budgeted, _, _, budget = backend._resolve_kv_policy(True, 4, 4096, 4096)
    unpinned, _, _, _ = backend._resolve_kv_policy(True, 4, 0, 262144)
    assert budget == 4096 and budgeted["note"] == "" and unpinned["note"] == ""
    for backend._kv_context_budget in (budget, None):
        backend._kv_quant = budgeted
        kwargs = backend._kv_quant_generate_kwargs()
        assert set(kwargs) == {"prompt_cache"}
        assert [getattr(entry, "bits", None) for entry in kwargs["prompt_cache"]] == [4, 4]


def _drive_vlm_generation(
    backend,
    monkeypatch,
    image = None,
    max_new_tokens = 1,
    session = None,
    videos = None,
    on_stream = None,
    **request,
):
    import sys
    import types

    seen = {}
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.prompt_utils = SimpleNamespace(
        MODEL_CONFIG = {}, apply_chat_template = lambda *_a, **_k: "prompt"
    )

    def _stream(*_a, **kwargs):
        seen.clear()
        seen.update(kwargs)
        if on_stream is not None:
            on_stream()
        yield SimpleNamespace(text = "ok", prompt_tokens = 3, generation_tokens = 1)

    mlx_vlm.stream_generate = _stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_a, **_k: "prompt",
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.detect_think_prefill", lambda *_a, **_k: ""
    )
    backend._model = SimpleNamespace(config = {"model_type": "gemma3"})
    backend._processor = SimpleNamespace(tokenizer = SimpleNamespace())
    backend._is_vlm = True
    monkeypatch.setattr(backend, "_release_vlm_snapshots", lambda: None)
    monkeypatch.setattr(backend, "_vlm_prompt_cache_session", lambda *_a, **_k: session)
    list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "hi"}],
            image = image,
            videos = videos,
            max_new_tokens = max_new_tokens,
            **request,
        )
    )
    return seen


def test_a_speculative_load_drafts_eligible_single_replies(monkeypatch):
    pytest.importorskip("mlx_lm")
    import sys
    import types

    from core.inference import mlx_inference

    class _Ids(list):
        def tolist(self):
            return list(self)

    utils = types.ModuleType("mlx_vlm.utils")
    utils.prepare_inputs = lambda *_a, **_k: {"input_ids": [_Ids([7, 8, 9])], "extra": 1}
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    prepared, verify_scope, scopes = [], [], []

    @contextmanager
    def exact_verify(model):
        verify_scope.append("open")
        try:
            yield model
        finally:
            verify_scope.append("closed")

    monkeypatch.setattr(
        sys.modules["unsloth_zoo.mlx.inference"], "exact_speculative_verify", exact_verify
    )
    draft = SimpleNamespace(draft_kind = "mtp", draft_n = 6, draft_n_accepted = 4)
    draft.prepare = (
        lambda ids, sampling, given: prepared.append((ids, sampling.temperature, sampling.top_k))
        or setattr(draft, "given", given)
        or scopes.append(list(verify_scope))
    )
    backend = mlx_inference.MLXInferenceBackend()
    backend._speculative_draft = draft

    # top_k -1 is the API's "disabled", which the drafter's sampling settings spell 0.
    seen = _drive_vlm_generation(
        backend,
        monkeypatch,
        temperature = 0.5,
        top_k = -1,
        repetition_penalty = 1.1,
        on_stream = lambda: scopes.append(list(verify_scope)),
    )
    assert seen["draft_model"] is draft and seen["extra"] == 1 and prepared == [([7, 8, 9], 0.5, 0)]
    # The draft is told every processor the reply runs: none reaches mlx-vlm as a shortcut.
    assert "repetition_penalty" not in seen and draft.given == seen["logits_processors"] != []
    monkeypatch.setattr(
        mlx_inference, "_mlx_sampling_processors", lambda **_: [lambda tokens, logits: logits]
    )
    timings = backend.last_generation_stats["timings"]
    assert (timings["draft_n"], timings["draft_n_accepted"]) == (6, 4)
    for request in (
        {"repetition_penalty": 1.1},
        {"tools": [{"type": "function"}]},
        {"tool_protocol_active": True},
    ):
        seen = _drive_vlm_generation(backend, monkeypatch, **request)
        assert "max_tokens" in seen and "draft_model" not in seen
        assert "draft_n" not in backend.last_generation_stats["timings"]
    # Only the drafted reply entered the verify scope: open from preparing the draft through decoding.
    assert len(prepared) == 1 and scopes == [["open"]] * 2 and verify_scope == ["open", "closed"]
    opened, backend._is_vlm, backend._vlm_batch_unavailable_reason = [], True, lambda r, **_k: None
    backend._vlm_resident_unavailable_reason = lambda r, gaps = iter(["no stream"]): next(gaps, None)
    assert backend.batch_unavailable_reason([{}, {}]) == "no stream"
    monkeypatch.setattr(
        _batch_engine(monkeypatch), "BatchStream", lambda *a, **k: opened.append(k["speculative"])
    )
    backend._generate_session_batch = lambda r, kind, **_: opened.append(kind) or ()
    mlx_inference._VisionBatchSession(backend, width = 2)
    list(backend.generate_chat_batch([{}, {}]))
    assert opened == [draft, mlx_inference._VisionBatchSession]
    del backend._generate_session_batch
    steps, cancel = [], threading.Event()
    step = lambda waiting: steps.append(waiting) or setattr(session, "rows_in_flight", 0) or ()
    session = SimpleNamespace(admit = lambda *a: None, rows_in_flight = 1, step = step, close = list)
    list(backend._generate_session_batch([{}], lambda *a, **k: session, cancel_event = cancel))
    assert steps == [cancel.is_set]  # a cancelled reply ends its speculative step early


def _stub_prepare_inputs(monkeypatch, per_medium = 520):
    import sys
    import types

    seen = {}

    def _prepare_inputs(
        _processor,
        *,
        prompts,
        images = None,
        audio = None,
        videos = None,
        **_kwargs,
    ):
        media = [*(images or ()), *(audio or ()), *(videos or ())]
        seen.clear()
        seen.update(_kwargs)
        return {"input_ids": SimpleNamespace(size = len(prompts.split()) + per_medium * len(media))}

    utils = types.ModuleType("mlx_vlm.utils")
    utils.prepare_inputs = _prepare_inputs
    utils.seen = seen
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    return utils


def _vision_backend(budget = 1024, window = 1024):
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = SimpleNamespace(model_type = "qwen2_5_vl"))
    backend._processor = object()
    backend._is_vlm = True
    backend._tokenizer = SimpleNamespace(encode = lambda text, **_k: [0] * len(text.split()))
    backend._served_context = window
    backend._kv_context_budget = budget
    return backend


def test_the_context_budget_counts_the_request_in_tokens_the_processor_would_produce(monkeypatch):
    """Media must be counted expanded: text alone is 7 tokens vs 582 for one image."""
    from core.inference import context_refusal
    from core.inference.mlx_inference import MLXInferenceBackend

    utils = _stub_prepare_inputs(monkeypatch)
    backend = _vision_backend()
    prompt = "placeholder describe this picture"

    assert backend._count_prompt_tokens(prompt) == 4
    assert backend._count_prompt_tokens(prompt, [object()]) == 524
    assert backend._count_prompt_tokens(prompt, [object(), object()]) == 1044

    backend._check_context_budget(prompt, 16)
    backend._check_context_budget(prompt, 16, images = [object()])
    with pytest.raises(context_refusal.ContextBudgetExceeded):
        backend._check_context_budget(prompt, 16, images = [object(), object()])
    with pytest.raises(context_refusal.ContextBudgetExceeded):
        backend._check_context_budget(prompt, 16, audio = [object(), object()])
    monkeypatch.setattr("core.inference.mlx_inference._video_frame_rate", lambda *_a: 1.5)
    with pytest.raises(context_refusal.ContextBudgetExceeded):
        backend._check_context_budget(prompt, 16, videos = ["a.mp4", "b.mp4"])
    with pytest.raises(context_refusal.ContextBudgetExceeded):
        backend._check_context_budget(prompt, 16, images = [object()], videos = ["a.mp4"])
    backend._check_context_budget(prompt, 16, videos = ["a.mp4"])
    assert utils.seen["fps"] == 1.5

    sized = backend._generation_limit(prompt, None, images = [object()])
    assert sized == 1024 - 524
    backend._generation_limit(prompt, sized, images = [object()])
    with pytest.raises(context_refusal.ContextBudgetExceeded):
        backend._generation_limit(prompt, 16, images = [object(), object()])
    with pytest.raises(context_refusal.ContextBudgetExceeded):
        backend._generation_limit(prompt, 16, videos = ["a.mp4", "b.mp4"])

    def _unpreparable(*_a, **_k):
        raise RuntimeError("no processor")

    utils.prepare_inputs = _unpreparable
    assert backend._count_prompt_tokens(prompt, [object(), object()]) == 4
    backend._check_context_budget(prompt, 16, images = [object(), object()])


def test_the_vlm_entry_point_hands_the_limit_a_clip_the_processor_can_read(monkeypatch):
    """Every clip is counted from a readable file, not the route's base64."""
    import base64
    import os

    from core.inference.mlx_inference import MLXInferenceBackend

    monkeypatch.setattr("core.inference.mlx_inference._mlx_vlm_decodes_video", lambda: True)
    backend = MLXInferenceBackend()
    backend._kv_context_budget = 1024
    seen = {}

    def _limit(*_a, **kwargs):
        seen.update(kwargs)
        seen["on_disk"] = [os.path.exists(clip) for clip in kwargs.get("videos") or ()]
        seen["contents"] = [Path(clip).read_bytes() for clip in kwargs.get("videos") or ()]
        return 1

    monkeypatch.setattr(backend, "_generation_limit", _limit)
    clips = [base64.b64encode(b"not a readable clip").decode(), "BBBB"]
    with contextlib.suppress(Exception):
        _drive_vlm_generation(backend, monkeypatch, videos = clips)
    assert seen["videos"] != clips
    assert seen["on_disk"] == [True, True]
    assert seen["contents"] == [base64.b64decode(clip) for clip in clips]
    assert not any(os.path.exists(path) for path in seen["videos"])


def test_media_is_prepared_only_under_a_budget_and_only_once(monkeypatch):
    from core.inference.mlx_inference import UNSET_GENERATION_BUDGET

    utils = _stub_prepare_inputs(monkeypatch)
    calls = []
    real = utils.prepare_inputs
    utils.prepare_inputs = lambda *a, **k: calls.append(1) or real(*a, **k)
    prompt = "placeholder describe this picture"

    unbudgeted = _vision_backend(budget = None, window = 8192)
    sized = unbudgeted._generation_limit(
        prompt, None, images = [object()], cap = UNSET_GENERATION_BUDGET
    )
    assert calls == []
    assert sized == min(8192 - 4, UNSET_GENERATION_BUDGET)

    budgeted = _vision_backend()
    assert budgeted._generation_limit(prompt, None, images = [object()]) == 1024 - 524
    assert calls == [1]


def test_a_max_tokens_at_the_whole_window_is_the_chat_saying_no_cap(monkeypatch):
    """Max Tokens at the window means "as much as fits", not a request for that many."""
    from core.inference import context_refusal

    _stub_prepare_inputs(monkeypatch)
    backend = _vision_backend()
    prompt = "placeholder describe this picture"

    assert backend._generation_limit(prompt, 1024) == 1024 - 4
    assert backend._generation_limit(prompt, 1024, images = [object()]) == 1024 - 524
    assert backend._generation_limit(prompt, 16) == 16
    assert backend._generation_limit(prompt, 2048) == 1024 - 4
    assert backend._generation_limit(prompt, 500, range(700)) == 1024 - 700
    with pytest.raises(context_refusal.ContextBudgetExceeded):
        backend._generation_limit(prompt, 1, range(1024))


def test_the_bound_is_checked_on_a_real_cache_at_the_size_that_was_asked_for():
    """make_cache ignores max_kv_size, and those caches range from constant-state to
    unbounded, so the argument does not say whether it applied. Nor does an
    architecture-chosen cap serve a narrower request."""
    from core.inference.mlx_inference import _kv_window_enforced

    pytest.importorskip("mlx_lm.models.cache")
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
    assert _kv_window_enforced(hybrid, False, 4096) is False
    assert _kv_window_enforced(chunked, False, 4096) is True
    assert _kv_window_enforced(undeclared, False, 4096) is False
    assert _kv_window_enforced(nested, False, 4096) is False
    assert _kv_window_enforced(native, False, 128) is False
    assert _kv_window_enforced(native, False, 4096) is True
    assert _kv_window_enforced(honours, False, 4) is False
    assert _kv_window_enforced(SimpleNamespace(), False, 4096) is None
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
    for broken in (None, object(), SimpleNamespace(caches = 7)):
        model = factory(lambda _m, max_kv_size = None, _b = broken: _b)
        assert _kv_window_enforced(model, False, 4096) is None


def test_the_window_reaches_the_runtime_on_every_generation_route(monkeypatch):
    """The runtimes read max_kv_size only when no prompt_cache is passed, so each route
    building its own cache carries the bound. The audio route once did not."""
    from core.inference.mlx_inference import MLXInferenceBackend
    from core.inference.mlx_inference import _MLXPromptCacheHistory
    import types as _types

    kwargs = MLXInferenceBackend._kv_window_generate_kwargs
    assert kwargs(SimpleNamespace(_kv_cache_window = None)) == {}
    assert kwargs(SimpleNamespace()) == {}
    assert kwargs(SimpleNamespace(_kv_cache_window = 8192)) == {"max_kv_size": 8192}

    pytest.importorskip("mlx_lm.models.cache")

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
    vlm._reads_vision = True
    vlm._processor = SimpleNamespace(tokenizer = SimpleNamespace())
    next(
        vlm._generate_vlm(
            [{"role": "user", "content": [{"type": "image"}]}],
            [object()],
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

    interactive = SimpleNamespace(headers = Headers({}), query_params = {}, cookies = {})
    backend = _RenderRecordingBackend()
    _count_hi(monkeypatch, backend, template = _PLAIN_TEMPLATE, request = interactive)
    from routes.inference import current_date_prompt_line

    line = current_date_prompt_line(request = interactive)
    assert line, "the harness must actually produce a date line"
    assert (
        backend.system == line
    ), f"the count dropped the date the completion adds: {backend.system!r}"
    assert backend.messages == [{"role": "user", "content": "hi"}]


def test_an_mlx_client_tool_count_keeps_the_template_default_after_the_date(monkeypatch):
    from starlette.datastructures import Headers

    from routes import inference as route

    monkeypatch.setattr(
        route,
        "_local_template_system_turn",
        lambda _today, image = False, tools = False, controls = (): (
            True,
            "You are Qwen." if tools else "",
        ),
    )
    interactive = SimpleNamespace(headers = Headers({}), query_params = {}, cookies = {})
    backend = _RenderRecordingBackend()
    _count_hi(
        monkeypatch,
        backend,
        template = _DEFAULT_TOOL_TEMPLATE,
        request = interactive,
        enable_tools = False,
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    from routes.inference import current_date_prompt_line

    line = current_date_prompt_line(request = interactive)
    assert backend.system == ""
    assert backend.messages[0] == {"role": "system", "content": f"{line}\n\nYou are Qwen."}


def test_an_mlx_count_prices_the_archive_tool_and_its_compaction_nudge(monkeypatch):
    """An archive adds search_conversation and a compaction nudge, both gated on the
    thread id, so a count that drops it under-reports."""
    from routes import inference as route
    from state import tool_policy

    monkeypatch.setattr(tool_policy, "_tool_policy_default", True)
    monkeypatch.setattr(route, "_thread_has_conversation_archive", lambda tid: bool(tid))
    backend = _RenderRecordingBackend()
    _count_hi(
        monkeypatch,
        backend,
        template = _TOOL_TEMPLATE,
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
    from state import tool_policy

    from starlette.datastructures import Headers

    monkeypatch.setattr(tool_policy, "_tool_policy_default", True)
    keyed = SimpleNamespace(
        headers = Headers({"authorization": "Bearer sk-unsloth-test"}),
        query_params = {},
        cookies = {},
    )
    backend = _RenderRecordingBackend()
    _count_hi(
        monkeypatch,
        backend,
        template = _TOOL_TEMPLATE,
        request = keyed,
        enabled_tools = ["web_search"],
    )
    from routes.inference import current_date_prompt_line

    line = current_date_prompt_line(request = keyed)
    assert line, "the harness must produce a date line"
    assert (
        line in backend.system
    ), f"the count dropped the date the tool-loop completion adds: {backend.system!r}"


@pytest.mark.parametrize(
    ("active", "advertised"),
    [
        ("/Users/alice/outputs/my-finetune-mlx", None),
        ("/hf/hub/models--org--repo/snapshots/abc123", "org/repo"),
        ("mlx-community/Qwen3.5-2B-4bit", None),
    ],
    ids = ["local_folder", "auto_switch_snapshot", "hub_repo_id"],
)
def test_an_mlx_count_reports_the_id_status_publishes(monkeypatch, active, advertised):
    """/status publishes the raw active name as model_identifier and the client adopts it as its
    checkpoint, dropping any count that names another model; a cleaned or advertised id here
    left a local-folder or auto-switched MLX / vLLM model on the estimate forever."""
    backend = _RenderRecordingBackend()
    backend.active_model_name = active
    backend._openai_advertised_id = advertised
    models = {active: next(iter(_mirror(_TOOL_TEMPLATE).values()))}
    served = _count_route(
        monkeypatch, backend, models = models, messages = [{"role": "user", "content": "hello"}]
    )
    assert json.loads(served.body)["model"] == active


@pytest.mark.parametrize("via_api_key", [True, False], ids = ["api_key", "ui_session"])
def test_an_mlx_count_redacts_a_local_folder_for_an_api_key_caller(monkeypatch, via_api_key):
    """/status hides host paths from API-key callers; the count must not hand the same path back."""
    from hub.utils.host_paths import redact_host_paths
    from routes import inference as route

    active = "/Users/alice/outputs/my-finetune-mlx"
    backend = _RenderRecordingBackend()
    backend.active_model_name = active
    backend.models = {active: next(iter(_mirror(_TOOL_TEMPLATE).values()))}
    monkeypatch.setattr(route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False))
    monkeypatch.setattr(route.active_generations, "count", lambda: 0)
    served = asyncio.run(
        route.chat_count_tokens(
            route.ChatCountTokensRequest(messages = [{"role": "user", "content": "hi"}]),
            current_subject = "tester",
            via_api_key = via_api_key,
        )
    )
    model = json.loads(served.body)["model"]
    if via_api_key:
        assert "/Users/alice" not in model
        status_view = redact_host_paths({"model_identifier": active}, via_api_key = True)
        assert model == status_view["model_identifier"]
    else:
        assert model == active


def test_an_mlx_count_is_dropped_when_a_same_model_reload_lands_under_it(monkeypatch):
    """The active name cannot see a same-ID reload, which is how a template override lands.
    A count routed from the old entry must not be published."""
    from fastapi import HTTPException

    backend = _RenderRecordingBackend()
    backend.load_generation = 7

    real_count = backend.count_chat_tokens

    def _reload_midway(*args, **kwargs):
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

    backend = _RenderRecordingBackend()
    counts = iter([0, 1])
    with pytest.raises(HTTPException) as excinfo:
        _count_hi(monkeypatch, backend, generations = lambda: next(counts, 1))
    assert excinfo.value.status_code == 503
    assert "generation is in progress" in str(excinfo.value.detail)


def test_the_mlx_mcp_snapshot_is_taken_under_the_same_guard_the_gguf_count_uses():
    """The MCP handlers hold this guard across the row change and the schema-cache
    invalidation, so a snapshot outside it can pair a new row with a stale schema."""
    from routes import inference as route
    import inspect

    body = inspect.getsource(route._mlx_count_chat_tokens)
    assert "mcp_server_snapshot_guard" in body, "the MLX snapshot is unguarded"
    guard = body.index("async with mcp_server_snapshot_guard():")
    snapshot = body.index("asyncio.to_thread(cached_mcp_tools)")
    assert guard < snapshot, "the guard must be held across the snapshot, not after it"


@pytest.fixture(autouse = True)
def _fresh_mlx_estimate_caches():
    from core.inference import mlx_inference, mlx_memory

    mlx_inference._kv_refusal_cache.clear()
    mlx_memory._clear_sizing_cache()
    yield
    mlx_inference._kv_refusal_cache.clear()
    mlx_memory._clear_sizing_cache()


def test_a_width_refusal_is_asked_once_per_checkpoint_and_width(monkeypatch, tmp_path):
    from core.inference import mlx_inference

    (tmp_path / "config.json").write_text("{}")
    asked = []
    monkeypatch.setattr(
        mlx_inference, "_asking_the_architecture", lambda d, q: asked.append(d) or True
    )
    assert mlx_inference.mlx_kv_quant_is_refused(str(tmp_path), 4) is True
    assert mlx_inference.mlx_kv_quant_is_refused(str(tmp_path), 4) is True and len(asked) == 1
    assert mlx_inference.mlx_kv_quant_is_refused(str(tmp_path), 8) is True and len(asked) == 2
    (tmp_path / "config.json").write_text('{"changed": 1}')
    assert mlx_inference.mlx_kv_quant_is_refused(str(tmp_path), 4) is True and len(asked) == 3


def test_what_the_fit_is_asked_and_when_it_is_asked_at_all(monkeypatch, tmp_path):
    from core.inference import mlx_inference

    priced, rewound = [], []
    monkeypatch.setitem(
        sys.modules,
        "core.inference.mlx_memory",
        types.SimpleNamespace(
            mlx_fit_context = lambda model_dir, **kw: (
                priced.append(kw | {"dir": model_dir})
                or rewound.append(f"priced@{kw['kv_bits']}")
                or 24_576
            )
        ),
    )
    monkeypatch.setattr(mlx_inference, "mlx_memory_budget", lambda **_: 8 * 1024**3)
    monkeypatch.setattr(
        mlx_inference, "_mlx_rng_key_words", lambda: rewound.append("held") or ("key",)
    )
    monkeypatch.setattr(mlx_inference, "_restore_mlx_rng_key", rewound.append)
    asked = []
    verdict = {"answer": ("full", "", True)}
    monkeypatch.setattr(
        mlx_inference,
        "_kv_quant_eligibility",
        lambda *a: asked.append(a) or verdict["answer"],
    )
    model = object()

    def fit(kv_bits = None, **kw):
        priced.clear(), asked.clear(), rewound.clear()
        return mlx_inference._fitted_context(
            model,
            str(tmp_path),
            262_144,
            load_in_4bit = True,
            retains_history = True,
            kv_bits = kv_bits,
            **kw,
        )

    assert fit() == (24_576, None)
    assert asked == [] and rewound == ["held", "priced@None", ("key",)]
    assert priced == [
        {
            "budget_bytes": 8 * 1024**3,
            "max_ctx": 262_144,
            "kv_bits": None,
            "load_in_4bit": True,
            "dir": str(tmp_path),
        }
    ]
    assert fit(vision = True) and priced[0]["vision"] is True
    # The width the cache will take is the width the fit is priced at.
    for answer, bits in ((("full", "", True), 4), (("partial", "w", True), 4)):
        verdict["answer"] = answer
        assert fit(4, is_vlm = True) == (24_576, answer)
        assert asked == [(model, True, 4)] and priced[0]["kv_bits"] == bits
    for answer in (("refused", "r", True), ("none", "n", True)):
        verdict["answer"] = answer
        assert fit(4) == (24_576, answer) and priced[0]["kv_bits"] is None

    monkeypatch.setitem(
        sys.modules,
        "core.inference.mlx_memory",
        types.SimpleNamespace(
            mlx_fit_context = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no"))
        ),
    )
    assert fit() == (None, None) and rewound == ["held", ("key",)]
    monkeypatch.setattr(mlx_inference, "mlx_memory_budget", lambda **_: None)
    assert fit() == (None, None) and rewound == []
    monkeypatch.setattr(mlx_inference, "mlx_memory_budget", lambda **_: 8 * 1024**3)
    monkeypatch.setattr(mlx_inference, "_snapshot_dir", lambda model, name: None)
    assert fit() == (None, None) and (priced, rewound) == ([], [])


def test_the_fit_budget_leaves_the_prompt_history_its_own_room(monkeypatch):
    from core.inference import mlx_inference

    fake = types.SimpleNamespace(
        metal = types.SimpleNamespace(is_available = lambda: True),
        device_info = lambda: {"max_recommended_working_set_size": 100_000_000_000},
    )
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    if "mlx" in sys.modules:
        monkeypatch.setattr(sys.modules["mlx"], "core", fake, raising = False)
    else:
        # The parent package is imported first, so it must resolve on hosts without MLX.
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core = fake))
    monkeypatch.delenv("UNSLOTH_MLX_PROMPT_CACHE_BYTES", raising = False)

    assert mlx_inference.mlx_memory_budget() == 85_000_000_000 - 15_000_000_000
    assert mlx_inference.mlx_memory_budget(retains_history = False) == 85_000_000_000
    fake.metal.is_available = lambda: False
    assert mlx_inference.mlx_memory_budget() is None


def test_the_fit_asks_about_the_repo_the_loader_actually_opens(monkeypatch, tmp_path):
    import utils.utils as backend_utils
    from core.inference import mlx_inference

    asked = []
    monkeypatch.setattr(
        backend_utils, "hf_cache_snapshot_dir", lambda name: asked.append(name) or tmp_path
    )
    assert mlx_inference._snapshot_dir(SimpleNamespace(_src_path = tmp_path), "org/alias") == str(
        tmp_path
    )
    assert asked == []
    assert mlx_inference._snapshot_dir(SimpleNamespace(), "org/alias") == str(tmp_path)
    assert asked == ["org/alias"]


def test_only_a_load_that_asked_for_nothing_is_fitted_to_the_machine(monkeypatch):
    """A pin is the user's own answer about memory and is served verbatim."""
    _install_fake_mlx(monkeypatch)
    _install_fake_fast_mlx(monkeypatch, [])

    from core.inference import mlx_inference

    FULL = ("full", "", True)
    asked, verdicts, probes = [], [], []
    monkeypatch.setattr(
        mlx_inference, "_kv_quant_eligibility", lambda *a: verdicts.append(a) or FULL
    )
    monkeypatch.setattr(mlx_inference, "_row_quantized_cache_gap", lambda: "unavailable")

    def load(
        fitted = (24_576, FULL),
        enforceable = bool,
        model = "fake/text",
        **kwargs,
    ):
        del asked[:], verdicts[:], probes[:]
        monkeypatch.setattr(
            mlx_inference,
            "_fitted_context",
            lambda _m, name, ceiling, **kw: asked.append((name, ceiling, kw)) or fitted,
        )
        monkeypatch.setattr(
            mlx_inference.MLXInferenceBackend,
            "_kv_cache_window_enforceable",
            lambda _s, served: probes.append(served) or enforceable(served),
        )
        backend = mlx_inference.MLXInferenceBackend()
        assert backend.load_model(
            SimpleNamespace(
                identifier = model,
                is_vision = kwargs.pop("vision", False),
                is_lora = kwargs.pop("lora", False),
            ),
            **kwargs,
        )
        return backend, backend.models[model]

    backend, pinned = load(max_seq_length = 4096, kv_quant = "4")
    assert (asked, verdicts) == ([], [(backend._model, False, 4)])
    assert (pinned["context_length"], pinned["context_length_fitted"]) == (4096, None)
    assert (pinned["mlx_kv_bits"], pinned["mlx_context_budget"]) == (4, 4096)
    backend, info = load(max_seq_length = 0, kv_quant = "4")
    assert (verdicts, probes) == ([], [24_576])
    assert asked == [
        (
            "fake/text",
            None,
            dict(
                load_in_4bit = True,
                retains_history = True,
                kv_bits = 4,
                is_vlm = False,
                eligibility = None,
            ),
        )
    ]
    assert (info["context_length"], info["context_length_fitted"]) == (24_576, 24_576)
    assert (info["mlx_kv_bits"], info["mlx_context_budget"]) == (4, 24_576)
    assert (info["context_length_enforced"], backend._kv_cache_window) == (False, None)

    backend, info = load((24_576, None), max_seq_length = 0)
    assert (info["mlx_kv_bits"], info["mlx_context_budget"]) == (None, None)
    assert (info["context_length_enforced"], backend._kv_cache_window) == (True, 24_576)
    refused = ("refused", "r", True)
    backend, info = load((24_576, refused), max_seq_length = 0, kv_quant = "4")
    assert (info["mlx_kv_bits"], backend._kv_cache_window, verdicts) == (None, 24_576, [])
    _, info = load(enforceable = lambda _s: False, max_seq_length = 0, kv_quant = "4")
    assert (info["mlx_kv_bits"], info["mlx_context_budget"], info["context_length"]) == (
        4,
        None,
        24_576,
    )
    backend, info = load(enforceable = lambda _s: False, max_seq_length = 4096, kv_quant = "4")
    assert (info["mlx_kv_bits"], verdicts) == (4, [(backend._model, False, 4)])

    monkeypatch.setitem(
        sys.modules, "mlx_vlm.turboquant", SimpleNamespace(TurboQuantKVCache = object)
    )
    monkeypatch.setattr(
        mlx_inference,
        "_turboquant_status",
        lambda _m, bits, refusal = "": dict(
            requested_kv_bits = bits, kv_bits = bits, eligibility = "full", reason = "", note = ""
        ),
    )
    load(max_seq_length = 0, kv_quant = "tq-4")
    assert verdicts == [] and [kw["kv_bits"] for _, _, kw in asked] == [None]

    for available, reserved in ((True, True), (False, False)):
        monkeypatch.setattr(
            mlx_inference, "mlx_vlm_snapshot_store_available", lambda a = available: a
        )
        load(model = "fake/vlm", vision = True, max_seq_length = 0)
        assert asked == [
            (
                "fake/vlm",
                None,
                dict(
                    load_in_4bit = True,
                    retains_history = reserved,
                    kv_bits = None,
                    is_vlm = True,
                    eligibility = None,
                ),
            )
        ]

    for kwargs in (
        dict(
            parallel_mode = "tensor",
            distributed_group = SimpleNamespace(rank = lambda: 0, size = lambda: 2),
        ),
        dict(dtype = "float32"),
        dict(lora = True),
    ):
        load(max_seq_length = 0, **kwargs)
        assert not asked


def test_the_width_probe_judges_the_tower_the_sizing_would_price(monkeypatch):
    """The load's eligibility rule, asked of the first tower that runs, and never evaluated."""
    from core.inference import mlx_inference

    fake_mx = types.SimpleNamespace(
        array = lambda value: value,
        zeros = lambda *_a, **_k: None,
        int32 = None,
        eval = lambda *_a: pytest.fail("a weightless tower must never be evaluated"),
    )
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mx)
    if "mlx" in sys.modules:
        monkeypatch.setattr(sys.modules["mlx"], "core", fake_mx, raising = False)
    else:
        monkeypatch.setitem(sys.modules, "mlx", types.SimpleNamespace(core = fake_mx))

    class Quantized:
        keys, state, nbytes = "k", (), 1

    class Entry:
        keys, state, window_size = "k", (), None

        def __init__(
            self,
            max_size = None,
            converts = True,
        ):
            self.max_size = max_size
            if converts:
                self.to_quantized = lambda **_kw: Quantized()

    class Tower:
        def __init__(
            self,
            *entries,
            runs = True,
            attends_quantized = True,
        ):
            self.entries, self.runs, self.attends_quantized = entries, runs, attends_quantized

        def __call__(
            self,
            *_a,
            cache = None,
            **_k,
        ):
            if not self.runs:
                raise ValueError("this tower cannot run")
            if not self.attends_quantized and any(isinstance(e, Quantized) for e in cache):
                raise TypeError("shared KV takes arrays only")

    rewound = []

    def refused(*towers):
        monkeypatch.setitem(
            sys.modules,
            "core.inference.mlx_memory",
            types.SimpleNamespace(
                _PROBE_SHORT = 8,
                _runtime_dtype = lambda: None,
                _snapshot_config = lambda _d: {"model_type": "x"},
                _probe_models = lambda *_a: iter(
                    [(lambda t = t: t, lambda tower: list(tower.entries), None) for t in towers]
                ),
                weightless_forward = contextlib.nullcontext,
            ),
        )
        marks = len(rewound)
        mlx_inference._kv_refusal_cache.clear()
        answer = mlx_inference.mlx_kv_quant_is_refused("/d", 4)
        assert rewound[marks] == "held" and rewound[-1] == ("key",)
        return answer

    monkeypatch.setattr(
        mlx_inference, "_mlx_rng_key_words", lambda: rewound.append("held") or ("key",)
    )
    monkeypatch.setattr(mlx_inference, "_restore_mlx_rng_key", rewound.append)

    assert refused(Tower(Entry())) is False
    assert refused(Tower(Entry(), Entry(max_size = 512))) is False
    assert refused(Tower(Entry(max_size = 512))) is True
    assert refused(Tower(Entry(converts = False))) is True
    assert refused(Tower()) is True
    # Gemma 4 E2B / E4B: the conversion succeeds, attending over it does not.
    assert refused(Tower(Entry(), Entry(max_size = 512), attends_quantized = False)) is True
    broken = Entry()
    broken.to_quantized = lambda **_kw: (_ for _ in ()).throw(ValueError("group size"))
    assert refused(Tower(broken)) is True
    assert refused(Tower(Entry(max_size = 512), runs = False), Tower(Entry())) is False
    assert refused(Tower(Entry(), runs = False), Tower(Entry(max_size = 512))) is True
    assert refused(Tower(Entry(max_size = 512), runs = False)) is False

    monkeypatch.setitem(
        sys.modules,
        "core.inference.mlx_memory",
        types.SimpleNamespace(
            _snapshot_config = lambda _d: (_ for _ in ()).throw(RuntimeError("no")),
        ),
    )
    marks = len(rewound)
    mlx_inference._kv_refusal_cache.clear()
    assert mlx_inference.mlx_kv_quant_is_refused("/d", 4) is False
    assert rewound[marks:] == ["held", ("key",)]


class _TwoRowStream:
    """A vision batch reporting two rows, recording whatever is taken back."""

    def __init__(
        self,
        events,
        retired = (),
    ):
        self._events, self._retired, self.withdrawn = events, set(retired), []

    def iter_step(self, waiting):
        yield from self._events
        self.exhausted = True

    def withdraw(self, row):
        assert self.exhausted, "the batch cannot change mid-step"
        self.withdrawn.append(row)
        if row in self._retired:
            return None
        return SimpleNamespace(prompt_token_count = 3, token_ids = [1, 2], **_DRAFTS)


_DRAFTS = {"draft_tokens": 8, "accepted_draft_tokens": 3}


def _open_vision_session(rows):
    """A session holding the given rows, with nothing else started."""
    from core.inference import mlx_inference

    session = mlx_inference._VisionBatchSession.__new__(mlx_inference._VisionBatchSession)
    session._rows = {row.handle: row for row in rows}
    session._by_row, session._settled = {row.row: row for row in rows}, {}
    session._speculative, session._decodes_ids = None, False
    return session


@pytest.mark.parametrize("stopping", (0, 1))
@pytest.mark.parametrize("terminal", (False, True))
def test_a_vision_row_cut_by_its_stop_sequence_leaves_the_batch(monkeypatch, stopping, terminal):
    """Whichever row stops is closed out where it stopped and its neighbour."""
    from core.inference import mlx_inference

    backend = _vlm_backend(monkeypatch, markers = None)
    handles = ("a", "b")
    session = _open_vision_session(
        [
            mlx_inference._VisionBatchRow(
                handle = handle,
                plan = SimpleNamespace(stream = _vlm_row(backend, stop = ["HALT"]), streamed = None),
                row = number,
            )
            for number, handle in enumerate(handles)
        ]
    )
    other, session._speculative = 1 - stopping, "draft"
    result = SimpleNamespace(finish_reason = "length", prompt_token_count = 3, token_ids = [1, 2])
    vars(result).update(_DRAFTS)
    session.stream = _TwoRowStream(
        [
            SimpleNamespace(
                index = stopping, delta = "one HALT two", result = result if terminal else None
            ),
            SimpleNamespace(index = other, delta = "still going", result = None),
        ],
        retired = (stopping,) if terminal else (),
    )

    reported = list(session.step())

    assert (handles[stopping], None) in reported, "the stopped row is reported finished"
    assert session._settled[handles[stopping]]["finish_reason"] == "stop"
    timings = session._settled[handles[stopping]]["timings"]
    assert (timings["draft_n"], timings["draft_n_accepted"]) == (8, 3)
    assert handles[stopping] not in session._rows, "and it has left the batch"
    assert handles[other] in session._rows, "its neighbour is still in the batch"
    assert (handles[other], "still going") in reported
    assert session.stream.withdrawn == (
        [] if terminal else [stopping]
    ), "a row the engine has already retired is not handed back again"


def test_a_resident_vision_tool_row_decodes_its_ids_and_keeps_the_stop_that_closes_its_call(
    monkeypatch,
):
    from core.inference import mlx_inference

    pieces = ("<|tool_call>", "call:terminal{command:id}", "<tool_call|>")
    turn = _SpmTurn(
        pieces, specials = pieces[::2], eos = "<tool_call|>", chat_template = "x", ends = "stop"
    )
    backend = _vlm_backend(monkeypatch, markers = None)
    backend._tokenizer = backend._processor.tokenizer = turn
    plan = backend._plan_vlm_row(
        [{"role": "user", "content": "hi"}],
        None,
        **dict(temperature = 0, top_p = 1, top_k = 0, min_p = 0, max_new_tokens = 9),
        repetition_penalty = 1.0,
        tools = _TOOL,
        decodes_ids = True,
    )
    session = _open_vision_session([mlx_inference._VisionBatchRow(handle = "r", plan = plan, row = 0)])
    ended = SimpleNamespace(finish_reason = "stop", prompt_token_count = 3, token_ids = turn.ids[:2])
    event = SimpleNamespace(index = 0, delta = "", token_ids = tuple(turn.ids), result = ended)
    session.stream = SimpleNamespace(step = lambda: [event])
    assert list(session.step())[-2:] == [("r", "".join(pieces)), ("r", None)]


def test_a_text_load_batch_row_carries_one_bos_as_the_single_path_does(monkeypatch):
    engine = _batch_engine(monkeypatch)
    monkeypatch.setattr(engine, "GenerationRequest", SimpleNamespace, raising = False)
    monkeypatch.setattr(engine, "SamplingParams", dict, raising = False)
    monkeypatch.setattr(engine, "BatchRowRefused", RuntimeError, raising = False)
    monkeypatch.setattr(engine, "vlm_batch_adds_special_tokens", lambda *a: True, raising = False)
    backend = _vlm_backend(monkeypatch, markers = None)
    backend._tokenizer.bos_token = "<s>"
    backend._plan_vlm_request = lambda request, **_k: SimpleNamespace(
        prompt = "<s>hi",
        images = None,
        videos = None,
        max_tokens = 4,
        sampling = {},
        processors = None,
        think_prefix = "",
        stream = None,
        streamed = None,
    )
    session = _open_vision_session([])
    session.backend, session._adapter_state, session._resumes_rows = backend, None, False
    added = []
    session.stream = SimpleNamespace(
        add = lambda request: added.append(request.prompt) or len(added) - 1
    )

    def stream_batch(model, processor, batch, defaults):
        added.extend(request.prompt for request in batch)
        asked.append(defaults)
        raise LookupError

    asked = []
    monkeypatch.setattr(engine, "stream_batch", stream_batch, raising = False)
    monkeypatch.setattr(engine, "GenerationDefaults", SimpleNamespace, raising = False)
    backend._reads_vision, backend._generation_lock = False, threading.Lock()
    with pytest.raises(LookupError):
        next(backend._generate_vlm_batch([{}]))
    from dataclasses import make_dataclass

    fields = (
        "max_tokens",
        "prefill_batch_size",
        "completion_batch_size",
        ("prefill_step_size", int, None),
    )
    monkeypatch.setattr(
        engine, "GenerationDefaults", make_dataclass("Defaults", fields), raising = False
    )
    backend._prefill_step = 8192
    with pytest.raises(LookupError):
        next(backend._generate_vlm_batch([{}]))
    assert asked[-1].prefill_step_size == 2048 and added.pop() == "hi"
    backend._prefill_step = None
    for reads_vision in (False, True):
        backend._reads_vision = reads_vision
        session.admit({}, reads_vision)
    monkeypatch.setattr(engine, "vlm_batch_adds_special_tokens", lambda *a: False)
    backend._reads_vision = False
    session.admit({}, "gemma")
    assert added == ["hi", "hi", "<s>hi", "<s>hi"]


@pytest.mark.parametrize("speculative", (None, "draft"))
def test_a_vision_row_times_its_prefill_wherever_the_batch_runs_it(monkeypatch, speculative):
    from core.inference import mlx_inference

    engine = _batch_engine(monkeypatch)
    monkeypatch.setattr(engine, "GenerationRequest", SimpleNamespace, raising = False)
    monkeypatch.setattr(engine, "SamplingParams", dict, raising = False)
    monkeypatch.setattr(engine, "BatchRowRefused", RuntimeError, raising = False)
    monkeypatch.setattr(engine, "vlm_batch_adds_special_tokens", lambda *a: True, raising = False)
    backend = _vlm_backend(monkeypatch, markers = None)
    plan = SimpleNamespace(
        prompt = "hi",
        images = None,
        max_tokens = 16,
        sampling = {},
        processors = None,
        think_prefix = "",
        stream = _vlm_row(backend),
        streamed = None,
        speculates = True,
    )
    backend._plan_vlm_request = lambda request, **_k: plan
    backend._reads_vision = False
    session = _open_vision_session([])
    session.backend, session._adapter_state, session._resumes_rows = backend, None, False
    session._speculative = speculative
    clock = [0.0]
    monkeypatch.setattr(mlx_inference.time, "perf_counter", lambda: clock[0])

    def prefill():
        clock[0] += 2.0

    steps = []

    def events(waiting = None):
        steps.append(None)
        if len(steps) == 1:
            if not speculative:
                prefill()
            yield SimpleNamespace(index = 0, delta = "a", result = None)
            return
        clock[0] += 1.0
        result = SimpleNamespace(
            finish_reason = "length", prompt_token_count = 1000, token_ids = [1] * 10
        )
        vars(result).update(_DRAFTS)
        yield SimpleNamespace(index = 0, delta = "", result = result)

    def add(request):
        if speculative:
            prefill()
        else:
            # A plain add() only prepares the row; that time is not its prefill.
            clock[0] += 3.0
        return 0

    session.stream = SimpleNamespace(
        add = add,
        step = lambda: list(events()),
        iter_step = events,
    )
    session.admit({}, "r")
    while session.rows_in_flight:
        list(session.step())

    timings = session.take_stats("r")["timings"]
    assert timings["prompt_per_second"] == pytest.approx(500.0)
    assert timings["predicted_per_second"] == pytest.approx(10.0)


def test_a_fixed_vision_batch_marks_each_row_s_clips_and_shares_their_decode_budget(monkeypatch):
    import base64

    from core.inference import mlx_inference

    engine, backend = _batch_engine(monkeypatch), _vlm_backend(monkeypatch, markers = None)
    for name in ("GenerationRequest", "GenerationDefaults"):
        monkeypatch.setattr(engine, name, SimpleNamespace, raising = False)
    monkeypatch.setattr(engine, "SamplingParams", dict, raising = False)
    monkeypatch.setattr(engine, "vlm_batch_adds_special_tokens", lambda *a: False, raising = False)
    too_large, refuse = (_ for _ in ()).throw, mlx_inference._ClipsTooLarge
    fps = lambda p, proc, rows = 1: rows / 4 if rows < 3 else too_large(refuse())
    monkeypatch.setattr(mlx_inference, "_video_sampling_fps", fps)
    seen, paths = [], []
    clips, other = ([base64.b64encode(data).decode()] for data in (b"one", b"two"))
    backend._render_vlm_prompt = lambda messages, images, videos, **k: (
        seen.append(([p["type"] for p in messages[-1]["content"]], videos)) or ("p", None, None)
    )
    backend._media_generation_limit = lambda p, n, ids, images, videos: seen.append(videos) or n
    backend._kv_window_generate_kwargs, backend._generation_lock = dict, threading.Lock()
    backend._release_vlm_snapshots = lambda: seen.append("freed")

    def stream_batch(model, processor, batch, defaults):
        seen.extend((row.fps, [Path(path).read_bytes() for path in row.videos]) for row in batch)
        paths.extend(path for row in batch for path in row.videos)
        return iter(())

    monkeypatch.setattr(engine, "stream_batch", stream_batch, raising = False)
    request = {"messages": [{"role": "user", "content": "hi"}], "videos": clips}
    with pytest.raises(RuntimeError, match = "unfinished"):
        list(backend._generate_vlm_batch([request, request, {**request, "videos": other}]))
    planned = [(["video", "text"], clips), clips] * 2 + [(["video", "text"], other), other]
    assert seen[:7] == planned + ["freed"]
    rates = [(0.5, [b"one"])] * 2 + [(0.25, [b"two"])]
    assert seen[7:] == rates and paths[0] == paths[1] != paths[2], "a group shares files and budget"
    seen.clear()
    with pytest.raises(RuntimeError, match = "unfinished"):
        list(backend._generate_vlm_batch([request] * 3))
    assert seen[-3:] == [(0.25, [b"one"])] * 3 and len(set(paths[3:])) == 3, "too large to share"
    assert not any(map(os.path.exists, paths)), "removed when the batch ends"


def test_a_resident_vision_row_resumes_from_the_snapshot_store_and_reports_it(monkeypatch):
    from core.inference import mlx_inference

    engine = _batch_engine(monkeypatch)
    monkeypatch.setattr(engine, "GenerationRequest", SimpleNamespace, raising = False)
    monkeypatch.setattr(engine, "SamplingParams", dict, raising = False)
    monkeypatch.setattr(engine, "BatchRowRefused", RuntimeError, raising = False)
    monkeypatch.setattr(engine, "row_prompt_cache_unavailable_reason", lambda: None, raising = False)
    monkeypatch.setattr(engine, "vlm_batch_adds_special_tokens", lambda *a: True, raising = False)
    backend = _vlm_backend(monkeypatch, markers = None)
    store = mlx_inference.VLMPromptSnapshotStore(max_bytes = 10**9)
    for key in ("m", "m|img"):
        store._entries[(key, (1,))] = ([], 0)
    backend._vlm_snapshot_store = store
    backend._vlm_prompt_cache_store = lambda: store
    backend._vlm_is_diffusion_model = lambda model: False
    backend._vlm_snapshot_scope = lambda state, images: (None, "m|img" if images else "m", list)

    backend._plan_vlm_request = lambda request, **_k: SimpleNamespace(
        prompt = "p",
        images = request.get("images"),
        max_tokens = 4,
        sampling = {},
        processors = None,
        think_prefix = "",
        stream = _vlm_row(backend),
        streamed = None,
    )
    session = _open_vision_session([])
    session.backend, session._adapter_state = backend, None
    session._resumes_rows = mlx_inference._row_prompt_cache_gap() is None
    added = []
    session.stream = SimpleNamespace(add = lambda request: added.append(request) or len(added) - 1)

    session.admit({}, "text")
    assert added[0].prompt_cache_state._store is store and len(store) == 2
    backend._model = SimpleNamespace(config = SimpleNamespace(image_token_id = 9))
    session.admit({"images": [object()]}, "image")
    assert list(store._entries) == [("m|img", (1,))]
    assert added[1].prompt_cache_state._store is store
    backend._model.config.use_bidirectional_attention = "vision"
    session.admit({"images": [object()]}, "block")
    assert added[2].prompt_cache_state._store is None and len(store) == 0

    result = SimpleNamespace(
        finish_reason = "length", prompt_token_count = 10, cached_token_count = 6, token_ids = [1]
    )
    session.stream.step = lambda: iter([SimpleNamespace(index = 0, delta = "", result = result)])
    list(session.step())
    stats = session.take_stats("text")
    assert stats["usage"]["prompt_tokens"] == 10 and stats["timings"]["prompt_n"] == 4
    assert stats["usage"]["prompt_tokens_details"]["cached_tokens"] == 6
    session.stream.withdraw = lambda row: result
    list(session.withdraw(["image"]))
    assert session.take_stats("image")["usage"]["prompt_tokens_details"]["cached_tokens"] == 6

    store._entries[("m|img", (1,))] = ([], 0)
    monkeypatch.setattr(engine, "row_prompt_cache_unavailable_reason", lambda: "old zoo")
    session._resumes_rows = mlx_inference._row_prompt_cache_gap() is None
    session.admit({"images": [object()]}, "old")
    assert not hasattr(added[3], "prompt_cache_state") and len(store) == 0

    backend._kv_quant = {"kv_bits": 4}
    with pytest.raises(mlx_inference.RowRefused):
        session.admit({}, "unquantized")
    session._resumes_rows = True
    backend._vlm_prompt_cache_store = lambda: None
    session.admit({"images": [object()]}, "quantized")
    assert added[4].prompt_cache_state._store is None


def _batch_engine(monkeypatch):
    """The zoo batch engine, or a stand-in where the backend CI installs no unsloth-zoo."""
    try:
        from unsloth_zoo.mlx import generate as engine
    except ImportError:
        engine = types.ModuleType("unsloth_zoo.mlx.generate")
        engine.stream_unavailable_reason = lambda *a, **k: None
        for name in ("unsloth_zoo", "unsloth_zoo.mlx"):
            if not isinstance(sys.modules.get(name), types.ModuleType):
                monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.generate", engine)
        monkeypatch.setattr(sys.modules["unsloth_zoo.mlx"], "generate", engine, raising = False)
    return engine


def test_a_text_load_whose_engine_cannot_batch_says_so_before_it_commits(monkeypatch):
    """The reply is owed the decode it would have had, not the error building raises."""
    from core.inference.mlx_inference import MLXInferenceBackend

    engine = _batch_engine(monkeypatch)

    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._model, backend._tokenizer, backend._is_vlm = object(), object(), False
    for gap in ("BatchGenerator.insert missing", None):
        monkeypatch.setattr(engine, "stream_unavailable_reason", lambda *a, **k: gap)
        assert backend.resident_unavailable_reason({}) == gap
        assert backend.batch_unavailable_reason([{}, {}]) == gap


@pytest.mark.parametrize(
    "extra",
    (
        {"images": [object()]},
        {"images_base64": ["AAAA"]},
        {"videos": ["AAAAGGZ0eXBtcDQy"]},
        {"videos_base64": ["AAAAGGZ0eXBtcDQy"]},
    ),
)
def test_a_reply_the_batch_cannot_serve_decodes_alone(monkeypatch, extra):
    """Replayed tool images and a clip live only on the single path."""
    from core.inference.mlx_inference import MLXInferenceBackend

    engine = _batch_engine(monkeypatch)

    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._model, backend._tokenizer, backend._is_vlm = object(), object(), False
    monkeypatch.setattr(engine, "stream_unavailable_reason", lambda *a, **k: None)
    plain = {"messages": [], "images": [], "image_ordinal": 0}

    assert backend.resident_unavailable_reason(plain) is None
    assert backend.batch_unavailable_reason([plain, plain]) is None
    assert backend.resident_unavailable_reason({**plain, **extra}) is not None
    assert backend.batch_unavailable_reason([plain, {**plain, **extra}]) is not None


@pytest.mark.parametrize(
    "extra, refusal",
    (
        ({"tools": [{"type": "function"}]}, "a reply runs the tool protocol"),
        ({"tool_protocol_active": True}, "a reply runs the tool protocol"),
        (
            {"response_format": {"type": "json_object"}},
            "a reply is constrained to a response format",
        ),
        ({"grammar": 'root ::= "a"'}, "a reply is constrained to a response format"),
    ),
)
def test_a_tool_or_constrained_reply_joins_a_batch_that_decodes_it_from_its_ids(
    monkeypatch, extra, refusal
):
    from dataclasses import make_dataclass

    from core.inference import mlx_inference

    engine = _batch_engine(monkeypatch)
    monkeypatch.setattr(engine, "stream_unavailable_reason", lambda *a, **k: None)
    monkeypatch.setattr(mlx_inference, "_batched_vision_engine_gap", lambda _engine: None)
    backend = mlx_inference.MLXInferenceBackend.__new__(mlx_inference.MLXInferenceBackend)
    backend._model, backend._tokenizer, backend._is_vlm = object(), object(), False
    assert backend.batch_unavailable_reason([{}, extra]) is None
    backend._model, backend._is_vlm = SimpleNamespace(_is_vlm_model = True), True
    backend._processor, backend._speculative_draft = object(), None
    backend._kv_quant, backend._kv_context_budget = {"kv_bits": None}, None
    backend._turboquant = False
    assert backend.resident_unavailable_reason({}) is None
    monkeypatch.setattr(engine, "GenerationDefaults", SimpleNamespace, raising = False)
    monkeypatch.setattr(engine, "BatchStream", lambda *_a, **_k: None, raising = False)
    for fields, draft, joins in (
        (("token_ids",), None, True),
        ((), None, False),
        (("token_ids",), 1, False),
        (("drafts",), 1, False),
        (("token_ids", "drafts"), 1, True),
    ):
        for name in ("GenerationEvent", "GenerationRequest"):
            monkeypatch.setattr(engine, name, make_dataclass(name, fields), raising = False)
        backend._speculative_draft = draft
        assert backend.resident_unavailable_reason(extra) == (None if joins else refusal)
        assert mlx_inference._VisionBatchSession(backend, width = 2)._decodes_ids == joins
    backend._speculative_draft = None  # The fixed multi-reply command decodes the batch's text.
    assert backend.batch_unavailable_reason([{}, extra]) == refusal
    assert backend.resident_unavailable_reason({"response_format": {"type": "text"}}) is None


def test_a_quantized_or_budgeted_kv_cache_does_not_batch(monkeypatch):
    from core.inference.mlx_inference import MLXInferenceBackend

    engine = _batch_engine(monkeypatch)

    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._model, backend._tokenizer, backend._is_vlm = object(), object(), False
    backend._kv_quant, backend._kv_context_budget = {"kv_bits": None}, None
    monkeypatch.setattr(engine, "stream_unavailable_reason", lambda *a, **k: None)
    assert backend.resident_unavailable_reason({}) is None

    for quant, budget in (({"kv_bits": 8}, None), ({"kv_bits": None}, 4096)):
        backend._kv_quant, backend._kv_context_budget = quant, budget
        assert backend.resident_unavailable_reason({}) is not None
        assert backend.batch_unavailable_reason([{}, {}]) is not None


def test_a_quantized_vision_load_batches_only_as_resident_rows(monkeypatch):
    from core.inference.mlx_inference import MLXInferenceBackend

    engine = _batch_engine(monkeypatch)
    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._model, backend._processor, backend._is_vlm = object(), object(), True
    backend._kv_quant, backend._kv_context_budget = {"kv_bits": 4}, 4096
    backend._turboquant = False
    backend._vlm_batch_unavailable_reason = lambda requests, **_k: None
    monkeypatch.setattr(engine, "stream_unavailable_reason", lambda *a, **k: None)
    capable = lambda: None
    monkeypatch.setattr(
        engine, "row_quantized_prompt_cache_unavailable_reason", capable, raising = False
    )
    assert backend.resident_unavailable_reason({}) is None
    assert backend.batch_unavailable_reason([{}, {}]) is not None
    backend._turboquant = True
    assert backend.resident_unavailable_reason({}) is None
    assert backend.batch_unavailable_reason([{}, {}]) is not None
    monkeypatch.setattr(
        engine, "row_quantized_prompt_cache_unavailable_reason", lambda: "old mlx-vlm"
    )
    assert backend.resident_unavailable_reason({}) == "old mlx-vlm"


@pytest.mark.parametrize("gap", (None, "speculative", "old zoo", "no decoder", "no vision"))
def test_a_video_reply_batches_only_as_a_fixed_command_of_a_load_that_reads_it(monkeypatch, gap):
    from core.inference import mlx_inference

    engine, backend = _batch_engine(monkeypatch), _vlm_backend(monkeypatch, markers = None)
    backend._speculative_draft = gap == "speculative" or None
    backend._reads_vision, available = gap != "no vision", lambda request: None
    backend._vlm_batch_unavailable_reason = backend._vlm_resident_unavailable_reason = available
    monkeypatch.setattr(mlx_inference, "_mlx_vlm_decodes_video", lambda: gap != "no decoder")
    fields = SimpleNamespace(__dataclass_fields__ = ("image", "videos")[: 1 + (gap != "old zoo")])
    monkeypatch.setattr(engine, "GenerationRequest", fields, raising = False)
    clip, sent = {"videos": [_CLIP_B64]}, {"videos_base64": [_CLIP_B64]}
    for row in (clip, sent):
        assert (backend.batch_unavailable_reason([{}, row]) is None) == (gap is None)
    assert backend.resident_unavailable_reason(sent) == "a reply carries a video clip"
    assert backend.resident_unavailable_reason({}) is None


def test_a_turboquant_vision_batch_quantizes_rows_as_the_single_path_does(monkeypatch):
    from core.inference import mlx_inference

    engine = _batch_engine(monkeypatch)
    monkeypatch.setattr(engine, "GenerationDefaults", SimpleNamespace, raising = False)
    monkeypatch.setattr(
        engine, "BatchStream", lambda model, processor, defaults: defaults, raising = False
    )
    backend = mlx_inference.MLXInferenceBackend.__new__(mlx_inference.MLXInferenceBackend)
    backend._model = backend._processor = object()
    backend._kv_quant, backend._turboquant = {"kv_bits": 3.5}, True
    defaults = mlx_inference._VisionBatchSession(backend, width = 2).stream
    assert (defaults.kv_bits, defaults.kv_quant_scheme, defaults.quantized_kv_start) == (
        3.5,
        "turboquant",
        0,
    )
    backend._kv_quant, backend._turboquant = {"kv_bits": 4}, False
    assert not hasattr(mlx_inference._VisionBatchSession(backend, width = 2).stream, "kv_bits")


def test_a_batched_vision_row_keeps_to_the_context_budget(monkeypatch):
    from core.inference import context_refusal

    backend = _vlm_backend(monkeypatch, markers = None)
    backend._kv_context_budget = 1024
    plan = lambda: backend._plan_vlm_row(
        [{"role": "user", "content": "hi"}],
        None,
        temperature = 0.7,
        top_p = 0.9,
        top_k = 0,
        min_p = 0.0,
        max_new_tokens = 500,
        repetition_penalty = 1.0,
    )
    backend._count_prompt_tokens = lambda *a, **k: 1000
    assert plan().max_tokens == 24
    backend._count_prompt_tokens = lambda *a, **k: 1024
    with pytest.raises(context_refusal.ContextBudgetExceeded):
        plan()


try:
    import mlx.core as _mx
    _METAL = _mx.metal.is_available()
except Exception:
    _METAL = False

metal_only = pytest.mark.skipif(not _METAL, reason = "requires Apple Silicon Metal")

BATCH_MODEL = "mlx-community/SmolLM2-135M-Instruct"
BATCH_PROMPT = [{"role": "user", "content": "Name a colour and explain why."}]
BATCH_VLM_MODEL = "mlx-community/Qwen3.5-0.8B-4bit"


@pytest.fixture(scope = "module")
def batch_backend(allow_outbound_network):
    from core.inference.mlx_inference import MLXInferenceBackend

    loaded = MLXInferenceBackend()
    with allow_outbound_network():
        loaded.load_model(BATCH_MODEL)
    return loaded


@pytest.fixture(scope = "module")
def batch_vlm_backend(allow_outbound_network):
    from core.inference.mlx_inference import MLXInferenceBackend

    loaded = MLXInferenceBackend()
    with allow_outbound_network():
        loaded.load_model(SimpleNamespace(identifier = BATCH_VLM_MODEL, is_vision = True))
    return loaded


def _batch_request(**overrides):
    base = dict(messages = BATCH_PROMPT, max_new_tokens = 24, temperature = 0.9)
    base.update(overrides)
    return base


def _drain_batch(backend, requests, **kwargs):
    """Cumulative snapshots per row, plus the stats latched at the end."""
    replies: dict[int, list[str]] = {row: [] for row in range(len(requests))}
    finished = []
    for row, snapshot in backend.generate_chat_batch(requests, **kwargs):
        if snapshot is None:
            finished.append(row)
        else:
            replies[row].append(snapshot)
    assert sorted(finished) == sorted(replies), "every row must report completion once"
    return replies, list(backend.last_batch_generation_stats)


@metal_only
def test_a_batch_costs_one_forward_per_token_not_one_per_reply(batch_backend):
    model = batch_backend._model
    original = type(model).__call__
    calls = {"n": 0}

    def counted(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    type(model).__call__ = counted
    try:
        _replies, stats = _drain_batch(
            batch_backend, [_batch_request(seed = 200 + i) for i in range(4)]
        )
    finally:
        type(model).__call__ = original

    generated = sum(entry["usage"]["completion_tokens"] for entry in stats)
    assert generated >= 4 * 20
    assert (
        calls["n"] < generated / 2
    ), f"{calls['n']} forwards for {generated} tokens across 4 replies"


@metal_only
def test_an_out_of_range_logit_bias_id_cannot_corrupt_a_batch(batch_backend):
    """MLX does no bounds checking on the gather these processors do."""
    plain = _batch_request(seed = 21)
    stray = _batch_request(seed = 21, logit_bias = {10**9: 5.0, -4: 3.0})
    _first, _ = _drain_batch(batch_backend, [plain, plain])
    opener = batch_backend._tokenizer.encode(_first[0][-1])[0]
    real = _batch_request(seed = 21, logit_bias = {int(opener): -100.0})

    replies, stats = _drain_batch(batch_backend, [plain, stray, stray, real])
    assert all(entry["usage"]["completion_tokens"] > 0 for entry in stats)
    assert replies[1] == replies[0]
    assert replies[2] == replies[0]
    assert replies[3] != replies[0]


def test_a_repeated_token_is_charged_the_penalty_it_earned():
    """The frequency processor counts occurrences in float32 and scales once."""
    mx = pytest.importorskip("mlx.core")

    from core.inference.mlx_inference import _make_mlx_frequency_penalty_processor

    processor = _make_mlx_frequency_penalty_processor(0.3)
    logits = mx.zeros((1, 8), dtype = mx.float16)
    processor(mx.array([0, 1]), logits)
    tokens = mx.array([0, 1] + [5] * 1000)
    charged = processor(tokens, logits)
    assert abs(float(charged[0, 5]) + 300.0) < 0.5, float(charged[0, 5])
    assert float(charged[0, 4]) == 0.0


@metal_only
def test_a_loaded_vision_model_says_a_batch_would_leave_its_window_behind(batch_vlm_backend):
    stored = batch_vlm_backend.models[batch_vlm_backend.active_model_name]

    assert stored["context_unbounded_when_batched"] is True


@metal_only
def test_a_vision_reply_stopped_partway_reports_the_tokens_it_actually_used(batch_vlm_backend):
    """A stopped row has no result, so its usage has to come from what it produced."""
    import threading

    from PIL import Image

    def rows():
        return [
            dict(
                messages = [{"role": "user", "content": "Describe this in detail."}],
                image = Image.new("RGB", (64, 64), (12, 200, 60)),
                max_new_tokens = 64,
                temperature = 0.0,
            )
            for _ in range(2)
        ]

    _replies, whole = _drain_batch(batch_vlm_backend, rows())

    cancel = threading.Event()
    delivered: dict[int, list[str]] = {0: [], 1: []}
    for row, snapshot in batch_vlm_backend.generate_chat_batch(rows(), cancel_event = cancel):
        if snapshot is not None:
            delivered[row].append(snapshot)
            if all(len(seen) >= 3 for seen in delivered.values()):
                cancel.set()

    cut_short = batch_vlm_backend.last_batch_generation_stats
    assert all(delivered.values()), "nothing was delivered before the batch was cancelled"
    for row, seen in delivered.items():
        usage, entire = cut_short[row]["usage"], whole[row]["usage"]
        assert (
            usage["prompt_tokens"] == entire["prompt_tokens"]
        ), "a stopped row read a different prompt than the same row read whole"
        assert len(seen) <= usage["completion_tokens"] < entire["completion_tokens"], (
            f"row {row} delivered {len(seen)} snapshots and reported "
            f"{usage['completion_tokens']} of {entire['completion_tokens']} tokens"
        )


import numpy as np  # noqa: E402

from core.inference import grammar_constraint as gc
from core.inference.grammar_constraint import (
    ResponseFormatError,
    build_constraint,
    constraint_spec_from_response_format,
)

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
    "additionalProperties": False,
}

JSON_SCHEMA_FORMAT = {"type": "json_schema", "schema": SCHEMA}


def _char_tokenizer(
    *extra_specials,
    markers_are_special = False,
    extra_chars = "",
):
    if not gc.LLGUIDANCE_AVAILABLE:
        pytest.skip("llguidance.mlx did not import")
    tk = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    chars = list('{}"[]:,0123456789abcdefghijklmnopqrstuvwxyz \n.-' + extra_chars)
    backend = tk.Tokenizer(tk.models.BPE(vocab = {c: i for i, c in enumerate(chars)}, merges = []))
    backend.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = False)
    backend.decoder = tk.decoders.ByteLevel()
    backend.add_special_tokens([tk.AddedToken("<eos>", special = True)])
    markers = [tk.AddedToken(t, special = markers_are_special) for t in ("</think>", *extra_specials)]
    if markers_are_special:
        backend.add_special_tokens(markers)
    else:
        backend.add_tokens(markers)
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object = backend, eos_token = "<eos>")
    tokenizer.chat_template = "{{ messages[0].content }}<think>"
    return tokenizer


@pytest.fixture(scope = "module")
def tiny_tokenizer():
    return _char_tokenizer()


def _metal_mx():
    mx = pytest.importorskip("mlx.core")
    if not mx.metal.is_available():
        pytest.skip("llguidance.mlx masks with a Metal kernel")
    return mx


def _allowed(constraint, width):
    mx = _metal_mx()
    masked = np.asarray(constraint.mask_logits(mx.zeros((width,), dtype = mx.float32)))
    return set(np.nonzero(~np.isneginf(masked))[0].tolist())


def test_every_spelling_of_a_supplied_schema_agrees_and_a_non_schema_is_refused(tiny_tokenizer):
    wrapped = constraint_spec_from_response_format(
        {"type": "json_schema", "json_schema": {"name": "city", "schema": SCHEMA}}
    ).grammar
    for spelling in (
        JSON_SCHEMA_FORMAT,
        {"type": "json_schema", "schema": dict(reversed(list(SCHEMA.items())))},
        {"type": "json_object", "schema": SCHEMA},
    ):
        assert constraint_spec_from_response_format(spelling).grammar == wrapped
    for supplied in (False, True, "nonsense", [1, 2]):
        with pytest.raises(ResponseFormatError):
            constraint_spec_from_response_format({"type": "json_object", "schema": supplied})
    plain = constraint_spec_from_response_format({"type": "json_object"}).grammar
    for spelling in (
        {"type": "json_object", "schema": None},
        {"type": "json_object", "schema": {}},
        {"type": "json_schema", "schema": {"type": "object"}},
    ):
        assert constraint_spec_from_response_format(spelling).grammar == plain
    digit = tiny_tokenizer.encode("4", add_special_tokens = False)[0]
    width = len(tiny_tokenizer)
    any_value = build_constraint({"type": "json_schema", "schema": {}}, tiny_tokenizer, "p")
    assert digit in _allowed(any_value, width)
    assert digit not in _allowed(
        build_constraint({"type": "json_object"}, tiny_tokenizer, "p"), width
    )


def test_the_schema_itself_constrains_not_merely_json_syntax(tiny_tokenizer):
    constraint = build_constraint(JSON_SCHEMA_FORMAT, tiny_tokenizer, "prompt")
    _allowed(constraint, len(tiny_tokenizer))
    for token_id in tiny_tokenizer.encode('{"', add_special_tokens = False):
        constraint.advance(int(token_id))
    assert _allowed(constraint, len(tiny_tokenizer)) == set(
        tiny_tokenizer.encode("a", add_special_tokens = False)
    )
    for token_id in tiny_tokenizer.encode('a":', add_special_tokens = False):
        constraint.advance(int(token_id))
    quote = tiny_tokenizer.encode('"', add_special_tokens = False)[0]
    assert quote not in _allowed(constraint, len(tiny_tokenizer))


def test_advance_raises_on_a_token_the_grammar_rejects(tiny_tokenizer):
    constraint = build_constraint(JSON_SCHEMA_FORMAT, tiny_tokenizer, "prompt")
    _allowed(constraint, len(tiny_tokenizer))
    illegal = tiny_tokenizer.encode("]", add_special_tokens = False)[0]
    with pytest.raises(gc.GrammarDesyncError, match = "desynced on token") as excinfo:
        constraint.advance(int(illegal))
    assert not isinstance(excinfo.value, ResponseFormatError)


def test_binding_waits_for_the_logits_width_a_padded_lm_head_widens(tiny_tokenizer):
    gc._TOKENIZER_CACHE.clear()
    constraint = build_constraint(JSON_SCHEMA_FORMAT, tiny_tokenizer, "prompt")
    padded = len(tiny_tokenizer) + 16
    legal = _allowed(constraint, padded)
    assert legal == set(tiny_tokenizer.encode("{", add_special_tokens = False))
    assert max(legal) < len(tiny_tokenizer)


def test_prelude_is_bounded_so_an_unclosed_think_block_cannot_eat_the_budget(
    tiny_tokenizer, monkeypatch
):
    monkeypatch.setattr(gc, "_PRELUDE_MAX_TOKENS", 6)
    constraint = build_constraint(
        JSON_SCHEMA_FORMAT, tiny_tokenizer, "chat\n<think>\n", reasoning_is_extracted = True
    )
    _allowed(constraint, len(tiny_tokenizer))
    for token_id in tiny_tokenizer.encode("abcdef", add_special_tokens = False):
        constraint.advance(int(token_id))
    close_id = tiny_tokenizer.encode("</think>", add_special_tokens = False)[0]
    assert _allowed(constraint, len(tiny_tokenizer)) == {close_id}
    constraint.advance(int(close_id))
    assert _allowed(constraint, len(tiny_tokenizer)) == set(
        tiny_tokenizer.encode("{", add_special_tokens = False)
    )


def test_reasoning_cannot_spell_the_closer_as_text_and_forge_the_document_boundary():
    tokenizer = _char_tokenizer(extra_chars = "</>")

    def _fresh():
        constraint = build_constraint(
            JSON_SCHEMA_FORMAT, tokenizer, "chat\n<think>", reasoning_is_extracted = True
        )
        constraint._bind(len(tokenizer))
        return constraint

    def _step(constraint, token_id):
        gc._llg_mlx.fill_next_token_bitmask(constraint._matcher, constraint._bitmask)
        constraint.advance(int(token_id))

    def _chars(text):
        return [tokenizer.convert_tokens_to_ids(c) for c in text]

    constraint = _fresh()
    for token_id in _chars("a<b</think"):
        _step(constraint, token_id)
    with pytest.raises(gc.GrammarDesyncError):
        _step(constraint, tokenizer.convert_tokens_to_ids(">"))

    close_id = tokenizer.encode("</think>", add_special_tokens = False)
    assert len(close_id) == 1
    constraint = _fresh()
    for token_id in (
        _chars("a<b</th") + close_id + tokenizer.encode('{"a":1}', add_special_tokens = False)
    ):
        _step(constraint, token_id)


@pytest.mark.parametrize("reply_keeps_special_tokens", [False, True])
def test_a_close_marker_is_refused_only_where_the_reply_would_lose_it(reply_keeps_special_tokens):
    tokenizer = _char_tokenizer(markers_are_special = True)
    ids = tokenizer.encode("</think>", add_special_tokens = False)
    assert (
        tokenizer.decode(ids, skip_special_tokens = True) != "</think>"
    ), "fixture no longer models a marker a skip_special_tokens decode strips"

    def _build():
        return build_constraint(
            JSON_SCHEMA_FORMAT,
            tokenizer,
            "chat\n<think>",
            reasoning_is_extracted = True,
            reply_keeps_special_tokens = reply_keeps_special_tokens,
        )

    if not reply_keeps_special_tokens:
        with pytest.raises(ResponseFormatError):
            _build()
        return
    constraint = _build()
    assert constraint.allows_reasoning
    for token_id in tokenizer.encode('x</think>{"a":1}', add_special_tokens = False):
        assert int(token_id) in _allowed(constraint, len(tokenizer))
        constraint.advance(int(token_id))


class _Recorder:
    def __init__(self):
        self.advanced = []

    def mask_logits(self, logits):
        return logits

    def advance(self, token_id):
        self.advanced.append(int(token_id))


def test_the_grammar_mask_runs_first_and_its_inf_survives_every_knob():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from core.inference.mlx_inference import _mlx_sampling_processors

    class _MaskOdd(_Recorder):
        def mask_logits(self, logits):
            keep = mx.array([[1.0, 0.0, 1.0, 0.0]])
            return mx.where(keep > 0, logits, mx.array(float("-inf")))

    grammar = _MaskOdd()
    processors = _mlx_sampling_processors(
        repetition_penalty = 1.3,
        presence_penalty = 0.5,
        frequency_penalty = 0.5,
        logit_bias = {1: 100.0},
        grammar_constraint = grammar,
    )
    zeros = mx.zeros((1, 4), dtype = mx.float32)
    tokens = mx.array([0, 1])
    leading = np.asarray(processors[0](tokens, zeros))[0]
    assert np.isneginf(leading[1]) and np.isneginf(leading[3])

    logits = zeros
    for processor in processors:
        logits = processor(tokens, logits)
    out = np.asarray(logits)[0]
    assert np.isneginf(out[1]) and np.isneginf(out[3])
    assert np.isfinite(out[0]) and np.isfinite(out[2])

    assert grammar.advanced == []
    processors[0](mx.array([0, 1, 2, 3]), zeros)
    processors[0](mx.array([0, 1, 2, 3, 4]), zeros)
    assert grammar.advanced == [2, 3, 4]


def _two_stop_tokenizer():
    tokenizer = _char_tokenizer("<end_of_text>", markers_are_special = True)
    tokenizer.chat_template = "{{ messages[0].content }}"
    runtime_stop = tokenizer.convert_tokens_to_ids("<end_of_text>")
    assert runtime_stop != tokenizer.eos_token_id, "fixture no longer models a disagreement"
    return tokenizer, runtime_stop


def _stop_offered_after_the_document(target, tokenizer):
    constraint = build_constraint(JSON_SCHEMA_FORMAT, target, "prompt")
    for token_id in tokenizer.encode('{"a":1}', add_special_tokens = False):
        _allowed(constraint, len(tokenizer))
        constraint.advance(int(token_id))
    assert constraint._matcher.is_stopped(), "the document did not complete"
    return _allowed(constraint, len(tokenizer))


def test_the_grammar_stops_on_the_token_the_decode_loop_stops_on():
    mlx_lm = pytest.importorskip("mlx_lm")
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    tokenizer, runtime_stop = _two_stop_tokenizer()
    wrapped = TokenizerWrapper(tokenizer, eos_token_ids = [runtime_stop])
    # mlx-lm 0.32 also keeps the tokenizer's own end token in the wrapper's stop set.
    assert runtime_stop in wrapped.eos_token_ids
    assert _stop_offered_after_the_document(wrapped, tokenizer) == set(wrapped.eos_token_ids)
    assert _stop_offered_after_the_document(tokenizer, tokenizer) == {tokenizer.eos_token_id}


def test_a_schema_forced_through_a_stop_token_is_refused_while_it_can_still_be_reported():
    pytest.importorskip("mlx_lm")
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    tokenizer, runtime_stop = _two_stop_tokenizer()
    wrapped = TokenizerWrapper(tokenizer, eos_token_ids = [runtime_stop])

    def replay(schema, text):
        constraint = build_constraint({"type": "json_schema", "schema": schema}, wrapped, "p")
        for token in tokenizer.encode(text, add_special_tokens = False):
            _allowed(constraint, len(tokenizer))
            constraint.advance(int(token))
        return _allowed(constraint, len(tokenizer))

    with pytest.raises(ResponseFormatError, match = "reserves as a control token"):
        replay({"const": tokenizer.convert_ids_to_tokens(runtime_stop)}, '"')
    assert runtime_stop in replay({"type": "integer"}, "42")
    assert runtime_stop in replay(SCHEMA, '{"a":1}')


def test_a_scalar_end_token_does_not_hide_the_runtime_stop_set():
    pytest.importorskip("mlx_vlm")
    from mlx_vlm.utils import StoppingCriteria

    tokenizer, runtime_stop = _two_stop_tokenizer()
    assert getattr(tokenizer, "eos_token_ids", None) == tokenizer.eos_token_id, "not a scalar"
    criteria = StoppingCriteria([runtime_stop], tokenizer = tokenizer)
    tokenizer.stopping_criteria = criteria
    for target in (tokenizer, SimpleNamespace(tokenizer = tokenizer)):
        assert _stop_offered_after_the_document(target, tokenizer) == set(criteria.eos_token_ids)


def test_advance_before_the_first_mask_raises_rather_than_dropping_the_token(tiny_tokenizer):
    constraint = build_constraint(JSON_SCHEMA_FORMAT, tiny_tokenizer, "prompt")
    with pytest.raises(gc.GrammarDesyncError, match = "before the grammar was bound"):
        constraint.advance(tiny_tokenizer.encode("{", add_special_tokens = False)[0])


def _uncopyable_naive_detokenizer(detokenizers):
    """The pre-0.6.0 naive detokenizer, whose copy raises; pinned by behaviour, not the installed wheel."""

    class _Uncopyable(detokenizers.NaiveStreamingDetokenizer):
        def __copy__(self):
            raise AttributeError(
                "property 'text' of 'NaiveStreamingDetokenizer' object has no setter"
            )

    return _Uncopyable


class _SpmTurn:
    """SPM releases text only at space-marked pieces, so a control can land in a later segment."""

    def __init__(
        self,
        pieces,
        specials = (),
        eos = None,
        chat_template = None,
        ends = "exhausted",
        reports_finish_reason = True,
        block = False,
        shares_detokenizer = False,
        detokenizer_class = "spm",
    ):
        detokenizers = pytest.importorskip("mlx_vlm.tokenizer_utils")
        self.vocab = {}
        self.ids = [self.vocab.setdefault(piece, len(self.vocab)) for piece in pieces]
        self._by_id = {token_id: piece for piece, token_id in self.vocab.items()}
        self.all_special_ids = tuple(self.vocab[piece] for piece in specials)
        self.all_special_tokens = tuple(specials)
        self.chat_template = chat_template
        self._ends = ends
        self._reports_finish_reason = reports_finish_reason
        self._block = block
        self._shares_detokenizer = shares_detokenizer
        if eos is not None:
            eos_ids = [self.vocab[piece] for piece in ((eos,) if isinstance(eos, str) else eos)]
            self.eos_token_ids = tuple(eos_ids)
            self.eos_token_id = eos_ids[-1]
        self.detokenizer = (
            _uncopyable_naive_detokenizer(detokenizers)(self)
            if detokenizer_class == "naive"
            else detokenizers.SPMStreamingDetokenizer(self, trim_space = False)
        )

    def convert_ids_to_tokens(self, token_id):
        return self._by_id[token_id]

    def decode(
        self,
        token_ids,
        skip_special_tokens = False,
        **_kwargs,
    ):
        return "".join(
            "" if (skip_special_tokens and i in self.all_special_ids) else self._by_id.get(i, "")
            for i in token_ids
        ).replace("▁", " ")

    def stream(self, *_a, **_k):
        """ends picks the last yield's shape; the mlx-vlm floor has no reports_finish_reason field."""
        if self._block:
            yield self._yielded(self.decode(self.ids), self.ids[-1], len(self.ids), "stop")
            return
        # mlx-vlm's floor streams through the processor's own detokenizer, not a copy, so
        # anything else driving that instance corrupts both.
        detokenizer = self.detokenizer if self._shares_detokenizer else copy.copy(self.detokenizer)
        detokenizer.reset()
        streamed = self.ids if self._ends == "exhausted" else self.ids[:-1]
        for position, token_id in enumerate(streamed, start = 1):
            detokenizer.add_token(token_id, skip_special_token_ids = [])
            yield self._yielded(detokenizer.last_segment, token_id, position, None)
        if self._ends == "length":
            detokenizer.add_token(self.ids[-1], skip_special_token_ids = [])
        detokenizer.finalize()
        yield self._yielded(
            detokenizer.last_segment,
            self.ids[-1],
            len(self.ids),
            "stop" if self._ends == "stop" else "length",
        )

    def _yielded(self, text, token_id, position, finish_reason):
        fields = dict(text = text, token = token_id, prompt_tokens = 3, generation_tokens = position)
        if self._reports_finish_reason:
            fields["finish_reason"] = finish_reason
        return SimpleNamespace(**fields)


def _run_spm_vlm_turn(
    monkeypatch,
    turn,
    tool_name = "get_weather",
    stop = None,
):
    from core.inference import mlx_inference

    @contextmanager
    def _adapter_state(_model, _state):
        yield

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _adapter_state)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.detect_think_prefill", lambda *_a, **_k: ""
    )
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.prompt_utils = SimpleNamespace(
        MODEL_CONFIG = {"deepseek_vl_v2": object()},
        apply_chat_template = lambda *_a, **_k: "<image> model-aware",
    )
    mlx_vlm.stream_generate = turn.stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _t, _m, **_k: "<image> model-aware",
    )

    backend = mlx_inference.MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "deepseek_vl_v2"})
    backend._processor = SimpleNamespace(
        tokenizer = turn, chat_template = turn.chat_template, detokenizer = turn.detokenizer
    )
    backend._tokenizer = turn
    backend._reads_vision = True
    return list(
        backend._generate_vlm(
            [{"role": "user", "content": [{"type": "image"}]}],
            [object()],
            0,
            1,
            0,
            0,
            32,
            1,
            None,
            _adapter_state = False,
            tools = [{"type": "function", "function": {"name": tool_name}}],
            stop = stop,
        )
    )


def test_mlx_vlm_keeps_a_native_tool_wrapper_exactly_once(monkeypatch):
    """Strict parsing needs the wrapper to tell a native call from markerless prose, and a second
    copy is no better: the parser reads the pair as a malformed envelope."""
    turn = _SpmTurn(
        (
            "<|tool_call>",
            "call",
            ":",
            "terminal",
            "{",
            "command",
            ":",
            '"id"',
            "}",
            "<tool_call|>",
        ),
        specials = ("<|tool_call>", "<tool_call|>"),
    )
    snapshots = _run_spm_vlm_turn(monkeypatch, turn, tool_name = "terminal")
    assert snapshots[-1] == '<|tool_call>call:terminal{command:"id"}<tool_call|>'


def test_mlx_vlm_keeps_prose_that_spells_a_suppressed_control(monkeypatch):
    """Suppression is about where a control came from, not what the text spells."""
    turn = _SpmTurn(
        ("Write", "▁`", "<", "pad", ">", "`", "▁literally.", "<pad>"),
        specials = ("<pad>",),
    )
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == "Write `<pad>` literally."


def test_mlx_vlm_does_not_leak_a_preserved_stop_token(monkeypatch):
    """This path appends each settled piece straight into the snapshot, so an allowlisted control
    used as the runtime EOS would trail every ordinary answer."""
    turn = _SpmTurn(
        ("Hi", "▁there", "<|end_message|>"),
        specials = ("<|end_message|>",),
        eos = "<|end_message|>",
        ends = "stop",
    )
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == "Hi there"


def test_mlx_vlm_settles_a_turn_a_runtime_ends_without_saying_so(monkeypatch):
    """mlx-vlm gained a finish reason well after the floor Studio supports."""
    turn = _SpmTurn(
        ("Hi", "▁there", "<|end_message|>"),
        specials = ("<|end_message|>",),
        eos = "<|end_message|>",
        ends = "stop",
        reports_finish_reason = False,
    )
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == "Hi there"


def test_mlx_vlm_keeps_a_stop_token_that_closes_a_tool_envelope(monkeypatch):
    """TML Inkling uses ``<|end_message|>`` both as that closer and as a possible EOS, and the
    runtime stops before its detokenizer sees that id."""
    turn = _SpmTurn(
        (
            "<|content_invoke_tool_json|>",
            '{"name":',
            '▁"get_weather",',
            '▁"args":',
            "▁{}}",
            "<|end_message|>",
        ),
        specials = ("<|content_invoke_tool_json|>", "<|end_message|>"),
        eos = "<|end_message|>",
        ends = "stop",
    )
    envelope = '<|content_invoke_tool_json|>{"name": "get_weather", "args": {}}'
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == envelope + "<|end_message|>"


def test_mlx_vlm_keeps_a_stop_id_the_runtime_generated_straight_through(monkeypatch):
    """``_mlx_stop_token_ids`` reads the config before the tokenizer attribute and the two disagree
    on some repos, so a held id can be one the runtime generated and carried on from."""
    turn = _SpmTurn(
        (
            "<|content_invoke_tool_json|>",
            '{"name":',
            '▁"get_weather",',
            '▁"args":',
            "▁{}}",
            "<|end_message|>",
            "<eos>",
            "▁Anything",
            "▁else?",
        ),
        specials = ("<|content_invoke_tool_json|>", "<|end_message|>", "<eos>"),
        eos = ("<|end_message|>", "<eos>"),
    )
    envelope = '<|content_invoke_tool_json|>{"name": "get_weather", "args": {}}'
    snapshot = _run_spm_vlm_turn(monkeypatch, turn)[-1]
    assert snapshot == envelope + "<|end_message|> Anything else?"


def test_mlx_vlm_drops_a_generated_stop_id_outside_the_allowlist(monkeypatch):
    """Carrying on past a stop id says it ended nothing, not that the reply may show it."""
    turn = _SpmTurn(
        ("Nearly", "▁done", "<eos>", "▁and", "▁back"),
        specials = ("<eos>",),
        eos = "<eos>",
    )
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == "Nearly done and back"


def test_mlx_vlm_keeps_a_closer_a_later_stop_id_followed(monkeypatch):
    """The config lists more ids than the runtime stopped on, so a closer can be followed by
    another stop id."""
    turn = _SpmTurn(
        (
            "<|content_invoke_tool_json|>",
            '{"name":"get_weather","args":{}}',
            "<|end_message|>",
            "<eos>",
        ),
        specials = ("<|content_invoke_tool_json|>", "<|end_message|>", "<eos>"),
        eos = ("<|end_message|>", "<eos>"),
        ends = "stop",
    )
    envelope = '<|content_invoke_tool_json|>{"name":"get_weather","args":{}}'
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == envelope + "<|end_message|>"


def test_mlx_vlm_keeps_every_closer_a_nested_call_needs(monkeypatch):
    """Half-closed is as rejectable as unclosed."""
    turn = _SpmTurn(
        (
            "<tool_call>",
            "<function=get_weather>",
            "<parameter=city>",
            "Paris",
            "</parameter>",
            "</function>",
            "</tool_call>",
        ),
        specials = ("<tool_call>", "</parameter>", "</function>", "</tool_call>"),
        eos = ("</parameter>", "</function>", "</tool_call>"),
        ends = "stop",
    )
    call = "<tool_call><function=get_weather><parameter=city>Paris"
    snapshot = _run_spm_vlm_turn(monkeypatch, turn)[-1]
    assert snapshot == call + "</parameter></function></tool_call>"


def test_mlx_stream_detokenizer_handles_one_that_cannot_be_copied():
    """Its ``text`` is a property over an inherited slot, so ``copy.copy`` raises. It is also the
    one mlx-lm builds per read, which is what makes a second read independent."""
    detokenizers = pytest.importorskip("mlx_lm.tokenizer_utils")
    from core.inference.mlx_inference import _mlx_stream_detokenizer

    class _Tok:
        def decode(self, token_ids):
            return "".join(f"<{token_id}>" for token_id in token_ids)

        def encode(
            self,
            text,
            add_special_tokens = True,
        ):
            return [ord(c) for c in text]

    class _PerRead:
        """Stands in for ``TokenizerWrapper``, whose ``detokenizer`` is a property."""

        @property
        def detokenizer(self):
            return _uncopyable_naive_detokenizer(detokenizers)(_Tok())

    source = _PerRead()
    with pytest.raises(AttributeError):
        copy.copy(source.detokenizer)
    own = _mlx_stream_detokenizer(source)
    assert own is not None
    own.add_token(7)
    own.finalize()
    assert own.text == "<7>"


def test_mlx_stream_detokenizer_rebuilds_the_one_a_retained_source_cannot_copy():
    """Below mlx-vlm 0.6.0 copying the naive detokenizer raises, so rebuild it, never drop the filter."""
    detokenizers = pytest.importorskip("mlx_vlm.tokenizer_utils")
    from core.inference.mlx_inference import _mlx_stream_detokenizer

    class _Tok:
        def decode(self, token_ids, **_kwargs):
            return "".join(f"<{token_id}>" for token_id in token_ids)

    class _Retained:
        def __init__(self):
            self.detokenizer = _uncopyable_naive_detokenizer(detokenizers)(_Tok())

    source = _Retained()
    with pytest.raises(AttributeError):
        copy.copy(source.detokenizer)

    own = _mlx_stream_detokenizer(source)
    assert own is not None, "a detokenizer that cannot be copied can still be rebuilt"
    assert own is not source.detokenizer

    own.add_token(7)
    own.finalize()
    assert own.text == "<7>"
    source.detokenizer.add_token(9)
    source.detokenizer.finalize()
    assert source.detokenizer.text == "<9>"


def test_mlx_vlm_suppresses_a_control_on_a_runtime_whose_detokenizer_cannot_be_copied(monkeypatch):
    """The end of that: on the supported mlx-vlm floor the reply still owes only what the
    allowlist keeps, rather than every control the runtime rendered."""
    turn = _SpmTurn(
        ("▁Hello", "<pad>", "▁world"),
        specials = ("<pad>",),
        ends = "exhausted",
        shares_detokenizer = True,
        detokenizer_class = "naive",
    )
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == " Hello world"


@pytest.mark.parametrize("ends", ("stop", "exhausted"))
def test_mlx_vlm_owes_a_closer_exactly_once_without_a_detokenizer_of_its_own(monkeypatch, ends):
    """A runtime that stopped before rendering the stop id never wrote it; one that did has already
    said it."""
    turn = _SpmTurn(
        (
            "<|content_invoke_tool_json|>",
            '{"name":"get_weather","args":{}}',
            "<|end_message|>",
        ),
        specials = ("<|content_invoke_tool_json|>", "<|end_message|>"),
        eos = "<|end_message|>",
        ends = ends,
    )
    monkeypatch.setattr(
        "core.inference.mlx_inference._mlx_stream_detokenizer", lambda _source: None
    )
    envelope = '<|content_invoke_tool_json|>{"name":"get_weather","args":{}}'
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == envelope + "<|end_message|>"


@pytest.mark.parametrize(
    "trailing, expected_tail", ((("▁Done.",), " Done."), (("\n", "Done."), "\nDone."))
)
def test_mlx_vlm_does_not_repeat_a_closer_the_runtime_already_wrote(
    monkeypatch, trailing, expected_tail
):
    """The runtime's text already carries every control but the one it stopped before -- whether an
    SPM space marker released it mid-turn or the whole reply arrived in the final flush."""
    turn = _SpmTurn(
        ("<tool_call>", '{"name":"get_weather","arguments":{}}', "</tool_call>")
        + trailing
        + ("<eos>",),
        specials = ("<tool_call>", "</tool_call>", "<eos>"),
        eos = ("</tool_call>", "<eos>"),
        ends = "stop",
    )
    monkeypatch.setattr(
        "core.inference.mlx_inference._mlx_stream_detokenizer", lambda _source: None
    )
    call = '<tool_call>{"name":"get_weather","arguments":{}}</tool_call>'
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == call + expected_tail


def test_mlx_vlm_does_not_drive_the_detokenizer_the_runtime_streams_through(monkeypatch):
    """The supported mlx-vlm floor streams through the processor's own instance."""
    turn = _SpmTurn(
        ("<|channel>", "thought", "\n", "weighing", "▁it", "<channel|>", "▁the", "▁answer"),
        specials = ("<|channel>", "<channel|>"),
        chat_template = "...<|channel>thought\n...<channel|>",
        shares_detokenizer = True,
    )
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == "<think>weighing it</think> the answer"


def test_mlx_vlm_drops_an_orphan_closer_that_opened_nothing(monkeypatch):
    """Mentioning a marker is not opening an envelope."""
    turn = _SpmTurn(
        ("The", "▁[ARGS]", "▁marker", "▁is", "▁neat", "<|end_message|>"),
        specials = ("<|end_message|>",),
        eos = "<|end_message|>",
        ends = "stop",
    )
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == "The [ARGS] marker is neat"


def test_mlx_vlm_keeps_a_block_the_runtime_settled_in_one_step(monkeypatch):
    """A diffusion generator denoises a whole block and names only its last id."""
    turn = _SpmTurn(("Hello", "\u2581world"), block = True)
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == "Hello world"


def test_mlx_vlm_matches_a_stop_sequence_the_detokenizer_held_to_the_end(monkeypatch):
    """Text the cut held back is unmatched, not cleared."""
    turn = _SpmTurn(("Hi", "\u2581STOP"))
    assert _run_spm_vlm_turn(monkeypatch, turn, stop = "STOP")[-1] == "Hi "


def test_mlx_vlm_writes_a_buffered_reasoning_delimiter_once(monkeypatch):
    """The opener is buffered with the pieces after it, so a second copy written on its own step
    lands ahead of them: the normalizer then matches the later copy and emits the earlier one as
    answer text, which is how a gemma-4 thought leaked ``<|channel>`` into the reply."""
    turn = _SpmTurn(
        (
            "<|channel>",
            "thought",
            "\n",
            "weighing",
            "▁it",
            "<channel|>",
            "▁the",
            "▁answer",
        ),
        specials = ("<|channel>", "<channel|>"),
        chat_template = "...<|channel>thought\n...<channel|>",
    )
    assert _run_spm_vlm_turn(monkeypatch, turn)[-1] == "<think>weighing it</think> the answer"


def _fake_image(
    pixels,
    raw = None,
    orientation = None,
):
    """A PIL-like image whose RGB conversion holds ``pixels``; a palette image's own
    bytes are indices (``raw``) saying nothing about its colors."""
    rgb = SimpleNamespace(mode = "RGB", size = (2, 2), tobytes = lambda: pixels)
    return SimpleNamespace(
        mode = "P" if raw else "RGB",
        size = (2, 2),
        tobytes = lambda: raw or pixels,
        convert = lambda _mode: rgb,
        getexif = lambda: {0x0112: orientation} if orientation else {},
    )


def test_mlx_vlm_snapshots_are_keyed_by_what_the_vision_tower_sees():
    from core.inference.mlx_inference import MLXInferenceBackend as backend

    digest = backend._vlm_image_digest
    assert digest([_fake_image(b"aaaa")]) == digest([_fake_image(b"aaaa")])
    assert digest([_fake_image(b"aaaa")]) != digest([_fake_image(b"bbbb")])
    assert digest([_fake_image(b"red!", raw = b"idx")]) != digest([_fake_image(b"blue", raw = b"idx")])
    assert digest([_fake_image(b"aaaa")]) != digest([_fake_image(b"aaaa", orientation = 6)])
    assert digest([_fake_image(b"aaaa")]) != digest([_fake_image(b"aaaa"), _fake_image(b"bbbb")])


def test_vlm_add_special_tokens_falls_back_to_the_inline_rule(monkeypatch):
    from core.inference.mlx_inference import _vlm_add_special_tokens as rule

    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", types.ModuleType("mlx_vlm.utils"))
    template = SimpleNamespace(chat_template = "t")
    assert rule("gemma4", template) is False and rule("gemma4", SimpleNamespace()) is True
    assert rule("qwen2_vl", template) is True
    sys.modules["mlx_vlm.utils"].should_add_special_tokens = lambda *_: "mlx-vlm's answer"
    assert rule("gemma4", template) == "mlx-vlm's answer"


def test_mlx_vlm_pixels_follow_the_order_the_markers_appear_in():
    """MLX binds pixels to markers positionally. The replayed MCP pictures carry
    their markers on earlier turns, and the attachment's marker goes on the last
    user turn, so the attachment's pixels have to come last."""
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._is_vlm = True
    backend._multi_image_marker = "<image>"
    captured = []
    backend._generate_vlm = lambda messages, attached, *_args, **_kwargs: (
        captured.append((messages, attached)) or iter(())
    )
    replayed_first, replayed_second, attachment = object(), object(), object()
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "a"}]},
        {"role": "assistant", "content": "seen"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": "seen"},
        {"role": "user", "content": "and this one?"},
    ]

    list(
        backend.generate_chat_response(
            messages,
            image = attachment,
            images = [replayed_first, replayed_second],
        )
    )

    sent_messages, attached = captured[0]
    assert attached == [replayed_first, replayed_second, attachment]
    marker_turns = [
        index
        for index, message in enumerate(sent_messages)
        if isinstance(message.get("content"), list)
        and any(part.get("type") == "image" for part in message["content"])
    ]
    assert marker_turns[-1] == len(sent_messages) - 1


def test_mlx_attachment_displaces_a_replayed_image_on_the_same_turn():
    """A local turn takes one picture, so the attachment replaces the replay
    marker and its pixels together."""
    from core.inference.mlx_inference import MLXInferenceBackend, _count_vlm_images

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._is_vlm = True
    backend._multi_image_marker = "<image>"
    captured = []
    backend._generate_vlm = lambda messages, attached, *_a, **_k: (
        captured.append((messages, attached)) or iter(())
    )
    replayed, attachment = object(), object()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "and this one?"}],
        }
    ]

    list(backend.generate_chat_response(messages, image = attachment, images = [replayed]))

    sent, attached = captured[0]
    markers = sum(_count_vlm_images(m.get("content")) for m in sent)
    assert attached == [attachment]
    assert markers == len(attached), f"{markers} marker(s) for {len(attached)} images"
    assert sent[-1]["content"][-1] == {"type": "image"}


def test_mlx_does_not_double_mark_an_attachment_the_client_already_marked():
    from core.inference.mlx_inference import MLXInferenceBackend, _count_vlm_images

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._is_vlm = True
    backend._multi_image_marker = "<image>"
    captured = []
    backend._generate_vlm = lambda messages, attached, *_a, **_k: (
        captured.append((messages, attached)) or iter(())
    )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "x"}]}]

    list(backend.generate_chat_response(messages, image = object()))

    sent, attached = captured[0]
    assert sum(_count_vlm_images(m.get("content")) for m in sent) == len(attached) == 1


def test_mlx_normalizes_a_replay_only_conversation_for_a_processor_template():
    """A processor template needs every turn in part shape. A replay-only request
    (images without an attachment) is a mix of strings and marker lists, and gating
    the normalization on the singular image left it failing at render."""
    from types import SimpleNamespace

    from core.inference.mlx_inference import MLXInferenceBackend, _count_vlm_images

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._is_vlm = True
    backend._multi_image_marker = "<image>"
    backend._processor = SimpleNamespace(
        apply_chat_template = lambda *a, **k: "prompt", chat_template = "t"
    )
    captured = []
    backend._generate_vlm = lambda messages, attached, *a, **k: (
        captured.append((messages, attached)) or iter(())
    )
    messages = [
        {"role": "tool", "content": "[1 image returned]"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what was it"}]},
    ]

    list(backend.generate_chat_response(messages, images = [object()]))

    sent, attached = captured[0]
    assert all(isinstance(message.get("content"), list) for message in sent), sent
    markers = sum(_count_vlm_images(message.get("content")) for message in sent)
    assert markers == len(attached) == 1


def test_mlx_binds_an_earlier_attachment_to_its_own_turn_not_a_newer_replay():
    """Pre-added markers make the top-up a no-op, so an attachment and a tool picture swap turns."""
    from core.inference.mcp_images import placeholder_turn
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    backend._model = object()
    backend._is_vlm = True
    backend._multi_image_marker = "<image>"
    captured = []
    backend._generate_vlm = lambda messages, attached, *_args, **_kwargs: (
        captured.append((messages, attached)) or iter(())
    )
    attachment, replayed = object(), object()

    messages = [
        {"role": "user", "content": "here is my diagram"},
        {"role": "assistant", "content": "noted"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "mcp__s__shot", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "[1 image returned]"},
        placeholder_turn(1, 1),
        {"role": "user", "content": "which one is bluer?"},
    ]

    list(
        backend.generate_chat_response(
            messages,
            image = attachment,
            images = [replayed],
            image_ordinal = 0,
        )
    )

    sent_messages, attached = captured[0]
    marker_turns = [
        index
        for index, message in enumerate(sent_messages)
        if isinstance(message.get("content"), list)
        and any(part.get("type") == "image" for part in message["content"])
    ]
    assert len(marker_turns) == 2, sent_messages
    assert marker_turns[0] < marker_turns[1]
    assert attached == [attachment, replayed], "the pixels bound to each other's markers"


def _run_mlx_reasoning_stream(
    monkeypatch,
    turn,
    stop = None,
):
    _install_fake_mlx(monkeypatch)
    from core.inference import mlx_inference

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_args, **_kwargs: "prompt",
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
    mlx_lm_sample.make_sampler = lambda **_kwargs: object()
    mlx_lm_sample.make_logits_processors = lambda **_kwargs: None

    mlx_lm_pkg.stream_generate = lambda _model, _tokenizer, **_kwargs: turn.stream()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    backend = mlx_inference.MLXInferenceBackend()
    backend._model = object()
    backend._tokenizer = turn
    backend._is_vlm = False
    return list(
        backend.generate_chat_response(
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"type": "function", "function": {"name": "terminal"}}],
            max_new_tokens = len(turn.ids),
            stop = stop,
        )
    )


def test_mlx_reasoning_reply_does_not_end_in_a_preserved_eos_control(monkeypatch):
    """The reasoning branch appended it, ending an ordinary reply in raw markup."""
    turn = _SpmTurn(
        ("<|channel>", "thought", "▁reasoned", "<channel|>", "▁Done.", "<|end_message|>"),
        specials = ("<|channel>", "<channel|>", "<|end_message|>"),
        eos = "<|end_message|>",
        chat_template = "x",
        ends = "stop",
    )
    snapshots = _run_mlx_reasoning_stream(monkeypatch, turn)
    assert "<|end_message|>" not in snapshots[-1]
    assert snapshots[-1].endswith("Done.")


def test_mlx_reasoning_keeps_the_token_the_limit_cut_the_turn_after(monkeypatch):
    """mlx-lm reports that token without ever yielding it on its own step."""
    turn = _SpmTurn(
        ("<|channel>", "thought", "▁reasoned", "<channel|>", "▁the", "▁answer"),
        specials = ("<|channel>", "<channel|>"),
        chat_template = "x",
        ends = "length",
    )
    assert _run_mlx_reasoning_stream(monkeypatch, turn)[-1].endswith(" the answer")


def test_mlx_reasoning_matches_a_stop_sequence_the_detokenizer_held_to_the_end(monkeypatch):
    """As on the VLM path: the flush ending the turn is the first the sequence is visible in."""
    turn = _SpmTurn(
        ("<|channel>", "thought", "\u2581reasoned", "<channel|>", "\u2581Answer", "\u2581STOP"),
        specials = ("<|channel>", "<channel|>"),
        chat_template = "x",
    )
    snapshots = _run_mlx_reasoning_stream(monkeypatch, turn, stop = "STOP")
    assert snapshots[-1] == "<think> reasoned</think> Answer "


def test_mlx_reasoning_keeps_an_eos_control_that_closes_a_tool_envelope(monkeypatch):
    """The trim is envelope-aware: the same marker terminates a real Inkling call."""
    turn = _SpmTurn(
        (
            "<|channel>",
            "thought",
            "▁reasoned",
            "<channel|>",
            "<|content_invoke_tool_json|>",
            '{"name":"terminal","args":{}}',
            "<|end_message|>",
        ),
        specials = (
            "<|channel>",
            "<channel|>",
            "<|content_invoke_tool_json|>",
            "<|end_message|>",
        ),
        eos = "<|end_message|>",
        chat_template = "x",
        ends = "stop",
    )
    snapshots = _run_mlx_reasoning_stream(monkeypatch, turn)
    assert snapshots[-1].endswith("<|end_message|>")


def test_mlx_keeps_a_think_closer_whose_opener_came_from_the_prefill(monkeypatch):
    """The restored ``<think>`` lives in the PROMPT, not in the generated ids. Judging the
    stop token on generated text alone found no opener, dropped the ``</think>``, and left
    the reasoning block open so it swallowed the visible answer."""
    _install_fake_mlx(monkeypatch)
    from core.inference import mlx_inference

    @contextmanager
    def _adapter_state(_model, _state):
        yield

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _adapter_state)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.detect_think_prefill",
        lambda *_a, **_k: "<think>",
    )
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _t, _m, **_k: "prompt <think>",
    )

    THINK_CLOSE = 9

    class _Tok:
        chat_template = "x"
        all_special_ids = (THINK_CLOSE,)
        eos_token_id = THINK_CLOSE
        _IDS = {THINK_CLOSE: "</think>"}

        def convert_ids_to_tokens(self, token_id):
            return self._IDS[token_id]

        def decode(
            self,
            ids,
            skip_special_tokens = False,
            **_k,
        ):
            return "".join(
                self._IDS.get(i, "reasoning")
                for i in ids
                if not (skip_special_tokens and i in self.all_special_ids)
            )

    mlx_lm_pkg = types.ModuleType("mlx_lm")
    mlx_lm_sample = types.ModuleType("mlx_lm.sample_utils")
    mlx_lm_sample.make_sampler = lambda **_kw: object()
    mlx_lm_sample.make_logits_processors = lambda **_kw: None

    def _stream_generate(_model, _tokenizer, **_kw):
        for tok in (1, THINK_CLOSE):
            yield SimpleNamespace(token = tok, prompt_tokens = 2, generation_tokens = 1)

    mlx_lm_pkg.stream_generate = _stream_generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_pkg)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm_sample)

    backend = mlx_inference.MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "g"})
    backend._tokenizer = _Tok()
    backend._is_vlm = False

    snapshots = list(
        backend.generate_with_adapter_control(
            use_adapter = False,
            messages = [{"role": "user", "content": "ping"}],
            tools = [{"function": {"name": "web_search"}}],
            max_new_tokens = 2,
        )
    )
    final = snapshots[-1]
    assert final.startswith("<think>"), final
    assert "</think>" in final, f"the closer was trimmed, leaving the block open: {final!r}"


def test_mlx_vlm_prompt_cache_session_prefills_on_mlx_vlm_default_step(monkeypatch):
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend, VLMPromptSnapshotStore

    names = ("mlx_vlm", "mlx_vlm.generate", "mlx_vlm.generate.common", "mlx_vlm.models")
    modules = {name: types.ModuleType(name) for name in names + ("mlx_vlm.models.cache",)}
    modules["mlx_vlm.generate.common"].DEFAULT_PREFILL_STEP_SIZE = 1234
    modules["mlx_vlm.models.cache"].make_prompt_cache = lambda _model, max_kv_size = None: []
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    class LanguageModel:
        pass

    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    backend._vlm_snapshot_store = VLMPromptSnapshotStore(10**6)
    backend._vlm_snapshot_store_unavailable = False
    backend._vlm_is_diffusion_model = lambda _model: False
    backend._model = SimpleNamespace(config = SimpleNamespace(), language_model = LanguageModel())
    backend._kv_cache_window = None
    backend.active_model_name = "m"
    session = backend._vlm_prompt_cache_session("base")
    assert mlx_inference.vlm_prefill_step() == 1234 and session.step == 1234
    assert mlx_inference.shape_stable_prefix(1300, step = session.step) == 1234
    backend._prefill_step, backend._kv_runtime_quant_kwargs = 8192, lambda: {}
    grid = lambda: (lambda session: (session.step, session.window))(
        backend._vlm_prompt_cache_session("base")
    )
    assert grid() == (1234, 1234) and backend._vlm_batch_row_cache("base", ())._step == 1234
    zoo = types.SimpleNamespace(wide_prefill_step = lambda model: 1234 if model == "wide" else None)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.offload", zoo)
    probe = [
        mlx_inference._prefill_layer_step(*args)
        for args in (("wide", 4936), ("wide", 8192), ("whole", 4936))
    ]
    assert probe == [1234, None, None]
    backend._prefill_layer_step = 1234
    assert grid() == (1234, 8192) and backend._vlm_batch_row_cache("base", ())._step == 1234
    backend._kv_runtime_quant_kwargs = lambda: {"kv_bits": 4}
    assert grid() == (1234, 1234)
    backend._kv_runtime_quant_kwargs, backend._kv_cache_window = lambda: {}, 4096
    assert grid() == (1234, 1234)


_BUDGET_WINDOW = 2048
_BUDGET_PROMPT_N = 37
_BUDGET_BOS = "<s>"
_IMAGE_TURN = [{"role": "user", "content": [{"type": "image"}]}]
_TEXT_TURN = [{"role": "user", "content": "hi"}]


def _budget_backend(
    monkeypatch,
    served = _BUDGET_WINDOW,
    marker_tokens = 0,
):
    from core.inference import mlx_inference

    @contextmanager
    def _adapter_state(_model, _state):
        yield

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _adapter_state)
    backend = mlx_inference.MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "llama"})
    backend._tokenizer = SimpleNamespace(
        all_special_tokens = [],
        bos_token = _BUDGET_BOS if marker_tokens else None,
        encode = lambda _p, add_special_tokens = True: list(
            range(_BUDGET_PROMPT_N + (marker_tokens if add_special_tokens else 0))
        ),
        decode = lambda _ids, **_k: "ok",
    )
    backend._served_context = served
    return backend


def _run_text_budget(
    monkeypatch,
    max_new_tokens,
    served = _BUDGET_WINDOW,
    cached = None,
):
    from core.inference import mlx_inference

    seen = {}
    mlx_lm = types.ModuleType("mlx_lm")
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_k: object()
    mlx_lm.sample_utils = sample_utils

    def _stream(_model, _tokenizer, **kwargs):
        seen["max_tokens"] = kwargs.get("max_tokens")
        yield _Resp("ok", 1)

    mlx_lm.stream_generate = _stream
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(mlx_inference, "_mlx_sampling_processors", lambda **_k: None)

    backend = _budget_backend(monkeypatch, served = served)
    monkeypatch.setattr(
        backend,
        "_render_text_prompt",
        lambda *_a, **_k: SimpleNamespace(prompt = "P", reasoning_channel_markers = None),
        raising = False,
    )
    if cached is not None:
        monkeypatch.setattr(
            backend,
            "_prepare_prompt_cache",
            lambda _p, _a: ("TAIL", object(), "k", cached, len(cached) - 4),
        )
    list(backend._generate_text(_TEXT_TURN, 0.0, 1.0, 0, 0.0, max_new_tokens, 1.0, None))
    return seen["max_tokens"]


def _run_vlm_budget(
    monkeypatch,
    messages,
    image,
    max_new_tokens,
    served = _BUDGET_WINDOW,
    marker_tokens = 0,
):
    seen = {}
    mlx_vlm = types.ModuleType("mlx_vlm")

    def _stream(*_a, **kwargs):
        seen["max_tokens"] = kwargs.get("max_tokens")
        yield SimpleNamespace(text = "ok", prompt_tokens = 3, generation_tokens = 1)

    mlx_vlm.stream_generate = _stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _t, _m, **_k: f"{_BUDGET_BOS}<image> p",
    )

    backend = _budget_backend(monkeypatch, served = served, marker_tokens = marker_tokens)
    backend._is_vlm = True
    backend._reads_vision = True
    backend._model = SimpleNamespace()
    backend._processor = SimpleNamespace(tokenizer = backend._tokenizer)
    args = (messages, [image] if image is not None else None, 0, 1, 0, 0, max_new_tokens, 1, None)
    list(backend._generate_vlm(*args, _adapter_state = False))
    return seen["max_tokens"]


def test_mlx_text_resolves_an_unset_budget_against_the_served_window(monkeypatch):
    assert _run_text_budget(monkeypatch, None) == _BUDGET_WINDOW - _BUDGET_PROMPT_N


def test_mlx_text_budget_counts_the_cached_prefix_too(monkeypatch):
    # Sizing from the 4-token uncached tail would treat the cached prefix as free room.
    budget = _run_text_budget(monkeypatch, None, served = 4096, cached = list(range(300)))

    assert budget == 4096 - 300


def test_mlx_vlm_does_not_size_a_budget_from_a_prompt_carrying_an_image(monkeypatch):
    from core.inference.runtime_context import UNSET_GENERATION_BUDGET
    budget = _run_vlm_budget(monkeypatch, _IMAGE_TURN, object(), None, served = 32768)

    assert budget == UNSET_GENERATION_BUDGET


def test_mlx_vlm_image_budget_stays_inside_a_narrow_served_window(monkeypatch):
    # A pinned load rotates KV at the served window, so a flat 2048 would evict the image.
    budget = _run_vlm_budget(monkeypatch, _IMAGE_TURN, object(), None, served = 1024)

    assert budget == 1024 - _BUDGET_PROMPT_N


def test_mlx_vlm_counts_a_text_only_turn_by_the_vision_marker_rule(monkeypatch):
    budget = _run_vlm_budget(monkeypatch, _TEXT_TURN, None, None, served = 32768, marker_tokens = 1)

    assert budget == 32768 - _BUDGET_PROMPT_N - 1


def test_mlx_passes_an_explicit_budget_through_both_paths(monkeypatch):
    assert _run_text_budget(monkeypatch, 4096) == 4096
    assert _run_vlm_budget(monkeypatch, _IMAGE_TURN, object(), 4096) == 4096


def test_mlx_unset_budget_falls_back_when_the_prompt_cannot_be_counted(monkeypatch):
    from core.inference.runtime_context import UNSET_GENERATION_BUDGET

    backend = _budget_backend(monkeypatch)
    backend._tokenizer = None

    assert backend._unset_generation_budget("P") == UNSET_GENERATION_BUDGET


def test_mlx_count_strips_replayed_mcp_images_off_the_event_loop(monkeypatch):
    import base64
    import io
    import json
    import threading
    from PIL import Image
    from core.inference import mcp_images
    from routes import inference as route

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "blue").save(buffer, format = "PNG")
    envelope = (
        "screenshot result\n"
        + mcp_images.SENTINEL
        + json.dumps(
            [{"data": base64.b64encode(buffer.getvalue()).decode(), "mimeType": "image/png"}]
        )
    )
    history = [
        {"role": "user", "content": "Read the screenshot"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "mcp__screen__shot", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "mcp__screen__shot", "content": envelope},
        {"role": "user", "content": "Describe it"},
    ]
    loop_thread = threading.get_ident()
    split_threads = []
    original_split = mcp_images.split_images

    def track_split(content):
        if mcp_images.SENTINEL in content:
            split_threads.append(threading.get_ident())
        return original_split(content)

    monkeypatch.setattr(mcp_images, "split_images", track_split)
    monkeypatch.setattr(route, "split_mcp_images", track_split)
    backend = _RenderRecordingBackend()
    response = _count_route(monkeypatch, backend, messages = history, enable_tools = False)
    assert response.status_code == 200
    assert split_threads and all(thread != loop_thread for thread in split_threads)
    assert backend.messages[1]["tool_calls"][0]["function"]["name"] == "mcp__screen__shot"
    assert backend.messages[2]["content"] == "screenshot result"


_CLIP_B64 = "AAAAGGZ0eXBtcDQy"


def _require_video_stack_module():
    """`_require_video_stack`, returning the package so a test can restand its ``utils``."""
    _require_video_stack()
    import mlx_vlm

    return mlx_vlm


def _require_video_stack():
    """Both halves: guarding on one alone let a test needing both run on a runner with only the
    other, and the decoder lands in 0.5.0 while the install floor is 0.4.4."""
    utils = pytest.importorskip("mlx_vlm.utils")
    if not callable(getattr(utils, "load_video", None)):
        pytest.skip("the installed mlx-vlm predates the 0.5.0 clip decoder")
    pytest.importorskip("cv2")
    return utils


def _video_vlm_backend(monkeypatch, streams):
    from core.inference import mlx_inference

    @contextmanager
    def _adapter_state(_model, _state):
        yield

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _adapter_state)
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.prompt_utils = SimpleNamespace(MODEL_CONFIG = {})

    def _vlm_stream(*args, **kwargs):
        paths = kwargs.get("video")
        streams.append((args, kwargs, [Path(p).read_bytes() for p in paths] if paths else None))
        yield SimpleNamespace(text = "ok", prompt_tokens = 3, generation_tokens = 1)

    mlx_vlm.stream_generate = _vlm_stream
    real_vlm_utils = _require_video_stack()
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", real_vlm_utils)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _t, _m, **_k: "<video> marked",
    )
    backend = mlx_inference.MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "qwen3_5"})
    backend._processor = SimpleNamespace(tokenizer = SimpleNamespace())
    backend._is_vlm = True
    backend._reads_vision = True
    return backend


def test_mlx_vlm_a_video_turn_never_resumes_a_prompt_cache_snapshot(monkeypatch):
    """A snapshot spans the video rows but is keyed by token ids the clip does not vary, so a
    later clip would resume the earlier one's vision rows and its mRoPE grid."""
    streams = []
    backend = _video_vlm_backend(monkeypatch, streams)
    backend._vlm_is_diffusion_model = lambda _model: False
    monkeypatch.setattr(type(backend), "_vlm_prompt_cache_store", lambda self: object())
    monkeypatch.setattr(type(backend), "_vlm_media_token_ids", lambda self, _config: (7,))
    monkeypatch.setattr(type(backend), "_vlm_media_block", lambda self, *_a: None)
    cache_module = types.ModuleType("mlx_vlm.models.cache")
    cache_module.make_prompt_cache = lambda _model, max_kv_size = None: []
    monkeypatch.setitem(sys.modules, "mlx_vlm.models", types.ModuleType("mlx_vlm.models"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.cache", cache_module)

    assert backend._vlm_prompt_cache_session(False, None, "prompt") is not None
    assert backend._vlm_prompt_cache_session(False, None, "prompt", has_video = True) is None

    turn = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "what moves"}]}
    ]
    args = (turn, None, 0, 1, 0, 0, 1, 1, None)
    assert list(backend._generate_vlm(*args, _adapter_state = False, videos = [_CLIP_B64])) == ["ok"]
    ((_, stream_kwargs, _),) = streams
    assert "prompt_cache" not in stream_kwargs
    assert "prompt_cache_state" not in stream_kwargs
    backend._model.language_model = type("LanguageModel", (), {})()
    backend._prefill_step, backend._prefill_layer_step = 8192, 2048
    assert list(
        backend._generate_vlm(
            *(([{"role": "user", "content": "hi"}],) + args[1:]), _adapter_state = False
        )
    ) == ["ok"]
    assert (
        streams[-1][1]["prefill_step_size"] == 8192
        and streams[-1][1]["prompt_cache_state"].step == 2048
    )


def test_mlx_vlm_a_video_turn_releases_retained_snapshots(monkeypatch):
    """A video turn resumes nothing, so retained snapshots are pure occupancy against the
    frame budget it is about to allocate."""
    streams = []
    backend = _video_vlm_backend(monkeypatch, streams)
    released = []
    monkeypatch.setattr(type(backend), "_release_vlm_snapshots", lambda self: released.append(True))
    turn = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "what moves"}]}
    ]
    args = (turn, None, 0, 1, 0, 0, 1, 1, None)

    assert list(backend._generate_vlm(*args, _adapter_state = False, videos = [_CLIP_B64])) == ["ok"]
    assert released == [True]


def test_mlx_vlm_an_unset_budget_is_capped_for_a_video_turn(monkeypatch):
    """A clip expands furthest past its placeholder, so the counted prompt understates the real
    one by more than an image does."""
    from core.inference.runtime_context import UNSET_GENERATION_BUDGET

    streams = []
    backend = _video_vlm_backend(monkeypatch, streams)
    monkeypatch.setattr(
        type(backend),
        "_unset_generation_budget",
        lambda self, *_args, **_kwargs: UNSET_GENERATION_BUDGET * 8,
    )
    turn = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "what moves"}]}
    ]
    args = (turn, None, 0, 1, 0, 0, None, 1, None)

    assert list(backend._generate_vlm(*args, _adapter_state = False, videos = [_CLIP_B64])) == ["ok"]
    assert streams[-1][1]["max_tokens"] == UNSET_GENERATION_BUDGET


def test_mlx_vlm_video_clip_lives_on_disk_only_for_the_stream(monkeypatch):
    """The clip lives on disk only for the stream, including when the caller closes it early."""
    import base64

    streams = []
    backend = _video_vlm_backend(monkeypatch, streams)
    turn = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "what moves"}]}
    ]
    args = (turn, None, 0, 1, 0, 0, 1, 1, None)

    assert list(backend._generate_vlm(*args, _adapter_state = False, videos = [_CLIP_B64])) == ["ok"]
    ((stream_args, stream_kwargs, contents),) = streams
    assert stream_args[3] is None, "no image was attached"
    assert contents == [base64.b64decode(_CLIP_B64)]
    assert not any(Path(p).exists() for p in stream_kwargs["video"])

    gen = backend._generate_vlm(*args, _adapter_state = False, videos = [_CLIP_B64])
    assert next(gen) == "ok"
    (path,) = streams[-1][1]["video"]
    assert Path(path).exists()
    gen.close()
    assert not Path(path).exists()


def test_mlx_vlm_every_clip_reaches_mlx_vlm_in_order_at_one_rate(monkeypatch):
    """Releases before 0.7.3 apply one ``fps`` to every clip, so the slowest affordable rate
    is the one that keeps every clip inside its share of the decode budget."""
    import base64

    from core.inference import mlx_inference

    streams = []
    backend = _video_vlm_backend(monkeypatch, streams)
    # The slowest clip sits between faster ones, so neither the first nor the last rate passes.
    rates = {b"fast clip": 2.0, b"slow clip": 0.5, b"medium clip": 1.0}
    shares = []

    def _rate(
        path,
        _processor,
        clips = 1,
    ):
        shares.append(clips)
        return rates[Path(path).read_bytes()]

    monkeypatch.setattr(mlx_inference, "_video_frame_rate", _rate)
    clips = [base64.b64encode(content).decode() for content in rates]
    turn = [
        {
            "role": "user",
            "content": [{"type": "video"}] * 3 + [{"type": "text", "text": "which"}],
        }
    ]
    args = (turn, None, 0, 1, 0, 0, 1, 1, None)

    assert list(backend._generate_vlm(*args, _adapter_state = False, videos = clips)) == ["ok"]
    ((_, stream_kwargs, contents),) = streams
    assert contents == list(rates)
    assert stream_kwargs["fps"] == 0.5
    assert shares == [3, 3, 3]
    assert not any(Path(p).exists() for p in stream_kwargs["video"])


def _tiny_clip(
    path,
    width = 32,
    height = 24,
    frames = 12,
):
    # importorskip: two of the three callers reach here before any other guard.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (width, height))
    for index in range(frames):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, index : index + 4] = 255
        writer.write(frame)
    writer.release()
    return str(path)


def test_mlx_vlm_the_decoded_frame_stack_is_bounded(monkeypatch, tmp_path):
    """A small upload can decode to gigabytes; the rate is the one knob every release passes on."""
    import base64

    from core.inference import mlx_inference

    _require_video_stack()
    clip = _tiny_clip(tmp_path / "clip.mp4")
    per_frame = 2 * 3 * 32 * 24
    plain = SimpleNamespace()
    slower = SimpleNamespace(video_processor = SimpleNamespace(fps = 1.0))

    monkeypatch.setattr(mlx_inference, "_VIDEO_DECODE_BUDGET_BYTES", per_frame * 6)
    assert mlx_inference._video_frame_rate(clip, plain) == 2.0
    assert mlx_inference._video_frame_rate(clip, slower) == 1.0

    monkeypatch.setattr(mlx_inference, "_VIDEO_DECODE_BUDGET_BYTES", per_frame * 4)
    assert mlx_inference._video_frame_rate(clip, plain) == pytest.approx(4 / 3)

    # mlx-vlm holds every decoded clip at once, so two clips split the budget.
    monkeypatch.setattr(mlx_inference, "_VIDEO_DECODE_BUDGET_BYTES", per_frame * 8)
    assert mlx_inference._video_frame_rate(clip, plain) == 2.0
    assert mlx_inference._video_sampling_fps([clip, clip], plain) == pytest.approx(4 / 3)

    monkeypatch.setattr(mlx_inference, "_VIDEO_DECODE_BUDGET_BYTES", per_frame * 3)
    with pytest.raises(RuntimeError, match = "32x24 frames are too large"):
        mlx_inference._video_frame_rate(clip, plain)

    unopenable = tmp_path / "header"
    unopenable.write_bytes(base64.b64decode(_CLIP_B64))
    assert mlx_inference._video_frame_rate(str(unopenable), plain) is None


def test_mlx_vlm_a_clip_with_an_unreadable_rate_is_still_bounded(monkeypatch):
    """mlx-vlm decodes such a clip at an assumed rate instead of refusing it, so skipping the
    budget here is the one path around it."""
    from core.inference import mlx_inference

    _require_video_stack()
    cv2 = pytest.importorskip("cv2")
    measurements = {
        cv2.CAP_PROP_FRAME_WIDTH: 32.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 24.0,
        cv2.CAP_PROP_FRAME_COUNT: 12.0,
        cv2.CAP_PROP_FPS: 0.0,
    }

    class _RatelessCapture:
        def __init__(self, _path):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            return measurements[prop]

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", _RatelessCapture)
    monkeypatch.setattr(mlx_inference, "_VIDEO_DECODE_BUDGET_BYTES", 2 * 3 * 32 * 24 * 6)
    assert mlx_inference._video_frame_rate("clip.mp4", SimpleNamespace()) == 0.5


def test_mlx_vlm_the_frame_rate_follows_an_older_mlx_vlm_decoder(monkeypatch, tmp_path):
    """Older releases hand load_video only ``fps``, so its signature is the sampling source."""
    from core.inference import mlx_inference

    mlx_vlm = _require_video_stack_module()

    def load_video(
        video_path,
        fps = 1.5,
        nframes = None,
        min_frames = 8,
        max_frames = 768,
    ):
        raise AssertionError("never decoded here")

    monkeypatch.setattr(mlx_vlm, "utils", SimpleNamespace(load_video = load_video))
    clip = _tiny_clip(tmp_path / "clip.mp4")
    per_frame = 2 * 3 * 32 * 24

    monkeypatch.setattr(mlx_inference, "_VIDEO_DECODE_BUDGET_BYTES", per_frame * 12)
    assert mlx_inference._video_frame_rate(clip, SimpleNamespace()) == 1.5
    monkeypatch.setattr(mlx_inference, "_VIDEO_DECODE_BUDGET_BYTES", per_frame * 7)
    with pytest.raises(RuntimeError, match = "8 of them"):
        mlx_inference._video_frame_rate(clip, SimpleNamespace())


def test_mlx_vlm_the_frame_rate_rides_the_stream(monkeypatch, tmp_path):
    import base64
    import tempfile

    from core.inference import mlx_inference

    clip_b64 = base64.b64encode(Path(_tiny_clip(tmp_path / "clip.mp4")).read_bytes()).decode()
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(spool))
    streams = []
    backend = _video_vlm_backend(monkeypatch, streams)
    monkeypatch.setattr(mlx_inference, "_VIDEO_DECODE_BUDGET_BYTES", 2 * 3 * 32 * 24 * 4)
    turn = [
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "what moves"}]}
    ]

    assert list(
        backend._generate_vlm(
            turn, None, 0, 1, 0, 0, 1, 1, None, _adapter_state = False, videos = [clip_b64]
        )
    ) == ["ok"]
    assert streams[0][1]["fps"] == pytest.approx(4 / 3)

    monkeypatch.setattr(mlx_inference, "_VIDEO_DECODE_BUDGET_BYTES", 2 * 3 * 32 * 24 * 3)
    gen = backend._generate_vlm(
        turn, None, 0, 1, 0, 0, 1, 1, None, _adapter_state = False, videos = [clip_b64]
    )
    with pytest.raises(RuntimeError, match = "too large to decode"):
        next(gen)
    assert len(streams) == 1, "a refused clip never reaches mlx-vlm"
    assert list(spool.glob("unsloth-video-*")) == []


def test_mlx_vlm_a_video_turn_whose_render_is_unusable_is_refused_not_recovered(monkeypatch):
    """Registered renderers take image and audio counts only, so recovery would drop the clip."""
    backend = _video_vlm_backend(monkeypatch, [])
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _t, messages, **_k: f"{messages[0]['content'][0]} serialized",
    )
    turn = [{"role": "user", "content": [{"type": "image"}, {"type": "video"}]}]
    with pytest.raises(RuntimeError, match = "for a video turn"):
        list(
            backend._generate_vlm(
                turn, [object()], 0, 1, 0, 0, 1, 1, None, _adapter_state = False, videos = [_CLIP_B64]
            )
        )


def test_mlx_vlm_an_undecodable_clip_leaves_no_file_behind(monkeypatch, tmp_path):
    import binascii
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    backend = _video_vlm_backend(monkeypatch, [])
    turn = [{"role": "user", "content": [{"type": "video"}, {"type": "video"}]}]
    with pytest.raises(binascii.Error):
        list(
            backend._generate_vlm(
                turn, None, 0, 1, 0, 0, 1, 1, None, _adapter_state = False, videos = [_CLIP_B64, "A"]
            )
        )
    assert list(tmp_path.iterdir()) == []


def test_mlx_vlm_structured_video_items_must_match_attached_clips(monkeypatch):
    backend = _video_vlm_backend(monkeypatch, [])
    args = (None, 0, 1, 0, 0, 1, 1, None)
    without_part = [{"role": "user", "content": [{"type": "text", "text": "what moves"}]}]
    with pytest.raises(RuntimeError, match = "0 structured video item.*1 attached video"):
        list(backend._generate_vlm(without_part, *args, _adapter_state = False, videos = [_CLIP_B64]))
    with_part = [{"role": "user", "content": [{"type": "video"}]}]
    with pytest.raises(RuntimeError, match = "1 structured video item.*0 attached video"):
        list(backend._generate_vlm(with_part, *args, _adapter_state = False))


def test_mlx_generate_chat_response_attaches_the_video_part_and_forwards_the_clip(monkeypatch):
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    _require_video_stack()
    backend = MLXInferenceBackend()
    backend._model = object()
    backend._is_vlm = True
    backend._reads_vision = True
    captured = []
    backend._generate_vlm = lambda messages, *_args, **kwargs: (
        captured.append((messages, kwargs)) or iter(())
    )
    list(
        backend.generate_chat_response(
            [{"role": "user", "content": "what moves"}], videos = [_CLIP_B64]
        )
    )
    messages, kwargs = captured[0]
    assert messages[-1]["content"] == [{"type": "video"}, {"type": "text", "text": "what moves"}]
    assert kwargs["videos"] == [_CLIP_B64]
    list(
        backend.generate_chat_response(
            [{"role": "user", "content": "what moves"}], videos = [_CLIP_B64, "BBBB"]
        )
    )
    messages, kwargs = captured[1]
    assert messages[-1]["content"][:2] == [{"type": "video"}, {"type": "video"}]
    assert kwargs["videos"] == [_CLIP_B64, "BBBB"]

    backend._reads_vision = False
    with pytest.raises(RuntimeError, match = "loaded model does not read video"):
        list(
            backend.generate_chat_response([{"role": "user", "content": "hi"}], videos = [_CLIP_B64])
        )

    backend._is_vlm = True
    backend._reads_vision = True
    monkeypatch.setattr(mlx_inference, "_mlx_vlm_decodes_video", lambda: False)
    with pytest.raises(RuntimeError, match = "installed mlx-vlm does not read video"):
        list(
            backend.generate_chat_response([{"role": "user", "content": "hi"}], videos = [_CLIP_B64])
        )


def test_mlx_vlm_decodes_video_only_with_its_clip_decoder(monkeypatch):
    """The declared mlx-vlm range starts before load_video existed."""
    from core.inference import mlx_inference

    mlx_vlm = _require_video_stack_module()

    assert mlx_inference._mlx_vlm_decodes_video() is True
    monkeypatch.setattr(mlx_vlm, "utils", SimpleNamespace(prepare_inputs = object()))
    assert mlx_inference._mlx_vlm_decodes_video() is False


def test_mlx_reads_video_asks_the_processor_and_the_template(monkeypatch):
    """The processor must carry a video component and the template must place a marker."""
    from core.inference import mlx_inference
    from core.inference.mlx_inference import _VIDEO_PROBE_MESSAGES, _mlx_reads_video

    _require_video_stack()
    renders = {}

    def _render(_target, messages, **_kwargs):
        if messages is _VIDEO_PROBE_MESSAGES:
            outcome = renders["video"]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return "hi"

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation", _render
    )
    with_video = SimpleNamespace(video_processor = object(), tokenizer = SimpleNamespace())
    renders["video"] = "<|video_pad|> hi"
    assert _mlx_reads_video(with_video) is True
    assert _mlx_reads_video(SimpleNamespace(tokenizer = SimpleNamespace())) is False
    assert _mlx_reads_video(None) is False
    renders["video"] = "hi"
    assert _mlx_reads_video(with_video) is False
    renders["video"] = f"{_VIDEO_PROBE_MESSAGES[0]['content'][0]} hi"
    assert _mlx_reads_video(with_video) is False
    renders["video"] = ValueError("no video in this template")
    assert _mlx_reads_video(with_video) is False
    renders["video"] = "<|video_pad|> hi"
    monkeypatch.setattr(mlx_inference, "_mlx_vlm_decodes_video", lambda: False)
    assert _mlx_reads_video(with_video) is False, "a release without the decoder"


def test_mlx_reads_video_rejects_a_clip_rendered_as_prose(monkeypatch):
    """An installed override can render the part as ordinary text, which differs from the text
    render while marking nothing -- the model would then answer without the clip."""
    from core.inference.mlx_inference import _VIDEO_PROBE_MESSAGES, _mlx_reads_video

    _require_video_stack()
    renders = {}

    def _render(_target, messages, **_kwargs):
        return renders["video"] if messages is _VIDEO_PROBE_MESSAGES else "hi"

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation", _render
    )
    named = SimpleNamespace(
        video_processor = object(),
        video_token = "<|video_pad|>",
        tokenizer = SimpleNamespace(),
    )

    renders["video"] = "video hi"
    assert _mlx_reads_video(named) is False
    renders["video"] = "[part] hi"
    assert _mlx_reads_video(named) is False
    renders["video"] = "<|video_pad|> hi"
    assert _mlx_reads_video(named) is True

    unnamed = SimpleNamespace(video_processor = object(), tokenizer = SimpleNamespace())
    renders["video"] = "video hi"
    assert _mlx_reads_video(unnamed) is True


def _same_template_processor(template = "T", drops_video = True):
    """A processor that would render, with its tokenizer holding ``template``. ``drops_video`` is
    mlx-vlm's Gemma 4 processor, which loses a bare video part its tokenizer renders as written."""
    return SimpleNamespace(
        apply_chat_template = lambda *_a, **_k: "unused",
        chat_template = "T",
        video_processor = object(),
        video_token = "<|video|>",
        drops_video = drops_video,
        tokenizer = SimpleNamespace(chat_template = template),
    )


def _render_by_target(monkeypatch):
    def _render(target, messages, **_kwargs):
        has_video = any(
            isinstance(part, dict) and part.get("type") == "video"
            for message in messages
            for part in (message.get("content") or [])
        )
        marked = has_video and not getattr(target, "drops_video", False)
        return "<|video|> hi" if marked else "hi"

    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation", _render
    )


def test_a_video_turn_renders_through_the_tokenizer_only_when_the_processor_drops_it(monkeypatch):
    from core.inference.mlx_inference import MLXInferenceBackend, _mlx_reads_video

    _require_video_stack()
    _render_by_target(monkeypatch)
    drops = _same_template_processor()
    assert _mlx_reads_video(drops) is True
    # A template of the processor's own is what it must render, as before.
    assert _mlx_reads_video(_same_template_processor(template = "other")) is False

    backend = MLXInferenceBackend()
    video_turn = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": "hi"}]}]
    image_turn = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]
    backend._processor = drops
    prompt, target, _ = backend._render_vlm_prompt(video_turn, None, videos = ["clip.mp4"])
    assert (prompt, target) == ("<|video|> hi", drops.tokenizer)
    _, target, _ = backend._render_vlm_prompt(image_turn, [object()])
    assert target is drops

    # Some processors deliberately differ from their tokenizer, so one that keeps the marker renders.
    keeps = _same_template_processor(drops_video = False)
    backend._processor = keeps
    prompt, target, _ = backend._render_vlm_prompt(video_turn, None, videos = ["clip.mp4"])
    assert (prompt, target) == ("<|video|> hi", keeps)
    assert _mlx_reads_video(keeps) is True


def _sampling_release(monkeypatch, **utils_fields):
    """Stand mlx_vlm.utils up in one release's shape, so one processor can be read on each."""
    mlx_vlm = _require_video_stack_module()
    monkeypatch.setattr(mlx_vlm, "utils", SimpleNamespace(**utils_fields))


def _legacy_load_video(
    video_path,
    fps = 2.0,
    nframes = None,
    min_frames = 4,
    max_frames = 768,
):
    raise AssertionError("never decoded here")


def _resolving_utils():
    """The 0.7.0 shape: a resolver that reads the processor, terminating in the library defaults."""

    def resolve_video_sampling(processor, _overrides):
        component = getattr(processor, "video_processor", None)
        declared = {}
        if component is not None:
            hook = getattr(component, "video_sampling_defaults", None)
            declared = (
                hook()
                if callable(hook)
                else {n: getattr(component, n, None) for n in ("fps", "min_frames")}
            )
        return SimpleNamespace(
            fps = declared.get("fps") or 2.0,
            min_frames = declared.get("min_frames") or 4,
            nframes = declared.get("nframes"),
        )

    return {"load_video": _legacy_load_video, "resolve_video_sampling": resolve_video_sampling}


@pytest.mark.parametrize(
    "processor, expected_fps",
    [
        (SimpleNamespace(), 2.0),
        (SimpleNamespace(video_processor = SimpleNamespace(fps = 1.0)), 1.0),
        (SimpleNamespace(video_processor = SimpleNamespace(fps = 4.0)), 4.0),
        (SimpleNamespace(video_processor = SimpleNamespace()), 2.0),
        (
            SimpleNamespace(
                video_processor = SimpleNamespace(
                    video_sampling_defaults = lambda: {"fps": 0.5, "min_frames": 4}
                )
            ),
            0.5,
        ),
    ],
)
def test_mlx_vlm_every_release_reads_one_models_rate_the_same_way(
    monkeypatch, processor, expected_fps
):
    """A model's declared rate must not depend on which mlx-vlm is installed.

    Studio's mlx-vlm range spans both shapes: releases with no resolver
    and, from 0.7.0, releases that have one. Reading the rate only through the resolver left
    every install on the older shape sampling at the library default instead of the rate the
    checkpoint asked for.
    """
    from core.inference import mlx_inference

    seen = []
    for utils_fields in (
        {"load_video": _legacy_load_video},  # 0.5.0 and the whole 0.6 line
        _resolving_utils(),  # 0.7.0
    ):
        with pytest.MonkeyPatch.context() as patch:
            _sampling_release(patch, **utils_fields)
            seen.append(mlx_inference._video_sampling_target(processor))

    assert seen[0] == seen[1], "the installed mlx-vlm changed the sampling this model asked for"
    assert seen[0] == (expected_fps, 4, None)


def test_mlx_vlm_a_release_naming_no_fps_knob_still_bounds_a_clip(monkeypatch):
    """Forward compatibility: an unknown load_video signature must not make a clip a KeyError."""
    from core.inference import mlx_inference

    def load_video(
        video_path,
        sampling = None,
        **sampling_kwargs,
    ):
        raise AssertionError("never decoded here")

    _sampling_release(monkeypatch, load_video = load_video)
    assert mlx_inference._video_sampling_target(SimpleNamespace()) == (2.0, 4, None)
    assert mlx_inference._video_sampling_target(
        SimpleNamespace(video_processor = SimpleNamespace(fps = 1.0))
    ) == (1.0, 4, None)


def test_mlx_vlm_without_opencv_refuses_a_clip_by_name(monkeypatch):
    """Video is offered only when cv2 is present, since load_video and the frame budget both import it."""
    from core.inference import mlx_inference

    _require_video_stack()
    assert mlx_inference._mlx_vlm_decodes_video() is True

    monkeypatch.setitem(sys.modules, "cv2", None)
    assert mlx_inference._mlx_vlm_decodes_video() is False
    assert (
        _mlx_reads_video_probe(monkeypatch) is False
    ), "a model must not advertise video once the decoder is gone"


def _mlx_reads_video_probe(monkeypatch):
    from core.inference.mlx_inference import _mlx_reads_video
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda _t, _m, **_k: "<video> marked",
    )
    return _mlx_reads_video(SimpleNamespace(video_processor = object(), tokenizer = SimpleNamespace()))


_RENDERER = "core.inference.chat_template_helpers.apply_chat_template_for_generation"


def _vlm_runtime(monkeypatch, marker = "<|image|>"):
    """A backend already classified with ``marker`` as its image token, plus a stubbed mlx-vlm."""
    from core.inference import mlx_inference

    calls = []

    def _render(_target, messages, **_kwargs):
        def _flat(part):
            return "<|image|>" if part.get("type") == "image" else part.get("text", "")

        return "".join(
            body if isinstance(body, str) else "".join(map(_flat, body or ()))
            for body in (message.get("content") for message in messages)
        )

    def _vlm_stream(model, processor, prompt, images, **kwargs):
        calls.append({"prompt": prompt, "images": images})
        yield SimpleNamespace(text = "ok", prompt_tokens = 1, generation_tokens = 1)

    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.stream_generate = _vlm_stream
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(_RENDERER, _render)
    monkeypatch.setattr(
        mlx_inference, "_temporary_mlx_adapter_state", lambda *_a, **_k: contextlib.nullcontext()
    )

    backend = mlx_inference.MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "gemma4"})
    backend._processor = SimpleNamespace(
        chat_template = "template", tokenizer = SimpleNamespace(chat_template = "nested")
    )
    backend._is_vlm = True
    backend._multi_image_marker = marker
    return backend, calls


def test_several_images_reach_the_runtime_in_the_order_they_were_attached(monkeypatch):
    """Binding is positional, so a reordered list answers about the wrong picture, not an error."""
    backend, calls = _vlm_runtime(monkeypatch)
    first, second = object(), object()
    history = [{"role": "user", "content": "ask"}, {"role": "user", "content": "compare"}]

    assert list(
        backend.generate_chat_response(history, images = [first, second], max_new_tokens = 4)
    ) == ["ok"]
    assert calls[-1]["images"] == [first, second]
    assert calls[-1]["prompt"].count("<|image|>") == 2

    only = [{"role": "user", "content": "describe"}]
    list(backend.generate_chat_response(only, image = first, max_new_tokens = 4))
    list(backend.generate_chat_response(only, images = [first], max_new_tokens = 4))
    assert calls[-2] == calls[-1] == {"prompt": "<|image|>describe", "images": [first]}


@pytest.mark.parametrize(
    "marker, render, error",
    [
        ("<|image|>", "<|image|>only one", "marked 1 image"),
    ],
    ids = ["the render collapsed a marker"],
)
def test_a_render_that_cannot_bind_every_image_is_refused(monkeypatch, marker, render, error):
    """One image is still taken either way; too few markers would describe one picture twice."""
    backend, _calls = _vlm_runtime(monkeypatch, marker = marker)
    if render is not None:
        monkeypatch.setattr(_RENDERER, lambda *_a, **_k: render)

    turn = [{"role": "user", "content": "compare"}]
    with pytest.raises((ValueError, RuntimeError), match = error):
        list(backend.generate_chat_response(turn, images = [object(), object()]))


def test_a_single_image_template_is_served_one_picture_not_refused(monkeypatch):
    """A template that marks one image cannot bind several, but a resumed chat carrying
    replayed MCP pictures must still be answered: the caller's attachment, or else the
    newest picture, is served with its own marker and the others are left out."""
    backend, calls = _vlm_runtime(monkeypatch, marker = None)
    older, newer = object(), object()
    turn = [{"role": "user", "content": "compare"}]

    assert list(backend.generate_chat_response(turn, images = [older, newer])) == ["ok"]
    assert calls[-1]["images"] == [newer]
    assert calls[-1]["prompt"].count("<|image|>") == 1

    attachment = object()
    list(backend.generate_chat_response(turn, image = attachment, images = [older]))
    assert calls[-1]["images"] == [attachment]
    assert calls[-1]["prompt"].count("<|image|>") == 1


@pytest.mark.parametrize("merged_question", [False, True])
def test_single_image_replay_notes_describe_only_retained_pixels(monkeypatch, merged_question):
    from core.inference.mcp_images import IMAGE_TURN_TEXT, image_marker_parts, placeholder_turn

    backend, calls = _vlm_runtime(monkeypatch, marker = None)
    older, newer = object(), object()
    previous_image = placeholder_turn(1)
    if merged_question:
        previous_image["content"].append({"type": "text", "text": "Keep this user question."})
    history = [
        {"role": "user", "content": "Inspect the first screenshot."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "first",
                    "type": "function",
                    "function": {
                        "name": "mcp__test__screenshot",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "first", "content": "[1 image returned]"},
        previous_image,
        {"role": "assistant", "content": "The first screenshot was inspected."},
        {"role": "user", "content": "Inspect the next screenshot."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "second",
                    "type": "function",
                    "function": {
                        "name": "mcp__test__screenshot",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "second", "content": "[1 image returned]"},
        placeholder_turn(1),
    ]
    assert list(backend.generate_chat_response(history, images = [older, newer])) == ["ok"]
    assert calls[-1]["images"] == [newer]
    assert calls[-1]["prompt"].count("<|image|>") == 1
    if merged_question:
        assert "Keep this user question." in calls[-1]["prompt"]
        assert "(0 of 1)" in calls[-1]["prompt"]
    else:
        assert calls[-1]["prompt"].count(IMAGE_TURN_TEXT) == 1
    assert len(image_marker_parts(history)) == 2


@pytest.mark.parametrize("module", ["mlx_lm.models.cache", "mlx_vlm.models.cache"])
@pytest.mark.parametrize(
    "bits, head_dim", [(5, 128), (6, 128), (3, 256), (5, 256), (6, 256), (8, 128)]
)
def test_an_empty_cache_converted_before_generation_decodes_at_every_width(module, bits, head_dim):
    # Upstream sizes an empty QuantizedKVCache one packed word too wide at these widths.
    mx = pytest.importorskip("mlx.core")
    cache_module = pytest.importorskip(module)
    from core.inference.mlx_inference import _quantize_kv_entries

    (cache,) = _quantize_kv_entries([cache_module.KVCache()], bits)
    for _ in range(300):
        token = mx.random.normal((1, 2, 1, head_dim))
        keys, _values = cache.update_and_fetch(token, token)
        restored = mx.dequantize(*keys, group_size = 64, bits = bits)
        mx.eval(restored)
    assert restored.shape == (1, 2, 300, head_dim)


def test_a_random_model_decoding_through_mlx_lm_emits_only_the_schema():
    """Real mlx_lm pipelining (the next step is masked before this token is yielded) with a
    random model: only the grammar can make the reply a valid document."""
    mx = _metal_mx()
    pytest.importorskip("mlx_lm")
    from mlx_lm import stream_generate
    from mlx_lm.models import llama
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from core.inference.grammar_constraint import make_grammar_logits_processor

    tokenizer = _char_tokenizer()
    tokenizer.chat_template = "{{ messages[0].content }}"
    mx.random.seed(0)
    model = llama.Model(
        llama.ModelArgs(
            model_type = "llama",
            hidden_size = 32,
            num_hidden_layers = 1,
            intermediate_size = 64,
            num_attention_heads = 2,
            num_key_value_heads = 2,
            rms_norm_eps = 1e-5,
            vocab_size = len(tokenizer),
        )
    )
    schema = {
        "type": "object",
        "properties": {"a": {"enum": ["x", "y"]}, "b": {"type": "boolean"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    }
    wrapped = TokenizerWrapper(tokenizer)
    constraint = build_constraint({"type": "json_schema", "schema": schema}, wrapped, "p")
    text = "".join(
        chunk.text
        for chunk in stream_generate(
            model,
            wrapped,
            tokenizer.encode("go", add_special_tokens = False),
            max_tokens = 64,
            sampler = make_sampler(temp = 1.0),
            logits_processors = [make_grammar_logits_processor(constraint)],
        )
    )
    document = json.loads(text)
    assert document["a"] in ("x", "y") and isinstance(document["b"], bool)
    assert set(document) == {"a", "b"}


def _constrained_vlm_reply(
    monkeypatch,
    pieces,
    *,
    allows_reasoning,
    markers,
    specials = (),
):
    """Snapshots ``_generate_vlm`` yields for scripted text under a stand-in grammar."""
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.prompt_utils = SimpleNamespace(MODEL_CONFIG = {}, apply_chat_template = None)
    mlx_vlm.stream_generate = lambda *_a, **_k: iter(
        [SimpleNamespace(text = piece, prompt_tokens = 1, generation_tokens = 1) for piece in pieces]
    )
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setattr(
        "core.inference.chat_template_helpers.apply_chat_template_for_generation",
        lambda *_a, **_k: "user: hi\nassistant: <think>\n",
    )
    monkeypatch.setattr(
        mlx_inference, "_temporary_mlx_adapter_state", lambda *_a, **_k: contextlib.nullcontext()
    )
    monkeypatch.setattr(
        mlx_inference, "detect_reasoning_channel_markers", lambda *_a, **_k: markers
    )
    monkeypatch.setattr(
        mlx_inference,
        "_build_grammar_constraint",
        lambda *_a, **_k: SimpleNamespace(
            allows_reasoning = allows_reasoning,
            stops_on = lambda _ids: None,
            decoded_dropping = lambda _ids = None: None,
        ),
    )
    monkeypatch.setattr(mlx_inference, "_mlx_sampling_processors", lambda **_k: None)
    monkeypatch.setattr(mlx_inference, "_vlm_generation_is_diffusion", lambda _model: False)

    backend = MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "generic_vlm"})
    backend._processor = SimpleNamespace(chat_template = "template", all_special_tokens = specials)
    backend._is_vlm = True
    backend._reads_vision = True
    args = ([{"role": "user", "content": "hi"}], [], 0.7, 0.9, 40, 0.01, 64, 1.0, None)
    return list(backend._generate_vlm(*args, response_format = {"type": "json_object"}))


def test_a_document_only_vlm_reply_is_not_rewritten_by_the_reasoning_normalizer(monkeypatch):
    document = '{"city":"<think>Paris</think>"}'
    snapshots = _constrained_vlm_reply(
        monkeypatch,
        [document[:12], document[12:]],
        allows_reasoning = False,
        markers = ("<think>", "</think>"),
    )
    assert snapshots[-1] == document


def test_an_allowed_vlm_reasoning_block_keeps_its_opener_when_the_closer_is_special(monkeypatch):
    snapshots = _constrained_vlm_reply(
        monkeypatch,
        ["x", "</think>", '{"a":1}'],
        allows_reasoning = True,
        markers = None,
        specials = ("</think>",),
    )
    assert snapshots[-1] == '<think>\nx</think>{"a":1}'


def test_the_text_grammar_is_told_a_closer_survives_whenever_the_prefill_is(monkeypatch):
    """Client tools run the native decoder, which keeps a special </think>: the prefill then
    re-emits the opener, so the grammar must allow the close rather than refuse the request."""
    from core.inference import mlx_inference

    mlx_lm = types.ModuleType("mlx_lm")
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_k: object()
    mlx_lm.sample_utils = sample_utils
    mlx_lm.stream_generate = lambda *_a, **_k: iter([_Resp("ok", 1)])
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(mlx_inference, "_mlx_sampling_processors", lambda **_k: None)
    monkeypatch.setattr(mlx_inference, "decoder_preserves_token", lambda *_a, **_k: True)
    seen = {}

    def _capture(*_a, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(mlx_inference, "_build_grammar_constraint", _capture)
    backend = _budget_backend(monkeypatch)
    monkeypatch.setattr(
        backend,
        "_render_text_prompt",
        lambda *_a, **_k: SimpleNamespace(prompt = "P<think>", reasoning_channel_markers = None),
        raising = False,
    )
    list(
        backend._generate_text(
            _TEXT_TURN,
            0.0,
            1.0,
            0,
            0.0,
            8,
            1.0,
            None,
            tools = [{"type": "function", "function": {"name": "f"}}],
            response_format = {"type": "json_object"},
            reasoning_is_extracted = True,
        )
    )
    assert seen["reply_keeps_special_tokens"] is True


@pytest.mark.parametrize("constrained", [False, True])
def test_a_constrained_plain_reply_is_decoded_without_space_cleanup(monkeypatch, constrained):
    from core.inference import mlx_inference

    mlx_lm = types.ModuleType("mlx_lm")
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_k: object()
    mlx_lm.sample_utils = sample_utils
    mlx_lm.stream_generate = lambda *_a, **_k: iter([_Resp("ok", 1)])
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(mlx_inference, "_mlx_sampling_processors", lambda **_k: None)
    monkeypatch.setattr(
        mlx_inference,
        "_build_grammar_constraint",
        lambda *_a, **_k: SimpleNamespace(
            allows_reasoning = False, decoded_dropping = lambda _ids: None
        )
        if constrained
        else None,
    )
    backend = _budget_backend(monkeypatch)
    decodes = []

    def _decode(_ids, **kwargs):
        decodes.append(kwargs)
        return "ok"

    backend._tokenizer.decode = _decode
    monkeypatch.setattr(
        backend,
        "_render_text_prompt",
        lambda *_a, **_k: SimpleNamespace(prompt = "P", reasoning_channel_markers = None),
        raising = False,
    )
    list(
        backend._generate_text(
            _TEXT_TURN,
            0.0,
            1.0,
            0,
            0.0,
            8,
            1.0,
            None,
            response_format = {"type": "json_object"} if constrained else None,
        )
    )
    assert decodes
    for kwargs in decodes:
        assert kwargs.get("clean_up_tokenization_spaces") is (False if constrained else None)


def test_a_special_token_the_reply_would_drop_is_refused_not_silently_deleted():
    """llguidance spells a forced literal with a special id when one matches, and a decode that
    skips special tokens would delete it from a reply reported as valid."""
    tokenizer = _char_tokenizer("ab", markers_are_special = True)
    special = tokenizer.convert_tokens_to_ids("ab")
    fmt = {"type": "json_schema", "schema": {"const": "ab"}}

    def _after_quote(constraint):
        for token in tokenizer.encode('"', add_special_tokens = False):
            _allowed(constraint, len(tokenizer))
            constraint.advance(int(token))
        return _allowed(constraint, len(tokenizer))

    assert special in _after_quote(
        build_constraint(fmt, tokenizer, "p", reply_keeps_special_tokens = True)
    )
    with pytest.raises(ResponseFormatError, match = "reserves as a control token"):
        _after_quote(build_constraint(fmt, tokenizer, "p"))


def test_only_the_specials_the_decoder_drops_are_withheld_from_the_document():
    """A native decoder keeps its allowlist and drops every other special id, so the grammar is
    held to exactly that set, not to all specials or none."""
    tokenizer = _char_tokenizer("ab", "cd", markers_are_special = True)
    dropped, kept = (tokenizer.convert_tokens_to_ids(t) for t in ("ab", "cd"))

    def _after_quote(literal):
        constraint = build_constraint(
            {"type": "json_schema", "schema": {"const": literal}},
            tokenizer,
            "p",
            reply_keeps_special_tokens = True,
        )
        constraint.decoded_dropping({dropped})
        for token in tokenizer.encode('"', add_special_tokens = False):
            _allowed(constraint, len(tokenizer))
            constraint.advance(int(token))
        return _allowed(constraint, len(tokenizer))

    assert kept in _after_quote("cd")
    with pytest.raises(ResponseFormatError, match = "reserves as a control token"):
        _after_quote("ab")


def test_the_text_path_hands_the_constraint_its_decoders_dropped_ids(monkeypatch):
    from core.inference import mlx_inference

    mlx_lm = types.ModuleType("mlx_lm")
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_k: object()
    mlx_lm.sample_utils = sample_utils
    mlx_lm.stream_generate = lambda *_a, **_k: iter([_Resp("ok", 1)])
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(mlx_inference, "_mlx_sampling_processors", lambda **_k: None)
    seen = []
    constraint = SimpleNamespace(allows_reasoning = False, decoded_dropping = seen.append)
    monkeypatch.setattr(mlx_inference, "_build_grammar_constraint", lambda *_a, **_k: constraint)

    class _Decoder:
        def __init__(self, *_a, **_k):
            pass

        def dropped_ids(self):
            return frozenset({7})

    monkeypatch.setattr(mlx_inference, "NativeToolTokenDecoder", _Decoder)
    backend = _budget_backend(monkeypatch)
    monkeypatch.setattr(
        backend,
        "_render_text_prompt",
        lambda *_a, **_k: SimpleNamespace(prompt = "P", reasoning_channel_markers = None),
        raising = False,
    )
    run = lambda **kw: list(
        backend._generate_text(
            _TEXT_TURN,
            0.0,
            1.0,
            0,
            0.0,
            8,
            1.0,
            None,
            response_format = {"type": "json_object"},
            **kw,
        )
    )
    try:
        run(tools = [{"type": "function", "function": {"name": "f"}}])
    except Exception:
        pass
    assert seen and seen[0] == frozenset({7})
    seen.clear()
    run()
    assert seen == [None], "a plain skip_special_tokens decode drops every special id"


def test_a_tokenizer_end_token_the_runtime_does_not_stop_on_is_withheld_too():
    """llguidance ends only on the runtime stops, so the tokenizer's own end token is just
    another special id the decode drops, and a literal spelled only by it is refused."""
    pytest.importorskip("mlx_lm")
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    tokenizer, runtime_stop = _two_stop_tokenizer()
    wrapped = TokenizerWrapper(tokenizer, eos_token_ids = [runtime_stop])
    constraint = build_constraint(
        {"type": "json_schema", "schema": {"const": "<eos>"}}, wrapped, "p"
    )
    for token in tokenizer.encode('"', add_special_tokens = False):
        _allowed(constraint, len(tokenizer))
        constraint.advance(int(token))
    with pytest.raises(ResponseFormatError, match = "reserves as a control token"):
        _allowed(constraint, len(tokenizer))


def test_the_grammar_ends_on_the_stop_the_decode_loop_names():
    """mlx-vlm stops on config ids the tokenizer may not carry; told them, the finished
    document is closed by that id rather than the tokenizer's end token."""
    tokenizer, runtime_stop = _two_stop_tokenizer()
    constraint = build_constraint(JSON_SCHEMA_FORMAT, tokenizer, "p")
    constraint.stops_on([runtime_stop])
    for token_id in tokenizer.encode('{"a":1}', add_special_tokens = False):
        _allowed(constraint, len(tokenizer))
        constraint.advance(int(token_id))
    assert _allowed(constraint, len(tokenizer)) == {runtime_stop}


def test_turboquant_leaves_every_cache_for_the_runtime_to_build():
    lm_cache = pytest.importorskip("mlx_lm.models.cache")

    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    kinds = lambda: [
        type(entry).__name__
        for entry in backend._prepare_kv_entries([lm_cache.KVCache(), lm_cache.KVCache()])
    ]

    backend._kv_quant = {"kv_bits": 3.5}
    backend._turboquant = True
    assert kinds() == ["KVCache"] * 2
    backend._kv_quant = {"kv_bits": 4}
    assert kinds() == ["KVCache"] * 2

    backend._turboquant = False
    assert kinds() == ["QuantizedKVCache"] * 2


def test_turboquant_sends_its_width_on_both_vlm_cache_branches(monkeypatch):
    """No pre-converted entry carries the width, so a reused session must still send it."""
    from core.inference.mlx_inference import MLXInferenceBackend

    backend = MLXInferenceBackend()
    backend._reads_vision = True
    backend._kv_quant = {"kv_bits": 3.5}
    backend._turboquant = True

    class _Session(contextlib.AbstractContextManager):
        cache, media_block, step, produced_tokens, produced_seconds = [object()], None, 8, 0, 0.0
        window = step

        def __exit__(self, *_exc):
            return False

        def finish(self):
            return None

    session = _Session()
    fresh = dict(_drive_vlm_generation(backend, monkeypatch))
    reused = dict(_drive_vlm_generation(backend, monkeypatch, session = session))
    for kwargs in (fresh, reused):
        assert (kwargs["kv_bits"], kwargs["kv_quant_scheme"], kwargs["quantized_kv_start"]) == (
            3.5,
            "turboquant",
            0,
        )
    assert "prompt_cache" in reused and "prompt_cache" not in fresh


def test_a_turboquant_load_turns_a_pin_into_a_request_budget(monkeypatch):
    # Same rule as mx.quantize: an enforceable pin budgets requests, it does not refuse.
    pytest.importorskip("mlx_lm.models.cache")
    from core.inference import mlx_inference
    from core.inference.mlx_inference import MLXInferenceBackend

    def status(
        _model,
        bits,
        refusal = "",
    ):
        if refusal:
            return {"kv_bits": None, "eligibility": "refused", "reason": refusal}
        return {"kv_bits": bits, "eligibility": "full", "reason": ""}

    monkeypatch.setattr(mlx_inference, "_turboquant_status", status)
    honours = SimpleNamespace(layers = [object(), object()])

    def policy(
        requested,
        served,
        refusal = "",
    ):
        backend = MLXInferenceBackend()
        backend._model = honours
        backend._is_vlm = True
        backend._turboquant = True
        backend._turboquant_refusal = refusal
        quant, window, enforced, budget = backend._resolve_kv_policy(True, 3.5, requested, served)
        return quant["kv_bits"], window, enforced, budget

    assert policy(8192, 8192) == (3.5, None, False, 8192)
    assert policy(0, 262144) == (3.5, None, False, None)
    assert policy(8192, 8192, "refused") == (None, 8192, True, None)


def test_mlx_precision_folder_loads_from_its_own_download(monkeypatch, tmp_path):
    from core.inference.mlx_inference import MLXInferenceBackend

    _install_fake_mlx(monkeypatch)
    calls = []
    _install_fake_fast_mlx(monkeypatch, calls)
    fetched = []

    def _snapshot_download(
        repo_id,
        allow_patterns = None,
        **_,
    ):
        fetched.append((repo_id, allow_patterns))
        (tmp_path / "6-bit").mkdir(exist_ok = True)
        (tmp_path / "6-bit" / "config.json").write_text("{}")
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _snapshot_download)
    monkeypatch.setattr("utils.utils.hf_cache_snapshot_dir_for_repo", lambda repo: None)

    backend = MLXInferenceBackend()
    repo = "orcarouter/Qwen3.8-27B-Uncensored-MLX"
    config = SimpleNamespace(identifier = repo, is_vision = False, gguf_variant = "6-bit")
    assert backend.load_model(config)
    assert calls[-1][0] == (str(tmp_path / "6-bit"),)
    assert fetched == [(repo, ["6-bit/*"])]
    assert backend.active_model_name == repo
    assert backend.models[repo]["gguf_variant"] == "6-bit"

    assert backend.load_model(SimpleNamespace(identifier = repo, is_vision = False))
    assert calls[-1][0] == (repo,)
    assert backend.models[repo]["gguf_variant"] is None
    assert len(fetched) == 1


def test_a_batch_row_without_an_independent_detokenizer_is_refused_not_crashed():
    from core.inference import mlx_inference
    from core.inference.worker import RowRefused

    decoder = SimpleNamespace(suppresses = lambda _id: False)
    mlx_inference._require_row_detokenizer(None)
    mlx_inference._require_row_detokenizer(
        mlx_inference._MlxStreamedText(SimpleNamespace(), decoder, None, ())
    )
    with pytest.raises(RowRefused, match = "detokenizer"):
        mlx_inference._require_row_detokenizer(
            mlx_inference._MlxStreamedText(None, decoder, None, ())
        )


def test_a_flushed_tail_releases_its_text_before_a_stop_it_completes():
    from core.inference.mlx_inference import _RowStream

    stream = _RowStream(sequences = ("HALT",))
    assert stream.feed(text = "already ") == ["already "]
    assert stream.flush("remaining HALT") == ["already remaining "]
    assert stream.stopped and stream.settle() == []
    plain = _RowStream(sequences = ("HALT",))
    plain.feed(text = "all ")
    assert plain.flush("of it") == ["all of it"] and plain.settle() == []
