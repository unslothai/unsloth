# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unbuildable MLX architectures are refused from config.json before the weights download (#18972)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

try:
    from core.inference import mlx_model_support as support
except (
    ImportError
):  # a tree predating it: the load and route tests below still show what it changes
    support = None

# mlx-community/Inkling-2-Mini-4bit's config.json, trimmed to the routing keys. Its mtp_config ships no weights.
_INKLING2_MINI = {
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "mtp_config": {"hidden_size": 2304, "num_hidden_layers": 3, "block_size": 8},
    "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
    "text_config": {"model_type": "inkling2_loop_text", "hidden_size": 2304},
    "vision_config": {"model_type": "inkling2_vision", "hidden_size": 1024},
}

_VLM_0_8_0 = {"mlx_vlm.models.inkling", "mlx_vlm.models.inkling2", "mlx_vlm.models.qwen3_5"}
_VLM_0_8_0_DRAFTERS = {"mlx_vlm.speculative.drafters.inkling2_dspark"}
_LM_0_32_2 = {"mlx_lm.models.llama", "mlx_lm.models.qwen3"}
# mlx-vlm 0.8.0's MODEL_REMAPPING has no inkling2_loop; main (Blaizzy/mlx-vlm#2778) maps it to inkling2.
_REMAP_0_8_0 = {"inkling_mm_model": "inkling"}
_REMAP_MAIN = {**_REMAP_0_8_0, "inkling2_loop": "inkling2"}


def _libraries(
    monkeypatch,
    remap = _REMAP_0_8_0,
    modules = (),
    resolvers = True,
):
    """Fake mlx_vlm.utils / mlx_lm.utils and the modules find_spec locates."""
    vlm = types.ModuleType("mlx_vlm.utils")
    vlm.MODEL_REMAPPING = dict(remap)
    lm = types.ModuleType("mlx_lm.utils")
    lm.MODEL_REMAPPING = {"mistral": "llama"}
    if resolvers:
        vlm.get_model_type = lambda config: (lambda t: vlm.MODEL_REMAPPING.get(t, t))(
            str(config.get("model_type")).lower()
        )
        lm.model_package = lambda config: lm.MODEL_REMAPPING.get(
            config["model_type"], config["model_type"]
        )
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.ModuleType("mlx_vlm"))
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", vlm)
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", lm)
    present = set(modules) or (_VLM_0_8_0 | _VLM_0_8_0_DRAFTERS | _LM_0_32_2)
    if support is not None:
        monkeypatch.setattr(
            support, "find_spec", lambda name: object() if name in present else None
        )
    return vlm, lm


@pytest.mark.parametrize("resolvers", [True, False])
def test_inkling2_mini_is_refused_on_the_released_mlx_vlm(monkeypatch, resolvers):
    _libraries(monkeypatch, resolvers = resolvers)
    assert support.unsupported_model_type(_INKLING2_MINI) == "inkling2_loop"


def test_inkling2_mini_loads_once_mlx_vlm_maps_it(monkeypatch):
    _libraries(monkeypatch, remap = _REMAP_MAIN)
    assert support.unsupported_model_type(_INKLING2_MINI) is None


@pytest.mark.parametrize(
    "config",
    [
        {"model_type": "inkling2"},  # Inkling 2 Small: mlx-vlm 0.8.0 builds it
        {"model_type": "mistral"},  # mlx-lm, through its remapping
        {"model_type": "qwen3"},
    ],
)
def test_an_architecture_either_library_builds_is_not_refused(monkeypatch, config):
    _libraries(monkeypatch)
    assert support.unsupported_model_type(config) is None


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"model_type": ""},
        {"model_type": 7},
        {**_INKLING2_MINI, "model_file": "modeling.py"},  # checkpoint-local model code
    ],
)
def test_a_config_it_cannot_judge_is_never_refused(monkeypatch, config):
    _libraries(monkeypatch)
    assert support.unsupported_model_type(config) is None


def test_a_library_that_will_not_import_never_refuses(monkeypatch):
    _libraries(monkeypatch)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", None)  # import raises ImportError
    assert support.unsupported_model_type(_INKLING2_MINI) is None
    _libraries(monkeypatch)

    def broken(config):
        raise RuntimeError("registry changed shape")

    sys.modules["mlx_lm.utils"].model_package = broken
    assert support.unsupported_model_type(_INKLING2_MINI) is None


def test_the_message_names_the_architecture_and_the_gguf_build(monkeypatch):
    from routes.inference import _NOT_SUPPORTED_HINTS

    msg = support.unsupported_model_type_message(
        "inkling2_loop", "mlx-community/Inkling-2-Mini-4bit"
    )
    assert msg.startswith("mlx-community/Inkling-2-Mini-4bit uses the 'inkling2_loop' architecture")
    assert "mlx-vlm" in msg and "mlx-lm" in msg
    assert "unsloth/Inkling-2-Mini-GGUF" in msg
    other = support.unsupported_model_type_message("brand_new_arch")
    assert other.startswith("This model uses the 'brand_new_arch'") and "GGUF" in other
    for text in (msg, other):
        assert not any(hint.lower() in text.lower() for hint in _NOT_SUPPORTED_HINTS)


@pytest.mark.parametrize(
    "raw",
    [
        # mlx-vlm 0.8.0, as #18972 reported it
        "Model type inkling2_loop not supported. Error: No module named "
        "'mlx_vlm.speculative.drafters.inkling2_loop'",
        "Model type inkling2_loop not supported.",
    ],
)
def test_the_route_rewrites_the_libraries_own_refusal(raw):
    from routes.inference import _maybe_unsupported_message

    msg = _maybe_unsupported_message(raw)
    assert "'inkling2_loop' architecture" in msg and "unsloth/Inkling-2-Mini-GGUF" in msg
    assert "No module named" not in msg and "drafters" not in msg
    assert _maybe_unsupported_message("CUDA out of memory") == "CUDA out of memory"


def _backend(monkeypatch, config):
    """An MLX backend over fake mlx.core / unsloth-zoo whose loader records the loads it was asked for."""
    from core.inference import mlx_inference
    from utils.models import model_config

    loads = []

    class _Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loads.append(name)
            # What mlx-vlm 0.8.0 raises once the weights are on disk.
            raise ValueError(
                "Model type inkling2_loop not supported. Error: No module named "
                "'mlx_vlm.speculative.drafters.inkling2_loop'"
            )

    if "mlx.core" not in sys.modules:
        mlx = types.ModuleType("mlx")
        mlx.core = types.ModuleType("mlx.core")
        monkeypatch.setitem(sys.modules, "mlx", mlx)
        monkeypatch.setitem(sys.modules, "mlx.core", mlx.core)
    loader = types.ModuleType("unsloth_zoo.mlx.loader")
    loader.FastMLXModel = _Loader
    loader._is_vlm = lambda cfg: True
    loader._mlx_vlm_only_text_model = lambda cfg, model_type: False
    monkeypatch.setitem(sys.modules, "unsloth_zoo.mlx.loader", loader)
    monkeypatch.setattr(model_config, "_read_raw_config_json", lambda name, token = None: config)
    monkeypatch.setattr(
        mlx_inference.MLXInferenceBackend, "_configure_memory_limits", lambda self: None
    )
    return mlx_inference.MLXInferenceBackend(), loads


def test_the_load_is_refused_before_the_weights_download(monkeypatch):
    _libraries(monkeypatch)
    backend, loads = _backend(monkeypatch, _INKLING2_MINI)
    request = SimpleNamespace(
        identifier = "mlx-community/Inkling-2-Mini-4bit", is_vision = True, is_lora = False
    )
    with pytest.raises(ValueError) as refused:
        backend.load_model(request)
    assert loads == []
    assert "'inkling2_loop' architecture" in str(refused.value)
    assert "unsloth/Inkling-2-Mini-GGUF" in str(refused.value)
