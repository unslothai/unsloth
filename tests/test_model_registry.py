"""Register each model set and check the registered ids exist on the HF Hub."""

import subprocess
import sys
from dataclasses import dataclass

import pytest
from huggingface_hub import ModelInfo as HfModelInfo

from unsloth.registry import register_models, search_models
from unsloth.registry._deepseek import register_deepseek_models
from unsloth.registry._gemma import register_gemma_models
from unsloth.registry._llama import register_llama_models
from unsloth.registry._mistral import register_mistral_models
from unsloth.registry._phi import register_phi_models
from unsloth.registry._qwen import register_qwen_models
from unsloth.registry.registry import MODEL_REGISTRY, QUANT_TAG_MAP, QuantType
from unsloth.utils.hf_hub import get_model_info

MODEL_NAMES = [
    "llama",
    "qwen",
    "mistral",
    "phi",
    "gemma",
    "deepseek",
]
MODEL_REGISTRATION_METHODS = [
    register_llama_models,
    register_qwen_models,
    register_mistral_models,
    register_phi_models,
    register_gemma_models,
    register_deepseek_models,
]


@dataclass
class ModelTestParam:
    name: str
    register_models: callable


def _test_model_uploaded(model_ids: list[str]):
    missing_models = []
    for _id in model_ids:
        model_info: HfModelInfo = get_model_info(_id)
        if not model_info:
            missing_models.append(_id)

    return missing_models


TestParams = [
    ModelTestParam(name, models) for name, models in zip(MODEL_NAMES, MODEL_REGISTRATION_METHODS)
]


@pytest.mark.parametrize("model_test_param", TestParams, ids = lambda param: param.name)
def test_model_registration(model_test_param: ModelTestParam):
    MODEL_REGISTRY.clear()
    registration_method = model_test_param.register_models
    registration_method()
    registered_models = MODEL_REGISTRY.keys()
    missing_models = _test_model_uploaded(registered_models)
    assert not missing_models, f"{model_test_param.name} missing following models: {missing_models}"


def test_all_model_registration():
    register_models()
    registered_models = MODEL_REGISTRY.keys()
    missing_models = _test_model_uploaded(registered_models)
    assert not missing_models, f"Missing following models: {missing_models}"


def test_quant_type():
    # NOTE: for org="unsloth" models, QuantType.NONE aliases QuantType.UNSLOTH
    dynamic_quant_models = search_models(quant_types = [QuantType.UNSLOTH])
    assert all(m.quant_type == QuantType.UNSLOTH for m in dynamic_quant_models)
    quant_tag = QUANT_TAG_MAP[QuantType.UNSLOTH]
    assert all(quant_tag in m.model_path for m in dynamic_quant_models)


def test_importing_registry_does_not_register_models():
    """Importing the registry must not populate MODEL_REGISTRY on its own.

    ``_deepseek`` used to call ``register_deepseek_models(...)`` at module
    scope, so merely importing ``unsloth.registry`` registered models (and hit
    the hub) as an import side effect, unlike every other family which only
    registers on demand. Run in a subprocess so the check is independent of any
    ``register_models()`` calls other tests make on the shared registry.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import unsloth.registry\n"
            "from unsloth.registry.registry import MODEL_REGISTRY\n"
            "print('REGISTRY_SIZE', len(MODEL_REGISTRY))",
        ],
        capture_output = True,
        text = True,
        check = True,
    )
    size_lines = [
        line for line in result.stdout.splitlines() if line.startswith("REGISTRY_SIZE")
    ]
    assert size_lines == ["REGISTRY_SIZE 0"], result.stdout
