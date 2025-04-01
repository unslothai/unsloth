from dataclasses import dataclass

import pytest
from huggingface_hub import ModelInfo as HfModelInfo

from unsloth.registry import register_models
from unsloth.registry._deepseek import register_deepseek_models
from unsloth.registry._gemma import register_gemma_models
from unsloth.registry._llama import register_llama_models
from unsloth.registry._mistral import register_mistral_models
from unsloth.registry._phi import register_phi_models
from unsloth.registry._qwen import register_qwen_models
from unsloth.registry.registry import MODEL_REGISTRY, ModelInfo
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
    registration_models: callable


def _test_model_uploaded(model_ids: list[str]):
    missing_models = []
    for _id in model_ids:
        model_info: HfModelInfo = get_model_info(_id)
        if not model_info:
            missing_models.append(_id)

    return missing_models


TestParams = [
    ModelTestParam(name, models)
    for name, models in zip(MODEL_NAMES, MODEL_REGISTRATION_METHODS)
]

# Test that model registration methods register respective models
@pytest.mark.parametrize(
    "model_test_param", TestParams, ids=lambda param: param.name
)
def test_model_registration(model_test_param: ModelTestParam):
    MODEL_REGISTRY.clear()
    model_test_param.registration_models()
    registered_models = MODEL_REGISTRY.keys()
    missing_models = _test_model_uploaded(registered_models)
    assert not missing_models, (
        f"{model_test_param.name} missing following models: {missing_models}"
    )


# if __name__ == "__main__":
#     for method in [
#         get_llama_models,
#         get_llama_vision_models,
#         get_qwen_models,
#         get_qwen_vl_models,
#         get_phi_models,
#         get_phi_instruct_models,
#     ]:
#         models = method()
#         model_name = next(iter(models.values())).base_name
#         print(f"{model_name}: {len(models)} registered")
#         for model_info in models.values():
#             print(f"  {model_info.model_path}")
#         missing_models = test_model_uploaded(list(models.keys()))

#         if missing_models:
#             print("--------------------------------")
#             print(f"Missing models: {missing_models}")
#             print("--------------------------------")
