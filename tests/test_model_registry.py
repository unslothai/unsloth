from dataclasses import dataclass

import pytest
from huggingface_hub import ModelInfo as HfModelInfo
from unsloth.model_registry import (
    ModelInfo,
    get_llama_models,
    get_llama_vision_models,
    get_phi_instruct_models,
    get_phi_models,
    get_qwen_models,
    get_qwen_vl_models,
)
from unsloth.utils.hf_hub import get_model_info

MODEL_NAMES = [
    "llama",
    "llama_vision",
    "qwen",
    "qwen_vl",
    "phi",
    "phi_instruct",
]
REGISTERED_MODELS = [
    get_llama_models(),
    get_llama_vision_models(),
    get_qwen_models(),
    get_qwen_vl_models(),
    get_phi_models(),
    get_phi_instruct_models(),
]


@dataclass
class ModelTestParam:
    name: str
    models: dict[str, ModelInfo]


def _test_model_uploaded(model_ids: list[str]):
    missing_models = []
    for _id in model_ids:
        model_info: HfModelInfo = get_model_info(_id)
        if not model_info:
            missing_models.append(_id)

    return missing_models


TestParams = [
    ModelTestParam(name, models)
    for name, models in zip(MODEL_NAMES, REGISTERED_MODELS)
]


@pytest.mark.parametrize(
    "model_test_param", TestParams, ids=lambda param: param.name
)
def test_model_uploaded(model_test_param: ModelTestParam):
    missing_models = _test_model_uploaded(model_test_param.models)
    assert not missing_models, (
        f"{model_test_param.name} missing following models: {missing_models}"
    )


if __name__ == "__main__":
    for method in [
        get_llama_models,
        get_llama_vision_models,
        get_qwen_models,
        get_qwen_vl_models,
        get_phi_models,
        get_phi_instruct_models,
    ]:
        models = method()
        model_name = next(iter(models.values())).base_name
        print(f"{model_name}: {len(models)} registered")
        for model_info in models.values():
            print(f"  {model_info.model_path}")
        missing_models = test_model_uploaded(list(models.keys()))

        if missing_models:
            print("--------------------------------")
            print(f"Missing models: {missing_models}")
            print("--------------------------------")
