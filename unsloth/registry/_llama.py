from typing import Optional
from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_LLAMA_3_1_REGISTERED = False
_IS_LLAMA_3_2_REGISTERED = False
_IS_LLAMA_3_2_VISION_REGISTERED = False


class LlamaModelInfo(ModelInfo):
    """
    A class that represents information about a Llama model. This class is used to construct model names based on various parameters such as base name, version, size, quantization type, and instruction tag.
    """
    @classmethod
    def construct_model_name(cls, base_name: str, version: str, size: str, quant_type: QuantType, instruct_tag: Optional[str]) -> str:
        """
        Constructs a model name based on the provided parameters.
        
        Args:
            base_name (`str`): The base name of the model.
            version (`str`): The version of the model.
            size (`str`): The size of the model.
            quant_type (`QuantType`): The quantization type of the model.
            instruct_tag (`Optional[str]`): The instruction tag of the model.
        
        Returns:
            `str`: The constructed model name.
        """
        key = f"{base_name}-{version}-{size}B"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)


class LlamaVisionModelInfo(ModelInfo):
    """
    A class that represents information about a Llama vision model. This class is used to construct model names for vision models, which include a 'Vision' suffix.
    """
    @classmethod
    def construct_model_name(cls, base_name: str, version: str, size: str, quant_type: QuantType, instruct_tag: Optional[str]) -> str:
        """
        Constructs a model name for a vision model based on the provided parameters.
        
        Args:
            base_name (`str`): The base name of the model.
            version (`str`): The version of the model.
            size (`str`): The size of the model.
            quant_type (`QuantType`): The quantization type of the model.
            instruct_tag (`Optional[str]`): The instruction tag of the model.
        
        Returns:
            `str`: The constructed model name with a 'Vision' suffix.
        """
        key = f"{base_name}-{version}-{size}B-Vision"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)


# Llama 3.1
LlamaMeta_3_1 = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    instruct_tags=[None, "Instruct"],
    model_version="3.1",
    model_sizes=["8"],
    model_info_cls=LlamaModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

# Llama 3.2 Base Models
LlamaMeta_3_2_Base = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    instruct_tags=[None],
    model_version="3.2",
    model_sizes=["1", "3"],
    model_info_cls=LlamaModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

# Llama 3.2 Instruction Tuned Models
LlamaMeta_3_2_Instruct = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    instruct_tags=["Instruct"],
    model_version="3.2",
    model_sizes=["1", "3"],
    model_info_cls=LlamaModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH, QuantType.GGUF],
)

# Llama 3.2 Vision
LlamaMeta_3_2_Vision = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    instruct_tags=[None, "Instruct"],
    model_version="3.2",
    model_sizes=["11", "90"],
    model_info_cls=LlamaVisionModelInfo,
    is_multimodal=True,
    quant_types={
        "11": [QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
        "90": [QuantType.NONE],
    },
)


def register_llama_3_1_models(include_original_model: bool = False) -> None:
    """
    Registers Llama 3.1 models in the model registry. This function ensures that the models are only registered once.
    
    Args:
        include_original_model (`bool`, optional): Whether to include the original model in the registration. Defaults to False.
    """
    global _IS_LLAMA_3_1_REGISTERED
    if _IS_LLAMA_3_1_REGISTERED:
        return
    _register_models(LlamaMeta_3_1, include_original_model=include_original_model)
    _IS_LLAMA_3_1_REGISTERED = True

def register_llama_3_2_models(include_original_model: bool = False) -> None:
    """
    Registers Llama 3.2 models in the model registry. This function ensures that the models are only registered once.
    
    Args:
        include_original_model (`bool`, optional): Whether to include the original model in the registration. Defaults to False.
    """
    global _IS_LLAMA_3_2_REGISTERED
    if _IS_LLAMA_3_2_REGISTERED:
        return
    _register_models(LlamaMeta_3_2_Base, include_original_model=include_original_model)
    _register_models(LlamaMeta_3_2_Instruct, include_original_model=include_original_model)
    _IS_LLAMA_3_2_REGISTERED = True

def register_llama_3_2_vision_models(include_original_model: bool = False) -> None:
    """
    Registers Llama 3.2 vision models in the model registry. This function ensures that the models are only registered once.
    
    Args:
        include_original_model (`bool`, optional): Whether to include the original model in the registration. Defaults to False.
    """
    global _IS_LLAMA_3_2_VISION_REGISTERED
    if _IS_LLAMA_3_2_VISION_REGISTERED:
        return
    _register_models(LlamaMeta_3_2_Vision, include_original_model=include_original_model)
    _IS_LLAMA_3_2_VISION_REGISTERED = True


def register_llama_models(include_original_model: bool = False) -> None:
    """
    Registers all Llama models (3.1, 3.2, and 3.2 vision) in the model registry.
    
    Args:
        include_original_model (`bool`, optional): Whether to include the original model in the registration. Defaults to False.
    """
    register_llama_3_1_models(include_original_model=include_original_model)
    register_llama_3_2_models(include_original_model=include_original_model)
    register_llama_3_2_vision_models(include_original_model=include_original_model)

if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    MODEL_REGISTRY.clear()

    register_llama_models(include_original_model=True)

    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
