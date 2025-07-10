from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_GEMMA_3_BASE_REGISTERED = False
_IS_GEMMA_3_INSTRUCT_REGISTERED = False

class GemmaModelInfo(ModelInfo):
    """
    A class that provides information about Gemma models. This class is used to construct model names based on various parameters such as base name, version, size, quantization type, and instruction tag.
    """
    @classmethod
    def construct_model_name(cls, base_name: str, version: str, size: str, quant_type: QuantType, instruct_tag: str) -> str:
        """
        Constructs a model name based on the provided parameters.
        
        Args:
            base_name (str): The base name of the model.
            version (str): The version of the model.
            size (str): The size of the model.
            quant_type (QuantType): The quantization type of the model.
            instruct_tag (str): The instruction tag of the model.
            key (str): The key used for model name construction.
        
        Returns:
            str: The constructed model name.
        """
        key = f"{base_name}-{version}-{size}B"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)

# Gemma3 Base Model Meta
GemmaMeta3Base = ModelMeta(
    org="google",
    base_name="gemma",
    instruct_tags=["pt"],  # pt = base
    model_version="3",
    model_sizes=["1", "4", "12", "27"],
    model_info_cls=GemmaModelInfo,
    is_multimodal=True,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

# Gemma3 Instruct Model Meta
GemmaMeta3Instruct = ModelMeta(
    org="google",
    base_name="gemma",
    instruct_tags=["it"],  # it = instruction tuned
    model_version="3",
    model_sizes=["1", "4", "12", "27"],
    model_info_cls=GemmaModelInfo,
    is_multimodal=True,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH, QuantType.GGUF],
)

def register_gemma_3_base_models(include_original_model: bool = False) -> None:
    """
    Registers Gemma 3 base models in the model registry. This function ensures that the models are only registered once.
    
    Args:
        include_original_model (bool, optional): Whether to include the original model in the registration. Defaults to False.
    """
    global _IS_GEMMA_3_BASE_REGISTERED
    if _IS_GEMMA_3_BASE_REGISTERED:
        return
    _register_models(GemmaMeta3Base, include_original_model=include_original_model)
    _IS_GEMMA_3_BASE_REGISTERED = True

def register_gemma_3_instruct_models(include_original_model: bool = False) -> None:
    """
    Registers Gemma 3 instruction-tuned models in the model registry. This function ensures that the models are only registered once.
    
    Args:
        include_original_model (bool, optional): Whether to include the original model in the registration. Defaults to False.
    """
    global _IS_GEMMA_3_INSTRUCT_REGISTERED
    if _IS_GEMMA_3_INSTRUCT_REGISTERED:
        return
    _register_models(GemmaMeta3Instruct, include_original_model=include_original_model)
    _IS_GEMMA_3_INSTRUCT_REGISTERED = True

def register_gemma_models(include_original_model: bool = False) -> None:
    """
    Registers all Gemma models (both base and instruction-tuned) in the model registry.
    
    Args:
        include_original_model (bool, optional): Whether to include the original model in the registration. Defaults to False.
    """
    register_gemma_3_base_models(include_original_model=include_original_model)
    register_gemma_3_instruct_models(include_original_model=include_original_model)


if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    MODEL_REGISTRY.clear()
    
    register_gemma_models(include_original_model=True)
    
    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
