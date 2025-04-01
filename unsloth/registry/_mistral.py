import copy

from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_MISTRAL_SMALL_REGISTERED = False

_MISTRAL_SMALL_03_25_VERSION = "2503"
_MISTRAL_SMALL_01_25_VERSION = "2501"
_MISTRAL_SMALL_09_24_VERSION = "2409" # Not uploaded to unsloth

class MistralSmallModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        if version == _MISTRAL_SMALL_03_25_VERSION:
            key = f"{base_name}-3.1-{size}B-{instruct_tag}"
        else:
            key = f"{base_name}-{size}B-{instruct_tag}"
        key += f"-{version}"
        key = cls.append_quant_type(key, quant_type)
        
        return key


MistralSmall_2503_Base_Meta = ModelMeta(
    org="mistralai",
    base_name="Mistral-Small",
    instruct_tags=["Base"],
    model_version=_MISTRAL_SMALL_03_25_VERSION,
    model_sizes=["24"],
    model_info_cls=MistralSmallModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.UNSLOTH, QuantType.BNB],
)

MistralSmall_2503_Instruct_Meta = copy.deepcopy(MistralSmall_2503_Base_Meta)
MistralSmall_2503_Instruct_Meta.instruct_tags = ["Instruct"]
MistralSmall_2503_Instruct_Meta.quant_types = [QuantType.NONE, QuantType.UNSLOTH, QuantType.BNB, QuantType.GGUF]

MistralSmall_2501_Base_Meta = copy.deepcopy(MistralSmall_2503_Base_Meta)
MistralSmall_2501_Base_Meta.model_version = _MISTRAL_SMALL_01_25_VERSION

MistralSmall_2501_Instruct_Meta = copy.deepcopy(MistralSmall_2503_Instruct_Meta)
MistralSmall_2501_Instruct_Meta.model_version = _MISTRAL_SMALL_01_25_VERSION

def register_mistral_small_models(include_original_model: bool = False):
    global _IS_MISTRAL_SMALL_REGISTERED
    if _IS_MISTRAL_SMALL_REGISTERED:
        return
    _register_models(MistralSmall_2503_Base_Meta, include_original_model=include_original_model)
    _register_models(MistralSmall_2503_Instruct_Meta, include_original_model=include_original_model)
    _register_models(MistralSmall_2501_Base_Meta, include_original_model=include_original_model)
    _register_models(MistralSmall_2501_Instruct_Meta, include_original_model=include_original_model)

    _IS_MISTRAL_SMALL_REGISTERED = True

def register_mistral_models(include_original_model: bool = False):
    register_mistral_small_models(include_original_model=include_original_model)

if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    MODEL_REGISTRY.clear()
    
    register_mistral_models(include_original_model=True)
    
    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")    