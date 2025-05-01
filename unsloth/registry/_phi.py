from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_PHI_4_REGISTERED = False
_IS_PHI_4_INSTRUCT_REGISTERED = False

class PhiModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)

# Phi Model Meta
PhiMeta4 = ModelMeta(
    org="microsoft",
    base_name="phi",
    instruct_tags=[None],
    model_version="4",
    model_sizes=["1"],  # Assuming only one size
    model_info_cls=PhiModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

# Phi Instruct Model Meta
PhiInstructMeta4 = ModelMeta(
    org="microsoft",
    base_name="phi",
    instruct_tags=["mini-instruct"],
    model_version="4",
    model_sizes=["1"],  # Assuming only one size
    model_info_cls=PhiModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH, QuantType.GGUF],
)

def register_phi_4_models(include_original_model: bool = False):
    global _IS_PHI_4_REGISTERED
    if _IS_PHI_4_REGISTERED:
        return
    _register_models(PhiMeta4, include_original_model=include_original_model)
    _IS_PHI_4_REGISTERED = True

def register_phi_4_instruct_models(include_original_model: bool = False):
    global _IS_PHI_4_INSTRUCT_REGISTERED
    if _IS_PHI_4_INSTRUCT_REGISTERED:
        return
    _register_models(PhiInstructMeta4, include_original_model=include_original_model)
    _IS_PHI_4_INSTRUCT_REGISTERED = True

def register_phi_models(include_original_model: bool = False):
    register_phi_4_models(include_original_model=include_original_model)
    register_phi_4_instruct_models(include_original_model=include_original_model)

if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    MODEL_REGISTRY.clear()
    
    register_phi_models(include_original_model=True)
    
    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}") 