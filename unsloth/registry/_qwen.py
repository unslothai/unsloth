from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_QWEN_2_5_REGISTERED = False
_IS_QWEN_2_5_VL_REGISTERED = False
_IS_QWEN_QWQ_REGISTERED = False
class QwenModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}{version}-{size}B"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)


class QwenVLModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}{version}-VL-{size}B"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)

class QwenQwQModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{size}B"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)
    
class QwenQVQPreviewModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{size}B-Preview"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)
    
# Qwen2.5 Model Meta
Qwen_2_5_Meta = ModelMeta(
    org="Qwen",
    base_name="Qwen",
    instruct_tags=[None, "Instruct"],
    model_version="2.5",
    model_sizes=["3", "7"],
    model_info_cls=QwenModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

# Qwen2.5 VL Model Meta
Qwen_2_5_VLMeta = ModelMeta(
    org="Qwen",
    base_name="Qwen",
    instruct_tags=["Instruct"],  # No base, only instruction tuned
    model_version="2.5",
    model_sizes=["3", "7", "32", "72"],
    model_info_cls=QwenVLModelInfo,
    is_multimodal=True,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

# Qwen QwQ Model Meta
QwenQwQMeta = ModelMeta(
    org="Qwen",
    base_name="QwQ",
    instruct_tags=[None],
    model_version="",
    model_sizes=["32"],
    model_info_cls=QwenQwQModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH, QuantType.GGUF],
)

# Qwen QVQ Preview Model Meta
QwenQVQPreviewMeta = ModelMeta(
    org="Qwen",
    base_name="QVQ",
    instruct_tags=[None],
    model_version="",
    model_sizes=["72"],
    model_info_cls=QwenQVQPreviewModelInfo,
    is_multimodal=True,
    quant_types=[QuantType.NONE, QuantType.BNB],
)

def register_qwen_2_5_models(include_original_model: bool = False):
    global _IS_QWEN_2_5_REGISTERED
    if _IS_QWEN_2_5_REGISTERED:
        return
    _register_models(Qwen_2_5_Meta, include_original_model=include_original_model)
    _IS_QWEN_2_5_REGISTERED = True

def register_qwen_2_5_vl_models(include_original_model: bool = False):
    global _IS_QWEN_2_5_VL_REGISTERED
    if _IS_QWEN_2_5_VL_REGISTERED:
        return
    _register_models(Qwen_2_5_VLMeta, include_original_model=include_original_model)
    _IS_QWEN_2_5_VL_REGISTERED = True

def register_qwen_qwq_models(include_original_model: bool = False):
    global _IS_QWEN_QWQ_REGISTERED
    if _IS_QWEN_QWQ_REGISTERED:
        return
    _register_models(QwenQwQMeta, include_original_model=include_original_model)
    _register_models(QwenQVQPreviewMeta, include_original_model=include_original_model)
    _IS_QWEN_QWQ_REGISTERED = True

def register_qwen_models(include_original_model: bool = False):
    register_qwen_2_5_models(include_original_model=include_original_model)
    register_qwen_2_5_vl_models(include_original_model=include_original_model)
    register_qwen_qwq_models(include_original_model=include_original_model)

if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    MODEL_REGISTRY.clear()
    
    register_qwen_models(include_original_model=True)
    
    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
