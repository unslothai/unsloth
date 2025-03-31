from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_DEEPSEEKV3_REGISTERED = False
_IS_DEEPSEEKR1_REGISTERED = False

class DeepseekV3ModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-V{version}"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key

class DeepseekR1ModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}" if version else base_name
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key
    
# Deepseek V3 Model Meta
DeepseekV3Meta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek",
    instruct_tags=[None],
    model_version="3",
    model_sizes=[""],
    model_info_cls=DeepseekV3ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BF16],
)

DeepseekV3_0324Meta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek",
    instruct_tags=[None],
    model_version="3-0324",
    model_sizes=[""],
    model_info_cls=DeepseekV3ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.GGUF],
)

DeepseekR1Meta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1",
    instruct_tags=[None],
    model_version="",
    model_sizes=[""],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BF16, QuantType.GGUF],
)

DeepseekR1ZeroMeta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1",
    instruct_tags=[None],
    model_version="Zero",
    model_sizes=[""],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.GGUF],
)
def register_deepseek_v3_models(include_original_model: bool = False):
    global _IS_DEEPSEEKV3_REGISTERED
    if _IS_DEEPSEEKV3_REGISTERED:
        return
    _register_models(DeepseekV3Meta, include_original_model=include_original_model)
    _register_models(DeepseekV3_0324Meta, include_original_model=include_original_model)
    _IS_DEEPSEEKV3_REGISTERED = True


def register_deepseek_r1_models(include_original_model: bool = False):
    global _IS_DEEPSEEKR1_REGISTERED
    if _IS_DEEPSEEKR1_REGISTERED:
        return
    _register_models(DeepseekR1Meta, include_original_model=include_original_model)
    _register_models(DeepseekR1ZeroMeta, include_original_model=include_original_model)
    _IS_DEEPSEEKR1_REGISTERED = True

#register_deepseek_v3_models(include_original_model=True)
register_deepseek_r1_models(include_original_model=True)


if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
