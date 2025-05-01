from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_DEEPSEEK_V3_REGISTERED = False
_IS_DEEPSEEK_V3_0324_REGISTERED = False
_IS_DEEPSEEK_R1_REGISTERED = False
_IS_DEEPSEEK_R1_ZERO_REGISTERED = False
_IS_DEEPSEEK_R1_DISTILL_LLAMA_REGISTERED = False
_IS_DEEPSEEK_R1_DISTILL_QWEN_REGISTERED = False

class DeepseekV3ModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-V{version}"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)

class DeepseekR1ModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}" if version else base_name
        if size:
            key = f"{key}-{size}B"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)
    
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

DeepseekR1DistillLlamaMeta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1-Distill",
    instruct_tags=[None],
    model_version="Llama",
    model_sizes=["8", "70"],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types={"8": [QuantType.UNSLOTH, QuantType.GGUF], "70": [QuantType.GGUF]},
)

# Deepseek R1 Distill Qwen Model Meta
DeepseekR1DistillQwenMeta = ModelMeta(
    org="deepseek-ai",
    base_name="DeepSeek-R1-Distill",
    instruct_tags=[None],
    model_version="Qwen",
    model_sizes=["1.5", "7", "14", "32"],
    model_info_cls=DeepseekR1ModelInfo,
    is_multimodal=False,
    quant_types={
        "1.5": [QuantType.UNSLOTH, QuantType.BNB, QuantType.GGUF],
        "7": [QuantType.UNSLOTH, QuantType.BNB],
        "14": [QuantType.UNSLOTH, QuantType.BNB, QuantType.GGUF],
        "32": [QuantType.GGUF, QuantType.BNB],
    },
)
        
def register_deepseek_v3_models(include_original_model: bool = False):
    global _IS_DEEPSEEK_V3_REGISTERED
    if _IS_DEEPSEEK_V3_REGISTERED:
        return
    _register_models(DeepseekV3Meta, include_original_model=include_original_model)
    _IS_DEEPSEEK_V3_REGISTERED = True

def register_deepseek_v3_0324_models(include_original_model: bool = False):
    global _IS_DEEPSEEK_V3_0324_REGISTERED
    if _IS_DEEPSEEK_V3_0324_REGISTERED:
        return
    _register_models(DeepseekV3_0324Meta, include_original_model=include_original_model)
    _IS_DEEPSEEK_V3_0324_REGISTERED = True

def register_deepseek_r1_models(include_original_model: bool = False):
    global _IS_DEEPSEEK_R1_REGISTERED
    if _IS_DEEPSEEK_R1_REGISTERED:
        return
    _register_models(DeepseekR1Meta, include_original_model=include_original_model)
    _IS_DEEPSEEK_R1_REGISTERED = True

def register_deepseek_r1_zero_models(include_original_model: bool = False):
    global _IS_DEEPSEEK_R1_ZERO_REGISTERED
    if _IS_DEEPSEEK_R1_ZERO_REGISTERED:
        return
    _register_models(DeepseekR1ZeroMeta, include_original_model=include_original_model)
    _IS_DEEPSEEK_R1_ZERO_REGISTERED = True

def register_deepseek_r1_distill_llama_models(include_original_model: bool = False):
    global _IS_DEEPSEEK_R1_DISTILL_LLAMA_REGISTERED
    if _IS_DEEPSEEK_R1_DISTILL_LLAMA_REGISTERED:
        return
    _register_models(DeepseekR1DistillLlamaMeta, include_original_model=include_original_model)
    _IS_DEEPSEEK_R1_DISTILL_LLAMA_REGISTERED = True

def register_deepseek_r1_distill_qwen_models(include_original_model: bool = False):
    global _IS_DEEPSEEK_R1_DISTILL_QWEN_REGISTERED
    if _IS_DEEPSEEK_R1_DISTILL_QWEN_REGISTERED:
        return
    _register_models(DeepseekR1DistillQwenMeta, include_original_model=include_original_model)
    _IS_DEEPSEEK_R1_DISTILL_QWEN_REGISTERED = True

def register_deepseek_models(include_original_model: bool = False):
    register_deepseek_v3_models(include_original_model=include_original_model)
    register_deepseek_v3_0324_models(include_original_model=include_original_model)
    register_deepseek_r1_models(include_original_model=include_original_model)
    register_deepseek_r1_zero_models(include_original_model=include_original_model)
    register_deepseek_r1_distill_llama_models(include_original_model=include_original_model)
    register_deepseek_r1_distill_qwen_models(include_original_model=include_original_model)

def _list_deepseek_r1_distill_models():
    from unsloth.utils.hf_hub import ModelInfo as HfModelInfo
    from unsloth.utils.hf_hub import list_models
    models: list[HfModelInfo] = list_models(author="unsloth", search="Distill", limit=1000)
    distill_models = []
    for model in models:
        model_id = model.id
        model_name = model_id.split("/")[-1]
        # parse out only the version
        version = model_name.removeprefix("DeepSeek-R1-Distill-")
        distill_models.append(version)

    return distill_models


register_deepseek_models(include_original_model=True)

if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    MODEL_REGISTRY.clear()
    
    register_deepseek_models(include_original_model=True)
    
    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
    # distill_models = _list_deepseek_r1_distill_models()
    # for model in sorted(distill_models):
    #     if "qwen" in model.lower():
    #         print(model)