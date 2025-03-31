from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_LLAMA_REGISTERED = False
_IS_LLAMA_VISION_REGISTERED = False


class LlamaModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}-{size}B"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key


class LlamaVisionModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}-{size}B-Vision"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key


# Llama 3.1
LlamaMeta3_1 = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    instruct_tags=[None, "Instruct"],
    model_version="3.1",
    model_sizes=["8"],
    model_info_cls=LlamaModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

# Llama 3.2
LlamaMeta3_2 = ModelMeta(
    org="meta-llama",
    base_name="Llama",
    instruct_tags=[None, "Instruct"],
    model_version="3.2",
    model_sizes=["1", "3"],
    model_info_cls=LlamaModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH, QuantType.GGUF],
)

# Llama 3.2 Vision
LlamaMeta3_2_Vision = ModelMeta(
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


def register_llama_models(include_original_model: bool = False):
    global _IS_LLAMA_REGISTERED
    if _IS_LLAMA_REGISTERED:
        return
    _register_models(LlamaMeta3_1, include_original_model=include_original_model)
    _register_models(LlamaMeta3_2, include_original_model=include_original_model)
    _IS_LLAMA_REGISTERED = True


def register_llama_vision_models(include_original_model: bool = False):
    global _IS_LLAMA_VISION_REGISTERED
    if _IS_LLAMA_VISION_REGISTERED:
        return
    _register_models(LlamaMeta3_2_Vision, include_original_model=include_original_model)
    _IS_LLAMA_VISION_REGISTERED = True


# register_llama_models(include_original_model=True)
register_llama_vision_models(include_original_model=True)

if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info

    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
