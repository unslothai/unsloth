from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_QWEN_REGISTERED = False
_IS_QWEN_VL_REGISTERED = False

class QwenModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(
        cls, base_name, version, size, quant_type, instruct_tag
    ):
        key = f"{base_name}{version}-{size}B"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key


class QwenVLModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(
        cls, base_name, version, size, quant_type, instruct_tag
    ):
        key = f"{base_name}{version}-VL-{size}B"
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key


# Qwen Model Meta
QwenMeta = ModelMeta(
    org="Qwen",
    base_name="Qwen",
    instruct_tags=[None, "Instruct"],
    model_version="2.5",
    model_sizes=[3, 7],
    model_info_cls=QwenModelInfo,
    is_multimodal=False,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

# Qwen VL Model Meta
QwenVLMeta = ModelMeta(
    org="Qwen",
    base_name="Qwen",
    instruct_tags=["Instruct"],  # No base, only instruction tuned
    model_version="2.5",
    model_sizes=[3, 7, 32, 72],
    model_info_cls=QwenVLModelInfo,
    is_multimodal=True,
    quant_types=[QuantType.NONE, QuantType.BNB, QuantType.UNSLOTH],
)

def register_qwen_models():
    global _IS_QWEN_REGISTERED
    if _IS_QWEN_REGISTERED:
        return
    _register_models(QwenMeta)
    _IS_QWEN_REGISTERED = True

def register_qwen_vl_models():
    global _IS_QWEN_VL_REGISTERED
    if _IS_QWEN_VL_REGISTERED:
        return
    _register_models(QwenVLMeta)
    _IS_QWEN_VL_REGISTERED = True

register_qwen_models()
register_qwen_vl_models()


if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info
    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
