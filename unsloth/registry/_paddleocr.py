from unsloth.registry.registry import ModelInfo, ModelMeta, QuantType, _register_models

_IS_PADDLEOCR_REGISTERED = False


class PaddleOCRModelInfo(ModelInfo):
    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag):
        key = f"{base_name}-{version}"
        return super().construct_model_name(base_name, version, size, quant_type, instruct_tag, key)


# PaddleOCR-VL Model Meta
PaddleOCRVLMeta = ModelMeta(
    org="PaddlePaddle",
    base_name="PaddleOCR",
    instruct_tags=[None],
    model_version="VL-1.6",
    model_sizes=["0_9"],   # ~0.9B parameters (hidden=1024, 18 layers)
    model_info_cls=PaddleOCRModelInfo,
    is_multimodal=True,
    quant_types=[QuantType.NONE, QuantType.BNB],
)


def register_paddleocr_models(include_original_model: bool = False):
    global _IS_PADDLEOCR_REGISTERED
    if _IS_PADDLEOCR_REGISTERED:
        return
    _register_models(PaddleOCRVLMeta, include_original_model=include_original_model)
    _IS_PADDLEOCR_REGISTERED = True


if __name__ == "__main__":
    from unsloth.registry.registry import MODEL_REGISTRY, _check_model_info

    MODEL_REGISTRY.clear()

    register_paddleocr_models(include_original_model=True)

    for model_id, model_info in MODEL_REGISTRY.items():
        model_info = _check_model_info(model_id)
        if model_info is None:
            print(f"\u2718 {model_id}")
        else:
            print(f"\u2713 {model_id}")
