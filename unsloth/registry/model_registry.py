from functools import partial
from typing import Callable, Literal

from unsloth.registry._llama import LlamaMeta3_1, LlamaMeta3_2
from unsloth.registry.common import ModelInfo, ModelMeta

# _IS_LLAMA_REGISTERED = False
# _IS_LLAMA_VISION_REGISTERED = False

# _IS_QWEN_REGISTERED = False
# _IS_QWEN_VL_REGISTERED = False

_IS_GEMMA_REGISTERED = False

_IS_PHI_REGISTERED = False
_IS_PHI_INSTRUCT_REGISTERED = False



# class PhiModelInfo(ModelInfo):
#     @classmethod
#     def construct_model_name(
#         cls, base_name, version, size, quant_type, instruct_tag
#     ):
#         key = f"{base_name}-{version}"
#         key = cls.append_instruct_tag(key, instruct_tag)
#         key = cls.append_quant_type(key, quant_type)
#         return key





# # Qwen text only models
# # NOTE: Qwen vision models will be registered separately

# _PHI_INFO = {
#     "org": "microsoft",
#     "base_name": "phi",
#     "model_versions": ["4"],
#     "model_sizes": {"4": [None]},  # -1 means only 1 size
#     "instruct_tags": [None],
#     "is_multimodal": False,
#     "model_info_cls": PhiModelInfo,
# }

# _PHI_INSTRUCT_INFO = {
#     "org": "microsoft",
#     "base_name": "Phi",
#     "model_versions": ["4"],
#     "model_sizes": {"4": [None]},  # -1 means only 1 size
#     "instruct_tags": ["mini-instruct"],
#     "is_multimodal": False,
#     "model_info_cls": PhiModelInfo,
# }


MODEL_REGISTRY: dict[str, ModelInfo] = {}


def register_model(
    model_info_cls: ModelInfo,
    org: str,
    base_name: str,
    version: str,
    size: int,
    instruct_tag: str = None,
    quant_type: Literal["bnb", "unsloth"] = None,
    is_multimodal: bool = False,
    name: str = None,
):
    name = name or model_info_cls.construct_model_name(
        base_name=base_name,
        version=version,
        size=size,
        quant_type=quant_type,
        instruct_tag=instruct_tag,
    )
    key = f"{org}/{name}"

    if key in MODEL_REGISTRY:
        raise ValueError(f"Model {key} already registered")

    MODEL_REGISTRY[key] = model_info_cls(
        org=org,
        base_name=base_name,
        version=version,
        size=size,
        is_multimodal=is_multimodal,
        instruct_tag=instruct_tag,
        quant_type=quant_type,
        name=name,
    )


# def _register_models(model_info: dict):
#     org = model_info["org"]
#     base_name = model_info["base_name"]
#     instruct_tags = model_info["instruct_tags"]
#     model_versions = model_info["model_versions"]
#     model_sizes = model_info["model_sizes"]
#     is_multimodal = model_info["is_multimodal"]
#     model_info_cls = model_info["model_info_cls"]

#     for version in model_versions:
#         for size in model_sizes[version]:
#             for instruct_tag in instruct_tags:
#                 for quant_type in QUANT_TYPES:
#                     _org = "unsloth" if quant_type is not None else org
#                     register_model(
#                         model_info_cls=model_info_cls,
#                         org=_org,
#                         base_name=base_name,
#                         version=version,
#                         size=size,
#                         instruct_tag=instruct_tag,
#                         quant_type=quant_type,
#                         is_multimodal=is_multimodal,
#                     )


def _register_models(model_meta: ModelMeta):
    org = model_meta.org
    base_name = model_meta.base_name
    instruct_tags = model_meta.instruct_tags
    model_version = model_meta.model_version
    model_sizes = model_meta.model_sizes
    is_multimodal = model_meta.is_multimodal
    quant_types = model_meta.quant_types
    model_info_cls = model_meta.model_info_cls

    for size in model_sizes:
        for instruct_tag in instruct_tags:
            for quant_type in quant_types:
                _org = "unsloth" if quant_type is not None else org
                register_model(
                    model_info_cls=model_info_cls,
                    org=_org,
                    base_name=base_name,
                    version=model_version,
                    size=size,
                    instruct_tag=instruct_tag,
                    quant_type=quant_type,
                    is_multimodal=is_multimodal,
                )

def register_llama_models():
    global _IS_LLAMA_REGISTERED
    if _IS_LLAMA_REGISTERED:
        return
    _register_models(LlamaMeta3_1)
    _register_models(LlamaMeta3_2)
    _IS_LLAMA_REGISTERED = True


def register_llama_vision_models():
    global _IS_LLAMA_VISION_REGISTERED
    if _IS_LLAMA_VISION_REGISTERED:
        return
    _register_models(_LLAMA_VISION_INFO)
    _IS_LLAMA_VISION_REGISTERED = True


def register_qwen_models():
    global _IS_QWEN_REGISTERED
    if _IS_QWEN_REGISTERED:
        return

    _register_models(_QWEN_INFO)
    _IS_QWEN_REGISTERED = True


def register_qwen_vl_models():
    global _IS_QWEN_VL_REGISTERED
    if _IS_QWEN_VL_REGISTERED:
        return

    _register_models(_QWEN_VL_INFO)
    _IS_QWEN_VL_REGISTERED = True


def register_gemma_models():
    global _IS_GEMMA_REGISTERED
    _register_models(_GEMMA_INFO)
    _IS_GEMMA_REGISTERED = True


def register_phi_models():
    global _IS_PHI_REGISTERED
    if _IS_PHI_REGISTERED:
        return
    _register_models(_PHI_INFO)
    _IS_PHI_REGISTERED = True


def register_phi_instruct_models():
    global _IS_PHI_INSTRUCT_REGISTERED
    if _IS_PHI_INSTRUCT_REGISTERED:
        return

    _register_models(_PHI_INSTRUCT_INFO)
    _IS_PHI_INSTRUCT_REGISTERED = True


def _base_name_filter(model_info: ModelInfo, base_name: str):
    return model_info.base_name == base_name


def _get_models(filter_func: Callable[[ModelInfo], bool] = _base_name_filter):
    return {k: v for k, v in MODEL_REGISTRY.items() if filter_func(v)}


def get_llama_models(version: str = None):
    if not _IS_LLAMA_REGISTERED:
        register_llama_models()

    llama_models: dict[str, ModelInfo] = _get_models(
        partial(_base_name_filter, base_name=LlamaMeta3_1.base_name)
    )
    if version is not None:
        llama_models = {
            k: v for k, v in llama_models.items() if v.version == version
        }
    return llama_models


def get_llama_vision_models():
    if not _IS_LLAMA_VISION_REGISTERED:
        register_llama_vision_models()

    return _get_models(
        lambda model_info: model_info.base_name
        == _LLAMA_VISION_INFO["base_name"]
        and model_info.is_multimodal
    )


def get_qwen_models():
    if not _IS_QWEN_REGISTERED:
        register_qwen_models()

    return _get_models(
        lambda model_info: model_info.base_name == _QWEN_INFO["base_name"]
    )


def get_qwen_vl_models():
    if not _IS_QWEN_VL_REGISTERED:
        register_qwen_vl_models()
    return _get_models(
        lambda model_info: model_info.base_name == _QWEN_VL_INFO["base_name"]
    )


def get_gemma_models():
    if not _IS_GEMMA_REGISTERED:
        register_gemma_models()

    return _get_models(
        lambda model_info: model_info.base_name == _GEMMA_INFO["base_name"]
    )


def get_phi_models():
    if not _IS_PHI_REGISTERED:
        register_phi_models()
    return _get_models(
        lambda model_info: model_info.base_name == _PHI_INFO["base_name"]
    )


def get_phi_instruct_models():
    if not _IS_PHI_INSTRUCT_REGISTERED:
        register_phi_instruct_models()
    return _get_models(
        lambda model_info: model_info.base_name
        == _PHI_INSTRUCT_INFO["base_name"]
    )


if __name__ == "__main__":
    from huggingface_hub import HfApi

    api = HfApi()

    def get_model_info(
        model_id: str, properties: list[str] = None
    ) -> ModelInfo:
        try:
            model_info: ModelInfo = api.model_info(model_id, expand=properties)
        except Exception as e:
            print(f"Error getting model info for {model_id}: {e}")
            model_info = None
        return model_info

    register_llama_models()

    llama3_1_models = get_llama_models(version="3.2")
    missing_models = []
    for k, v in llama3_1_models.items():
        model_info = get_model_info(v.model_path)
        if model_info is None:
            # print unicode cross mark followed by model k
            print(f"\u2718 {k}")
            missing_models.append(k)
    
    if len(missing_models) == 0:
        # print unicode checkmark
        print(f"\u2713 All models found!")