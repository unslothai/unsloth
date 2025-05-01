import warnings
from dataclasses import dataclass, field
from enum import Enum


class QuantType(Enum):
    BNB = "bnb"
    UNSLOTH = "unsloth" # dynamic 4-bit quantization
    GGUF = "GGUF"
    NONE = "none"
    BF16 = "bf16" # only for Deepseek V3

# Tags for Hugging Face model paths
BNB_QUANTIZED_TAG = "bnb-4bit"
UNSLOTH_DYNAMIC_QUANT_TAG = "unsloth" + "-" + BNB_QUANTIZED_TAG
GGUF_TAG = "GGUF"
BF16_TAG = "bf16"

QUANT_TAG_MAP = {
    QuantType.BNB: BNB_QUANTIZED_TAG,
    QuantType.UNSLOTH: UNSLOTH_DYNAMIC_QUANT_TAG,
    QuantType.GGUF: GGUF_TAG,
    QuantType.NONE: None,
    QuantType.BF16: BF16_TAG,
} 

# NOTE: models registered with org="unsloth" and QUANT_TYPE.NONE are aliases of QUANT_TYPE.UNSLOTH
@dataclass
class ModelInfo:
    org: str
    base_name: str
    version: str
    size: int
    name: str = None  # full model name, constructed from base_name, version, and size unless provided
    is_multimodal: bool = False
    instruct_tag: str = None
    quant_type: QuantType = None
    description: str = None

    def __post_init__(self):
        self.name = self.name or self.construct_model_name(
            self.base_name,
            self.version,
            self.size,
            self.quant_type,
            self.instruct_tag,
        )

    @staticmethod
    def append_instruct_tag(key: str, instruct_tag: str = None):
        if instruct_tag:
            key = "-".join([key, instruct_tag])
        return key

    @staticmethod
    def append_quant_type(
        key: str, quant_type: QuantType = None
    ):
        if quant_type != QuantType.NONE:
            key = "-".join([key, QUANT_TAG_MAP[quant_type]])
        return key

    @classmethod
    def construct_model_name(cls, base_name, version, size, quant_type, instruct_tag, key=""):
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key

    @property
    def model_path(
        self,
    ) -> str:
        return f"{self.org}/{self.name}"


@dataclass
class ModelMeta:
    org: str
    base_name: str
    model_version: str
    model_info_cls: type[ModelInfo]
    model_sizes: list[str] = field(default_factory=list)
    instruct_tags: list[str] = field(default_factory=list)
    quant_types: list[QuantType] | dict[str, list[QuantType]] = field(default_factory=list)
    is_multimodal: bool = False


MODEL_REGISTRY: dict[str, ModelInfo] = {}


def register_model(
    model_info_cls: ModelInfo,
    org: str,
    base_name: str,
    version: str,
    size: int,
    instruct_tag: str = None,
    quant_type: QuantType = None,
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
        raise ValueError(f"Model {key} already registered, current keys: {MODEL_REGISTRY.keys()}")

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


def _check_model_info(model_id: str, properties: list[str] = ["lastModified"]):
    from huggingface_hub import HfApi
    from huggingface_hub import ModelInfo as HfModelInfo
    from huggingface_hub.utils import RepositoryNotFoundError

    api = HfApi()

    try:
        model_info: HfModelInfo = api.model_info(model_id, expand=properties)
    except Exception as e:
        if isinstance(e, RepositoryNotFoundError):
            warnings.warn(f"{model_id} not found on Hugging Face")
            model_info = None
        else:
            raise e
    return model_info


def _register_models(model_meta: ModelMeta, include_original_model: bool = False):
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
            # Handle quant types per model size
            if isinstance(quant_types, dict):
                _quant_types = quant_types[size]
            else:
                _quant_types = quant_types
            for quant_type in _quant_types:
                # NOTE: models registered with org="unsloth" and QUANT_TYPE.NONE are aliases of QUANT_TYPE.UNSLOTH
                _org = "unsloth" # unsloth models -- these are all quantized versions of the original model
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
            # include original model from releasing organization
            if include_original_model:
                register_model(
                    model_info_cls=model_info_cls,
                    org=org,
                    base_name=base_name,
                    version=model_version,
                    size=size,
                    instruct_tag=instruct_tag,
                    quant_type=QuantType.NONE,
                    is_multimodal=is_multimodal,
                )
