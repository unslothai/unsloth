from typing import Optional, Any
import warnings
from dataclasses import dataclass, field
from enum import Enum


class QuantType(Enum):
    """
    Enumeration of supported quantization types for models.
    
    Attributes:
        BNB: str = 'bnb' - 4-bit quantization using bitsandbytes
        UNSLOTH: str = 'unsloth' - dynamic 4-bit quantization
        GGUF: str = 'GGUF' - GGUF format quantization
        NONE: str = 'none' - no quantization
        BF16: str = 'bf16' - BFloat16 precision (used for Deepseek V3)
    """
    BNB: str  = "bnb"

    UNSLOTH: str = "unsloth" # dynamic 4-bit quantization
    GGUF: str = "GGUF"

    NONE: str = "none"

    BF16: str = "bf16" # only for Deepseek V3

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
    """
    Data class containing information about a model.
    
    Attributes:
        org: str - Organization name on Hugging Face
        base_name: str - Base name of the model
        version: str - Model version
        size: int - Model size
        name: str - Full model name (constructed from base_name, version, and size if not provided)
        is_multimodal: bool - Whether the model is multimodal
        instruct_tag: str - Instruction tuning tag
        quant_type: QuantType - Quantization type
        description: str - Model description
    
    Methods:
        construct_model_name: Constructs model name from components
        append_instruct_tag: Adds instruction tag to model name
        append_quant_type: Adds quantization tag to model name
        model_path: Returns full model path in format '{org}/{name}'
    """
    org: str
    base_name: str
    version: str
    size: int
    name: str = None  # full model name, constructed from base_name, version, and size unless provided
    is_multimodal: bool   = False

    instruct_tag: str     = None

    quant_type: QuantType = None

    description: str      = None


    def __post_init__(self):
        """
        Initializes the model name if not provided by constructing it from base_name, version, size, and quant_type.
        """
        self.name = self.name or self.construct_model_name(
            self.base_name,
            self.version,
            self.size,
            self.quant_type,
            self.instruct_tag,
        )

    @staticmethod
    def append_instruct_tag(key: str, instruct_tag: str = None) -> str:
        """
        Adds instruction tag to model name if provided.
        
        Args:
            key: str - Base name to append to
            instruct_tag: str - Instruction tag to append
        
        Returns:
            str - Model name with instruction tag appended
        """
        if instruct_tag:
            key = "-".join([key, instruct_tag])
        return key

    @staticmethod
    def append_quant_type(
        key: str, quant_type: QuantType = None
    ) -> str:
        """
        Adds quantization tag to model name if quant_type is not NONE.
        
        Args:
            key: str - Base name to append to
            quant_type: QuantType - Quantization type to append
        
        Returns:
            str - Model name with quantization tag appended
        """
        if quant_type != QuantType.NONE:
            key = "-".join([key, QUANT_TAG_MAP[quant_type]])
        return key

    @classmethod
    def construct_model_name(cls, base_name: str, version: str, size: str, quant_type: QuantType, instruct_tag: Optional[str], key: str="") -> str:
        """
        Constructs model name from components.
        
        Args:
            base_name: str - Base model name
            version: str - Model version
            size: str - Model size
            quant_type: QuantType - Quantization type
            instruct_tag: Optional[str] - Instruction tag
            key: str - Base key to start with
        
        Returns:
            str - Constructed model name
        """
        key = cls.append_instruct_tag(key, instruct_tag)
        key = cls.append_quant_type(key, quant_type)
        return key

    @property
    def model_path(
        self,
    ) -> str:
        """
        Returns the full model path in Hugging Face format.
        
        Returns:
            str - Full model path in format '{org}/{name}'
        """
        return f"{self.org}/{self.name}"


@dataclass
class ModelMeta:
    """
    Data class containing metadata for registering multiple models.
    
    Attributes:
        org: str - Organization name
        base_name: str - Base model name
        model_version: str - Model version
        model_info_cls: type[ModelInfo] - Class to use for model info
        model_sizes: list[str] - List of model sizes
        instruct_tags: list[str] - List of instruction tags
        quant_types: list[QuantType] | dict[str, list[QuantType]] - Quantization types per model size
        is_multimodal: bool - Whether models are multimodal
    """
    org: str
    base_name: str
    model_version: str
    model_info_cls: type[ModelInfo]
    model_sizes: list[str]   = field(default_factory=list)

    instruct_tags: list[str] = field(default_factory=list)

    quant_types: list[QuantType] | dict[str, list[QuantType]] = field(default_factory=list)
    is_multimodal: bool      = False



MODEL_REGISTRY: dict[str, ModelInfo] = {}


def register_model(
    model_info_cls: ModelInfo,
    org: str,
    base_name: str,
    version: str,
    size: int,
    instruct_tag: str     = None,
    quant_type: QuantType = None,
    is_multimodal: bool   = False,
    name: str             = None,
) -> None:
    """
    Registers a model in the global model registry.
    
    Args:
        model_info_cls: ModelInfo - Model info class to use
        org: str - Organization name
        base_name: str - Base model name
        version: str - Model version
        size: int - Model size
        instruct_tag: str - Instruction tag
        quant_type: QuantType - Quantization type
        is_multimodal: bool - Whether model is multimodal
        name: str - Optional explicit model name
    
    Raises:
        ValueError - If model is already registered
    """
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


def _check_model_info(model_id: str, properties: list[str] = ["lastModified"]) -> Optional[Any]:
    """
    Checks if a model exists on Hugging Face Hub.
    
    Args:
        model_id: str - Model ID in format '{org}/{name}'
        properties: list[str] - Properties to check (default: ['lastModified'])
    
    Returns:
        Optional[Any] - Model info if found, None if not found
    
    Raises:
        Exception - If error occurs other than RepositoryNotFoundError
    """
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


def _register_models(model_meta: ModelMeta, include_original_model: bool = False) -> None:
    """
    Registers multiple models based on ModelMeta configuration.
    
    Args:
        model_meta: ModelMeta - Model metadata
        include_original_model: bool - Whether to include original unquantized model
    
    Note:
        Models registered with org='unsloth' and QUANT_TYPE.NONE are aliases of QUANT_TYPE.UNSLOTH
    """
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