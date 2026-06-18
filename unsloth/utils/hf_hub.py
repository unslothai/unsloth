from huggingface_hub import HfApi, ModelInfo

_HFAPI: HfApi = None

POPULARITY_PROPERTIES = [
    "downloads",
    "downloadsAllTime",
    "trendingScore",
    "likes",
]
THOUSAND = 1000
MILLION = 1000000
BILLION = 1000000000


def formatted_int(value: int) -> str:
    if value < THOUSAND:
        return str(value)
    elif value < MILLION:
        return f"{float(value) / 1000:,.1f}K"
    elif value < BILLION:
        return f"{float(value) / 1000000:,.1f}M"
    else:
        return f"{float(value) / 1000000000:,.1f}B"


def get_model_info(
    model_id: str, properties: list[str] = ["safetensors", "lastModified"]
) -> ModelInfo:
    """
    Get info for a model. Defaults to minimal info; pass None for full info.
    properties: see https://huggingface.co/docs/huggingface_hub/api-ref/hf_hub/hf_api/model_info
    """
    global _HFAPI
    if _HFAPI is None:
        _HFAPI = HfApi()
    try:
        model_info: ModelInfo = _HFAPI.model_info(model_id, expand = properties)
    except Exception as e:
        print(f"Error getting model info for {model_id}: {e}")
        model_info = None
    return model_info


def list_models(
    properties: list[str] = None,
    full: bool = False,
    sort: str = "downloads",
    author: str = "unsloth",
    search: str = None,
    limit: int = 10,
) -> list[ModelInfo]:
    """
    List models from the Hugging Face Hub. If full is True, properties is ignored.
    properties: see https://huggingface.co/docs/huggingface_hub/api-ref/hf_hub/hf_api/list_models
    """
    global _HFAPI
    if _HFAPI is None:
        _HFAPI = HfApi()
    if full:
        properties = None

    models: list[ModelInfo] = _HFAPI.list_models(
        author = author,
        search = search,
        sort = sort,
        limit = limit,
        expand = properties,
        full = full,
    )
    return models
