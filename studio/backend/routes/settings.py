# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings policy: the account, shared and owner routers decide who may reach each
/api/settings path."""

import asyncio
import functools
import hashlib
import re
import threading
import time
from contextvars import ContextVar
from typing import Annotated, Any, Literal, Optional, get_args
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from auth.authentication import (
    allow_ambient_hf_token,
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from auth.storage import rotate_preview_link_secret
from auth import policy
from utils.account_context import (
    OWNER,
    bind_account,
    current_account,
    is_owner_context,
    reset_account,
)
from hub.utils.hf_tokens import cache_reads_authorized, cached_read_refused, hf_token_arg

from routes.provider_credentials import current_credential_write, require_ui_session

from storage import credential_secrets
from core.rag.config import (
    default_gguf_repo,
    effective_gguf_repo_for_embedding_model,
)
from loggers import get_logger
from utils.utils import safe_curated_detail, safe_error_detail, log_and_http_error
from utils.personalization_settings import (
    MAX_AVATAR_DATA_URL_BYTES,
    PERSONALIZATION_VERSION,
    drop_unknown_palette,
    get_personalization,
    set_personalization,
)
from utils.upload_limits import (
    MAX_UPLOAD_LIMIT_MB,
    MIN_UPLOAD_LIMIT_MB,
    default_upload_limit_mb,
    get_upload_limit_mb,
    set_upload_limit_mb,
    upload_limit_bytes,
    upload_limit_label,
)
from utils.cache_inventory import CACHE_KEYS, cache_inventory, purge_caches
from utils.xet_notice_settings import reserve_xet_notice
from utils.chat_preferences_settings import (
    get_show_model_disclaimer,
    migrate_show_model_disclaimer,
    set_show_model_disclaimer,
)
from utils.helper_precache_settings import (
    DEFAULT_HELPER_PRECACHE_ENABLED,
    get_helper_precache_enabled,
    helper_model_disabled_by_env,
    set_helper_precache_enabled,
)
from utils import systemone_settings
from utils.download_transport_settings import (
    get_download_transport_mode,
    set_download_transport_mode,
)
from utils.hub_settings import (
    HubSettings,
    active_source,
    claim_automatic_source,
    get_hub_settings,
    set_hub_settings,
    set_hub_source,
)
from picker.schemas import MAX_CHAT_TEMPLATE_BYTES, chat_template_byte_length
from utils.reasoning_budget import validate_reasoning_budget_message
from utils.coding_agents import CODING_AGENTS, detect_installed_coding_agents
from utils.multi_model_settings import (
    DEFAULT_MULTI_MODEL_ENABLED,
    get_multi_model_enabled,
    set_multi_model_enabled,
)
from utils.model_memory_settings import (
    DEFAULT_KEEP_RESIDENT,
    DEFAULT_NO_RAM_RESERVE,
    get_model_memory_settings,
    memlock_limit_bytes,
    set_model_memory_settings,
    should_mlock,
)
from utils.vram_budget_settings import (
    VRAM_FRACTION_DEFAULT,
    VRAM_FRACTION_MAX,
    VRAM_FRACTION_MIN,
    get_vram_budget_state,
    set_vram_budget_fraction,
)
from utils.openai_auto_switch_settings import (
    BATCH_SIZE_MAX,
    BATCH_SIZE_MIN,
    CACHE_RAM_MAX_MIB,
    CACHE_RAM_MIN_MIB,
    CTX_CHECKPOINTS_MAX,
    DEFAULT_AUTO_UNLOAD_API_ONLY,
    DEFAULT_AUTO_UNLOAD_KEEP_KV,
    DEFAULT_MEDIA_AUTO_SWITCH_ENABLED,
    DEFAULT_MEDIA_AUTO_UNLOAD_IDLE_SECONDS,
    DEFAULT_OPENAI_AUTO_DOWNLOAD_ENABLED,
    DEFAULT_OPENAI_AUTO_SWITCH_ENABLED,
    MAX_GPU_ID,
    PARALLEL_SLOTS_MAX,
    PARALLEL_SLOTS_MIN,
    cached_repo_alias_keys,
    is_cache_load_path_key,
    get_auto_unload_api_only,
    get_auto_unload_idle_seconds,
    get_auto_unload_keep_kv,
    get_media_auto_switch_enabled,
    get_media_auto_unload_idle_seconds,
    get_model_overrides,
    get_openai_auto_switch_enabled,
    resolve_model_override_key,
    resolve_model_override_keys,
    get_stored_auto_unload_idle_seconds,
    get_stored_media_auto_unload_idle_seconds,
    get_stored_openai_auto_download_enabled,
    idle_unload_is_configured,
    set_model_override,
    set_openai_auto_switch,
)
from utils.keyless_api_access import (
    access_exposure,
    get_keyless_api_access_settings,
    set_keyless_api_access,
)
from utils.preview_sharing_settings import (
    DEFAULT_PREVIEW_SHARING_ENABLED,
    get_preview_sharing_enabled,
    set_preview_sharing_enabled,
)
from utils.managed_provider_url_settings import (
    DEFAULT_MANAGED_PRIVATE_PROVIDER_URLS_ALLOWED,
    get_managed_private_provider_urls_allowed,
    private_urls_locked_by_environment,
    set_managed_private_provider_urls_allowed,
)
from utils.current_date_prompt_settings import (
    DEFAULT_CURRENT_DATE_PROMPT_ENABLED,
    get_current_date_prompt_enabled,
    set_current_date_prompt_enabled,
)
from utils.lan_access_settings import (
    lan_access_status,
    save_lan_access_port,
    set_lan_access_auto_start,
    start_lan_access,
    stop_lan_access,
)
from utils.remote_access_settings import (
    DEFAULT_REMOTE_ACCESS_AUTO_START,
    remote_access_status,
    set_remote_access_auto_start,
    start_remote_access,
    stop_remote_access,
)
from utils.embedding_model_settings import (
    MAX_EMBEDDING_MODEL_LENGTH,
    default_embedding_model,
    get_rag_embedding_model,
    get_stored_embedding_model,
    reset_rag_embedding_model,
    set_rag_embedding_model,
    validate_embedding_model,
)
from utils.hf_cache_settings import cache_status, get_hf_cache_paths, set_hf_cache_home
from utils.llama_cpp_path_settings import (
    MAX_CUSTOM_LLAMA_CPP_PATH_LENGTH,
    custom_llama_cpp_path_status,
    set_custom_llama_cpp_path,
)
from utils.media_generation_preset_settings import (
    delete_media_generation_preset,
    get_media_generation_preset_settings,
    set_media_generation_preset_settings,
    upsert_media_generation_preset,
)


async def _require_installation_owner(current_subject: str = Depends(get_current_subject)) -> None:
    await policy.require_owner()


async def _shared_policy_read(current_subject: str = Depends(get_current_subject)):
    marker = None
    if not current_account().is_owner:
        marker = bind_account(OWNER)
    try:
        yield
    finally:
        if marker is not None:
            reset_account(marker)


router = APIRouter()
_account_settings_router = APIRouter()
_owner_settings_router = APIRouter(dependencies = [Depends(_require_installation_owner)])
_shared_settings_router = APIRouter(dependencies = [Depends(_shared_policy_read)])

logger = get_logger(__name__)


class ImageGenerationPresetParams(BaseModel):
    """Bounds track DiffusionGenerateRequest. A preset the generate endpoint would refuse is not
    a usable preset: selecting it would make every following Generate fail validation."""

    model_config = ConfigDict(extra = "forbid")

    negativePrompt: str = ""
    width: int = Field(default = 1024, ge = 256, le = 2752, multiple_of = 16)
    height: int = Field(default = 1024, ge = 256, le = 2752, multiple_of = 16)
    steps: int = Field(default = 9, ge = 1, le = 100)
    guidance: float = Field(default = 0, ge = 0, le = 20)
    batchSize: int = Field(default = 1, ge = 1, le = 32)
    runs: int = Field(default = 1, ge = 1)


class VideoGenerationPresetParams(BaseModel):
    """Bounds track VideoGenerateRequest, as the image params track theirs."""

    model_config = ConfigDict(extra = "forbid")

    negativePrompt: str = ""
    width: int = Field(default = 768, ge = 32, le = 2048)
    height: int = Field(default = 512, ge = 32, le = 2048)
    durationSeconds: float = Field(default = 3, gt = 0, le = 3600)
    steps: int = Field(default = 8, ge = 1, le = 100)
    guidance: float = Field(default = 1, ge = 0, le = 20)
    flowShift: Optional[float] = Field(default = None, gt = 0, le = 100)
    audioFlowShift: Optional[float] = Field(default = None, gt = 0, le = 100)


class MediaGenerationPreset(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str = Field(..., min_length = 1, max_length = 80)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        name = value.strip()
        if not name or name == "Default":
            raise ValueError("Preset name is reserved or empty")
        return name


class ImageGenerationPreset(MediaGenerationPreset):
    params: ImageGenerationPresetParams


class VideoGenerationPreset(MediaGenerationPreset):
    params: VideoGenerationPresetParams


class MediaGenerationPresetState(BaseModel):
    """A saved generation recipe and the selection that owns it.

    Model-load options are deliberately not here: they take effect only on a reload, they follow
    the hardware and the checkpoint rather than the recipe, and the resident build already reports
    them, so a second stored copy would only ever compete with it.
    """

    model_config = ConfigDict(extra = "forbid")

    activePreset: str = Field(default = "Default", min_length = 1, max_length = 80)


class ImageGenerationPresetState(MediaGenerationPresetState):
    currentParams: ImageGenerationPresetParams = Field(default_factory = ImageGenerationPresetParams)


class VideoGenerationPresetState(MediaGenerationPresetState):
    currentParams: VideoGenerationPresetParams = Field(default_factory = VideoGenerationPresetParams)


class ImageGenerationPresetSettings(ImageGenerationPresetState):
    # No cap on the read: upsert_media_generation_preset owns the limit, and refusing to report a store
    # that somehow exceeds it would only turn a GET into a 500.
    customPresets: list[ImageGenerationPreset] = Field(default_factory = list)
    saved: bool = False


class VideoGenerationPresetSettings(VideoGenerationPresetState):
    customPresets: list[VideoGenerationPreset] = Field(default_factory = list)
    saved: bool = False


def _nested_model(annotation: Any) -> Optional[type[BaseModel]]:
    for candidate in (annotation, *get_args(annotation)):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _readable(model: type[BaseModel], value: Any) -> Any:
    """Drop what this build's schema does not define, keeping every field it does. `extra = "forbid"` is right
    for a submitted payload but wrong for reading storage back: a blob holding one field from a newer build
    would otherwise fail validation, and a stored recipe the user can no longer read is worse than one
    missing a field this build cannot render anyway."""
    if isinstance(value, list):
        return [_readable(model, item) for item in value]
    if not isinstance(value, dict):
        return value
    readable = {}
    for name, field in model.model_fields.items():
        if name not in value:
            continue
        nested = _nested_model(field.annotation)
        readable[name] = _readable(nested, value[name]) if nested else value[name]
    return readable


def _without_field_at_location(value: Any, location: tuple[Any, ...]) -> tuple[Any, bool]:
    """Return a copy with one invalid leaf removed from a nested model payload."""
    if not location:
        return value, False
    key, *rest = location
    if not isinstance(value, dict) or key not in value:
        return value, False
    result = dict(value)
    if not rest:
        result.pop(key)
        return result, True
    nested, removed = _without_field_at_location(result[key], tuple(rest))
    if removed:
        result[key] = nested
    return result, removed


def _validated_without_invalid_fields(
    schema: type[BaseModel], payload: dict
) -> tuple[BaseModel, list[tuple[Any, ...]]]:
    """Validate, dropping only the fields that fail. Resetting the whole recipe over one unreadable field would
    hand the client schema defaults, which it then autosaves over the rest of a good stored recipe."""
    remaining = payload
    removed_locations = []
    while True:
        try:
            return schema.model_validate(remaining), removed_locations
        except ValidationError as exc:
            for error in exc.errors():
                location = tuple(error.get("loc", ()))
                remaining, removed = _without_field_at_location(remaining, location)
                if removed:
                    removed_locations.append(location)
                    break
            else:
                return schema(), removed_locations


_MISSING = object()


def _value_at_location(value: Any, location: tuple[Any, ...]) -> Any:
    for key in location:
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _with_value_at_location(
    value: Any, location: tuple[Any, ...], replacement: Any
) -> tuple[Any, bool]:
    if not location:
        return replacement, True
    key, *rest = location
    if not isinstance(value, dict) or key not in value:
        return value, False
    result = dict(value)
    nested, replaced = _with_value_at_location(result[key], tuple(rest), replacement)
    if replaced:
        result[key] = nested
    return result, replaced


def _preserve_recovered_defaults(schema: type[BaseModel], stored: dict, submitted: dict) -> dict:
    """Do not mistake a recovery default for an edit to an unreadable stored field. A downgraded GET omits
    known fields whose values this schema cannot validate, then Pydantic supplies their defaults in the
    response; the client cannot tell those defaults from stored values and echoes them in its next state
    write. Preserve the raw leaf only while the submitted value is still the synthesized value."""
    recovered, locations = _validated_without_invalid_fields(schema, _readable(schema, stored))
    recovered_values = recovered.model_dump()
    merged = submitted
    for location in locations:
        previous = _value_at_location(stored, location)
        submitted_value = _value_at_location(submitted, location)
        recovered_value = _value_at_location(recovered_values, location)
        if (
            previous is not _MISSING
            and submitted_value is not _MISSING
            and recovered_value is not _MISSING
            and submitted_value == recovered_value
        ):
            merged, _ = _with_value_at_location(merged, location, previous)
    return merged


def _validated_readable_model(schema: type[BaseModel], payload: Any) -> Optional[BaseModel]:
    try:
        return schema.model_validate(_readable(schema, payload))
    except ValidationError:
        return None


def _get_generation_preset_settings(kind, schema):
    stored = get_media_generation_preset_settings(kind)
    try:
        response = schema.model_validate(_readable(schema, stored))
    except ValidationError:
        # A value this build cannot represent at all. Drop only what fails: one unreadable entry costs neither
        # the rest of the list nor the state.
        logger.warning("Dropping unreadable %s generation preset entries", kind)
        presets = schema.model_fields["customPresets"].annotation
        item = _nested_model(get_args(presets)[0] if get_args(presets) else presets)
        readable = []
        # Only a list is a preset collection. Recovery exists so a store this build cannot represent still
        # reads; iterating a scalar here would answer 500 instead. _custom_presets takes the same view on write.
        raw_presets = stored.get("customPresets")
        for raw in raw_presets if isinstance(raw_presets, list) else []:
            validated = _validated_readable_model(item, raw)
            if validated is not None:
                readable.append(validated)
        state = {
            key: value for key, value in _readable(schema, stored).items() if key != "customPresets"
        }
        response, _ = _validated_without_invalid_fields(
            schema, {**state, "customPresets": readable}
        )
    # Saved means the store owns the CURRENT recipe, not merely that something is stored.
    response.saved = isinstance(stored.get("currentParams"), dict)
    return response


@_account_settings_router.get(
    "/generation-presets/image",
    response_model = ImageGenerationPresetSettings,
)
def get_image_generation_preset_settings(
    current_subject: str = Depends(get_current_subject),
) -> ImageGenerationPresetSettings:
    return _get_generation_preset_settings("image", ImageGenerationPresetSettings)


@_account_settings_router.put("/generation-presets/image")
def update_image_generation_preset_settings(
    payload: ImageGenerationPresetState, current_subject: str = Depends(get_current_subject)
) -> dict[str, bool]:
    set_media_generation_preset_settings(
        "image",
        payload.model_dump(),
        lambda stored, submitted: _preserve_recovered_defaults(
            ImageGenerationPresetState, stored, submitted
        ),
    )
    return {"saved": True}


@_account_settings_router.get(
    "/generation-presets/video",
    response_model = VideoGenerationPresetSettings,
)
def get_video_generation_preset_settings(
    current_subject: str = Depends(get_current_subject),
) -> VideoGenerationPresetSettings:
    return _get_generation_preset_settings("video", VideoGenerationPresetSettings)


@_account_settings_router.put("/generation-presets/video")
def update_video_generation_preset_settings(
    payload: VideoGenerationPresetState, current_subject: str = Depends(get_current_subject)
) -> dict[str, bool]:
    set_media_generation_preset_settings(
        "video",
        payload.model_dump(),
        lambda stored, submitted: _preserve_recovered_defaults(
            VideoGenerationPresetState, stored, submitted
        ),
    )
    return {"saved": True}


def _upsert_custom_generation_preset(
    kind: Literal["image", "video"], payload: ImageGenerationPreset | VideoGenerationPreset
) -> dict[str, bool]:
    try:
        schema = type(payload)
        upsert_media_generation_preset(
            kind,
            payload.model_dump(),
            lambda stored: _validated_readable_model(schema, stored) is not None,
        )
    except ValueError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    return {"saved": True}


@_account_settings_router.put("/generation-presets/image/custom")
def upsert_custom_image_generation_preset(
    payload: ImageGenerationPreset, current_subject: str = Depends(get_current_subject)
) -> dict[str, bool]:
    return _upsert_custom_generation_preset("image", payload)


@_account_settings_router.put("/generation-presets/video/custom")
def upsert_custom_video_generation_preset(
    payload: VideoGenerationPreset, current_subject: str = Depends(get_current_subject)
) -> dict[str, bool]:
    return _upsert_custom_generation_preset("video", payload)


@_account_settings_router.delete("/generation-presets/{kind}/custom")
def delete_custom_generation_preset(
    kind: Literal["image", "video"],
    name: str,
    current_subject: str = Depends(get_current_subject),
) -> dict[str, bool]:
    name = name.strip()
    if not name or name == "Default" or len(name) > 80:
        raise HTTPException(status_code = 422, detail = "Invalid preset name")
    delete_media_generation_preset(kind, name)
    return {"deleted": True}


class MultiModelPayload(BaseModel):
    enabled: StrictBool


class MultiModelResponse(BaseModel):
    enabled: bool
    default_enabled: bool = DEFAULT_MULTI_MODEL_ENABLED


class UploadLimitPayload(BaseModel):
    max_upload_size_mb: int = Field(..., ge = MIN_UPLOAD_LIMIT_MB, le = MAX_UPLOAD_LIMIT_MB)


class UploadLimitResponse(BaseModel):
    max_upload_size_mb: int
    max_upload_size_bytes: int
    max_upload_size_label: str
    default_upload_size_mb: int
    min_upload_size_mb: int = MIN_UPLOAD_LIMIT_MB
    max_allowed_upload_size_mb: int = MAX_UPLOAD_LIMIT_MB


class HuggingFaceTokenPayload(BaseModel):
    token: str = Field(..., min_length = 1, max_length = 512)

    @field_validator("token")
    @classmethod
    def normalize_token(cls, value: str) -> str:
        normalized = value.strip(" \t\r\n\"'")
        if not normalized:
            raise ValueError("Hugging Face token cannot be empty")
        return normalized


class HuggingFaceTokenResponse(BaseModel):
    token: Optional[str] = None
    has_token: bool = False


@_account_settings_router.get("/hugging-face-token", response_model = HuggingFaceTokenResponse)
def get_hugging_face_token(
    _current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> HuggingFaceTokenResponse:
    require_ui_session(via_api_key)
    token = credential_secrets.get_hf_token()
    return HuggingFaceTokenResponse(token = token, has_token = token is not None)


@_account_settings_router.put("/hugging-face-token", response_model = HuggingFaceTokenResponse)
def update_hugging_face_token(
    payload: HuggingFaceTokenPayload,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> HuggingFaceTokenResponse:
    require_ui_session(via_api_key)

    # Warm the auth-owned key before the generation guard takes its write lock.
    credential_secrets.get_or_create_credential_encryption_key()
    with current_credential_write(credential):
        credential_secrets.save_hf_token(payload.token)
    return HuggingFaceTokenResponse(token = payload.token, has_token = True)


@_account_settings_router.put(
    "/hugging-face-token/migrate", response_model = HuggingFaceTokenResponse
)
def migrate_hugging_face_token(
    payload: HuggingFaceTokenPayload,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> HuggingFaceTokenResponse:
    """Insert a browser legacy token only when the installation has none."""
    require_ui_session(via_api_key)
    credential_secrets.get_or_create_credential_encryption_key()
    with current_credential_write(credential):
        credential_secrets.save_hf_token_if_absent(payload.token)
        token = credential_secrets.get_hf_token()
    return HuggingFaceTokenResponse(token = token, has_token = token is not None)


@_account_settings_router.delete("/hugging-face-token", response_model = HuggingFaceTokenResponse)
def clear_hugging_face_token(
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> HuggingFaceTokenResponse:
    require_ui_session(via_api_key)
    with current_credential_write(credential):
        credential_secrets.delete_hf_token()
    return HuggingFaceTokenResponse(token = None, has_token = False)


class SystemOneModelOption(BaseModel):
    name: str
    description: str
    download_bytes: int
    kind: Literal["catalog", "fine_tune"] = "catalog"
    label: Optional[str] = None
    available: bool = True
    unavailable_reason: Optional[str] = None
    # A GGUF with no PyTorch form: the runtime setting matters, and PyTorch cannot serve it.
    llama_cpp_only: bool = False


class SystemOneConnectionOption(BaseModel):
    name: str
    provider_id: str
    provider: str
    model: str


class SystemOneSettingsResponse(BaseModel):
    enabled: bool
    enabled_locked: bool
    model: str
    model_locked: bool
    device: str
    device_locked: bool
    gpu_available: bool
    models: list[SystemOneModelOption]
    loaded_model: Optional[str] = None
    loaded_device: Optional[str] = None
    loading_model: Optional[str] = None
    installing: bool = False
    error: Optional[str] = None
    mcp_url: str
    # Runtime setting, what a text request to the configured model uses now, and why Auto chose PyTorch.
    backend: str = "auto"
    native_ctx: int = 16384
    effective_backend: Optional[str] = None
    loaded_backend: Optional[str] = None
    fallback_reason: Optional[str] = None
    input_modalities: list[str] = ["text"]
    # "laya", "clef" or "gguf" for the configured model, so env-configured local checkpoints get runtime controls.
    layout: Optional[str] = None


class SystemOneSettingsPayload(BaseModel):
    enabled: Optional[bool] = None
    model: Optional[str] = None
    device: Optional[str] = None
    backend: Optional[str] = None
    native_ctx: Optional[int] = None
    expected_enabled: Optional[bool] = None
    expected_model: Optional[str] = None


class SystemOneDownloadPlan(BaseModel):
    repo: Optional[str] = None
    files: list[str]
    size_bytes: int
    cached: bool
    error: Optional[str] = None


class HelperPrecachePayload(BaseModel):
    enabled: bool


class HelperPrecacheResponse(BaseModel):
    enabled: bool
    default_enabled: bool = DEFAULT_HELPER_PRECACHE_ENABLED
    disabled_by_env: bool


class DownloadTransportPayload(BaseModel):
    mode: Literal["auto", "xet", "http"]


class DownloadTransportResponse(BaseModel):
    mode: str
    xet_available: bool
    xet_unavailable_reason: Optional[str] = None
    auto_resolves_to: str
    auto_reason: Optional[str] = None


class HubSettingsPayload(BaseModel):
    hf_endpoint: str = Field(max_length = 2048)
    datasets_server_follows_endpoint: StrictBool


class HubSourcePayload(BaseModel):
    source: Literal["huggingface", "modelscope"]


class HubSettingsResponse(BaseModel):
    hf_endpoint: str
    datasets_server_follows_endpoint: bool
    source: Literal["huggingface", "modelscope"]
    active_source: Literal["huggingface", "modelscope"]


class HubSourceNoticeResponse(BaseModel):
    granted: bool


class XetNoticeReservePayload(BaseModel):
    # A legacy localStorage count from a client that has not reported one before. Can only raise the
    # stored count (see reserve_xet_notice), so a client cannot talk its own way back under the limit.
    seen_hint: int = 0


class XetNoticeResponse(BaseModel):
    granted: bool
    shown: int
    limit: int


class IgpuCarveoutNoticeDismissPayload(BaseModel):
    # The allocation being dismissed at, so raising it and running short again can
    # speak once more. Absent means "keep whatever is recorded", never lowering it.
    current_gb: Optional[float] = None


class IgpuCarveoutNoticeResponse(BaseModel):
    dismissed_at_gb: Optional[float] = None


class ChatPreferencesPayload(BaseModel):
    show_model_disclaimer: StrictBool


class ChatPreferencesMigrationPayload(BaseModel):
    show_model_disclaimer: Optional[StrictBool] = None


class ChatPreferencesResponse(BaseModel):
    show_model_disclaimer: bool


class ModelMemoryPayload(BaseModel):
    # None leaves the stored value untouched, so the switches save independently.
    keep_resident: Optional[bool] = None
    no_ram_reserve: Optional[bool] = None


class ModelMemoryResponse(BaseModel):
    keep_resident: bool
    no_ram_reserve: bool
    default_keep_resident: bool = DEFAULT_KEEP_RESIDENT
    default_no_ram_reserve: bool = DEFAULT_NO_RAM_RESERVE
    # Whether --mlock is passed on the next load. False when no_ram_reserve
    # vetoes it; the UI surfaces that rather than failing silently.
    mlock_active: bool
    # False when the running llama.cpp child has no host copy to lock (full offload to a discrete GPU),
    # so a keep-resident user is told why no lock is taken. True with nothing loaded.
    mlock_applicable: bool = True
    reload_required: bool
    # Soft RLIMIT_MEMLOCK when finite. mlock cannot exceed it, so the UI warns that residency will not
    # fully pin a larger model. None means unlimited (macOS) or not applicable (Windows).
    memlock_limit_bytes: Optional[int] = None


class VramBudgetPayload(BaseModel):
    # None clears the stored budget so env/default applies again; it cannot also mean "leave untouched", hence
    # required rather than defaulted: with a default, a client that dropped the field would silently discard it.
    fraction: Optional[float] = Field(ge = VRAM_FRACTION_MIN, le = VRAM_FRACTION_MAX)

    @field_validator("fraction", mode = "before")
    @classmethod
    def _reject_bool(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("fraction must be a number, not a boolean")
        return value


class VramBudgetResponse(BaseModel):
    fraction: float
    # False when inherited from UNSLOTH_VRAM_FRACTION or the default, so the UI
    # knows whether clearing it would change anything.
    is_stored: bool
    default_fraction: float = VRAM_FRACTION_DEFAULT
    min_fraction: float = VRAM_FRACTION_MIN
    max_fraction: float = VRAM_FRACTION_MAX
    # Read when a load sizes itself, so a change cannot reach a running child.
    reload_required: bool


class HuggingFaceCachePayload(BaseModel):
    cache_home: Optional[str] = Field(default = None, max_length = 4096)


class HuggingFaceCacheResponse(BaseModel):
    cache_home: str
    hub_cache: str
    xet_cache: str
    source: Literal["default", "studio", "environment"]
    editable: bool
    is_custom: bool
    available: bool
    writable: bool
    free_bytes: Optional[int] = None
    environment_variable: Optional[str] = None


class CacheEntryResponse(BaseModel):
    key: str
    group: str
    # Clearing this costs a re-download, so the UI never folds it into a
    # "clear everything" action.
    opt_in: bool
    paths: list[str]
    size_bytes: int
    entry_count: int
    present: bool
    purgeable: bool
    blocked_reason: Optional[str] = None


class CacheInventoryResponse(BaseModel):
    caches: list[CacheEntryResponse]
    total_bytes: int
    reclaimable_bytes: int
    free_bytes: Optional[int] = None
    total_disk_bytes: Optional[int] = None


class CachePurgePayload(BaseModel):
    # Cache identifiers, never paths: the backend owns the mapping from a key to
    # a directory, so a caller cannot name one of its own.
    keys: list[str] = Field(min_length = 1, max_length = len(CACHE_KEYS))


class CachePurgeResultResponse(BaseModel):
    key: str
    freed_bytes: int
    removed_entries: int
    errors: list[str]


class CachePurgeResponse(BaseModel):
    results: list[CachePurgeResultResponse]
    freed_bytes: int
    inventory: CacheInventoryResponse


class LlamaCppPathPayload(BaseModel):
    path: Optional[str] = Field(default = None, max_length = MAX_CUSTOM_LLAMA_CPP_PATH_LENGTH)


class LlamaCppPathResponse(BaseModel):
    path: Optional[str] = None
    source: Literal["default", "studio", "environment"]
    editable: bool
    available: bool
    resolved_binary: Optional[str] = None
    environment_variable: Optional[str] = None
    reload_required: bool = False


class OpenAIAutoSwitchPayload(BaseModel):
    enabled: bool
    # None leaves the stored value untouched (partial updates can't clobber it).
    auto_unload_idle_seconds: Optional[int] = Field(default = None, ge = 0)
    auto_unload_keep_kv: Optional[bool] = None
    auto_download_model: Optional[bool] = None
    auto_unload_api_only: Optional[bool] = None
    # The image/video TTL is its own setting, not a share of the chat one.
    media_auto_unload_idle_seconds: Optional[int] = Field(default = None, ge = 0)
    media_auto_switch_model: Optional[bool] = None


class OpenAIAutoSwitchResponse(BaseModel):
    enabled: bool
    auto_unload_idle_seconds: int
    default_enabled: bool = DEFAULT_OPENAI_AUTO_SWITCH_ENABLED
    # True when the idle-unload loop will actually unload (effective TTL > 0). With UNSLOTH_MODEL_IDLE_TTL set
    # and nothing stored this is true even while enabled is false, so the UI can show idle-unload as active.
    idle_unload_active: bool = False
    auto_unload_keep_kv: bool = DEFAULT_AUTO_UNLOAD_KEEP_KV
    # Stored, not effective: the UI must round-trip the saved value across an auto-switch toggle.
    auto_download_model: bool = DEFAULT_OPENAI_AUTO_DOWNLOAD_ENABLED
    # When true, the idle unload spares models loaded from the UI, not just via the API.
    auto_unload_api_only: bool = DEFAULT_AUTO_UNLOAD_API_ONLY
    # Stored, then effective: the UI shows the saved seconds and flags when a veto
    # (residency, or API-loaded only) is holding the image/video unload off.
    media_auto_unload_idle_seconds: int = DEFAULT_MEDIA_AUTO_UNLOAD_IDLE_SECONDS
    media_idle_unload_active: bool = False
    # When true, a media request may load the image or video model it names.
    media_auto_switch_model: bool = DEFAULT_MEDIA_AUTO_SWITCH_ENABLED


# A quant suffix as modelOverrideKey builds it, matched against the loader's quant pattern rather
# than a length heuristic: a POSIX path may hold a colon and inherit another model's flags.
_MAX_VARIANT_SUFFIX_LEN = 64

# A local id is a path plus an optional quant suffix, and LoadRequest.model_path is unbounded: a
# limit under PATH_MAX would 422 the server sync while the local save succeeded.
MAX_MODEL_OVERRIDE_KEY_LEN = 4096 + 1 + _MAX_VARIANT_SUFFIX_LEN

# GgufVariantDetail.quant may be a path-qualified variant key, not just a quant suffix.
MAX_GGUF_VARIANT_KEY_LEN = 4096

# A list longer than MAX_GPU_ID cannot name a device the normalizer would store, so reject an
# oversized array at the boundary instead of walking it.
MAX_GPU_IDS = MAX_GPU_ID + 1


class ModelOverridePayload(BaseModel):
    """One model's saved launch config, applied when the API loads that model.

    Everything past ``model_id`` is optional and omitted means "app default", so a
    payload carrying only ``model_id`` clears the entry. The bounds here mirror
    ``LoadRequest`` so a bad value is rejected at the boundary instead of being
    silently dropped by the normalizer; the enum-ish fields (KV dtype, speculative
    mode) are left to it, since their valid sets follow the llama.cpp build.
    """

    model_id: str = Field(..., min_length = 1, max_length = MAX_MODEL_OVERRIDE_KEY_LEN)
    engine_parallelism: Optional[Literal["tensor", "pipeline", "data"]] = None
    engine_precision: Optional[Literal["auto", "bf16", "fp16", "int4", "int8", "fp8"]] = None
    engine: Optional[Literal["auto", "vllm", "sglang"]] = None
    # None leaves the stored value alone (the UI has no control for flags); [] clears them.
    llama_extra_args: Optional[list[str]] = None
    # ge=1: the setter drops a falsy value, so reject 0 here instead of discarding it silently.
    max_seq_length: Optional[int] = Field(default = None, ge = 1, le = 1048576)
    custom_context_length: Optional[int] = Field(default = None, ge = 1, le = 1048576)
    kv_cache_dtype: Optional[str] = Field(default = None, max_length = 32)
    # A discrete set, enforced by the normalizer; these bounds only block absurd values.
    mlx_kv_quant: Optional[str] = Field(default = None, max_length = 16)
    mlx_kv_bits: Optional[float] = Field(default = None, ge = 2, le = 8)

    @model_validator(mode = "after")
    def derive_mlx_kv_quant(self):
        """Fold the pair into the field storage keeps; null is how a client spells Auto."""

        if "mlx_kv_quant" not in self.model_fields_set and self.mlx_kv_bits is not None:
            from core.inference.mlx_inference import encode_mlx_kv_quant
            self.mlx_kv_quant = encode_mlx_kv_quant(self.mlx_kv_bits)
        self.mlx_kv_bits = None
        return self

    speculative_type: Optional[str] = Field(default = None, max_length = 32)
    spec_draft_n_max: Optional[int] = Field(default = None, ge = 1, le = 16)
    # Parallel decode slots (llama-server --parallel), GGUF-only; None follows the server default.
    n_parallel: Optional[int] = Field(default = None, ge = PARALLEL_SLOTS_MIN, le = PARALLEL_SLOTS_MAX)
    reasoning_budget: Optional[int] = Field(default = None, ge = -1, le = 2_147_483_647)
    reasoning_budget_message: Optional[str] = None
    # prompt batch sizes (--batch-size / --ubatch-size), gguf-only; none = llama.cpp defaults
    n_batch: Optional[int] = Field(default = None, ge = BATCH_SIZE_MIN, le = BATCH_SIZE_MAX)
    n_ubatch: Optional[int] = Field(default = None, ge = BATCH_SIZE_MIN, le = BATCH_SIZE_MAX)
    # model_override_load_kwargs already applies all four off a stored row, so a route that drops them leaves the
    # setting reaching a picker load and nothing else.
    load_mode: Optional[str] = Field(default = None, max_length = 32)
    spec_draft_cache_type: Optional[str] = Field(default = None, max_length = 32)
    # Stored on "is not None", not on truth: 0 checkpoints and a 0 or -1 cache are
    # meaningful values (none kept; cache disabled; no limit). Bounds mirror LoadRequest.
    ctx_checkpoints: Optional[int] = Field(default = None, ge = 0, le = CTX_CHECKPOINTS_MAX)
    cache_ram: Optional[int] = Field(default = None, ge = CACHE_RAM_MIN_MIB, le = CACHE_RAM_MAX_MIB)
    # Does this client know the four above exist? A save REPLACES the entry, so an omission from a build that
    # predates them is indistinguishable from a user clearing them. Only a client that sets this may clear by
    # omission; default False, so an old payload is the safe case.
    mirrors_server_tuning: bool = False
    # The reasoning pair came later than the four, so a build that mirrors them can still
    # predate it: its own flag, same contract.
    mirrors_reasoning_budget: bool = False
    tensor_parallel: bool = False
    disable_vision: bool = False
    mlx_int8_prefill: bool = False
    # Validated in bytes below: pydantic counts characters, so a multi-byte template would pass.
    chat_template_override: Optional[str] = None
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = None
    # -1 is Auto (llama.cpp --fit sizes the offload); the normalizer treats it as unset.
    gpu_layers: Optional[int] = Field(default = None, ge = -1, le = 1024)
    n_cpu_moe: Optional[int] = Field(default = None, ge = 0, le = 1024)
    tensor_split: Optional[list[float]] = Field(default = None, min_length = 2, max_length = MAX_GPU_IDS)
    gpu_ids: Optional[list[int]] = Field(default = None, max_length = MAX_GPU_IDS)
    # Which index space gpu_ids is in. Absent means physical, the only thing a client
    # written before this field could have meant.
    gpu_index_kind: Optional[Literal["physical", "vulkan"]] = None
    # An all-default save carries no fields, like a forget; None keeps the legacy contract.
    remove: Optional[bool] = None
    # Fill in, don't replace: the backfill reads the map once then writes each model.
    fill_absent_fields: bool = False

    @model_validator(mode = "after")
    def _tensor_split_matches_gpu_ids(self):
        if self.tensor_split is not None:
            from utils.openai_auto_switch_settings import normalize_tensor_split
            if normalize_tensor_split(self.tensor_split, self.gpu_ids) is None:
                raise ValueError(
                    "tensor_split must match an ordered selection of at least two unique GPUs"
                )
        return self

    @field_validator("tensor_split")
    @classmethod
    def _valid_tensor_split(cls, value: Optional[list[float]]) -> Optional[list[float]]:
        if value is None:
            return None
        from utils.openai_auto_switch_settings import normalize_tensor_split

        if normalize_tensor_split(value, list(range(len(value)))) is None:
            raise ValueError("tensor_split must be finite, non-negative, and have a positive total")
        return value

    @field_validator("chat_template_override")
    @classmethod
    def _limit_chat_template_bytes(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        size = chat_template_byte_length(value)
        if size is None:
            raise ValueError("Chat template contains unpaired surrogate characters.")
        if size > MAX_CHAT_TEMPLATE_BYTES:
            raise ValueError(f"Chat template exceeds the {MAX_CHAT_TEMPLATE_BYTES}-byte limit.")
        return value

    @field_validator("reasoning_budget_message")
    @classmethod
    def _validate_reasoning_budget_message(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else validate_reasoning_budget_message(value)

    @field_validator(
        "max_seq_length",
        "custom_context_length",
        "spec_draft_n_max",
        "n_parallel",
        "reasoning_budget",
        "n_batch",
        "n_ubatch",
        "ctx_checkpoints",
        "cache_ram",
        "gpu_layers",
        "n_cpu_moe",
        "gpu_ids",
        "tensor_split",
        mode = "before",
    )
    @classmethod
    def _no_booleans(cls, value: Any) -> Any:
        # bool subclasses int and pydantic parses non-strictly.
        if isinstance(value, bool):
            raise ValueError("Expected a number, got a boolean.")
        if isinstance(value, list) and any(isinstance(item, bool) for item in value):
            raise ValueError("Expected numbers, got a boolean.")
        return value


class ModelOverridesResponse(BaseModel):
    overrides: dict[str, dict]
    # Filled only when the caller named a model, resolved here rather than in the browser: the folding rules are
    # Python's (casefold is not toLowerCase), so a client mirroring them can only approximate, and an ambiguous
    # fold matches nothing on purpose.
    resolved: Optional[dict] = None
    resolved_key: Optional[str] = None
    # What an explicit remove cleared; empty for a save.
    removed_keys: list[str] = []


def _upload_limit_response(limit_mb: int) -> UploadLimitResponse:
    return UploadLimitResponse(
        max_upload_size_mb = limit_mb,
        max_upload_size_bytes = upload_limit_bytes(limit_mb),
        max_upload_size_label = upload_limit_label(limit_mb),
        default_upload_size_mb = default_upload_limit_mb(),
    )


def _helper_precache_response(enabled: bool | None = None) -> HelperPrecacheResponse:
    return HelperPrecacheResponse(
        enabled = get_helper_precache_enabled() if enabled is None else enabled,
        disabled_by_env = helper_model_disabled_by_env(),
    )


def _download_transport_response(mode: str | None = None) -> DownloadTransportResponse:
    # No Xet probe: this renders a row, not a download start. The free-RAM gate is asked for
    # anyway, since the row states what the next download will use.
    from hub.utils.download_registry import get_download_transport_capabilities
    caps = get_download_transport_capabilities(ram_gate = True)
    return DownloadTransportResponse(
        mode = get_download_transport_mode() if mode is None else mode,
        xet_available = caps.xet.available,
        xet_unavailable_reason = caps.xet.reason,
        auto_resolves_to = caps.auto_resolves_to,
        auto_reason = caps.auto_reason,
    )


def _chat_preferences_response(enabled: bool | None = None) -> ChatPreferencesResponse:
    return ChatPreferencesResponse(
        show_model_disclaimer = (get_show_model_disclaimer() if enabled is None else enabled)
    )


# Distinct from None, which is a real launch this policy does not govern.
_NO_LAUNCH = object()


def _active_launch_placement():
    """``(state, policy_active, mlock_applicable, direct_io, dio_applicable, dio_managed,
    pending_settings)`` for the running child.

    ``state`` is ``_NO_LAUNCH`` when nothing is running or coming up, so the
    caller can tell "no process" apart from "a process with no load-mode".
    """
    try:
        from routes.inference import get_llama_cpp_backend

        backend = get_llama_cpp_backend()
        # Read ONCE: two reads can straddle the marker clear, leaving a killed child's
        # placement answering for the launch that replaced it. One attribute carries both
        # "is a launch pending" and "what is it committed to".
        pending = getattr(backend, "_memory_pending_launch", None)
        if not backend.is_active and pending is None:
            return _NO_LAUNCH, False, True, None, False, False, None
        return (
            getattr(backend, "_memory_state", None),
            bool(getattr(backend, "_memory_policy_active", False)),
            bool(getattr(backend, "_memory_mlock_applicable", True)),
            getattr(backend, "_memory_direct_io", None),
            bool(getattr(backend, "_memory_dio_applicable", False)),
            # The pair the POLICY emitted, not the aggregate: a user's own `dio` must
            # not be withdrawn on their behalf.
            bool(getattr(backend, "_memory_dio_flags", None)),
            pending,
        )
    except Exception:
        return _NO_LAUNCH, False, True, None, False, False, None


def _launch_effect_of(settings):
    """The part of ``(keep_resident, no_ram_reserve)`` a launch can express.

    Mirrors ``should_mlock``: the page-lock is emitted only when residency is on and
    no-reserve is off, so with no-reserve on the residency toggle reaches no flag.
    """
    keep_resident, no_ram_reserve = settings
    return (keep_resident and not no_ram_reserve, no_ram_reserve)


def _model_memory_reload_required() -> bool:
    """True when the loaded process's memory placement contradicts the settings.

    Compares the state the child ACTUALLY launched with -- env defaults plus
    last-wins argv, so a user-supplied --mlock / --no-mmap counts -- against
    what the current settings would produce. The idle-unload veto applies
    immediately (the loop re-reads each poll), so only placement can be stale.

    Keyed on is_active, not is_loaded: a save that lands while a load is still
    passing its health check would otherwise report no reload while the child is
    already committed to the pre-save flags. _memory_launch_pending covers the
    same window before Popen, where the placement is decided but _process is
    still None.
    """
    state, policy_active, mlock_applicable, direct_io, dio_applicable, dio_managed, pending = (
        _active_launch_placement()
    )
    if state is _NO_LAUNCH:
        return False

    # A launch in flight has no resolved flags and the comparator reads None as "not
    # governed", so answer from its snapshot. Whenever one is pending, NOT only when
    # `state` is None: replacing a model leaves the old child's `_memory_state` behind.
    if pending is not None:
        from utils.model_memory_settings import get_model_memory_settings

        # By EFFECT, not the literal pair: no-reserve wins over keep-resident for every
        # loader flag, so flipping keep-resident under it only moves the idle-unload veto,
        # which the loop re-reads each poll. The raw tuple asked for an inexpressible reload.
        return _launch_effect_of(get_model_memory_settings()) != _launch_effect_of(pending)

    # Same predicate the duplicate-load comparator uses.
    from core.inference.llama_server_args import memory_state_satisfies_settings

    return not memory_state_satisfies_settings(
        state, policy_active, mlock_applicable, direct_io, dio_applicable, dio_managed
    )


def _model_memory_mlock_active(want_mlock: bool) -> bool:
    """Whether page-locking is actually in force, not merely asked for. This drives the locked-memory cap
    warning, so taking it from the toggles alone would tell a discrete-GPU user to raise a limit nothing
    consults. With nothing running this is the intent; once a child exists it is what that child got, since
    a full offload to a discrete GPU skips the lock and a diffusion runner has no load-mode at all. A
    user's own --mlock counts, since the resolver reads the launched argv."""
    if not want_mlock:
        return False
    state, _policy_active, _applicable, _direct_io, _dio_applicable, _dio_managed, _pending = (
        _active_launch_placement()
    )
    if state is _NO_LAUNCH:
        return True
    return bool(state and state[0])


def _model_memory_mlock_applicable() -> bool:
    state, _policy_active, applicable, _direct_io, _dio_applicable, _dio_managed, _pending = (
        _active_launch_placement()
    )
    return state is _NO_LAUNCH or bool(applicable)


def _model_memory_response() -> ModelMemoryResponse:
    keep_resident, no_ram_reserve = get_model_memory_settings()
    mlock_active = _model_memory_mlock_active(should_mlock())
    return ModelMemoryResponse(
        keep_resident = keep_resident,
        no_ram_reserve = no_ram_reserve,
        mlock_active = mlock_active,
        mlock_applicable = _model_memory_mlock_applicable(),
        reload_required = _model_memory_reload_required(),
        memlock_limit_bytes = memlock_limit_bytes() if mlock_active else None,
    )


def _vram_budget_reload_required(fraction: float) -> bool:
    """True when a child is running that was sized against a different budget. Compares against the fraction the
    child actually launched with, so re-saving the same value does not nag for a reload. Exact equality is
    fine: both sides come from the same clamp."""
    try:
        from routes.inference import get_llama_cpp_backend

        backend = get_llama_cpp_backend()
        # A planned-but-unspawned load has no _process, so is_active is False while the child is already
        # committed to its captured fraction; answer from the pending value there.
        pending = getattr(backend, "_vram_fraction_pending", None)
        if pending is not None:
            return float(pending) != float(fraction)
        if not backend.is_active:
            return False
        launched = getattr(backend, "_vram_fraction_launched", None)
        # A child predating this field cannot be compared, so say no rather than nagging on every save.
        if launched is None:
            return False
        return float(launched) != float(fraction)
    except Exception:
        return False


def _vram_budget_response() -> VramBudgetResponse:
    fraction, is_stored = get_vram_budget_state()
    return VramBudgetResponse(
        fraction = fraction,
        is_stored = is_stored,
        reload_required = _vram_budget_reload_required(fraction),
    )


def _hugging_face_cache_response() -> HuggingFaceCacheResponse:
    return HuggingFaceCacheResponse(**cache_status(get_hf_cache_paths()))


def _llama_cpp_path_reload_required() -> bool:
    """Whether a running or pending GGUF server predates the path selection."""
    try:
        from routes.inference import get_llama_cpp_backend

        backend = get_llama_cpp_backend()
        pending = getattr(backend, "_binary_revision_pending", None)
        if pending is not None:
            return backend._binary_changed_since_revision(pending)
        return bool(backend.is_active and backend._binary_changed_since_launch())
    except Exception:
        return False


def _llama_cpp_path_response() -> LlamaCppPathResponse:
    return LlamaCppPathResponse(
        **custom_llama_cpp_path_status(),
        reload_required = _llama_cpp_path_reload_required(),
    )


@_owner_settings_router.get("/hugging-face-cache", response_model = HuggingFaceCacheResponse)
def get_hugging_face_cache(
    current_subject: str = Depends(get_current_subject),
) -> HuggingFaceCacheResponse:
    return _hugging_face_cache_response()


@_owner_settings_router.put("/hugging-face-cache", response_model = HuggingFaceCacheResponse)
def update_hugging_face_cache(
    payload: HuggingFaceCachePayload, current_subject: str = Depends(get_current_subject)
) -> HuggingFaceCacheResponse:
    try:
        set_hf_cache_home(payload.cache_home)
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    return _hugging_face_cache_response()


@_owner_settings_router.get("/caches", response_model = CacheInventoryResponse)
async def get_caches(
    refresh: bool = False,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> CacheInventoryResponse:
    """Size every cache this install writes to, plus the free space around them.

    ``refresh`` re-walks every cache instead of reusing a size measured in the
    last minute, for the Recheck the UI offers after something big was written.
    It is the interactive button, and a walk of a large hub or triton cache is
    seconds of stat calls in the shared executor with no memo in front of it, so
    only a UI session may ask for one. A plain read stays open to an API key.
    """
    if refresh:
        require_ui_session(via_api_key)
    # A cold walk of a large hub or triton cache is seconds of stat calls, so it
    # stays off the event loop.
    inventory = await asyncio.to_thread(cache_inventory, refresh = refresh)
    return CacheInventoryResponse(**inventory)


@_owner_settings_router.post("/caches/purge", response_model = CachePurgeResponse)
async def purge_caches_endpoint(
    payload: CachePurgePayload,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> CachePurgeResponse:
    """Empty the named caches. Only the interactive UI may delete anything."""
    require_ui_session(via_api_key)
    try:
        result = await asyncio.to_thread(purge_caches, payload.keys)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            str(exc),
            event = "settings.purge_caches_failed",
            log = logger,
        ) from exc
    return CachePurgeResponse(**result)


@_owner_settings_router.get("/llama-cpp-path", response_model = LlamaCppPathResponse)
def get_llama_cpp_path(current_subject: str = Depends(get_current_subject)) -> LlamaCppPathResponse:
    return _llama_cpp_path_response()


@_owner_settings_router.put("/llama-cpp-path", response_model = LlamaCppPathResponse)
def update_llama_cpp_path(
    payload: LlamaCppPathPayload,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> LlamaCppPathResponse:
    # Only the interactive Unsloth UI may change this executable setting.
    require_ui_session(via_api_key)
    try:
        set_custom_llama_cpp_path(payload.path)
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            str(exc),
            event = "settings.update_llama_cpp_path_failed",
            log = logger,
        ) from exc
    return _llama_cpp_path_response()


@_shared_settings_router.get("/multi-model", response_model = MultiModelResponse)
def get_multi_model(current_subject: str = Depends(get_current_subject)) -> MultiModelResponse:
    return MultiModelResponse(enabled = get_multi_model_enabled())


@_owner_settings_router.put("/multi-model", response_model = MultiModelResponse)
def update_multi_model(
    payload: MultiModelPayload,
    background_tasks: BackgroundTasks,
    current_subject: str = Depends(get_current_subject),
) -> MultiModelResponse:
    """Keep the loaded models when another loads. Takes effect on the next load."""
    try:
        enabled = set_multi_model_enabled(payload.enabled)
    except Exception as exc:
        raise log_and_http_error(
            exc,
            500,
            safe_error_detail(exc, fallback = "Could not save the multiple models setting."),
            event = "settings.update_multi_model_failed",
            log = logger,
        ) from exc
    logger.info("settings.multi_model_updated subject=%s enabled=%s", current_subject, enabled)
    if not enabled:
        # Back to one model: the idle kept ones go after the reply (a teardown can take minutes),
        # a busy one once it is ejected. One that fails to unload stays tracked as stuck.
        background_tasks.add_task(_unload_idle_models)
    return MultiModelResponse(enabled = enabled)


def _unload_idle_models() -> None:
    from core.inference import model_slots
    from routes.inference import release_chat_after_kept_models

    try:
        model_slots.unload_idle()
    except Exception:
        logger.warning("settings.multi_model_unload_idle_failed", exc_info = True)
    # The primary's own unload kept CHAT while these were loaded.
    release_chat_after_kept_models()


@_shared_settings_router.get("/upload-limit", response_model = UploadLimitResponse)
def get_upload_limit(current_subject: str = Depends(get_current_subject)) -> UploadLimitResponse:
    return _upload_limit_response(get_upload_limit_mb())


@_owner_settings_router.put("/upload-limit", response_model = UploadLimitResponse)
def update_upload_limit(
    payload: UploadLimitPayload, current_subject: str = Depends(get_current_subject)
) -> UploadLimitResponse:
    try:
        limit_mb = set_upload_limit_mb(payload.max_upload_size_mb)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid upload limit."),
            event = "settings.update_upload_limit_failed",
            log = logger,
        ) from exc
    return _upload_limit_response(limit_mb)


@_shared_settings_router.get("/helper-precache", response_model = HelperPrecacheResponse)
def get_helper_precache(
    current_subject: str = Depends(get_current_subject),
) -> HelperPrecacheResponse:
    return _helper_precache_response()


@_owner_settings_router.put("/helper-precache", response_model = HelperPrecacheResponse)
def update_helper_precache(
    payload: HelperPrecachePayload, current_subject: str = Depends(get_current_subject)
) -> HelperPrecacheResponse:
    try:
        enabled = set_helper_precache_enabled(payload.enabled)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid Helper LLM pre-cache setting."),
            event = "settings.update_helper_precache_failed",
            log = logger,
        ) from exc
    return _helper_precache_response(enabled)


def _clef_availability(checkpoint, reason: Optional[str]) -> dict:
    from core.systemone import laya_runtime

    if getattr(checkpoint, "layout", "laya") == laya_runtime.GGUF:
        # Selectable under a PyTorch runtime: the runtime row then says to switch it.
        try:
            laya_runtime.select(checkpoint, preference = "auto")
        except laya_runtime.Unavailable as exc:
            return {"llama_cpp_only": True, "available": False, "unavailable_reason": exc.message}
        return {"llama_cpp_only": True}
    if reason is None or getattr(checkpoint, "layout", "laya") != "clef":
        return {}
    # llama.cpp serves Clef without CUDA or ROCm.
    if laya_runtime.native_ready(checkpoint):
        return {}
    return {"available": False, "unavailable_reason": reason}


def _systemone_response(request: Request) -> SystemOneSettingsResponse:
    from pathlib import Path

    from core.systemone import catalog, laya_runtime
    from routes.systemone import MCP_PATH

    clef_reason = catalog.clef_unsupported_reason(wait = False)
    enabled = systemone_settings.get_enabled()
    runtime = laya_runtime.status()
    configured = catalog.default_checkpoint()
    model = configured.name
    if is_owner_context():
        fine_tunes = catalog.fine_tunes()
    else:
        # Other accounts see only the configured model, never the owner's other output folders.
        fine_tunes = [configured] if catalog.is_fine_tune_name(configured.name) else []
        if runtime["loaded_model"] != model:
            runtime["loaded_model"] = runtime["device"] = None
        if runtime["loading_model"] != model:
            runtime["loading_model"] = None
    error = runtime["error"]
    if runtime["error_model"] not in (None, model):
        error = None
    port = getattr(request.app.state, "server_port", None) or request.scope["server"][1]
    effective, fallback = laya_runtime.effective_backend(configured)
    if runtime["loaded_model"] == model and runtime["fallback_reason"]:
        fallback = runtime["fallback_reason"]
    return SystemOneSettingsResponse(
        enabled = enabled,
        enabled_locked = systemone_settings.enabled_locked(),
        model = model,
        model_locked = systemone_settings.model_locked(),
        # llama.cpp defaults to the GPU when no device is stored; report where it actually runs.
        device = systemone_settings.clef_device()
        if effective == "llama.cpp"
        else systemone_settings.get_device(),
        device_locked = systemone_settings.device_locked(),
        gpu_available = systemone_settings.gpu_available(),
        models = [
            SystemOneModelOption(
                name = c.name,
                description = c.description,
                download_bytes = c.download_bytes,
                label = c.label,
                **_clef_availability(c, clef_reason),
            )
            for c in catalog.CHECKPOINTS.values()
        ]
        + [
            SystemOneModelOption(
                name = c.name,
                description = c.description,
                download_bytes = 0,
                kind = "fine_tune",
                label = Path(c.source).name,
                **_clef_availability(c, clef_reason),
            )
            for c in fine_tunes
        ],
        loaded_model = runtime["loaded_model"],
        loaded_device = runtime["device"],
        loading_model = runtime["loading_model"],
        installing = runtime["installing"],
        error = error,
        mcp_url = f"http://127.0.0.1:{port}{MCP_PATH}/",
        backend = systemone_settings.get_backend(),
        native_ctx = systemone_settings.get_native_ctx(),
        effective_backend = effective,
        loaded_backend = runtime["loaded_backend"] if runtime["loaded_model"] else None,
        fallback_reason = fallback if effective == "pytorch" or effective is None else None,
        input_modalities = laya_runtime.input_modalities(configured),
        layout = getattr(configured, "layout", None),
    )


_SYSTEMONE_SETTINGS_LOCK = threading.Lock()


def _systemone_values(payload: SystemOneSettingsPayload) -> dict[str, Any]:
    try:
        return systemone_settings.validate(
            **payload.model_dump(
                include = {"enabled", "model", "device", "backend", "native_ctx"},
                exclude_none = True,
            )
        )
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_curated_detail(exc, fallback = "Invalid Decision API setting."),
            event = "settings.update_systemone_failed",
            log = logger,
        ) from exc


def _check_systemone_expectations(payload: SystemOneSettingsPayload) -> None:
    from core.systemone import catalog
    changed = (
        payload.expected_enabled is not None
        and systemone_settings.get_enabled() != payload.expected_enabled
    ) or (
        payload.expected_model is not None
        and catalog.default_checkpoint().name != payload.expected_model
    )
    if changed:
        raise HTTPException(status_code = 409, detail = "Decision API settings changed. Try again.")


# Not the shared router, which reads as the owner for everyone: this answer depends on who asks.
@_account_settings_router.get("/systemone", response_model = SystemOneSettingsResponse)
def get_systemone_settings(
    request: Request, current_subject: str = Depends(get_current_subject)
) -> SystemOneSettingsResponse:
    return _systemone_response(request)


async def _refresh_decision_models(payload: SystemOneSettingsPayload) -> None:
    from core.systemone import catalog
    from routes.systemone import refresh_listed_decision_models
    if catalog.parse_connection(payload.model):
        await refresh_listed_decision_models()


@_owner_settings_router.put("/systemone", response_model = SystemOneSettingsResponse)
async def update_systemone_settings(
    payload: SystemOneSettingsPayload,
    request: Request,
    current_subject: str = Depends(get_current_subject),
) -> SystemOneSettingsResponse:
    await _refresh_decision_models(payload)
    return await asyncio.to_thread(_save_systemone_settings, payload, request)


def _save_systemone_settings(
    payload: SystemOneSettingsPayload, request: Request
) -> SystemOneSettingsResponse:
    from core.systemone import laya_runtime
    with _SYSTEMONE_SETTINGS_LOCK:
        _check_systemone_expectations(payload)
        values = _systemone_values(payload)
        if values:
            # The resident model was built from the old settings; drop it so the next request uses the new ones.
            try:
                laya_runtime.unload()
            except laya_runtime.Unavailable as exc:
                raise HTTPException(status_code = 409, detail = exc.message) from None
            systemone_settings.save(values)
    return _systemone_response(request)


@_owner_settings_router.post("/systemone/validate", status_code = 204)
async def validate_systemone_settings(
    payload: SystemOneSettingsPayload, current_subject: str = Depends(get_current_subject)
) -> None:
    await _refresh_decision_models(payload)
    await asyncio.to_thread(_validate_systemone_settings, payload)


def _validate_systemone_settings(payload: SystemOneSettingsPayload) -> None:
    from core.systemone import laya_runtime
    with _SYSTEMONE_SETTINGS_LOCK:
        _check_systemone_expectations(payload)
        values = _systemone_values(payload)
        if values:
            try:
                laya_runtime.ensure_can_unload()
            except laya_runtime.Unavailable as exc:
                raise HTTPException(status_code = 409, detail = exc.message) from None


@_owner_settings_router.get(
    "/systemone/connections", response_model = list[SystemOneConnectionOption]
)
async def list_systemone_connections(
    current_subject: str = Depends(get_current_subject),
) -> list[SystemOneConnectionOption]:
    from core.systemone import catalog
    from routes.systemone import refresh_listed_decision_models

    await refresh_listed_decision_models()
    return [
        SystemOneConnectionOption(
            name = catalog.Connection(row["id"], model).name,
            provider_id = row["id"],
            provider = row["display_name"],
            model = model,
        )
        for row, models in await asyncio.to_thread(catalog.decision_connections)
        for model in models
    ]


@_owner_settings_router.get("/systemone/resolve", response_model = SystemOneDownloadPlan)
def resolve_systemone_download(
    model: Optional[str] = None,
    backend: Optional[str] = None,
    current_subject: str = Depends(get_current_subject),
) -> SystemOneDownloadPlan:
    from core.systemone import catalog, laya_runtime

    checkpoint = (
        catalog.default_checkpoint()
        if model is None
        else catalog.parse_connection(model) or catalog.resolve(model)
    )
    if checkpoint is None:
        raise HTTPException(status_code = 400, detail = "Unknown Decision API model.")
    if isinstance(checkpoint, catalog.Connection):
        return SystemOneDownloadPlan(files = [], size_bytes = 0, cached = True)
    if backend is not None and backend not in systemone_settings.BACKENDS:
        raise HTTPException(status_code = 400, detail = "Unknown Decision API runtime.")
    return SystemOneDownloadPlan(**laya_runtime.download_plan(checkpoint, preference = backend))


@_owner_settings_router.post("/systemone/unload", response_model = SystemOneSettingsResponse)
def unload_systemone_model(
    request: Request, current_subject: str = Depends(get_current_subject)
) -> SystemOneSettingsResponse:
    from core.systemone import laya_runtime
    try:
        laya_runtime.unload()
    except laya_runtime.Unavailable as exc:
        raise HTTPException(status_code = 409, detail = exc.message) from None
    return _systemone_response(request)


@_shared_settings_router.get("/download-transport", response_model = DownloadTransportResponse)
def get_download_transport(
    current_subject: str = Depends(get_current_subject),
) -> DownloadTransportResponse:
    return _download_transport_response()


@_owner_settings_router.put("/download-transport", response_model = DownloadTransportResponse)
def update_download_transport(
    payload: DownloadTransportPayload, current_subject: str = Depends(get_current_subject)
) -> DownloadTransportResponse:
    try:
        mode = set_download_transport_mode(payload.mode)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid download transport."),
            event = "settings.update_download_transport_failed",
            log = logger,
        ) from exc
    return _download_transport_response(mode)


def _hub_settings_response(settings: HubSettings) -> HubSettingsResponse:
    return HubSettingsResponse(
        hf_endpoint = settings.hf_endpoint,
        datasets_server_follows_endpoint = settings.datasets_server_follows_endpoint,
        source = settings.source,
        active_source = active_source(),
    )


# Owner only: the endpoint can name a private address that other accounts' clients must not learn.
@_owner_settings_router.get("/hub", response_model = HubSettingsResponse)
def get_hub(current_subject: str = Depends(get_current_subject)) -> HubSettingsResponse:
    return _hub_settings_response(get_hub_settings())


@_owner_settings_router.put("/hub", response_model = HubSettingsResponse)
def update_hub(
    payload: HubSettingsPayload,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> HubSettingsResponse:
    # The endpoint receives the installation's Hugging Face token, like the token routes above.
    require_ui_session(via_api_key)
    try:
        settings = set_hub_settings(payload.hf_endpoint, payload.datasets_server_follows_endpoint)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid Hugging Face endpoint."),
            event = "settings.update_hub_failed",
            log = logger,
        ) from exc
    return _hub_settings_response(settings)


@_owner_settings_router.put("/hub/source", response_model = HubSettingsResponse)
def update_hub_source(
    payload: HubSourcePayload,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> HubSettingsResponse:
    require_ui_session(via_api_key)
    return _hub_settings_response(set_hub_source(payload.source))


@_owner_settings_router.post("/hub/source-notice", response_model = HubSourceNoticeResponse)
def claim_hub_source_notice(
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> HubSourceNoticeResponse:
    """Keep the automatic ModelScope default; granted once, to the UI that tells the owner."""
    require_ui_session(via_api_key)
    return HubSourceNoticeResponse(granted = claim_automatic_source())


@_owner_settings_router.post("/xet-notice/reserve", response_model = XetNoticeResponse)
def post_xet_notice_reserve(
    payload: XetNoticeReservePayload, current_subject: str = Depends(get_current_subject)
) -> XetNoticeResponse:
    """Take one of the remaining notices. POST because it mutates the count."""
    try:
        result = reserve_xet_notice(payload.seen_hint)
    except Exception as exc:
        raise log_and_http_error(
            exc,
            500,
            safe_error_detail(exc, fallback = "Could not reserve the Xet download notice."),
            event = "settings.reserve_xet_notice_failed",
            log = logger,
        ) from exc
    return XetNoticeResponse(**result)


@_account_settings_router.post(
    "/igpu-carveout-notice/dismiss", response_model = IgpuCarveoutNoticeResponse
)
def post_igpu_carveout_notice_dismiss(
    payload: IgpuCarveoutNoticeDismissPayload, current_subject: str = Depends(get_current_subject)
) -> IgpuCarveoutNoticeResponse:
    """Stop offering the integrated-GPU memory advice at this allocation.

    Stored server-side rather than in the browser: an Unsloth origin is not stable,
    so a per-origin store hands out a fresh notice every time the port moves.
    """
    from utils.igpu_carveout_notice_settings import dismiss_notice

    try:
        stored = dismiss_notice(payload.current_gb)
    except Exception as exc:
        raise log_and_http_error(
            exc,
            500,
            safe_error_detail(exc, fallback = "Could not dismiss the GPU memory notice."),
            event = "settings.dismiss_igpu_carveout_notice_failed",
            log = logger,
        ) from exc
    return IgpuCarveoutNoticeResponse(dismissed_at_gb = stored)


@_account_settings_router.get("/chat-preferences", response_model = ChatPreferencesResponse)
def get_chat_preferences(
    current_subject: str = Depends(get_current_subject),
) -> ChatPreferencesResponse:
    return _chat_preferences_response()


@_account_settings_router.put("/chat-preferences", response_model = ChatPreferencesResponse)
def update_chat_preferences(
    payload: ChatPreferencesPayload, current_subject: str = Depends(get_current_subject)
) -> ChatPreferencesResponse:
    try:
        enabled = set_show_model_disclaimer(payload.show_model_disclaimer)
    except Exception as exc:
        raise log_and_http_error(
            exc,
            500,
            safe_error_detail(exc, fallback = "Could not save chat preferences."),
            event = "settings.update_chat_preferences_failed",
            log = logger,
        ) from exc
    return _chat_preferences_response(enabled)


@_account_settings_router.post("/chat-preferences/migrate", response_model = ChatPreferencesResponse)
def migrate_chat_preferences(
    payload: ChatPreferencesMigrationPayload, current_subject: str = Depends(get_current_subject)
) -> ChatPreferencesResponse:
    try:
        enabled = migrate_show_model_disclaimer(payload.show_model_disclaimer)
    except Exception as exc:
        raise log_and_http_error(
            exc,
            500,
            safe_error_detail(exc, fallback = "Could not migrate chat preferences."),
            event = "settings.migrate_chat_preferences_failed",
            log = logger,
        ) from exc
    return _chat_preferences_response(enabled)


@_owner_settings_router.get("/model-memory", response_model = ModelMemoryResponse)
def get_model_memory(current_subject: str = Depends(get_current_subject)) -> ModelMemoryResponse:
    return _model_memory_response()


@_owner_settings_router.put("/model-memory", response_model = ModelMemoryResponse)
def update_model_memory(
    payload: ModelMemoryPayload, current_subject: str = Depends(get_current_subject)
) -> ModelMemoryResponse:
    try:
        set_model_memory_settings(
            keep_resident = payload.keep_resident,
            no_ram_reserve = payload.no_ram_reserve,
        )
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid model memory setting."),
            event = "settings.update_model_memory_failed",
            log = logger,
        ) from exc
    return _model_memory_response()


LAST_LOCAL_MODEL_SETTING_KEY = "last_local_model_load"
_LAST_LOCAL_MODEL_LOCK = threading.Lock()


def _last_local_model_key(subject: str) -> str:
    """Per-subject key: one shared row would hand user B user A's last model."""
    if not current_account().is_owner:
        subject = current_account().account_id
    subject = (subject or "").strip()
    if not subject:
        return LAST_LOCAL_MODEL_SETTING_KEY
    # Hashed so an arbitrary subject cannot collide with another key.
    digest = hashlib.sha256(subject.encode("utf-8")).hexdigest()[:32]
    return f"{LAST_LOCAL_MODEL_SETTING_KEY}:{digest}"


def _read_last_local_model(subject: str) -> "dict | None":
    """The subject's record, falling back to the pre-scoping shared row so an
    upgrade keeps the model the install already remembered."""
    from storage.studio_db import get_app_setting

    stored = get_app_setting(_last_local_model_key(subject), None)
    if not isinstance(stored, dict):
        stored = get_app_setting(LAST_LOCAL_MODEL_SETTING_KEY, None)
    return stored if isinstance(stored, dict) else None


# Clients stamp loads, so cap how far ahead of server time a client clock may claim.
_LAST_LOCAL_MODEL_CLOCK_SLACK_MS = 5 * 60 * 1000


class LastLocalModelPayload(BaseModel):
    id: str = Field(..., min_length = 1, max_length = MAX_MODEL_OVERRIDE_KEY_LEN)
    kind: Literal["gguf", "model"]
    gguf_variant: Optional[str] = Field(default = None, max_length = MAX_GGUF_VARIANT_KEY_LEN)
    # Epoch ms of the load; orders writes from surfaces that keep their own local shadow.
    loaded_at: Optional[int] = Field(default = None, ge = 0)
    # The client clock when the request was sent: the skew (server_now - client_now) translates loaded_at
    # into the server frame. Never persisted.
    client_now: Optional[int] = Field(default = None, ge = 0)


class LastLocalModelResponse(BaseModel):
    id: Optional[str] = None
    kind: Optional[Literal["gguf", "model"]] = None
    gguf_variant: Optional[str] = None
    loaded_at: Optional[int] = None
    # Lets the client translate loaded_at back into its own clock frame.
    server_now: Optional[int] = None


@_account_settings_router.get("/last-local-model", response_model = LastLocalModelResponse)
def get_last_local_model(
    current_subject: str = Depends(get_current_subject),
) -> LastLocalModelResponse:
    stored = _read_last_local_model(current_subject)
    _now = int(time.time() * 1000)
    if stored is None:
        return LastLocalModelResponse(server_now = _now)
    try:
        payload = LastLocalModelPayload(**stored)
    except Exception:
        return LastLocalModelResponse(server_now = _now)
    return LastLocalModelResponse(**payload.model_dump(exclude = {"client_now"}), server_now = _now)


@_account_settings_router.put("/last-local-model", response_model = LastLocalModelResponse)
def update_last_local_model(
    payload: LastLocalModelPayload, current_subject: str = Depends(get_current_subject)
) -> LastLocalModelResponse:
    from storage.studio_db import upsert_app_settings

    # loaded_at orders stamped writes so a delayed older PUT cannot overwrite a newer
    # load; the stored record is returned. Unstamped writes stay last-write-wins.
    _server_now = int(time.time() * 1000)
    _key = _last_local_model_key(current_subject)
    with _LAST_LOCAL_MODEL_LOCK:
        if payload.loaded_at is not None:
            if payload.client_now is not None:
                # Into the server frame: fresh loads land near now, re-issued shadows stay old.
                _shifted = payload.loaded_at + (_server_now - payload.client_now)
                payload = payload.model_copy(update = {"loaded_at": max(0, _shifted)})
            _cap = _server_now + _LAST_LOCAL_MODEL_CLOCK_SLACK_MS
            if payload.loaded_at > _cap:
                payload = payload.model_copy(update = {"loaded_at": _cap})
            stored = _read_last_local_model(current_subject)
            if stored is not None:
                try:
                    current = LastLocalModelPayload(**stored)
                except Exception:
                    current = None
                if (
                    current is not None
                    and current.loaded_at is not None
                    and payload.loaded_at < current.loaded_at
                ):
                    return LastLocalModelResponse(
                        **current.model_dump(exclude = {"client_now"}), server_now = _server_now
                    )
        upsert_app_settings({_key: payload.model_dump(exclude = {"client_now"})})
    return LastLocalModelResponse(
        **payload.model_dump(exclude = {"client_now"}), server_now = _server_now
    )


class DiffusionAcceleratorFallbackRecord(BaseModel):
    accelerator: str
    fallback: Optional[str] = None
    # Qualifying failures under the current fingerprint; `proven` means one named the BUILD.
    strikes: int = 0
    proven: bool = False
    diverting: bool = False
    # Taken under a different driver, bundle or set of cards, so it is already inert.
    stale: bool = False


class DiffusionAcceleratorFallbackResponse(BaseModel):
    records: list[DiffusionAcceleratorFallbackRecord] = []
    # False when UNSLOTH_DIFFUSION_SD_CPP_VULKAN_FALLBACK is off, where no record can divert.
    enabled: bool = True
    diverting: bool = False


PINNED_MODELS_SETTING_KEY = "model_picker_pinned"
PINNED_CONNECTED_MODELS_SETTING_KEY = "model_picker_pinned_connected"
# Embedding models pinned to the RAG menu.
PINNED_EMBEDDING_MODELS_SETTING_KEY = "rag_embedding_pinned"
MAX_PINNED_MODELS = 512
# Room for a "::quant" suffix or an "external::<connection>::" prefix on top of a model id.
_MAX_PIN_KEY_LEN = MAX_MODEL_OVERRIDE_KEY_LEN + 512
_PinKey = Annotated[str, StringConstraints(min_length = 1, max_length = _MAX_PIN_KEY_LEN)]


class PinnedModelsPayload(BaseModel):
    """Either list may be omitted; only what is sent is replaced."""

    model_config = ConfigDict(extra = "forbid")

    pinned: Optional[list[_PinKey]] = Field(default = None, max_length = MAX_PINNED_MODELS)
    connected: Optional[list[_PinKey]] = Field(default = None, max_length = MAX_PINNED_MODELS)
    embedding: Optional[list[_PinKey]] = Field(default = None, max_length = MAX_PINNED_MODELS)


class PinnedModelsResponse(BaseModel):
    # None = never stored, so the browser seeds it.
    pinned: Optional[list[str]] = None
    connected: Optional[list[str]] = None
    embedding: Optional[list[str]] = None


def _pinned_models_response() -> PinnedModelsResponse:
    from storage.studio_db import get_app_settings

    stored = get_app_settings(
        [
            PINNED_MODELS_SETTING_KEY,
            PINNED_CONNECTED_MODELS_SETTING_KEY,
            PINNED_EMBEDDING_MODELS_SETTING_KEY,
        ]
    )

    def _ids(value: Any) -> Optional[list[str]]:
        return [v for v in value if isinstance(v, str)] if isinstance(value, list) else None

    return PinnedModelsResponse(
        pinned = _ids(stored.get(PINNED_MODELS_SETTING_KEY)),
        connected = _ids(stored.get(PINNED_CONNECTED_MODELS_SETTING_KEY)),
        embedding = _ids(stored.get(PINNED_EMBEDDING_MODELS_SETTING_KEY)),
    )


@_account_settings_router.get("/pinned-models", response_model = PinnedModelsResponse)
def get_pinned_models(current_subject: str = Depends(get_current_subject)) -> PinnedModelsResponse:
    """Per-account picker pins: an account switch clears the browser copy."""
    return _pinned_models_response()


@_account_settings_router.put("/pinned-models", response_model = PinnedModelsResponse)
def update_pinned_models(
    payload: PinnedModelsPayload, current_subject: str = Depends(get_current_subject)
) -> PinnedModelsResponse:
    from storage.studio_db import upsert_app_settings

    updates: dict[str, Any] = {}
    if payload.pinned is not None:
        updates[PINNED_MODELS_SETTING_KEY] = list(dict.fromkeys(payload.pinned))
    if payload.connected is not None:
        updates[PINNED_CONNECTED_MODELS_SETTING_KEY] = list(dict.fromkeys(payload.connected))
    if payload.embedding is not None:
        updates[PINNED_EMBEDDING_MODELS_SETTING_KEY] = list(dict.fromkeys(payload.embedding))
    if updates:
        upsert_app_settings(updates, read_back = False)
    return _pinned_models_response()


def _diffusion_accelerator_fallback_response() -> DiffusionAcceleratorFallbackResponse:
    from core.inference.sd_cpp_backend import accelerator_runtime_failure_state
    return DiffusionAcceleratorFallbackResponse(**accelerator_runtime_failure_state())


@_owner_settings_router.get(
    "/diffusion-accelerator-fallback", response_model = DiffusionAcceleratorFallbackResponse
)
def get_diffusion_accelerator_fallback(
    current_subject: str = Depends(get_current_subject),
) -> DiffusionAcceleratorFallbackResponse:
    """Which native diffusion accelerators this host has been recorded as unable to run.

    Upstream publishes one generic ROCm stable-diffusion.cpp build, not one per gfx arch, so a card
    it carries no kernels for cannot start it and the host moves to Vulkan (#9278, #8814).
    """
    return _diffusion_accelerator_fallback_response()


@_owner_settings_router.delete(
    "/diffusion-accelerator-fallback", response_model = DiffusionAcceleratorFallbackResponse
)
def clear_diffusion_accelerator_fallback(
    current_subject: str = Depends(get_current_subject),
) -> DiffusionAcceleratorFallbackResponse:
    """Forget the records, so the next load tries this host's own accelerator again.

    A driver upgrade or a new card retires them through the fingerprint; this is the way back for a
    fix it cannot see. Reinstalling does not clear them: the record lives in settings, not the tree.
    """
    from core.inference.sd_cpp_backend import clear_accelerator_runtime_failures

    clear_accelerator_runtime_failures()
    return _diffusion_accelerator_fallback_response()


@_owner_settings_router.get("/vram-budget", response_model = VramBudgetResponse)
def get_vram_budget(current_subject: str = Depends(get_current_subject)) -> VramBudgetResponse:
    return _vram_budget_response()


@_owner_settings_router.put("/vram-budget", response_model = VramBudgetResponse)
def update_vram_budget(
    payload: VramBudgetPayload, current_subject: str = Depends(get_current_subject)
) -> VramBudgetResponse:
    try:
        set_vram_budget_fraction(payload.fraction)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid VRAM budget."),
            event = "settings.update_vram_budget_failed",
            log = logger,
        ) from exc
    return _vram_budget_response()


class CodingAgentsResponse(BaseModel):
    # All agents `unsloth start` supports, in the CLI's declared order.
    agents: tuple[str, ...] = CODING_AGENTS
    # Subset of `agents` whose CLI binary was found on PATH; the frontend uses
    # this to default the API-keys panel to a command the user can run as-is.
    detected: list[str]


@_owner_settings_router.get("/coding-agents", response_model = CodingAgentsResponse)
def get_coding_agents(current_subject: str = Depends(get_current_subject)) -> CodingAgentsResponse:
    return CodingAgentsResponse(detected = detect_installed_coding_agents())


@_owner_settings_router.get("/openai-auto-switch", response_model = OpenAIAutoSwitchResponse)
def get_openai_auto_switch(
    current_subject: str = Depends(get_current_subject),
) -> OpenAIAutoSwitchResponse:
    return OpenAIAutoSwitchResponse(
        enabled = get_openai_auto_switch_enabled(),
        auto_unload_idle_seconds = get_stored_auto_unload_idle_seconds(),
        idle_unload_active = get_auto_unload_idle_seconds() > 0,
        auto_unload_keep_kv = get_auto_unload_keep_kv(),
        auto_download_model = get_stored_openai_auto_download_enabled(),
        auto_unload_api_only = get_auto_unload_api_only(),
        media_auto_unload_idle_seconds = get_stored_media_auto_unload_idle_seconds(),
        media_idle_unload_active = get_media_auto_unload_idle_seconds() > 0,
        media_auto_switch_model = get_media_auto_switch_enabled(),
    )


@_owner_settings_router.put("/openai-auto-switch", response_model = OpenAIAutoSwitchResponse)
def update_openai_auto_switch(
    payload: OpenAIAutoSwitchPayload, current_subject: str = Depends(get_current_subject)
) -> OpenAIAutoSwitchResponse:
    try:
        (
            enabled,
            idle_seconds,
            keep_kv,
            auto_download,
            api_only,
            media_idle_seconds,
            media_auto_switch,
        ) = set_openai_auto_switch(
            payload.enabled,
            payload.auto_unload_idle_seconds,
            payload.auto_unload_keep_kv,
            payload.auto_download_model,
            payload.auto_unload_api_only,
            payload.media_auto_unload_idle_seconds,
            payload.media_auto_switch_model,
        )
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid OpenAI auto-switch setting."),
            event = "settings.update_openai_auto_switch_failed",
            log = logger,
        ) from exc
    idle_unload_active = get_auto_unload_idle_seconds() > 0
    if not keep_kv or not idle_unload_is_configured():
        from core.inference.llama_keepwarm import purge_kv_resume
        purge_kv_resume()
    return OpenAIAutoSwitchResponse(
        enabled = enabled,
        auto_unload_idle_seconds = idle_seconds,
        idle_unload_active = idle_unload_active,
        auto_unload_keep_kv = keep_kv,
        auto_download_model = auto_download,
        auto_unload_api_only = api_only,
        media_auto_unload_idle_seconds = media_idle_seconds,
        media_idle_unload_active = get_media_auto_unload_idle_seconds() > 0,
        media_auto_switch_model = media_auto_switch,
    )


@_account_settings_router.get(
    "/openai-auto-switch/overrides", response_model = ModelOverridesResponse
)
def get_openai_auto_switch_overrides(
    model_id: Optional[str] = None,
    alias_id: Optional[str] = None,
    gguf_variant: Optional[str] = None,
    current_subject: str = Depends(get_current_subject),
) -> ModelOverridesResponse:
    """Every stored override, and optionally the one a named model's load would use.

    The resolution is the loader's own (``resolve_override_for_load``), so what a
    panel shows and what a load applies cannot disagree.
    """
    resolved_key: Optional[str] = None
    resolved: Optional[dict] = None
    if model_id:
        from utils.openai_auto_switch_settings import resolve_override_for_load
        resolved_key, resolved = resolve_override_for_load(model_id, alias_id, gguf_variant)
    return ModelOverridesResponse(
        overrides = get_model_overrides(),
        resolved = resolved,
        resolved_key = resolved_key,
    )


def _bare_model_id(model_id: str) -> Optional[str]:
    """``repo`` for a ``repo:QUANT`` key, or None when there is no quant suffix."""
    from utils.openai_auto_switch_settings import split_quant_suffix

    # Must look like a quant, not a short path segment; a bpw modifier and stem label both count.
    split = split_quant_suffix(model_id)
    return split[0] if split is not None else None


def _fallback_supplies_extra_args(model_id: str, target_id: str) -> bool:
    """Whether a load for this model would still pick flags off another entry. The carry-over copies a legacy
    bare ``repo`` row's flags onto the first ``repo:QUANT`` save and leaves the bare row in place, and a
    load reads the qualified key first and the bare one after it, so clearing the box for the quant is only
    a clear while the quant keeps a row of its own. Answered rather than repaired: stripping the flags off
    the bare row was the first fix and it is too broad, since that row is the fallback for every quant that
    has no row."""
    from utils.openai_auto_switch_settings import get_model_override

    for candidate in (
        _bare_model_id(model_id),
        _legacy_standalone_gguf_key(model_id),
    ):
        if (
            candidate
            and candidate != target_id
            and get_model_override(candidate).get("llama_extra_args")
        ):
            return True
    return False


def _fallback_supplies_reasoning_flag(model_id: str, target_id: str) -> bool:
    """Whether a load for this model would still pick a reasoning flag off another entry.

    The -1/"" pair is stored rather than dropped so a qualified row survives as a tombstone that
    shadows such a flag. A later save leaving the controls at their defaults omits the pair, and
    without this the row can empty out, be deleted, and hand the reset value straight back.
    """
    from core.inference.llama_server_args import (
        parse_reasoning_budget_message_override,
        parse_reasoning_budget_override,
    )
    from utils.openai_auto_switch_settings import get_model_override

    for candidate in (
        _bare_model_id(model_id),
        _legacy_standalone_gguf_key(model_id),
    ):
        if not candidate or candidate == target_id:
            continue
        stored = get_model_override(candidate)
        if stored.get("reasoning_budget", -1) != -1 or stored.get("reasoning_budget_message"):
            return True
        stored_args = stored.get("llama_extra_args")
        if not stored_args:
            continue
        try:
            if (
                parse_reasoning_budget_override(stored_args) is not None
                or parse_reasoning_budget_message_override(stored_args) is not None
            ):
                return True
        except ValueError:
            # A malformed stored flag is the loader's problem, not this save's.
            continue
    return False


def _other_quants_remain(bare_id: str, removed_ids: list[str]) -> bool:
    """Whether a quant of ``bare_id`` other than the ones being removed still has an entry. Such a quant has its
    own settings and never reads the bare fallback, so this is not "is anyone inheriting" but "is this
    forget the last one for the model"."""
    from utils.openai_auto_switch_settings import split_quant_suffix

    removed = {key.strip().lower() for key in removed_ids}
    prefix = bare_id.strip().lower()
    for key, entry in get_model_overrides().items():
        if not isinstance(entry, dict) or key.strip().lower() in removed:
            continue
        split = split_quant_suffix(key)
        if split is not None and split[0].strip().lower() == prefix:
            return True
    return False


def _legacy_standalone_gguf_key(model_id: str) -> Optional[str]:
    """The stored ``<path>:LABEL`` entry for a bare standalone .gguf path, if any. A loose file has no quant to
    choose between, so it is keyed by the bare path, but the label derived from its filename is never empty
    and that is how the picker keyed the same file before, so an upgraded install carries entries under it.
    The auto-switch loader reads that spelling after the bare path misses."""
    import os

    if not model_id.lower().endswith(".gguf"):
        return None
    # Already qualified, so the caller named the entry it meant, as the loader does.
    if _bare_model_id(model_id) is not None:
        return None
    from hub.utils.gguf import extract_quant_label

    label = extract_quant_label(os.path.basename(model_id))
    if not label:
        return None
    # Through the resolver: the browser lowercases the variant, and an ambiguous fold misses.
    return resolve_model_override_key(f"{model_id}:{label}")


def _fill_target_id(target_id: str) -> str:
    """Where a one-time backfill write for ``target_id`` has to land. A fill only adds, so unlike a save it
    cannot retire the other spelling of a cached repo. Creating the snapshot-path key while the server
    already holds the repo id would leave two entries for one quant, and the loader reads the load path
    before the advertised id, so an upgraded browser's pre-upgrade copy would shadow the newer server
    config. Only in that direction: a repo-id key never outranks an existing path entry, and two snapshot
    paths name two caches."""
    from core.inference.model_ids import hf_cache_repo_id
    from utils.openai_auto_switch_settings import split_quant_suffix

    # Already stored, so this write creates no second key to outrank anything.
    if isinstance(get_model_overrides().get(target_id), dict):
        return target_id
    split = split_quant_suffix(target_id)
    # A bare id backs every quant and is read last, and only a cache path outranks.
    if split is None or hf_cache_repo_id(split[0]) is None:
        return target_id
    for alias_id in cached_repo_alias_keys(target_id):
        alias_split = split_quant_suffix(alias_id)
        if alias_split is not None and hf_cache_repo_id(alias_split[0]) is None:
            return alias_id
    return target_id


# One override write at a time. A save stores its target key and then reads the map back to retire the other
# spelling of the same cached repo, and a remove clears up to four keys, each its own transaction: atomic on
# their own, but not as a sequence. This route is a plain `def`, so FastAPI runs it in a threadpool, and two
# clients saving one quant under both spellings could each write before either cleanup ran and then retire the
# other's row, leaving no override at all from two saves that both returned 200.
_override_write_lock = threading.Lock()


def _serialized_override_write(func):
    """Run ``func`` under _override_write_lock. functools.wraps carries __wrapped__, which
    inspect.signature follows, so FastAPI still sees the endpoint's own parameters and dependencies."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _override_write_lock:
            return func(*args, **kwargs)

    return wrapper


@_account_settings_router.put(
    "/openai-auto-switch/overrides", response_model = ModelOverridesResponse
)
@_serialized_override_write
def update_openai_auto_switch_override(
    payload: ModelOverridePayload, current_subject: str = Depends(get_current_subject)
) -> ModelOverridesResponse:
    from core.inference.llama_server_args import (
        drop_managed_flags,
        parse_ctx_override,
        strip_shadowing_flags,
        validate_extra_args,
    )
    from utils.openai_auto_switch_settings import MAX_SEQ_LENGTH_CEILING, get_model_override

    try:
        if payload.fill_absent_fields and payload.remove is True:
            # A fill that is also a delete has no meaning; picking one loses or resurrects.
            raise ValueError("fill_absent_fields cannot be combined with remove.")
        # Only model_id is the documented "remove"; otherwise omitted flags carry over.
        requested_extra_args = payload.llama_extra_args
        # fill_absent_fields and mirrors_server_tuning are write modes. Leaving either in would make every payload
        # look non-empty (they are bools, so exclude_none does not drop them) and break the legacy "no fields
        # means remove".
        saved_fields = payload.model_dump(
            exclude = {
                "model_id",
                "llama_extra_args",
                "remove",
                "fill_absent_fields",
                "mirrors_server_tuning",
                "mirrors_reasoning_budget",
            },
            exclude_none = True,
        )
        if payload.remove is not None:
            is_removal = payload.remove
        else:
            is_removal = (
                not payload.tensor_parallel
                and not payload.disable_vision
                and not payload.mlx_int8_prefill
                and not {
                    key: value
                    for key, value in saved_fields.items()
                    if key not in ("tensor_parallel", "disable_vision", "mlx_int8_prefill")
                }
            )
        if requested_extra_args is None and not is_removal:
            stored = get_model_override(payload.model_id)
            # A fill keeps the stored flags without echoing them back through validation: one
            # denylisted since it was saved would 400 the migration, which then retries forever.
            if not (payload.fill_absent_fields and stored):
                requested_extra_args = stored.get("llama_extra_args")
                if requested_extra_args is None:
                    bare_id = _bare_model_id(payload.model_id)
                    if bare_id:
                        requested_extra_args = get_model_override(bare_id).get("llama_extra_args")
                if requested_extra_args is None:
                    # And for a standalone .gguf upgraded from the build that keyed it by its filename label: the bare
                    # path written here is read before that key, so its flags would go dark with no page able to show them.
                    legacy_id = _legacy_standalone_gguf_key(payload.model_id)
                    if legacy_id:
                        requested_extra_args = get_model_override(legacy_id).get("llama_extra_args")
                if requested_extra_args is None:
                    # Same for the other spelling of a cached repo, which this save retires
                    # below: its flags have nowhere else to live, and the page cannot show them.
                    for alias_id in cached_repo_alias_keys(payload.model_id):
                        requested_extra_args = get_model_override(alias_id).get("llama_extra_args")
                        if requested_extra_args is not None:
                            break
        fields_set = payload.model_fields_set
        reset_reasoning_budget = "reasoning_budget" in fields_set and payload.reasoning_budget == -1
        reset_reasoning_budget_message = (
            "reasoning_budget_message" in fields_set and payload.reasoning_budget_message == ""
        )
        if not payload.fill_absent_fields and requested_extra_args:
            requested_extra_args = strip_shadowing_flags(
                requested_extra_args,
                strip_context = False,
                strip_cache = False,
                strip_spec = False,
                strip_template = False,
                strip_split_mode = False,
                strip_reasoning_budget = reset_reasoning_budget,
                strip_reasoning_budget_message = reset_reasoning_budget_message,
            )
        # Not validated on an explicit remove: a 400 would only leave the override in place.
        if payload.remove is True:
            extra_args = []
        elif payload.llama_extra_args is None:
            # Carried over, not sent: a flag denylisted since it was written is dropped rather than refused, or an
            # unrelated save fails naming a flag the user cannot fix from this payload.
            extra_args, dropped_flags = drop_managed_flags(requested_extra_args)
            if dropped_flags:
                logger.warning(
                    "model_override.dropped_managed_flags model_id=%s flags=%s",
                    payload.model_id,
                    ", ".join(dropped_flags),
                )
        else:
            extra_args = validate_extra_args(requested_extra_args)
        # Same shape as the extra-args carry-over above, for the same reason: a save replaces the entry, so a field
        # the caller never knew about must survive it. Gated on is_removal, not on payload.remove: the documented
        # legacy contract is a payload carrying only model_id, which leaves remove None.
        _tuning_fields = ("load_mode", "spec_draft_cache_type", "ctx_checkpoints", "cache_ram")
        _reasoning_fields = ("reasoning_budget", "reasoning_budget_message")
        _kept_tuning = {name: getattr(payload, name) for name in _tuning_fields + _reasoning_fields}
        # Each group is carried only for a client that does not mirror it.
        _carried_fields = (() if payload.mirrors_server_tuning else _tuning_fields) + (
            () if payload.mirrors_reasoning_budget else _reasoning_fields
        )
        if _carried_fields and not is_removal:
            # The same spellings the extra-args carry-over walks: a cached repo is not an ordinary folded match,
            # so a save under the repo id would find nothing and retire the alias with its tuning.
            _alias_ids = [payload.model_id]
            for _candidate in (
                _bare_model_id(payload.model_id),
                _legacy_standalone_gguf_key(payload.model_id),
                *cached_repo_alias_keys(payload.model_id),
            ):
                if _candidate and _candidate not in _alias_ids:
                    _alias_ids.append(_candidate)
            # Load order, not collection order: a lookup reads the concrete load path before the advertised repo
            # id, so reading the repo row first adopts tuning no load has used.
            _alias_ids.sort(key = lambda _key: not is_cache_load_path_key(_key))
            # Taken as a unit from the first row that exists, not field by field down the list: a load stops at the
            # first non-empty row rather than merging, so filling a gap in the winner from a loser would switch
            # dormant tuning on.
            for _alias_id in _alias_ids:
                _stored_tuning = get_model_override(_alias_id)
                if not _stored_tuning:
                    continue
                for name in _carried_fields:
                    if _kept_tuning[name] is None:
                        _kept_tuning[name] = _stored_tuning.get(name)
                break
        removed_keys: list[str] = []
        if payload.remove is True:
            # An explicit remove wins over any other field. Remove the key a load resolves to, not the literal one sent
            # (the browser normalizes casing), and every spelling: clearing one of two leaves the survivor as the sole
            # fold match.
            target_ids = resolve_model_override_keys(payload.model_id) or [
                payload.model_id,
            ]
            removed_keys.extend(target_ids)
            # A standalone .gguf is keyed by its bare path now, but a load also reads the
            # filename-derived <path>:LABEL an upgraded install holds, which would outlive this.
            legacy_id = _legacy_standalone_gguf_key(payload.model_id)
            if legacy_id and legacy_id not in target_ids:
                removed_keys.append(legacy_id)
            # The mirror image of the carry-over above: a save under repo:QUANT copies the flags off a legacy bare
            # `repo` entry and leaves it in place, and the loader falls back to it when the qualified key misses, so
            # clearing only the qualified key hands the same flags straight back. Only once it is nobody else's
            # fallback, though: it backs every quant with no entry of its own, so forgetting Q4 must not strip Q8.
            bare_id = _bare_model_id(payload.model_id)
            if (
                bare_id
                and bare_id not in target_ids
                and not _other_quants_remain(
                    bare_id,
                    target_ids,
                )
            ):
                removed_keys.append(bare_id)
            # And the other spelling of a cached repo: the loader reads the load path before
            # the advertised id, so clearing only the id leaves the path entry still applying.
            for alias_id in cached_repo_alias_keys(payload.model_id):
                if alias_id not in removed_keys:
                    removed_keys.append(alias_id)
            for removed_id in removed_keys:
                set_model_override(removed_id, llama_extra_args = [], max_seq_length = None)
        else:
            # Save under the key a load resolves to, as the removal branch does: the literal
            # id would leave two keys for one model, making every other casing ambiguous.
            target_id = resolve_model_override_key(payload.model_id) or payload.model_id
            if payload.fill_absent_fields:
                # A fill retires nothing below, so it must not create the higher-priority spelling of a row
                # the server already holds.
                target_id = _fill_target_id(target_id)
            # An explicit clear keeps a row even when nothing else is set, so long as a fallback would otherwise answer
            # for this model: "no launch flags" and "nothing stored" are the same thing everywhere else, and different
            # here. Written on the quant's own key, so no other quant moves.
            keep_empty = (
                payload.llama_extra_args == []
                and not payload.fill_absent_fields
                and _fallback_supplies_extra_args(payload.model_id, target_id)
            )
            # A default the caller did not send still has to be written while a broader entry
            # would otherwise answer with the flag this row exists to shadow.
            _kept_reasoning_budget = _kept_tuning["reasoning_budget"]
            _kept_reasoning_budget_message = _kept_tuning["reasoning_budget_message"]
            if (
                not payload.fill_absent_fields
                and _kept_reasoning_budget is None
                and _kept_reasoning_budget_message is None
                and _fallback_supplies_reasoning_flag(payload.model_id, target_id)
            ):
                _kept_reasoning_budget = -1
                _kept_reasoning_budget_message = ""
            # A -c sent with this save is what its load runs at (llama.cpp takes the last -c); store it as
            # the context or auto-switch strips it as stale (#11511). Carried-over flags and fills keep that rule.
            max_seq_length = payload.max_seq_length
            custom_context_length = payload.custom_context_length
            if payload.llama_extra_args is not None and not payload.fill_absent_fields:
                try:
                    explicit_ctx = parse_ctx_override(extra_args)
                except ValueError:
                    explicit_ctx = None
                # Past the stored ceiling the field would be dropped, leaving the flag unchecked.
                if explicit_ctx and explicit_ctx <= MAX_SEQ_LENGTH_CEILING:
                    if max_seq_length is not None:
                        max_seq_length = explicit_ctx
                    if custom_context_length is not None:
                        custom_context_length = explicit_ctx
            tensor_split = payload.tensor_split
            if "tensor_split" not in fields_set:
                previous = get_model_override(target_id)
                if previous.get("gpu_ids") == payload.gpu_ids and previous.get(
                    "gpu_index_kind", "physical"
                ) == (payload.gpu_index_kind or "physical"):
                    tensor_split = previous.get("tensor_split")
            set_model_override(
                target_id,
                llama_extra_args = extra_args,
                keep_empty_extra_args = keep_empty,
                engine_parallelism = payload.engine_parallelism
                if payload.engine_parallelism is not None or is_removal
                else get_model_override(target_id).get("engine_parallelism"),
                engine_precision = payload.engine_precision
                if payload.engine_precision is not None or is_removal
                else get_model_override(target_id).get("engine_precision"),
                engine = payload.engine
                if payload.engine is not None or is_removal
                else get_model_override(target_id).get("engine"),
                max_seq_length = max_seq_length,
                custom_context_length = custom_context_length,
                kv_cache_dtype = payload.kv_cache_dtype,
                mlx_kv_quant = payload.mlx_kv_quant,
                speculative_type = payload.speculative_type,
                spec_draft_n_max = payload.spec_draft_n_max,
                n_parallel = payload.n_parallel,
                reasoning_budget = (
                    None
                    if payload.fill_absent_fields and reset_reasoning_budget
                    else _kept_reasoning_budget
                ),
                reasoning_budget_message = (
                    None
                    if payload.fill_absent_fields and reset_reasoning_budget_message
                    else _kept_reasoning_budget_message
                ),
                n_batch = payload.n_batch,
                n_ubatch = payload.n_ubatch,
                load_mode = _kept_tuning["load_mode"],
                spec_draft_cache_type = _kept_tuning["spec_draft_cache_type"],
                ctx_checkpoints = _kept_tuning["ctx_checkpoints"],
                cache_ram = _kept_tuning["cache_ram"],
                tensor_parallel = payload.tensor_parallel,
                disable_vision = payload.disable_vision,
                mlx_int8_prefill = payload.mlx_int8_prefill,
                chat_template_override = payload.chat_template_override,
                gpu_memory_mode = payload.gpu_memory_mode,
                gpu_layers = payload.gpu_layers,
                n_cpu_moe = payload.n_cpu_moe,
                gpu_ids = payload.gpu_ids,
                tensor_split = tensor_split,
                gpu_index_kind = payload.gpu_index_kind,
                fill_absent_fields = payload.fill_absent_fields,
            )
            # A repo cached outside the active HF cache is keyed here by its repo id
            if not payload.fill_absent_fields:
                for alias_id in cached_repo_alias_keys(target_id):
                    set_model_override(alias_id, llama_extra_args = [], max_seq_length = None)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid model launch override."),
            event = "settings.update_model_override_failed",
            log = logger,
        ) from exc
    return ModelOverridesResponse(overrides = get_model_overrides(), removed_keys = removed_keys)


class EmbeddingModelPayload(BaseModel):
    embedding_model: str = Field(..., min_length = 1, max_length = MAX_EMBEDDING_MODEL_LENGTH)
    # The repo /resolve named, stored so the loader opens what was downloaded.
    gguf_repo: Optional[str] = Field(default = None, max_length = MAX_EMBEDDING_MODEL_LENGTH)
    # And the backend it needs, so a model with no GGUF is not sent to llama-server.
    backend: Optional[Literal["llama", "sentence-transformers"]] = None
    # Token for gated/private repos during verification (not stored).
    hf_token: Optional[str] = Field(default = None, max_length = 512)
    # Skip HF verification (offline installs, local paths HF can't see).
    force: bool = False


class EmbeddingModelResponse(BaseModel):
    embedding_model: str
    embedding_gguf_repo: str
    default_embedding_model: str
    default_embedding_gguf_repo: str
    is_custom: bool
    # Whether THIS model is held in memory right now, for the status line.
    loaded: bool = False
    # Whether ANY embedder is resident.
    backend_loaded: bool = False


def _embedding_model_response() -> EmbeddingModelResponse:
    model = get_rag_embedding_model()
    return EmbeddingModelResponse(
        embedding_model = model,
        embedding_gguf_repo = effective_gguf_repo_for_embedding_model(model),
        default_embedding_model = default_embedding_model(),
        default_embedding_gguf_repo = default_gguf_repo(),
        is_custom = get_stored_embedding_model() is not None,
        loaded = _embedder_is_loaded(model),
        backend_loaded = _any_embedder_is_loaded(),
    )


def _embedder_is_loaded(model: str) -> bool:
    from core.rag import embeddings
    try:
        return embeddings.backend_is_loaded(model)
    except Exception:  # noqa: BLE001 - probe must never block reading settings
        return False


def _any_embedder_is_loaded() -> bool:
    """Whether any embedder is resident, whichever model it belongs to."""
    from core.rag import embeddings
    try:
        return embeddings.backend_is_loaded()
    except Exception:  # noqa: BLE001 - probe must never block reading settings
        return False


def _ambient_hf_token() -> Optional[str]:
    """The HF token the loader would use (HF_TOKEN env or the cached login), so a gated
    repo is scanned rather than failing open. None if unavailable."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _model_names_gguf_repo(model: str) -> bool:
    """Whether ``model`` is a repo id naming GGUF weights, per the embedder's rule."""
    from core.rag import embeddings
    try:
        return embeddings._model_names_gguf_repo(model)
    except Exception:  # noqa: BLE001 - a name test that cannot answer blocks nothing
        return False


def _llama_runtime_available() -> bool:
    """Whether a llama-server binary this install can launch is present. Shares the embedder's own probe
    so the resolver and the loader cannot disagree about whether the backend exists."""
    from core.rag import embeddings
    try:
        return embeddings._llama_server_runtime_available()
    except Exception:  # noqa: BLE001 - an unanswerable probe must not block saving
        return True


def _llama_backend_active(model: str | None = None) -> bool:
    """Whether llama serves the active model, or would serve ``model`` if supplied. Delegates to the embeddings
    module so a runtime fallback from sentence-transformers to llama-server is honored: in that state the
    process loads only inert GGUF, so the ST pickle gate below must not hard-block a repo whose GGUF
    companion is clean. Before any backend is built this reflects the resolver."""
    from core.rag import embeddings
    try:
        if model is not None:
            return embeddings.resolved_backend_for_model(model) == "llama-server"
        return embeddings.active_backend_is_llama()
    except Exception:  # noqa: BLE001 - backend probe must never block saving
        return False


def _resolves_as_local_gguf(model: str) -> bool:
    """True when ``model`` is a local .gguf file or a directory holding one, so a save on the
    llama-server backend needs no HF verification: the artifact itself is the proof."""
    from core.rag.embed_llama_server import LlamaServerBackend
    try:
        return LlamaServerBackend._resolve_local_gguf(model) is not None
    except Exception:  # noqa: BLE001 - dir without .gguf, filesystem oddity
        return False


def _local_gguf_backend_error(model: str) -> str | None:
    """409 detail when ``model`` is a local dir without a .gguf but this install embeds via llama-server
    (macOS/CPU default), which needs one: a sentence-transformers-only folder would verify fine yet fail at
    first index. ``force`` skips this check like HF verification."""
    from pathlib import Path

    from utils.paths import normalize_path

    # Normalized as _resolve_local_gguf normalizes it, or a WSL drive-letter dir reads as "not a
    # directory" and the 409 that would have explained it never fires.
    if not Path(normalize_path(model)).expanduser().is_dir():
        return None
    from core.rag.embed_llama_server import LlamaServerBackend

    if not _llama_backend_active(model):
        return None
    try:
        LlamaServerBackend._resolve_local_gguf(model)
        return None
    except RuntimeError:
        return (
            f"{model!r} contains no .gguf file, but this install embeds with the "
            "llama-server backend which requires one. Add a GGUF file to the "
            "folder or use a Hugging Face repo."
        )
    except Exception:  # noqa: BLE001 - filesystem oddity: don't block saving
        return None


def _hf_gguf_backend_error(model: str, hf_token: Optional[str]) -> str | None:
    """409 detail when the llama-server backend would find no .gguf for an HF repo: neither the derived
    companion repo nor the repo itself has one, so saves that verify as embedding models would fail at
    first index. ``force`` skips this like HF verification."""
    from pathlib import Path

    if Path(model).expanduser().exists():
        return None
    if not _llama_backend_active(model):
        return None
    candidates = _embedding_gguf_candidates(model)
    if _remote_embedding_gguf_plan(candidates, hf_token) is not None:
        return None
    if _search_hub_for_gguf(model, hf_token) is not None:
        return None
    # Safetensors on sentence-transformers is a working answer, not a failure.
    if (
        _sentence_transformers_fallback_allowed(model)
        and _safetensors_plan(model, hf_token) is not None
    ):
        return None
    return _no_embedding_weights_error(candidates)


def _no_embedding_weights_error(candidates: list[str]) -> str:
    """Error after the caller has already exhausted GGUF and ST resolution."""
    checked = " or ".join(repr(c) for c in candidates)
    return (
        f"No GGUF weights found in {checked}, and no safetensors to fall back to. "
        "Only the model's own publisher is used as a source."
    )


def _st_cannot_load_error(model: str, candidates: list[str]) -> str:
    """Error when no GGUF was found and the model's safetensors need a newer sentence-transformers or transformers."""
    checked = " or ".join(repr(c) for c in candidates)
    return (
        f"No GGUF weights found in {checked}, and {model!r} needs a newer sentence-transformers or "
        "transformers than this install has."
    )


@_owner_settings_router.get("/embedding-model", response_model = EmbeddingModelResponse)
def get_embedding_model(
    current_subject: str = Depends(get_current_subject),
) -> EmbeddingModelResponse:
    return _embedding_model_response()


class EmbeddingModelResolveResponse(BaseModel):
    embedding_model: str
    backend: Literal["llama", "sentence-transformers"]
    # Repo the picker hands the download manager, and the files to take from it. Split GGUF plans contain
    # every shard in the selected family; both None when nothing needs fetching or when ``error`` is set.
    download_repo: Optional[str] = None
    files: Optional[list[str]] = None
    cached: bool = False
    size_bytes: Optional[int] = None
    error: Optional[str] = None


def _embedding_gguf_candidates(model: str) -> list[str]:
    """Repos the loader would try for ``model``'s GGUF, in its order."""
    from core.rag import config as rag_config

    # An env override is the loader's only source.
    if rag_config.gguf_repo_is_explicit():
        return rag_config.gguf_repo_candidates(model)
    try:
        from utils.embedding_model_settings import get_stored_gguf_repo
        stored = get_stored_gguf_repo(model)
    except Exception:  # noqa: BLE001 - resolver still has derived candidates
        stored = None
    return list(
        dict.fromkeys([*([stored] if stored else []), *rag_config.gguf_repo_candidates(model)])
    )


# A GGUF conversion must come from the same owner as the model.
_GGUF_MIRROR_SEARCH_LIMIT = 25
_GGUF_LIST_DEADLINE_S = 20.0
_EMBEDDING_RESOLVE_DEADLINE: ContextVar[float | None] = ContextVar(
    "embedding-resolve-deadline", default = None
)


def _call_with_embedding_resolve_budget(fn, *, name: str):
    """Run one remote probe inside the resolution's single time budget."""
    deadline = _EMBEDDING_RESOLVE_DEADLINE.get()
    timeout = (
        _GGUF_LIST_DEADLINE_S
        if deadline is None
        else max(0.0, min(_GGUF_LIST_DEADLINE_S, deadline - time.monotonic()))
    )
    if timeout <= 0:
        raise TimeoutError("embedding model resolution deadline expired")
    from utils.utils import call_with_deadline

    return call_with_deadline(fn, timeout, name = name)


def _with_embedding_resolve_budget(fn):
    """Give one GET/PUT resolution a deadline shared by every Hub fallback, and one scope for the
    sentence-transformers load proofs, so a check the spent deadline skips cannot undo an earlier one."""

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        if _EMBEDDING_RESOLVE_DEADLINE.get() is not None:
            return fn(*args, **kwargs)
        from core.rag.embeddings import st_load_proof_scope

        marker = _EMBEDDING_RESOLVE_DEADLINE.set(time.monotonic() + _GGUF_LIST_DEADLINE_S)
        try:
            with st_load_proof_scope():
                return fn(*args, **kwargs)
        finally:
            _EMBEDDING_RESOLVE_DEADLINE.reset(marker)

    return _wrapped


def _list_repo_files_bounded(repo: str, hf_token: Optional[str]) -> list[str]:
    """List a Hub repo without letting a blackholed route pin Settings forever."""
    from huggingface_hub import list_repo_files
    return _call_with_embedding_resolve_budget(
        lambda: list_repo_files(repo, token = hf_token),
        name = "embed-settings-repo-listing",
    )


def _gguf_conversion_name_matches(hit_name: str, base: str) -> bool:
    """Whether a search hit names a conversion of exactly ``base``. Prefix matching is unsafe (``foo`` must
    never resolve to ``foo-bar-GGUF``); the Hub filter already requires GGUF, so this accepts only the
    common conversion suffix spellings and the exact model name."""
    name = hit_name.casefold()
    base = base.casefold()
    return name == base or name in {f"{base}-gguf", f"{base}_gguf", f"{base}.gguf"}


def _gguf_files_for_pick(names: list[str], picked: str) -> Optional[list[str]]:
    """The complete downloadable file family for a picked GGUF. llama-server opens split siblings implicitly,
    so a single selected shard is not a usable plan, and incomplete published families are rejected.
    Deferred to the loader so the plan offered here and the transfer name one set."""
    from core.rag.embed_llama_server import LlamaServerBackend
    return LlamaServerBackend._split_family(names, picked)


def _pick_downloadable_gguf(names: list[str]) -> Optional[list[str]]:
    """Pick the loader's preferred GGUF, skipping torn split families."""
    from core.rag.embed_llama_server import LlamaServerBackend

    _picked, files = LlamaServerBackend._pick_complete_gguf(names)
    return files or None


def _search_hub_for_gguf(model: str, hf_token: Optional[str]) -> Optional[tuple[str, list[str]]]:
    """``(repo, files)`` for a GGUF conversion of ``model`` published by the same owner under a name the -GGUF
    candidates do not cover. Same owner only: a third party's "Qwen3-Embedding-8B-GGUF" is an unverified
    re-upload, and picking unsloth/X must download unsloth's own weights."""
    from core.rag import config as rag_config

    # The loader cannot open a discovered mirror while an explicit repo override is active, so
    # returning one would create a download that can never satisfy it.
    if rag_config.gguf_repo_is_explicit():
        return None
    owner, _, name = model.rpartition("/")
    if not owner:
        return None
    try:
        from huggingface_hub import HfApi
    except Exception:  # noqa: BLE001 - hub client unavailable
        return None
    base = rag_config._QUANT_SUFFIX_RE.sub("", name).lower()
    if not base:
        return None
    try:
        hits = _call_with_embedding_resolve_budget(
            lambda: list(
                HfApi().list_models(
                    search = base,
                    author = owner,
                    filter = ["gguf"],
                    sort = "downloads",
                    limit = _GGUF_MIRROR_SEARCH_LIMIT,
                    token = hf_token,
                )
            ),
            name = "embed-settings-model-search",
        )
    except Exception:  # noqa: BLE001 - offline or rate limited
        return None
    for hit in hits:
        hit_owner, _, hit_name = hit.id.rpartition("/")
        if hit_owner.casefold() != owner.casefold() or not _gguf_conversion_name_matches(
            hit_name, base
        ):
            continue
        try:
            files = _pick_downloadable_gguf(_list_repo_files_bounded(hit.id, hf_token))
        except Exception:  # noqa: BLE001 - unreadable listing: try the next
            continue
        if files:
            return hit.id, files
    return None


def _cached_embedding_gguf(candidates: list[str], *, require_variant: bool) -> Optional[str]:
    """First candidate already holding a usable GGUF on disk. No network."""
    from core.rag.embed_llama_server import LlamaServerBackend

    for candidate in candidates:
        try:
            if LlamaServerBackend._resolve_cached_gguf(candidate, require_variant = require_variant):
                return candidate
        except Exception:  # noqa: BLE001 - a bad cache entry is just a miss
            continue
    return None


def _cached_embedding_gguf_files(repo: str, files: list[str]) -> bool:
    """Whether the exact resolved GGUF family is complete in ``repo``'s snapshot."""
    from pathlib import Path, PurePosixPath
    from core.rag.embed_llama_server import LlamaServerBackend

    try:
        snapshot = LlamaServerBackend._cached_snapshot_dir(repo)
        if snapshot is None or not files:
            return False
        for name in files:
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (snapshot / Path(*path.parts)).is_file()
            ):
                return False
        return True
    except Exception:  # noqa: BLE001 - an unreadable cache is a miss
        return False


def _remote_embedding_gguf_plan(
    candidates: list[str], hf_token: Optional[str]
) -> Optional[tuple[str, list[str]]]:
    """``(repo, files)`` for the first candidate publishing a usable GGUF family."""
    for candidate in candidates:
        try:
            names = _list_repo_files_bounded(candidate, hf_token)
        except Exception:  # noqa: BLE001 - missing/gated repo: try the next
            continue
        try:
            files = _pick_downloadable_gguf(names)
        except Exception:  # noqa: BLE001 - unreadable listing: try the next
            files = None
        if files:
            return candidate, files
    return None


# safetensors first: a repo carrying both formats would otherwise be fetched twice.
_ST_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def _st_backend_available() -> bool:
    """Whether sentence-transformers could actually run here. A GGUF-only install
    has no torch, so the safetensors fallback is not on offer there."""
    try:
        from core.rag import embeddings
        return embeddings.sentence_transformers_runtime_available()
    except Exception:  # noqa: BLE001 - a broken import path is a no
        return False


def _is_st_weight_name(basename: str) -> bool:
    """Whether a filename is a checkpoint, not just something ending in a suffix. Shared with the loader
    so the plan and the cache check cannot disagree."""
    from utils.utils import is_st_weight_name
    return is_st_weight_name(basename)


def _st_weight_source(model: str, hf_token: Optional[str]) -> Optional[tuple[str, list[str]]]:
    """``(repo, weight files)`` for the repo an ST load of ``model`` would open. A slashless name such as
    ``all-MiniLM-L6-v2`` resolves under the ``sentence-transformers/`` namespace, which is what the loader's
    own ``st_repo_id_candidates`` encodes; probing only the literal id refused the alias outright and a
    forced save then pinned it cache-only."""
    from utils.utils import st_repo_id_candidates

    for candidate in st_repo_id_candidates(model) or [model]:
        files = _st_weight_files(candidate, hf_token)
        if files:
            return (candidate, files)
    return None


def _st_weight_files(model: str, hf_token: Optional[str]) -> Optional[list[str]]:
    """The repo's own weight files, or None when it publishes none we can load."""
    try:
        files = _list_repo_files_bounded(model, hf_token)
    except Exception:  # noqa: BLE001 - missing/gated repo or offline
        return None
    for suffix in _ST_WEIGHT_SUFFIXES:
        weights = [
            filename
            for filename in files
            if filename.rsplit("/", 1)[-1].lower().endswith(suffix)
            and _is_st_weight_name(filename.rsplit("/", 1)[-1])
        ]
        if weights:
            return weights
    return None


def _cached_snapshot_has_st_weights(model: str) -> bool:
    """Whether the cached snapshot holds a checkpoint ST itself can open. ``hf_cache_snapshot_is_loadable``
    counts ``.gguf``, which is right for the llama backend and wrong here: a cached GGUF-only repo would
    come back ready with no checkpoint ST can load. No network."""
    try:
        from utils.utils import snapshot_has_st_weights
        return snapshot_has_st_weights(model)
    except Exception:  # noqa: BLE001 - an unreadable cache is not a proof of weights
        return False


def _cached_st_source(model: str):
    """``(repo id, snapshot dir)`` the cached ST weights for ``model`` came from. Same predicate as
    ``_cached_snapshot_has_st_weights``, keeping the repo it matched under rather than reducing it to a yes:
    for a slashless alias that repo is the ``sentence-transformers/`` one, and the PUT verifies and scans
    it."""
    try:
        from utils.utils import cached_st_source
        return cached_st_source(model)
    except Exception:  # noqa: BLE001 - an unreadable cache is not a proof of weights
        return None


def _cached_st_weight_names(model: str) -> list[str]:
    """Snapshot-relative names of the ST weights already on disk for ``model``."""
    try:
        from utils.utils import hf_cache_snapshot_dir

        snapshot = hf_cache_snapshot_dir(model)
        if snapshot is None:
            return []
        return sorted(
            str(path.relative_to(snapshot))
            for path in snapshot.rglob("*")
            if _is_st_weight_name(path.name) and path.is_file()
        )
    except Exception:  # noqa: BLE001 - the plan stands without a file list
        return []


def _safetensors_plan(model: str, hf_token: Optional[str]) -> Optional[tuple[str, list[str]]]:
    """``(repo, files)`` for running ``model`` on sentence-transformers instead. An embedder with no GGUF still
    works from its own safetensors, for about 1 GB more memory, which beats refusing the model or pulling a
    stranger's conversion."""
    if not _st_backend_available():
        return None
    # A complete local snapshot is the same proof the listing gives, and works
    # offline, where the listing fails and a downloaded model became unselectable.
    from utils.utils import cached_st_repo

    # The repo the snapshot is actually filed under: a slashless name caches under sentence-transformers/, and
    # naming the literal id sends the download manager at a repo that usually does not exist.
    cached_repo = cached_st_repo(model)
    if cached_repo:
        return (cached_repo, _cached_st_weight_names(cached_repo))
    return _st_weight_source(model, hf_token)


def _sentence_transformers_fallback_allowed(model: str) -> bool:
    """Whether a newly selected model can actually be served by ST in this process."""
    try:
        from core.rag import embeddings
        return embeddings.sentence_transformers_fallback_allowed(model)
    except Exception:  # noqa: BLE001 - an unknown backend is not a safe fallback
        return False


def _sentence_transformers_can_load(model: str) -> bool:
    """Whether the installed sentence-transformers can open ``model``, within the resolution's budget."""
    try:
        from core.rag import embeddings
    except Exception:  # noqa: BLE001 - unimportable embedder: no proof either way
        return True
    # Before the budget, so a timeout cannot undo the plan's earlier proof.
    if embeddings.sentence_transformers_known_unloadable(model):
        return False
    try:
        return _call_with_embedding_resolve_budget(
            lambda: embeddings.sentence_transformers_can_load(model),
            name = "embed-settings-st-load-check",
        )
    except Exception:  # noqa: BLE001 - spent budget: no proof either way
        return True


def _hf_files_size(repo: str, files: list[str], hf_token: Optional[str]) -> Optional[int]:
    """Total bytes of ``files`` in ``repo``, for the confirm dialog. None when the hub does not say."""
    try:
        from huggingface_hub import model_info

        info = _call_with_embedding_resolve_budget(
            lambda: model_info(repo, files_metadata = True, token = hf_token),
            name = "embed-settings-file-size",
        )
        wanted = set(files)
        total = sum(
            sibling.size or 0 for sibling in (info.siblings or []) if sibling.rfilename in wanted
        )
        return total or None
    except Exception:  # noqa: BLE001 - size is advisory, never a blocker
        return None


def _hf_snapshot_size(repo: str, hf_token: Optional[str]) -> Optional[int]:
    """Bytes the model download worker will fetch for a full snapshot."""
    try:
        from huggingface_hub import model_info
        from hub.utils.snapshot_filters import snapshot_download_size

        info = _call_with_embedding_resolve_budget(
            lambda: model_info(repo, files_metadata = True, token = hf_token),
            name = "embed-settings-snapshot-size",
        )
        total = snapshot_download_size(info.siblings or [])
        return total or None
    except Exception:  # noqa: BLE001 - size is advisory, never a blocker
        return None


def _local_sentence_transformer_is_present(model: str) -> bool:
    """Whether ``model`` is an existing local path ST can open directly."""
    try:
        from pathlib import Path
        from utils.paths import is_local_path, normalize_path

        if not is_local_path(model):
            return False
        p = Path(normalize_path(model)).expanduser()
        # ST cannot open a .gguf. Falling through reaches the no-loadable-weights
        # error rather than reporting it ready and failing at the first index.
        if p.is_file() and p.suffix.lower() == ".gguf":
            return False
        if not p.exists():
            return False
        # A directory has to hold a checkpoint, not merely exist: modules.json
        # alone also passes is_embedding_model's local-path check.
        if not p.is_dir():
            # SentenceTransformer takes a directory or a repo id, never a bare checkpoint file.
            return False
        if not any(_is_st_weight_name(child.name) and child.is_file() for child in p.rglob("*")):
            return False
        # And a WHOLE one: half a shard family, or a module modules.json declares and the directory lacks, reads as
        # ready and fails at the first index.
        from utils.utils import checkpoint_directory_is_complete

        return checkpoint_directory_is_complete(p)
    except Exception:  # noqa: BLE001 - filesystem oddity is a cache miss
        return False


@_with_embedding_resolve_budget
def _resolve_embedding_model_plan(
    resolved: str, token: Optional[str]
) -> EmbeddingModelResolveResponse:
    """Server-owned artifact/backend plan shared by GET and PUT.

    The PUT must not persist a client assertion that the GET never validated,
    so both routes use this exact resolver.

    The cache lookups below read the operator's disk without consulting the credential,
    so a caller who cannot reach the repo must not learn its cached state from them. A
    local path the caller named itself is not the Hub cache and stays available.

    Per REPO, not per request: a lookup often answers with a stored override, a
    ``sentence-transformers/`` alias or a derived ``-GGUF`` conversion rather than
    ``resolved``, and /auth-check answers 200 for any string on a public base.
    """

    def _authorized(repo: Optional[str]) -> bool:
        """The repo a cache lookup actually matched, asked about in its own right.

        Through the shared gate, so the forced-anonymous sentinel keeps a cached PUBLIC
        embedder instead of being told to download one it already has: the raw check
        refuses that caller whatever the repo is, which the template, dataset and GGUF
        paths all stopped doing. is_cached is True because the lookup that produced this
        repo already found it on disk.
        """
        if not repo:
            return False
        return not cached_read_refused(token, repo_id = repo, is_cached = lambda: True)

    # No pre-gate on ``resolved``: every cache lookup below is authorized against the repo it
    # actually matched, so gating the lookup as well probed the Hub on every resolve, including
    # the local and sentence-transformers paths that never consult it, and could spend half the
    # resolver's deadline before a miss. It was also the wrong question, per the note above.

    # Resolve for the model being selected.
    on_llama = _llama_backend_active(resolved)
    backend: Literal["llama", "sentence-transformers"] = (
        "llama" if on_llama else "sentence-transformers"
    )

    if not on_llama:
        # A valid local SentenceTransformer path is already the artifact; it is
        # not a Hub repo for the download manager to fetch.
        if _local_sentence_transformer_is_present(resolved):
            return EmbeddingModelResolveResponse(
                embedding_model = resolved, backend = backend, cached = True
            )
        # The alias-aware predicate alone, which already pairs the ST file family with the loadable check per candidate;
        # the repo the cache hit came from is what the PUT verifies and scans.
        # Looked up BEFORE any authorization: for a slashless alias the snapshot is filed
        # under sentence-transformers/, so gating on the literal name's verdict skipped the
        # lookup entirely and a public alias that is fully cached came back as a download.
        # Learning WHICH repo answered is not the leak; handing its state back is, and that
        # repo is the one authorized here.
        cached_source = _cached_st_source(resolved)
        if cached_source is not None and not _authorized(cached_source[0]):
            cached_source = None
        cached = cached_source is not None
        source = None if cached else _st_weight_source(resolved, token)
        if not cached and source is None:
            # is_embedding_model gates on tags, so a feature-extraction repo publishing no loadable checkpoint would be
            # offered as a download ST cannot open.
            return EmbeddingModelResolveResponse(
                embedding_model = resolved,
                backend = backend,
                error = (
                    f"No sentence-transformers weights found in {resolved!r}. "
                    "The repository publishes no checkpoint this backend can load."
                ),
            )
        # The repo that actually publishes the weights, which for a slashless alias is the
        # sentence-transformers/ one, not the literal name.
        if cached:
            download_repo = cached_source[0]
        elif source is None:
            download_repo = resolved
        else:
            download_repo = source[0]
        return EmbeddingModelResolveResponse(
            embedding_model = resolved,
            backend = backend,
            download_repo = download_repo,
            cached = cached,
            size_bytes = None if cached else _hf_snapshot_size(download_repo, token),
        )

    local_gguf = _resolves_as_local_gguf(resolved)
    # Routing a model here does not make the backend runnable: without a binary the plan is advertised as valid and the
    # first warm fails in _resolve_binary.
    llama_only = (
        local_gguf
        or _model_names_gguf_repo(resolved)
        # An explicit llama policy (or a runtime pin) refuses the safetensors fallback for every model, not
        # only GGUF-named ones, so an ordinary repo id is just as unservable here without a binary.
        or not _sentence_transformers_fallback_allowed(resolved)
    )
    if llama_only and not _llama_runtime_available():
        return EmbeddingModelResolveResponse(
            embedding_model = resolved,
            backend = backend,
            error = (
                f"{resolved!r} can only be embedded by the llama-server backend "
                "here, and no llama-server binary was found. Install llama.cpp or "
                "set LLAMA_SERVER_PATH / UNSLOTH_LLAMA_CPP_PATH."
            ),
        )

    # A local .gguf (file or folder) is already the artifact; nothing to fetch.
    if local_gguf:
        return EmbeddingModelResolveResponse(embedding_model = resolved, backend = backend, cached = True)
    local_error = _local_gguf_backend_error(resolved)
    if local_error:
        return EmbeddingModelResolveResponse(
            embedding_model = resolved, backend = backend, error = local_error
        )

    candidates = _embedding_gguf_candidates(resolved)
    # Match the loader's online fast path exactly: only the preferred repo and
    # only the configured variant can suppress the download offer.
    cached_repo = _cached_embedding_gguf(candidates[:1], require_variant = True)
    if cached_repo and not _authorized(cached_repo):
        cached_repo = None
    if cached_repo:
        return EmbeddingModelResolveResponse(
            embedding_model = resolved,
            backend = backend,
            download_repo = cached_repo,
            cached = True,
        )
    plan = _remote_embedding_gguf_plan(candidates, token) or _search_hub_for_gguf(resolved, token)
    if plan is None:
        # The loader's offline fallback accepts any complete cached quant from
        # any candidate only after its bounded online listing fails.
        cached_repo = _cached_embedding_gguf(candidates, require_variant = False)
        if cached_repo and not _authorized(cached_repo):
            cached_repo = None
        if cached_repo:
            return EmbeddingModelResolveResponse(
                embedding_model = resolved,
                backend = backend,
                download_repo = cached_repo,
                cached = True,
            )
        # No GGUF from this publisher: run it on its own safetensors only when
        # configuration/runtime policy can actually select ST for this model.
        st_allowed = _sentence_transformers_fallback_allowed(resolved)
        # And only when ST can open it, or the download never loads.
        st_unloadable = st_allowed and not _sentence_transformers_can_load(resolved)
        st_plan = _safetensors_plan(resolved, token) if st_allowed and not st_unloadable else None
        # The GGUF branches above are gated and this one was not. The plan answers with the
        # repo the snapshot is FILED under, which for a slashless alias is not the name the
        # caller typed, and the response then reports it cached: that is how a denied caller
        # discovers the operator's private weights and force-saves them as the embedder.
        # Reading our own disk to find the repo is fine; naming it back is what is gated.
        if st_plan is not None and not _authorized(st_plan[0]):
            st_plan = None
        if st_plan is None:
            return EmbeddingModelResolveResponse(
                embedding_model = resolved,
                backend = backend,
                error = (
                    _st_cannot_load_error(resolved, candidates)
                    if st_unloadable
                    else _no_embedding_weights_error(candidates)
                ),
            )
        st_repo, _st_files = st_plan
        return EmbeddingModelResolveResponse(
            embedding_model = resolved,
            backend = "sentence-transformers",
            download_repo = st_repo,
            # Same alias-aware predicate, asked about the repo the plan named
            # rather than the alias the user typed, which the gate above authorized.
            cached = _cached_snapshot_has_st_weights(st_repo),
            size_bytes = _hf_snapshot_size(st_repo, token),
        )
    repo, files = plan
    # A gated repo can publish its filenames, so a plan coming back is not authorization to
    # report the operator's copy of it. The repo the plan names need not be the one the
    # caller asked about, so it is authorized in its own right like every other candidate.
    if _authorized(repo) and _cached_embedding_gguf_files(repo, files):
        return EmbeddingModelResolveResponse(
            embedding_model = resolved,
            backend = backend,
            download_repo = repo,
            files = files,
            cached = True,
        )
    return EmbeddingModelResolveResponse(
        embedding_model = resolved,
        backend = backend,
        download_repo = repo,
        files = files,
        size_bytes = _hf_files_size(repo, files, token),
    )


@_owner_settings_router.get(
    "/embedding-model/resolve", response_model = EmbeddingModelResolveResponse
)
def resolve_embedding_model(
    model: str,
    # Header, not a query param: keeps a gated-repo token out of URLs and logs.
    hf_token: Optional[str] = Header(None, alias = "X-Unsloth-HF-Token"),
    allow_ambient_token: bool = Depends(allow_ambient_hf_token),
    current_subject: str = Depends(get_current_subject),
) -> EmbeddingModelResolveResponse:
    """What saving ``model`` would need fetched, and whether it is already here.

    The picker calls this so it can offer the download up front instead of letting
    it happen invisibly at first index. ``error`` is the detail the PUT would
    refuse with, so the two cannot disagree about what is usable."""
    try:
        resolved = validate_embedding_model(model)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid embedding model."),
            event = "settings.resolve_embedding_model_failed",
            log = logger,
        ) from exc
    # Classified, not just trimmed: a bare strip makes a UI session look like an API key and
    # costs it its own cached marker. The GET must refuse exactly what the PUT refuses.
    token = hf_token_arg(hf_token, allow_ambient_token = allow_ambient_token)
    return _resolve_embedding_model_plan(resolved, token)


@_owner_settings_router.put("/embedding-model", response_model = EmbeddingModelResponse)
def update_embedding_model(
    payload: EmbeddingModelPayload,
    allow_ambient_token: bool = Depends(allow_ambient_hf_token),
    current_subject: str = Depends(get_current_subject),
) -> EmbeddingModelResponse:
    """Set the RAG embedding model. Unless ``force`` is set, the repo is verified
    to be an embedding model via HF metadata; an unverifiable model (wrong type,
    typo, gated repo, or no network) returns 409 so the UI can offer "save anyway".
    A repo flagged unsafe by HF's security scan returns 403 instead: a hard block
    that ``force`` cannot bypass, so the UI must not offer "save anyway".
    Documents indexed under the previous model must be re-uploaded."""
    from utils.models import is_embedding_model

    try:
        model = validate_embedding_model(payload.embedding_model)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid embedding model."),
            event = "settings.update_embedding_model_failed",
            log = logger,
        ) from exc
    hf_token = hf_token_arg(payload.hf_token, allow_ambient_token = allow_ambient_token)
    from utils.utils import hf_env_offline

    # Offline, both the Hub malware scan and the is-embedding check are unreachable and degrade
    # to the local cache below; capture the state once.
    local_only_load = hf_env_offline()
    # Resolve again server-side. The client fields are only an optimistic echo
    # of GET /resolve; neither a repository nor a backend is trusted on its word.
    plan = _resolve_embedding_model_plan(model, hf_token)
    requested_repo = (payload.gguf_repo or "").strip() or None
    if payload.backend is not None and payload.backend != plan.backend:
        raise HTTPException(
            status_code = 400,
            detail = "The embedding backend no longer matches the server resolution. Resolve it again.",
        )
    if requested_repo is not None and requested_repo != plan.download_repo:
        raise HTTPException(
            status_code = 400,
            detail = "The embedding download repository was not validated for this model.",
        )
    destination_is_llama = plan.backend == "llama"
    # Verify and scan the repo the loader will actually open: a slashless alias resolves under
    # sentence-transformers/, so the literal name scans a repo that usually does not exist. Only the ST path can
    # diverge: a llama download_repo is the GGUF companion, which is not what is scanned here.
    verify_target = model
    if not destination_is_llama and plan.download_repo and plan.download_repo != model:
        verify_target = plan.download_repo
    # The env/default model needs no verification; saving it is a no-op override. A local GGUF on the
    # llama-server backend is accepted as-is: it is exactly what the backend loads, and HF metadata cannot
    # verify a local path.
    is_local_gguf = destination_is_llama and _resolves_as_local_gguf(model)
    # The pickle gate matters only for the sentence-transformers backend; on llama-server the embedder loads
    # inert GGUFs from effective_gguf_repo(), so scanning the ST pickle wrongly rejects a clean companion.
    scan_st_pickle = (
        model != default_embedding_model() and not is_local_gguf and not destination_is_llama
    )
    if scan_st_pickle:
        # Malware/pickle gate before persisting a repo the embedder later loads; runs even under force, which only
        # skips the is-embedding type check for repos HF cannot verify. Local paths and unreachable scans fail open
        # inside evaluate_file_security.
        from utils.security import evaluate_file_security, security_load_subdirs
        from core.rag.embeddings import _st_module_subdirs

        # Fall back to the loader's own token so a gated/private repo is actually scanned
        # (a token-less scan fails open for exactly the repo that would still load).
        scan_token = hf_token or _ambient_hf_token()
        # Offline: subdir probes would hit the network and hang; the offline gate walks the whole cached
        # snapshot, so no load-subdir hints are needed.
        if local_only_load:
            load_subdirs = ()
        else:
            # Include ST module dirs (0_Transformer/) so a flagged pickle directly under one
            # blocks instead of passing as an unreferenced nested shard.
            load_subdirs = tuple(
                dict.fromkeys(
                    (
                        *security_load_subdirs(verify_target, scan_token),
                        *_st_module_subdirs(verify_target, scan_token),
                    )
                )
            )
        if evaluate_file_security(
            verify_target,
            hf_token = scan_token,
            load_subdirs = load_subdirs,
            local_only_load = local_only_load,
        ).blocked:
            # 403, not 409: the client routes every 409 into the forceable "save anyway" flow, but this is a
            # hard, non-forceable security refusal.
            if local_only_load:
                detail = (
                    f"{model!r} has cached pickle weights that cannot be security-scanned "
                    "offline and no safetensors alternative, so it cannot be used as the "
                    "embedding model. Re-download it with safetensors weights while online."
                )
            else:
                detail = (
                    f"{model!r} is flagged as unsafe by Hugging Face's security scan and "
                    "cannot be used as the embedding model."
                )
            raise HTTPException(status_code = 403, detail = detail)
    if model != default_embedding_model() and not payload.force and not is_local_gguf:
        from core.rag import config as rag_config

        # A GGUF-named repo on llama-server is loaded from its .gguf files, which rarely carry ST metadata, so verify
        # GGUF availability instead of the embedding-metadata gate.
        gguf_named = destination_is_llama and rag_config._names_gguf(model)
        if not gguf_named and not is_embedding_model(verify_target, hf_token = hf_token):
            # Offline, is_embedding_model can only confirm the ST layout, so a cached and loadable transformers-native
            # embedder (gte-modernbert and the like) is accepted rather than 409'd where online would not. Uncached
            # still 409s.
            from utils.utils import hf_cache_snapshot_is_loadable

            # Require a genuinely loadable cache (config + weights), not just a resolved refs/main,
            # so a metadata-only partial cache still gets the forceable 409.
            # A cached private repo accepted here becomes this deployment's embedder.
            offline_cached = (
                local_only_load
                and cache_reads_authorized(hf_token, repo_id = verify_target)
                and hf_cache_snapshot_is_loadable(verify_target)
            )
            if not offline_cached:
                raise HTTPException(
                    status_code = 409,
                    detail = (
                        f"Could not verify {model!r} as an embedding model on "
                        "Hugging Face (it may be the wrong model type, gated, or "
                        "you may be offline)."
                    ),
                )
        # Any plan error counts, not just llama ones: is_embedding_model gates on tags, so a repo with no loadable
        # checkpoint passes it and would be persisted anyway.
        if plan.error:
            raise HTTPException(status_code = 409, detail = plan.error)
    trusted_backend = None
    trusted_gguf_repo = None
    trusted_gguf_files = None
    trusted_download_pending = False
    if plan.error is None:
        trusted_backend = "llama-server" if destination_is_llama else "sentence-transformers"
        # The exact family this transfer delivers. A repo publishing no RAG_EMBED_GGUF_VARIANT is served another
        # quant on purpose, which the loader's variant lookup cannot recognize as what was downloaded, so the model
        # would stay cache-only.
        trusted_gguf_files = plan.files if destination_is_llama else None
        # A sentence-transformers download repo is not a GGUF source. Keeping
        # it out also prevents a later runtime fallback from mislabelling it.
        trusted_gguf_repo = plan.download_repo if destination_is_llama else None
        # The setting may activate so both settings surfaces stay in sync, but its loader stays
        # cache-only until the transfer completes, or a close/cancel becomes an implicit download.
        trusted_download_pending = bool(plan.download_repo and not plan.cached)
    else:
        # Save anyway, over a failed plan: nothing validated to record, but the marker still has to go on or
        # both loaders take their uncached path and fetch invisibly at the first index.
        trusted_download_pending = True
    set_rag_embedding_model(
        model,
        gguf_repo = trusted_gguf_repo,
        backend = trusted_backend,
        download_pending = trusted_download_pending,
        gguf_files = trusted_gguf_files,
    )
    logger.info(
        "settings.embedding_model_updated subject=%s model=%s forced=%s",
        current_subject,
        model,
        payload.force,
    )
    return _embedding_model_response()


@_owner_settings_router.post("/embedding-model/unload", response_model = EmbeddingModelResponse)
def unload_embedding_model(
    current_subject: str = Depends(get_current_subject),
) -> EmbeddingModelResponse:
    """Drop the embedder and stop its llama-server. Indexing rebuilds it on demand."""
    from core.rag import embeddings

    released = embeddings.release_backend()
    logger.info(
        "settings.embedding_model_unloaded subject=%s released=%s", current_subject, released
    )
    return _embedding_model_response()


@_owner_settings_router.delete("/embedding-model", response_model = EmbeddingModelResponse)
def reset_embedding_model(
    current_subject: str = Depends(get_current_subject),
) -> EmbeddingModelResponse:
    """Clear the override, returning to the env/default model."""
    reset_rag_embedding_model()
    logger.info("settings.embedding_model_reset subject=%s", current_subject)
    return _embedding_model_response()


class PreviewLinkRotateResponse(BaseModel):
    rotated: bool = True


@_owner_settings_router.post("/preview-links/rotate", response_model = PreviewLinkRotateResponse)
def rotate_preview_links(
    current_subject: str = Depends(get_current_subject),
) -> PreviewLinkRotateResponse:
    """Rotate the preview-link signing secret, revoking every previously shared `/p` link."""
    rotate_preview_link_secret()
    logger.info("settings.preview_links_rotated subject=%s", current_subject)
    return PreviewLinkRotateResponse(rotated = True)


class KeylessApiAccessPayload(BaseModel):
    scope: Literal["off", "inference", "full"]
    tools: Optional[StrictBool] = None


class KeylessApiAccessResponse(BaseModel):
    scope: Literal["off", "inference", "full"]
    tools: bool
    exposure: Optional[Literal["colab", "public_url", "private_lan", "network"]] = None


class PreviewSharingPayload(BaseModel):
    enabled: bool


class PreviewSharingResponse(BaseModel):
    enabled: bool
    default_enabled: bool = DEFAULT_PREVIEW_SHARING_ENABLED


class ManagedProviderUrlsPayload(BaseModel):
    allowed: StrictBool


class ManagedProviderUrlsResponse(BaseModel):
    allowed: bool
    default_allowed: bool = DEFAULT_MANAGED_PRIVATE_PROVIDER_URLS_ALLOWED
    # UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS=1 holds the answer: the UI says why rather than
    # showing a switch that silently reverts.
    locked_by_environment: bool = False


class CurrentDatePromptPayload(BaseModel):
    enabled: StrictBool


class CurrentDatePromptResponse(BaseModel):
    enabled: bool
    default_enabled: bool = DEFAULT_CURRENT_DATE_PROMPT_ENABLED


class RemoteAccessAutoStartPayload(BaseModel):
    enabled: StrictBool


class RemoteAccessResponse(BaseModel):
    state: Literal["off", "starting", "online", "stopping", "error"]
    url: Optional[str] = None
    error: Optional[str] = None
    auto_start: bool
    default_auto_start: bool = DEFAULT_REMOTE_ACCESS_AUTO_START
    available: bool
    managed_by: Optional[Literal["launch", "settings", "colab"]] = None
    can_start: bool
    can_stop: bool
    block_reason: Optional[str] = None
    password_pending: bool = False
    streaming_supported: bool = True


def _require_ui_session(via_api_key: bool = Depends(authenticated_via_api_key)) -> None:
    if via_api_key:
        raise HTTPException(status_code = 403, detail = "Remote access requires a UI session.")


def _remote_access_response(request: Request) -> RemoteAccessResponse:
    return RemoteAccessResponse(**remote_access_status(request.app.state))


@_owner_settings_router.get("/remote-access", response_model = RemoteAccessResponse)
def get_remote_access(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> RemoteAccessResponse:
    return _remote_access_response(request)


@_owner_settings_router.post("/remote-access/start", response_model = RemoteAccessResponse)
def start_remote_access_route(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> RemoteAccessResponse:
    try:
        response = RemoteAccessResponse(**start_remote_access(request.app.state))
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    logger.info("settings.remote_access_start_requested subject=%s", current_subject)
    return response


@_owner_settings_router.post("/remote-access/stop", response_model = RemoteAccessResponse)
def stop_remote_access_route(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> RemoteAccessResponse:
    try:
        status = stop_remote_access(request.app.state)
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    status.update(
        state = "off",
        url = None,
        error = None,
        managed_by = None,
        can_start = False,
        can_stop = False,
    )
    response = RemoteAccessResponse(**status)
    logger.info("settings.remote_access_stop_requested subject=%s", current_subject)
    return response


@_owner_settings_router.put("/remote-access/auto-start", response_model = RemoteAccessResponse)
def update_remote_access_auto_start(
    request: Request,
    payload: RemoteAccessAutoStartPayload,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> RemoteAccessResponse:
    if bool(getattr(request.app.state, "remote_access_is_colab", False)):
        raise HTTPException(status_code = 409, detail = "colab")
    set_remote_access_auto_start(payload.enabled)
    logger.info(
        "settings.remote_access_auto_start_updated subject=%s enabled=%s",
        current_subject,
        payload.enabled,
    )
    return _remote_access_response(request)


class LanAccessAutoStartPayload(BaseModel):
    enabled: StrictBool


class LanAccessPortPayload(BaseModel):
    port: Optional[StrictInt] = Field(ge = 1, le = 65535)


class LanAccessResponse(BaseModel):
    state: Literal["off", "online", "error"]
    urls: list[str] = []
    public_urls: list[str] = []
    error: Optional[str] = None
    auto_start: bool

    configured_port: Optional[int] = None
    active_port: Optional[int] = None
    managed_by: Optional[Literal["launch", "settings"]] = None
    can_start: bool
    can_stop: bool
    block_reason: Optional[str] = None
    bind_host: Optional[str] = None
    wildcard_bind: bool = False
    serves_web_ui: bool = True
    keyless_lan_eligible: bool = False
    keyless_scope: Literal["off", "inference", "full"] = "off"
    keyless_tools: bool = False


def _lan_access_response(request: Request) -> LanAccessResponse:
    return LanAccessResponse(**lan_access_status(request.app))


@_owner_settings_router.get("/lan-access", response_model = LanAccessResponse)
def get_lan_access(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> LanAccessResponse:
    return _lan_access_response(request)


@_owner_settings_router.post("/lan-access/start", response_model = LanAccessResponse)
def start_lan_access_route(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> LanAccessResponse:
    try:
        response = LanAccessResponse(**start_lan_access(request.app))
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    logger.info("settings.lan_access_start_requested subject=%s", current_subject)
    return response


@_owner_settings_router.post("/lan-access/stop", response_model = LanAccessResponse)
def stop_lan_access_route(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> LanAccessResponse:
    try:
        response = LanAccessResponse(**stop_lan_access(request.app))
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    logger.info("settings.lan_access_stop_requested subject=%s", current_subject)
    return response


@_owner_settings_router.put("/lan-access/auto-start", response_model = LanAccessResponse)
def update_lan_access_auto_start(
    request: Request,
    payload: LanAccessAutoStartPayload,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> LanAccessResponse:
    if bool(getattr(request.app.state, "lan_access_is_colab", False)):
        raise HTTPException(status_code = 409, detail = "colab")
    set_lan_access_auto_start(payload.enabled)
    logger.info(
        "settings.lan_access_auto_start_updated subject=%s enabled=%s",
        current_subject,
        payload.enabled,
    )
    return _lan_access_response(request)


@_owner_settings_router.put("/lan-access/port", response_model = LanAccessResponse)
def update_lan_access_port(
    request: Request,
    payload: LanAccessPortPayload,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> LanAccessResponse:
    try:
        response = LanAccessResponse(**save_lan_access_port(request.app, payload.port))
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    logger.info(
        "settings.lan_access_port_updated subject=%s port=%s",
        current_subject,
        payload.port if payload.port is not None else "automatic",
    )
    return response


@_owner_settings_router.get("/preview-sharing", response_model = PreviewSharingResponse)
def get_preview_sharing(
    current_subject: str = Depends(get_current_subject),
) -> PreviewSharingResponse:
    return PreviewSharingResponse(enabled = get_preview_sharing_enabled())


@_owner_settings_router.put("/preview-sharing", response_model = PreviewSharingResponse)
def update_preview_sharing(
    payload: PreviewSharingPayload, current_subject: str = Depends(get_current_subject)
) -> PreviewSharingResponse:
    """Enable/disable the public `/p` preview surface. When off, links 404 even with a token."""
    try:
        enabled = set_preview_sharing_enabled(payload.enabled)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid preview sharing setting."),
            event = "settings.update_preview_sharing_failed",
            log = logger,
        ) from exc
    logger.info("settings.preview_sharing_updated subject=%s enabled=%s", current_subject, enabled)
    return PreviewSharingResponse(enabled = enabled)


def _managed_provider_urls_response() -> ManagedProviderUrlsResponse:
    # The EFFECTIVE answer, not the stored preference: a switch reading back on while every save
    # is refused would be the worst of the three things this could say.
    return ManagedProviderUrlsResponse(
        allowed = get_managed_private_provider_urls_allowed(),
        locked_by_environment = private_urls_locked_by_environment(),
    )


@_shared_settings_router.get("/managed-provider-urls", response_model = ManagedProviderUrlsResponse)
def get_managed_provider_urls(
    current_subject: str = Depends(get_current_subject),
) -> ManagedProviderUrlsResponse:
    """Readable by any account: a managed one has to be able to tell a refusal the owner can lift
    from one nobody on this installation can, and it learns the same bit by trying to save a URL."""
    return _managed_provider_urls_response()


@_owner_settings_router.put("/managed-provider-urls", response_model = ManagedProviderUrlsResponse)
def update_managed_provider_urls(
    payload: ManagedProviderUrlsPayload,
    current_subject: str = Depends(get_current_subject),
    # Installation policy: set at the console, not from a remote key that happens to be owned.
    _ui_session: None = Depends(_require_ui_session),
) -> ManagedProviderUrlsResponse:
    """Allow or refuse private and LAN provider base URLs for the installation's managed accounts.

    Off by default. The preference is stored either way, so removing
    ``UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS`` later restores what the owner chose here rather
    than a default.
    """
    try:
        allowed = set_managed_private_provider_urls_allowed(payload.allowed)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid managed provider URL setting."),
            event = "settings.update_managed_provider_urls_failed",
            log = logger,
        ) from exc
    logger.info(
        "settings.managed_provider_urls_updated subject=%s allowed=%s", current_subject, allowed
    )
    return _managed_provider_urls_response()


@_account_settings_router.get("/current-date-prompt", response_model = CurrentDatePromptResponse)
def get_current_date_prompt(
    current_subject: str = Depends(get_current_subject),
) -> CurrentDatePromptResponse:
    return CurrentDatePromptResponse(enabled = get_current_date_prompt_enabled())


@_account_settings_router.put("/current-date-prompt", response_model = CurrentDatePromptResponse)
def update_current_date_prompt(
    payload: CurrentDatePromptPayload, current_subject: str = Depends(get_current_subject)
) -> CurrentDatePromptResponse:
    """Enable/disable telling the model today's date in chat and Deep Research prompts."""
    try:
        enabled = set_current_date_prompt_enabled(payload.enabled)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid current date prompt setting."),
            event = "settings.update_current_date_prompt_failed",
            log = logger,
        ) from exc
    logger.info(
        "settings.current_date_prompt_updated subject=%s enabled=%s", current_subject, enabled
    )
    return CurrentDatePromptResponse(enabled = enabled)


def _require_ui_session_for_keyless(via_api_key: bool = Depends(authenticated_via_api_key)) -> None:
    """Only a signed-in UI session may change who needs a key. An sk-unsloth key must not be able to switch
    authentication off for the whole install, and a keyless caller must not be able to widen its own scope;
    both are ``authenticated_via_api_key``, so one check covers them."""
    if via_api_key:
        raise HTTPException(
            status_code = 403,
            detail = "Keyless API access can only be changed from the Unsloth UI.",
        )


def _keyless_api_access_response(request: Request) -> KeylessApiAccessResponse:
    scope, tools = get_keyless_api_access_settings()
    return KeylessApiAccessResponse(
        scope = scope,
        tools = tools,
        exposure = access_exposure(request.app.state),
    )


@_owner_settings_router.get("/keyless-api-access", response_model = KeylessApiAccessResponse)
def get_keyless_api_access(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session_for_keyless),
) -> KeylessApiAccessResponse:
    return _keyless_api_access_response(request)


@_owner_settings_router.put("/keyless-api-access", response_model = KeylessApiAccessResponse)
def update_keyless_api_access(
    request: Request,
    payload: KeylessApiAccessPayload,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session_for_keyless),
) -> KeylessApiAccessResponse:
    """Choose which routes are served without an API key, and whether tools come too."""
    scope, tools = set_keyless_api_access(payload.scope, tools = payload.tools)
    logger.info(
        "settings.keyless_api_access_updated subject=%s scope=%s tools=%s exposure=%s",
        current_subject,
        scope,
        tools,
        access_exposure(request.app.state),
    )
    return _keyless_api_access_response(request)


def _is_bundled_avatar_url(value: str) -> bool:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return False
    path = unquote(parsed.path).lstrip("/")
    if ".." in path.split("/"):
        return False
    marker = "Sloth emojis/"
    if marker not in path:
        return False
    return path[path.index(marker) :].lower().endswith(".png")


class PersonalizationProfile(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    displayName: str = Field("", max_length = 200)
    nickname: str = Field("", max_length = 200)
    avatarDataUrl: Optional[str] = Field(None, max_length = MAX_AVATAR_DATA_URL_BYTES)
    avatarShape: Literal["circle", "rounded"] = "circle"
    showGreetingSloth: bool = True

    @field_validator("avatarDataUrl")
    @classmethod
    def _validate_avatar(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        if not value.startswith("data:image/") and not _is_bundled_avatar_url(value):
            raise ValueError("avatarDataUrl must be an image data URL or bundled avatar.")
        return value


class PersonalizationCustomColors(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    accent: Optional[str] = Field(None, pattern = r"^#[0-9a-fA-F]{6}$")
    background: Optional[str] = Field(None, pattern = r"^#[0-9a-fA-F]{6}$")
    foreground: Optional[str] = Field(None, pattern = r"^#[0-9a-fA-F]{6}$")


class PersonalizationCustomColorModes(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    light: PersonalizationCustomColors = Field(default_factory = PersonalizationCustomColors)
    dark: PersonalizationCustomColors = Field(default_factory = PersonalizationCustomColors)


MAX_IMPORTED_FONTS = 3
# ~1.5 MB font file as base64; matches MAX_IMPORTED_FONT_DATA_URL_LENGTH in the frontend.
MAX_FONT_DATA_URL_LENGTH = 2_200_000
# Aggregate cap across all imported fonts, matching MAX_TOTAL_IMPORTED_FONT_DATA_URL_LENGTH in the
# frontend so a synced payload always fits the browser's localStorage quota.
MAX_TOTAL_FONT_DATA_URL_LENGTH = 4_400_000

# Characters that could terminate a CSS declaration, escape the quoted font-family value or smuggle extra
# fallbacks/comments if a stored name reached a stylesheet. The server is the authoritative gate; the frontend
# strips the same set.
_FONT_NAME_FORBIDDEN = set(";{}()<>\"'\\/,`")


def _check_font_name(value: str) -> str:
    if any(c in _FONT_NAME_FORBIDDEN or ord(c) < 0x20 for c in value):
        raise ValueError("Font name contains invalid characters.")
    return value


# Matches FONT_DATA_URL_PATTERN in the frontend appearance-custom-store.
_FONT_DATA_URL_PATTERN = re.compile(
    r"^data:(?:font/(?:woff2?|ttf|otf|sfnt)"
    r"|application/(?:octet-stream|x-font-\w+|font-\w+));base64,[A-Za-z0-9+/=]+$"
)


class PersonalizationImportedFont(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    name: str = Field(..., min_length = 1, max_length = 100)
    dataUrl: str = Field(..., max_length = MAX_FONT_DATA_URL_LENGTH)

    @field_validator("name")
    @classmethod
    def _validate_font_name(cls, value: str) -> str:
        return _check_font_name(value)

    @field_validator("dataUrl")
    @classmethod
    def _validate_font_data_url(cls, value: str) -> str:
        # fullmatch, not match: re's ``$`` also matches just before a trailing newline, which the frontend's
        # JS pattern (``$`` = end of string) rejects.
        if not _FONT_DATA_URL_PATTERN.fullmatch(value):
            raise ValueError("dataUrl must be a base64 font data URL.")
        return value


# Optional user-menu items; the boolean is each id's default visibility. Settings-tab shortcuts
# ship hidden.
SIDEBAR_MENU_ITEM_DEFAULTS = {
    "api": True,
    "darkMode": True,
    "guidedTour": True,
    "profile": False,
    "appearance": False,
    "resources": False,
    "chat": False,
    "connections": False,
}

# Navigable sidebar rows the user can pin/reorder; the boolean is each id's default pin state. Order and pin
# state MUST match the frontend's shipped layout (SIDEBAR_NAV_ITEM_IDS / SIDEBAR_NAV_DEFAULT_PINNED in
# features/settings/stores/appearance-custom-store.ts): the client sends every id on each save, so a missing id
# 422s the whole personalization PUT.
SIDEBAR_NAV_ITEM_DEFAULTS = {
    "hub": True,
    "projects": True,
    "library": True,
    "images": True,
    "video": False,
    "audio": False,
    "train": True,
    "recipes": False,
    "export": False,
    "api": False,
}

MAX_SIDEBAR_NAV_INPUT_ITEMS = 4 * len(SIDEBAR_NAV_ITEM_DEFAULTS)

# The sidebarMenu validator below dedupes ids and re-fills any missing ones
MAX_SIDEBAR_MENU_INPUT_ITEMS = 4 * len(SIDEBAR_MENU_ITEM_DEFAULTS)


class PersonalizationSidebarMenuItem(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    id: Literal[
        "api",
        "darkMode",
        "guidedTour",
        "profile",
        "appearance",
        "resources",
        "chat",
        "connections",
    ]
    visible: bool = True


def _default_sidebar_menu() -> "list[PersonalizationSidebarMenuItem]":
    return [
        PersonalizationSidebarMenuItem(id = item_id, visible = visible)
        for item_id, visible in SIDEBAR_MENU_ITEM_DEFAULTS.items()
    ]


SidebarNavItemId = Literal[
    "hub",
    "projects",
    "library",
    "images",
    "video",
    "audio",
    "train",
    "recipes",
    "export",
    "api",
]


class PersonalizationSidebarNavItem(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    id: SidebarNavItemId
    pinned: bool = True


def _default_sidebar_nav() -> "list[PersonalizationSidebarNavItem]":
    return [
        PersonalizationSidebarNavItem(id = item_id, pinned = pinned)
        for item_id, pinned in SIDEBAR_NAV_ITEM_DEFAULTS.items()
    ]


class PersonalizationCustomization(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    colors: PersonalizationCustomColorModes = Field(default_factory = PersonalizationCustomColorModes)
    uiFont: Optional[str] = Field(None, max_length = 200)
    headingFont: Optional[str] = Field(None, max_length = 200)
    chatFont: Optional[str] = Field(None, max_length = 200)
    codeFont: Optional[str] = Field(None, max_length = 200)
    importedFonts: list[PersonalizationImportedFont] = Field(
        default_factory = list, max_length = MAX_IMPORTED_FONTS
    )

    @field_validator("importedFonts")
    @classmethod
    def _validate_total_font_size(
        cls, value: list[PersonalizationImportedFont]
    ) -> list[PersonalizationImportedFont]:
        if sum(len(f.dataUrl) for f in value) > MAX_TOTAL_FONT_DATA_URL_LENGTH:
            raise ValueError("Imported fonts exceed the total size limit.")
        return value

    @field_validator("uiFont", "headingFont", "chatFont", "codeFont")
    @classmethod
    def _validate_selected_fonts(cls, value: Optional[str]) -> Optional[str]:
        # Selected font names reach CSS the same way imported names do.
        return value if value is None else _check_font_name(value)

    uiFontSize: Optional[int] = Field(None, ge = 12, le = 20)
    codeFontSize: Optional[int] = Field(None, ge = 10, le = 20)
    chatWidth: Literal["standard", "wide", "full"] = "standard"
    sentAttachments: Literal["auto", "list", "chips"] = "auto"
    contrast: int = Field(50, ge = 0, le = 100)
    pointerCursors: bool = False
    reduceMotion: Literal["system", "on", "off"] = "system"
    fontSmoothing: bool = True
    sidebarMenu: list[PersonalizationSidebarMenuItem] = Field(
        default_factory = _default_sidebar_menu,
        max_length = MAX_SIDEBAR_MENU_INPUT_ITEMS,
    )
    # Order is the sidebar's render order, so the validator keeps the client's.
    sidebarNav: list[PersonalizationSidebarNavItem] = Field(
        default_factory = _default_sidebar_nav,
        max_length = MAX_SIDEBAR_NAV_INPUT_ITEMS,
    )
    # Rows still following an automatic rule rather than a choice the user made. None means the
    # record predates the field, which the client tells apart from an explicit empty list: a
    # server-filled default would reapply a rule the user had already overruled.
    sidebarNavAuto: Optional[list[SidebarNavItemId]] = Field(
        None, max_length = MAX_SIDEBAR_NAV_INPUT_ITEMS
    )

    @field_validator("sidebarNavAuto")
    @classmethod
    def _validate_sidebar_nav_auto(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return None
        seen: set[str] = set()
        return [item for item in value if not (item in seen or seen.add(item))]

    @field_validator("sidebarMenu")
    @classmethod
    def _validate_sidebar_menu(
        cls, value: list[PersonalizationSidebarMenuItem]
    ) -> list[PersonalizationSidebarMenuItem]:
        # Drop duplicate ids (keep the first) and re-append any missing ids so
        # the stored list always covers every optional menu item exactly once.
        seen: set[str] = set()
        items = [item for item in value if not (item.id in seen or seen.add(item.id))]
        for item_id, visible in SIDEBAR_MENU_ITEM_DEFAULTS.items():
            if item_id not in seen:
                items.append(PersonalizationSidebarMenuItem(id = item_id, visible = visible))
        return items

    @field_validator("sidebarNav")
    @classmethod
    def _validate_sidebar_nav(
        cls, value: list[PersonalizationSidebarNavItem]
    ) -> list[PersonalizationSidebarNavItem]:
        # Like sidebarMenu, but order is preserved: dedupe, then append missing.
        seen: set[str] = set()
        items = [item for item in value if not (item.id in seen or seen.add(item.id))]
        for item_id, pinned in SIDEBAR_NAV_ITEM_DEFAULTS.items():
            if item_id not in seen:
                items.append(PersonalizationSidebarNavItem(id = item_id, pinned = pinned))
        return items


class PersonalizationAppearance(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    theme: Literal["light", "dark", "system"] = "system"
    palette: Literal[
        "standard",
        "classic",
        "minimal",
        "blueberry",
        "butterfly-pea",
        "cherry",
        "cinnamon",
        "cotton-candy",
        "dragon-fruit",
        "earl-grey",
        "espresso",
        "honey",
        "licorice",
        "macaron",
        "matcha",
        "mint",
        "neon-cyberpunk",
        "oat-milk",
        "peach",
        "pina-paraiso",
        "plum",
        "tangerine",
        "taro",
        "wasabi",
        "yuzu",
    ] = "standard"
    language: Optional[str] = Field(None, max_length = 20)
    customization: PersonalizationCustomization = Field(
        default_factory = PersonalizationCustomization
    )


_PALETTE_IDS = frozenset(get_args(PersonalizationAppearance.model_fields["palette"].annotation))


class PersonalizationPayload(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    version: int = PERSONALIZATION_VERSION
    profile: PersonalizationProfile = Field(default_factory = PersonalizationProfile)
    appearance: PersonalizationAppearance = Field(default_factory = PersonalizationAppearance)


class PersonalizationResponse(PersonalizationPayload):
    saved: bool = False
    # False when the stored record predates a field, so the client keeps local
    # overrides instead of treating a server-filled default as an explicit value.
    customizationSaved: bool = False
    chatWidthSaved: bool = False
    sentAttachmentsSaved: bool = False
    paletteSaved: bool = False
    greetingSlothSaved: bool = False


@_account_settings_router.get("/personalization", response_model = PersonalizationResponse)
def get_personalization_settings(
    current_subject: str = Depends(get_current_subject),
) -> PersonalizationResponse:
    stored = drop_unknown_palette(get_personalization(), _PALETTE_IDS)
    response = PersonalizationResponse.model_validate(stored or {})
    response.saved = bool(stored)
    appearance = stored.get("appearance") if isinstance(stored, dict) else None
    customization = appearance.get("customization") if isinstance(appearance, dict) else None
    profile = stored.get("profile") if isinstance(stored, dict) else None
    response.customizationSaved = isinstance(appearance, dict) and "customization" in appearance
    response.chatWidthSaved = isinstance(customization, dict) and "chatWidth" in customization
    response.sentAttachmentsSaved = (
        isinstance(customization, dict) and "sentAttachments" in customization
    )
    response.paletteSaved = isinstance(appearance, dict) and "palette" in appearance
    response.greetingSlothSaved = isinstance(profile, dict) and "showGreetingSloth" in profile
    return response


def _merge_personalization(base: dict, overlay: dict) -> dict:
    # Recursively overlay only the request's set fields onto the stored record, so a stale client that
    # omits newer keys does not materialize their defaults and defeat the *Saved legacy detection.
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = _merge_personalization(existing, value)
        else:
            merged[key] = value
    return merged


@_account_settings_router.put("/personalization", response_model = PersonalizationPayload)
def update_personalization_settings(
    payload: PersonalizationPayload, current_subject: str = Depends(get_current_subject)
) -> PersonalizationPayload:
    try:
        # exclude_unset so absent fields are not persisted as defaults; merge so
        # fields the request omits keep whatever the record already stored.
        incoming = payload.model_dump(exclude_unset = True)
        merged = _merge_personalization(get_personalization(), incoming)
        set_personalization(merged)
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid personalization settings."),
            event = "settings.update_personalization_failed",
            log = logger,
        ) from exc
    # Return the stored record, not the defaults-filled request, so the response
    # matches storage (and the next GET) for fields the client omitted. An unknown
    # stored palette is filtered like GET does; clients send a palette with every
    # save, so the next save replaces it.
    return PersonalizationPayload.model_validate(drop_unknown_palette(merged, _PALETTE_IDS))


# Backs Settings > Logs: the session log always existed, but its path was only printed to a console
# the desktop user never sees.
class DebugLogSourceModel(BaseModel):
    id: str
    family: str
    label: str
    realpath: str
    size_bytes: int
    modified_at: float
    is_current: bool


class DebugLogSourcesResponse(BaseModel):
    sources: list[DebugLogSourceModel]
    default_source_id: Optional[str] = None
    matched_source_id: Optional[str] = None
    file_logging_disabled: bool = False
    # Where the logs actually live, so a caller does not have to guess. The
    # desktop "Open logs folder" button otherwise falls back to a hard-coded
    # ~/.unsloth/studio/logs, which is wrong whenever UNSLOTH_STUDIO_HOME or
    # STUDIO_HOME is set. Additive and optional: an older client ignores it.
    log_root: Optional[str] = None


class DebugLogResponse(BaseModel):
    status: Literal["ok", "empty", "missing", "unreadable", "disabled"]
    reason: Optional[str] = None
    source_id: Optional[str] = None
    realpath: Optional[str] = None
    lines: list[str] = Field(default_factory = list)
    cursor: Optional[str] = None
    reset: bool = False
    reset_reason: Optional[str] = None
    dropped_bytes: int = 0
    truncated_head: bool = False
    # The reader stopped at the response cap, and without saying so the caller cannot tell a complete
    # answer from a partial one, which is invisible in manual mode where no next poll is coming.
    more_pending: bool = False
    # File logging is off, so anything readable here is a PREVIOUS session and will never grow. The status stays "ok"
    # because the content is real and worth reading; saying nothing made a stale log look live.
    file_logging_disabled: bool = False
    size_bytes: int = 0


@_owner_settings_router.get("/debug/logs/sources", response_model = DebugLogSourcesResponse)
def get_debug_log_sources(
    diagnostic_path: Optional[str] = None,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> DebugLogSourcesResponse:
    """Every log file the viewer may read, newest first within each family.

    Individual files, not one entry per family: the llama runner writes one file
    per load ATTEMPT, so after a retry the useful one is often not the newest.
    """
    from utils import debug_log_sources

    sources = debug_log_sources.list_sources()
    # The first candidate root is the one the walk prefers. File logging may
    # be disabled before logs/ is created, so reveal the existing home then.
    roots = debug_log_sources.candidate_roots()
    log_root = None
    if roots:
        logs_dir = roots[0] / "logs"
        log_root = str(logs_dir if logs_dir.is_dir() else roots[0])
    return DebugLogSourcesResponse(
        sources = [DebugLogSourceModel(**vars(source)) for source in sources],
        default_source_id = debug_log_sources.default_source_id(),
        matched_source_id = debug_log_sources.source_id_for_path(diagnostic_path, sources),
        file_logging_disabled = debug_log_sources.file_logging_disabled(),
        log_root = log_root,
    )


@_owner_settings_router.get("/debug/logs", response_model = DebugLogResponse)
def get_debug_log(
    source: Optional[str] = None,
    cursor: Optional[str] = None,
    lines: int = 1000,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> DebugLogResponse:
    """The tail of one log, then only what was appended after `cursor`.

    Every content state answers 200. This is polled once a second in Live mode,
    and a 404 or a 500 on "the file is not there yet" would make the viewer
    flash an error on every tick; the caller reads `status` instead.
    """
    from utils import debug_log_reader, debug_log_sources

    source_id = source or debug_log_sources.default_source_id()
    if not source_id:
        disabled = debug_log_sources.file_logging_disabled()
        return DebugLogResponse(
            status = "disabled" if disabled else "missing",
            reason = (
                "File logging is turned off (UNSLOTH_STUDIO_NO_FILE_LOG=1)."
                if disabled
                else "No log files have been written yet."
            ),
        )

    path = debug_log_sources.resolve_source_id(source_id)
    if path is None:
        # An id the enumeration no longer produces. 404 here (unlike the content
        # states above) so a stale picker refetches its sources.
        raise HTTPException(status_code = 404, detail = "Unknown log source.")

    try:
        result = debug_log_reader.read_since(path, cursor, lines)
    except FileNotFoundError:
        return DebugLogResponse(
            status = "missing",
            reason = "The log file was removed.",
            source_id = source_id,
        )
    except (OSError, PermissionError) as exc:
        # The message embeds the path, so it goes through redaction too.
        from utils.log_redaction import redact_log_text
        return DebugLogResponse(
            status = "unreadable",
            reason = redact_log_text(str(exc)),
            source_id = source_id,
        )

    return DebugLogResponse(
        status = "empty" if (result.size_bytes == 0 and not result.lines) else "ok",
        source_id = source_id,
        realpath = str(path),
        lines = result.lines,
        cursor = result.cursor,
        reset = result.reset,
        reset_reason = result.reset_reason,
        dropped_bytes = result.dropped_bytes,
        truncated_head = result.truncated_head,
        more_pending = result.more_pending,
        file_logging_disabled = debug_log_sources.source_is_frozen(source_id),
        size_bytes = result.size_bytes,
    )


# One build at a time, process-wide. The route is a sync `def`, so it runs in
# the 40-thread anyio pool shared with every other sync endpoint; a few
# concurrent exports starve it, and anyio cannot cancel a running thread, so it
# does not recover. A second caller is told to wait rather than queued.
_DEBUG_LOG_EXPORT_LOCK = threading.Semaphore(1)


@_owner_settings_router.get("/debug/logs/export")
def export_debug_logs(
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> StreamingResponse:
    """Every log the picker lists, redacted, as one ZIP.

    Same two dependencies as the routes above: a bundle of logs and the paths
    they came from is UI-operator material, so an API-key or keyless caller is
    refused. On `_owner_settings_router` for the same reason they are, since
    that router carries `_require_installation_owner`: on the bare `router` the
    BUNDLE would be reachable by an account refused each log individually.

    Built before the response exists rather than inside the generator, so a
    failure is a 500 instead of a truncated download.
    """
    from utils import debug_log_export

    if not _DEBUG_LOG_EXPORT_LOCK.acquire(blocking = False):
        raise HTTPException(
            status_code = 429,
            detail = "A log export is already running. Wait for it to finish and try again.",
        )
    try:
        archive = debug_log_export.build_log_archive()
    finally:
        _DEBUG_LOG_EXPORT_LOCK.release()

    stamp = time.strftime("%Y%m%d-%H%M%S")

    def _chunks():
        try:
            while True:
                chunk = archive.read(debug_log_export.STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk
        finally:
            archive.close()

    return StreamingResponse(
        _chunks(),
        media_type = "application/zip",
        headers = {
            # Neither shipping caller reads this back: the browser names the Blob
            # itself and the desktop path names the file in Rust. It is here for
            # a curl or address-bar caller, so do not assume the button uses it.
            "Content-Disposition": f'attachment; filename="unsloth-logs-{stamp}.zip"',
            # A stable authenticated GET is otherwise cacheable: the archive could
            # outlive the download in the on-disk cache, and a second export could
            # be answered from it rather than from the logs as they are now.
            # `no-store` not `no-cache`: it must not be WRITTEN, not revalidated.
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
        },
        # Belt and braces with the `finally` above: on a client abort Starlette
        # cancels the task group without raising GeneratorExit, so that `finally`
        # waits for a cyclic GC pass, holding up to SPOOL_MAX_BYTES meanwhile.
        background = BackgroundTask(archive.close),
    )


class SandboxToolStatus(BaseModel):
    backend: str
    available: bool
    reason: str
    protection_state: str = "unavailable"
    limitations: list[str] = Field(default_factory = list)
    remediation: str = ""


class SandboxWindowsStatus(BaseModel):
    runtime_installed: bool
    # None: this Windows can run MXC; "arch" (not x64) or "build" (older than 26100) otherwise.
    runtime_unsupported: Optional[Literal["arch", "build"]] = None
    allow_dacl_fallback: bool
    allow_dacl_fallback_saved: bool
    dacl_locked_by_environment: bool
    persistent_read_grants: bool
    persistent_read_grants_saved: bool
    grants_locked_by_environment: bool
    # None: MXC could not tell; [] prepared; otherwise the wxc-host-prep verbs still missing.
    host_prep_missing: Optional[list[str]] = None
    prepare_repeats_after_restart: bool = True
    # True: MXC runs in Windows' built-in container (BaseContainer); False: this Windows has none; None: unknown.
    builtin_container: Optional[bool] = None


class SandboxSetupStatus(BaseModel):
    action: Optional[str] = None
    # sudo | pkexec | uac, or None.
    elevation: Optional[str] = None
    manual_command: str = ""
    reason: str = ""
    needs_consent: bool = False
    can_run: bool = False


class SandboxStatusResponse(BaseModel):
    platform: str
    python: SandboxToolStatus
    terminal: SandboxToolStatus
    terminal_shell: Optional[str] = None
    windows: Optional[SandboxWindowsStatus] = None
    setup: Optional[SandboxSetupStatus] = None
    checked_at: float
    grants_restored: Optional[int] = None


class SandboxSettingsPayload(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    allow_dacl_fallback: Optional[StrictBool] = None
    persistent_read_grants: Optional[StrictBool] = None


class SandboxSetupPayload(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    operation: Literal["linux-install", "windows-setup", "windows-runtime"]
    consent_dacl_fallback: StrictBool = False


class SandboxSetupJob(BaseModel):
    state: Literal["idle", "running", "succeeded", "declined", "failed"]
    id: Optional[str] = None
    operation: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    output_tail: list[str] = Field(default_factory = list)
    steps: list[str] = Field(default_factory = list)
    manual_command: str = ""
    note: str = ""


class SandboxPrepareJob(BaseModel):
    state: Literal["idle", "running", "succeeded", "declined", "failed"]
    id: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    output_tail: list[str] = Field(default_factory = list)
    steps: list[str] = Field(default_factory = list)


_SANDBOX_STATUS_TTL_SECONDS = 30.0
_sandbox_status_lock = threading.Lock()
_sandbox_status_cache: Optional[tuple[float, SandboxStatusResponse]] = None
# Bumped by every change: a status built before a save must not be cached after it.
_sandbox_status_generation = 0


def _forget_sandbox_status() -> None:
    global _sandbox_status_cache, _sandbox_status_generation
    with _sandbox_status_lock:
        _sandbox_status_cache = None
        _sandbox_status_generation += 1


def _sandbox_tool_status(capability) -> SandboxToolStatus:
    return SandboxToolStatus(
        backend = capability.backend,
        available = capability.available,
        reason = capability.reason,
        protection_state = capability.protection_state,
        limitations = list(capability.limitations),
        remediation = capability.remediation,
    )


def _sandbox_terminal_target() -> tuple[str, Optional[str]]:
    import shutil
    import sys

    from core.inference import tools

    if sys.platform != "win32":
        return shutil.which("bash") or "bash", None
    profile = tools._terminal_profile(False)
    if profile == "cmd_isolated":
        return tools._windows_system_cmd(), profile
    return tools._windows_bash() or tools._windows_system_cmd(), profile


def _sandbox_windows_status() -> SandboxWindowsStatus:
    from core.inference import (
        mxc_adapter,
        mxc_policy,
        mxc_read_grants,
        mxc_runtime,
        sandbox_setup_plan,
    )
    from utils import mxc_isolation_settings as saved

    try:
        mxc_runtime.installation_identity()
        installed = True
    except Exception:
        installed = False
    missing: Optional[list[str]] = None
    if installed:
        steps = mxc_runtime.probe_host_prep_steps(env = mxc_adapter._control_environment())
        missing = None if steps is None else list(steps)
    return SandboxWindowsStatus(
        runtime_installed = installed,
        runtime_unsupported = sandbox_setup_plan.windows_runtime_unsupported(),
        allow_dacl_fallback = mxc_policy.dacl_fallback_enabled(),
        allow_dacl_fallback_saved = saved.dacl_fallback_setting(),
        dacl_locked_by_environment = saved.locked_by_environment(mxc_policy.DACL_FALLBACK_ENV),
        persistent_read_grants = mxc_read_grants.enabled(),
        persistent_read_grants_saved = saved.persistent_grants_setting(),
        grants_locked_by_environment = saved.locked_by_environment(
            mxc_read_grants.PERSISTENT_GRANTS_ENV
        ),
        host_prep_missing = missing,
    )


def _sandbox_windows_block(python, dacl_at_probe: bool) -> SandboxWindowsStatus:
    """wxc-exec does not name its tier, but with the fallback off it runs only in BaseContainer."""
    from core.inference import mxc_probe

    windows = _sandbox_windows_status()
    builtin = None
    # A save between the probe and this read would pair one setting's verdict with the other.
    if windows.runtime_installed and not (windows.allow_dacl_fallback or dacl_at_probe):
        if python.available and python.backend == "mxc-processcontainer":
            builtin = True
        elif python.reason == mxc_probe.NO_BUILTIN_CONTAINER_REASON:
            builtin = False
    return windows.model_copy(update = {"builtin_container": builtin})


def _build_sandbox_status(force: bool) -> SandboxStatusResponse:
    """Blocking (live probes); run off the event loop. Never elevates: probes only."""
    import sys

    from core.inference import os_sandbox

    windows_refresh = force and sys.platform == "win32"
    if windows_refresh:
        # The Terminal's bash-or-cmd choice reads cached verdicts; a Refresh must choose from fresh ones.
        from core.inference import mxc_probe, sandbox_probe, tools

        sandbox_probe.reset_probe_cache()
        mxc_probe.invalidate_cache()
        tools.reset_terminal_profile_cache()
    # After the resets above: they raise the floor an earlier generation is dropped under.
    generation = os_sandbox.tool_isolation_generation()
    dacl_at_probe = False
    if sys.platform == "win32":
        from core.inference import mxc_policy
        dacl_at_probe = mxc_policy.dacl_fallback_enabled()
    python = os_sandbox.capability_snapshot(
        force = force, execution_kind = "python", selected_executable = sys.executable
    )
    terminal_exe, shell = _sandbox_terminal_target()
    # On Windows the choice above just probed this executable.
    terminal = os_sandbox.capability_snapshot(
        force = force and not windows_refresh,
        execution_kind = "terminal",
        selected_executable = terminal_exe,
    )
    for tool, capability in (("python", python), ("terminal", terminal)):
        os_sandbox.note_tool_isolation(
            tool,
            capability.available,
            backend = capability.backend,
            reason = capability.reason,
            generation = generation,
        )
    return SandboxStatusResponse(
        platform = sys.platform,
        python = _sandbox_tool_status(python),
        terminal = _sandbox_tool_status(terminal),
        terminal_shell = shell,
        windows = _sandbox_windows_block(python, dacl_at_probe) if sys.platform == "win32" else None,
        setup = _sandbox_setup_status(python.available and terminal.available),
        checked_at = time.time(),
    )


def _sandbox_setup_status(available: bool) -> Optional[SandboxSetupStatus]:
    from core.inference import sandbox_setup_plan
    try:
        plan = sandbox_setup_plan.detect(available)
    except Exception as exc:  # noqa: BLE001 - the status stays useful without the setup hint
        logger.warning("settings.sandbox_setup_plan_failed: %s", exc)
        return None
    return SandboxSetupStatus(
        action = plan.action,
        elevation = plan.elevation,
        manual_command = plan.manual_command,
        reason = plan.reason,
        needs_consent = plan.needs_consent,
    )


def _for_request(status: SandboxStatusResponse, request: Request) -> SandboxStatusResponse:
    """Blocking. The setup button: a direct local request, or a Linux install that prompts nobody here."""
    from core.inference import sandbox_setup_plan
    from utils.client_ip import is_direct_local_request

    setup = status.setup
    if setup is None:
        return status
    local = bool(setup.action) and is_direct_local_request(request)
    update: dict = {"can_run": False}
    if setup.action == sandbox_setup_plan.LINUX_INSTALL:
        can_run, elevation = sandbox_setup_plan.linux_install_allowed(local = local)
        update = {"can_run": can_run, "elevation": elevation}
    elif local:
        update = {"can_run": True}
    return status.model_copy(update = {"setup": setup.model_copy(update = update)})


def _sandbox_status(refresh: bool = False) -> SandboxStatusResponse:
    global _sandbox_status_cache
    with _sandbox_status_lock:
        cached, generation = _sandbox_status_cache, _sandbox_status_generation
    if not refresh and cached is not None and time.monotonic() < cached[0]:
        return cached[1]
    status = _build_sandbox_status(force = refresh)
    with _sandbox_status_lock:
        if generation == _sandbox_status_generation:
            _sandbox_status_cache = (time.monotonic() + _SANDBOX_STATUS_TTL_SECONDS, status)
    return status


def _sandbox_invalidate() -> None:
    from core.inference import mxc_probe, tools

    mxc_probe.invalidate_cache()
    tools.reset_terminal_profile_cache()
    _forget_sandbox_status()


def _sandbox_apply(payload: SandboxSettingsPayload) -> Optional[int]:
    """Blocking: save, reset every cached verdict, and take the read grants back once they are off."""
    from core.inference import mxc_policy, mxc_read_grants
    from utils import mxc_isolation_settings as saved

    if payload.allow_dacl_fallback is not None:
        saved.set_dacl_fallback_setting(payload.allow_dacl_fallback)
    if payload.persistent_read_grants is not None:
        saved.set_persistent_grants_setting(payload.persistent_read_grants)
    saved.forget_cached_setting()
    _sandbox_invalidate()
    if not (mxc_policy.dacl_fallback_enabled() and mxc_read_grants.enabled()):
        return len(mxc_read_grants.revoke_recorded())
    return None


def _sandbox_job_response(job) -> SandboxPrepareJob:
    return (
        SandboxPrepareJob(**job.as_dict()) if job is not None else SandboxPrepareJob(state = "idle")
    )


@_owner_settings_router.get("/sandbox", response_model = SandboxStatusResponse)
async def get_sandbox_status(
    request: Request,
    refresh: bool = False,
    current_subject: str = Depends(get_current_subject),
) -> SandboxStatusResponse:
    """What Python and the Terminal get from the OS sandbox on this machine, plus the Windows opt-in."""
    try:
        status = await asyncio.to_thread(_sandbox_status, refresh)
        return await asyncio.to_thread(_for_request, status, request)
    except Exception as exc:
        raise log_and_http_error(
            exc,
            500,
            "Could not read the sandbox status.",
            event = "settings.sandbox_status_failed",
            log = logger,
        ) from exc


@_owner_settings_router.put("/sandbox", response_model = SandboxStatusResponse)
async def update_sandbox_settings(
    payload: SandboxSettingsPayload,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    # Host policy: changed at the console, never by an API key the owner happens to hold.
    _ui_session: None = Depends(_require_ui_session),
) -> SandboxStatusResponse:
    """Save the Windows MXC opt-in and the persistent read grant choice. Applies to the next launch."""
    import sys

    from core.inference import mxc_policy, mxc_read_grants
    from utils import mxc_isolation_settings as saved

    if sys.platform != "win32":
        raise HTTPException(status_code = 409, detail = "These settings only apply on Windows.")
    locks = (
        (payload.allow_dacl_fallback, mxc_policy.DACL_FALLBACK_ENV),
        (payload.persistent_read_grants, mxc_read_grants.PERSISTENT_GRANTS_ENV),
    )
    for value, env in locks:
        if value is not None and saved.locked_by_environment(env):
            raise HTTPException(
                status_code = 409,
                detail = f"{env} is set in the environment Unsloth runs in, which decides this.",
            )
    try:
        restored = await asyncio.to_thread(_sandbox_apply, payload)
        status = await asyncio.to_thread(_sandbox_status, False)
    except Exception as exc:
        raise log_and_http_error(
            exc,
            500,
            "Could not save the sandbox settings.",
            event = "settings.sandbox_update_failed",
            log = logger,
        ) from exc
    logger.info(
        "settings.sandbox_updated subject=%s dacl=%s grants=%s",
        current_subject,
        payload.allow_dacl_fallback,
        payload.persistent_read_grants,
    )
    status = await asyncio.to_thread(_for_request, status, request)
    return status.model_copy(update = {"grants_restored": restored})


@_owner_settings_router.get("/sandbox/prepare", response_model = SandboxPrepareJob)
def get_sandbox_prepare(current_subject: str = Depends(get_current_subject)) -> SandboxPrepareJob:
    return _sandbox_job_response(_newest_host_job(prepares_host = True))


@_owner_settings_router.post("/sandbox/prepare", response_model = SandboxPrepareJob)
async def start_sandbox_prepare(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> SandboxPrepareJob:
    """Run MXC's elevated host preparation; Windows shows its administrator prompt on this computer."""
    import sys

    from core.inference import mxc_host_prep_job, mxc_runtime, sandbox_setup_job
    from utils.client_ip import is_direct_local_request

    # Stricter than client_ip(): a loopback peer carrying proxy headers is a remote browser relayed here.
    if not is_direct_local_request(request):
        raise HTTPException(
            status_code = 403,
            detail = (
                "Prepare this PC from the computer running Unsloth: the Windows administrator "
                "prompt appears there, not in this browser."
            ),
        )
    if sys.platform != "win32":
        raise HTTPException(status_code = 409, detail = "Host preparation is Windows-only.")
    try:
        await asyncio.to_thread(mxc_runtime.installation_identity)
    except Exception as exc:
        raise HTTPException(
            status_code = 409,
            detail = "The MXC runtime is not installed; install it from Settings > Sandbox first.",
        ) from exc
    mxc_host_prep_job.add_finish_hook(_forget_sandbox_status)
    try:
        job = await asyncio.to_thread(mxc_host_prep_job.start)
    except sandbox_setup_job.SetupUnavailable as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    logger.info("settings.sandbox_prepare_started subject=%s job=%s", current_subject, job.id)
    return _sandbox_job_response(job)


def _sandbox_setup_response(job) -> SandboxSetupJob:
    if job is None:
        return SandboxSetupJob(state = "idle")
    data = job.as_dict()
    data.setdefault("operation", "windows-setup")
    return SandboxSetupJob(**data)


@_owner_settings_router.get("/sandbox/setup", response_model = SandboxSetupJob)
def get_sandbox_setup(current_subject: str = Depends(get_current_subject)) -> SandboxSetupJob:
    return _sandbox_setup_response(_newest_host_job())


def _newest_host_job(prepares_host: bool = False):
    """``prepares_host``: the prepare row reads this, so a runtime-only install is not its result."""
    from core.inference import mxc_host_prep_job, sandbox_setup_job, sandbox_setup_plan

    job = sandbox_setup_job.current()
    if prepares_host and job is not None and job.operation != sandbox_setup_plan.WINDOWS_SETUP:
        job = None
    prep = mxc_host_prep_job.current()
    if prep is not None and (job is None or prep.started_at > job.started_at):
        return prep
    return job


@_owner_settings_router.post("/sandbox/setup", response_model = SandboxSetupJob)
async def start_sandbox_setup(
    payload: SandboxSetupPayload,
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> SandboxSetupJob:
    """Install or prepare the OS sandbox here; the password or administrator prompt appears on this computer.

    Steps that need no prompt (the Windows runtime-only install, a Linux install as root or with
    passwordless sudo) also work from a remote browser.
    """
    import sys

    from core.inference import mxc_policy, sandbox_setup_job, sandbox_setup_plan
    from utils import mxc_isolation_settings as saved
    from utils.client_ip import is_direct_local_request

    # Stricter than client_ip(): a loopback peer carrying proxy headers is a remote browser relayed here.
    local = is_direct_local_request(request)
    if not local and not await asyncio.to_thread(
        sandbox_setup_plan.remote_start_allowed, payload.operation
    ):
        raise HTTPException(
            status_code = 403,
            detail = (
                "Set up the sandbox from the computer running Unsloth: the password or administrator "
                "prompt appears there, not in this browser."
            ),
        )
    platform_operations = (
        (sandbox_setup_plan.WINDOWS_SETUP, sandbox_setup_plan.WINDOWS_RUNTIME)
        if sys.platform == "win32"
        else (sandbox_setup_plan.LINUX_INSTALL,)
        if sys.platform.startswith("linux")
        else ()
    )
    if payload.operation not in platform_operations:
        raise HTTPException(
            status_code = 409, detail = f"{payload.operation} does not apply to this computer."
        )
    consent = (
        payload.operation == sandbox_setup_plan.WINDOWS_SETUP
        and payload.consent_dacl_fallback
        and not mxc_policy.dacl_fallback_enabled()
    )
    if consent and saved.locked_by_environment(mxc_policy.DACL_FALLBACK_ENV):
        raise HTTPException(
            status_code = 409,
            detail = (
                f"{mxc_policy.DACL_FALLBACK_ENV} is set in the environment Unsloth runs in, "
                "which decides this."
            ),
        )
    sandbox_setup_job.add_finish_hook(_forget_sandbox_status)
    try:
        job = await asyncio.to_thread(sandbox_setup_job.start, payload.operation, interactive = local)
    except sandbox_setup_job.SetupUnavailable as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    if consent:
        # Only once accepted: turned on first, the plan reads "already works" and never prepares.
        await asyncio.to_thread(_sandbox_apply, SandboxSettingsPayload(allow_dacl_fallback = True))
        sandbox_setup_plan.invalidate()
    _forget_sandbox_status()
    logger.info(
        "settings.sandbox_setup_started subject=%s operation=%s job=%s",
        current_subject,
        payload.operation,
        job.id,
    )
    return _sandbox_setup_response(job)


router.include_router(_account_settings_router)
router.include_router(_shared_settings_router)
router.include_router(_owner_settings_router)
