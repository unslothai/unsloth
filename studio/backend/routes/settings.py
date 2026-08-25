# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import functools
import hashlib
import re
import threading
import time
from typing import Any, Literal, Optional, get_args
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from auth.storage import rotate_preview_link_secret

from routes.provider_credentials import current_credential_write, require_ui_session

from storage import credential_secrets
from core.rag.config import default_gguf_repo, effective_gguf_repo
from loggers import get_logger
from utils.utils import safe_error_detail, log_and_http_error
from utils.personalization_settings import (
    MAX_AVATAR_DATA_URL_BYTES,
    PERSONALIZATION_VERSION,
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
from utils.xet_notice_settings import reserve_xet_notice
from utils.helper_precache_settings import (
    DEFAULT_HELPER_PRECACHE_ENABLED,
    get_helper_precache_enabled,
    helper_model_disabled_by_env,
    set_helper_precache_enabled,
)
from picker.schemas import MAX_CHAT_TEMPLATE_BYTES, chat_template_byte_length
from utils.coding_agents import CODING_AGENTS, detect_installed_coding_agents
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
from utils.preview_sharing_settings import (
    DEFAULT_PREVIEW_SHARING_ENABLED,
    get_preview_sharing_enabled,
    set_preview_sharing_enabled,
)
from utils.lan_access_settings import (
    lan_access_status,
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

router = APIRouter()

logger = get_logger(__name__)


class ImageGenerationPresetParams(BaseModel):
    """Bounds track DiffusionGenerateRequest. A preset the generate endpoint would refuse is not
    a usable preset: selecting it would make every following Generate fail validation."""

    model_config = ConfigDict(extra = "forbid")

    negativePrompt: str = ""
    width: int = Field(default = 1024, ge = 256, le = 2048, multiple_of = 16)
    height: int = Field(default = 1024, ge = 256, le = 2048, multiple_of = 16)
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
    # No cap on the read: upsert_media_generation_preset owns the limit, and refusing to
    # report a store that somehow exceeds it would only turn a GET into a 500.
    customPresets: list[ImageGenerationPreset] = Field(default_factory = list)
    saved: bool = False


class VideoGenerationPresetSettings(VideoGenerationPresetState):
    # No cap on the read: upsert_media_generation_preset owns the limit, and refusing to
    # report a store that somehow exceeds it would only turn a GET into a 500.
    customPresets: list[VideoGenerationPreset] = Field(default_factory = list)
    saved: bool = False


def _nested_model(annotation: Any) -> Optional[type[BaseModel]]:
    for candidate in (annotation, *get_args(annotation)):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _readable(model: type[BaseModel], value: Any) -> Any:
    """Drop what this build's schema does not define, keeping every field it does.

    `extra = "forbid"` is right for a submitted payload but wrong for reading storage back: a blob
    holding one field from a newer build would otherwise fail validation, and a stored recipe the
    user can no longer read is worse than one missing a field this build cannot render anyway.
    """
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
    """Validate, dropping only the fields that fail.

    Resetting the whole recipe over one unreadable field would hand the client schema defaults,
    which it then autosaves over the rest of a perfectly good stored recipe.
    """
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
    """Do not mistake a recovery default for an edit to an unreadable stored field.

    A downgraded GET omits known fields whose values this schema cannot validate, then Pydantic
    supplies their defaults in the response. The client cannot tell those defaults from stored
    values and echoes them in its next state write. Preserve the raw leaf only while the submitted
    value is still the synthesized value; a real edit remains authoritative.
    """
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
        # A value this build cannot represent at all. Drop only what fails: one unreadable entry
        # costs neither the rest of the list nor the state, which is validated on its own here.
        logger.warning("Dropping unreadable %s generation preset entries", kind)
        presets = schema.model_fields["customPresets"].annotation
        item = _nested_model(get_args(presets)[0] if get_args(presets) else presets)
        readable = []
        # Only a list is a preset collection. Recovery exists so a store this build cannot
        # represent still reads; iterating a scalar here would answer 500 instead, which is the
        # one outcome it is meant to prevent. _custom_presets takes the same view on the write.
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
    # Saved means the store owns the CURRENT recipe, not merely that something is stored. A blob
    # holding named presets but no recipe -- a preset write that landed while the state write did
    # not -- would otherwise hand back schema defaults dressed as the user's own choice, and the
    # client suppresses the resident model's defaults for exactly as long as it believes that.
    response.saved = isinstance(stored.get("currentParams"), dict)
    return response


@router.get(
    "/generation-presets/image",
    response_model = ImageGenerationPresetSettings,
)
def get_image_generation_preset_settings(
    current_subject: str = Depends(get_current_subject),
) -> ImageGenerationPresetSettings:
    return _get_generation_preset_settings("image", ImageGenerationPresetSettings)


@router.put("/generation-presets/image")
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


@router.get(
    "/generation-presets/video",
    response_model = VideoGenerationPresetSettings,
)
def get_video_generation_preset_settings(
    current_subject: str = Depends(get_current_subject),
) -> VideoGenerationPresetSettings:
    return _get_generation_preset_settings("video", VideoGenerationPresetSettings)


@router.put("/generation-presets/video")
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


@router.put("/generation-presets/image/custom")
def upsert_custom_image_generation_preset(
    payload: ImageGenerationPreset, current_subject: str = Depends(get_current_subject)
) -> dict[str, bool]:
    return _upsert_custom_generation_preset("image", payload)


@router.put("/generation-presets/video/custom")
def upsert_custom_video_generation_preset(
    payload: VideoGenerationPreset, current_subject: str = Depends(get_current_subject)
) -> dict[str, bool]:
    return _upsert_custom_generation_preset("video", payload)


@router.delete("/generation-presets/{kind}/custom")
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


@router.get("/hugging-face-token", response_model = HuggingFaceTokenResponse)
def get_hugging_face_token(
    _current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> HuggingFaceTokenResponse:
    require_ui_session(via_api_key)
    token = credential_secrets.get_hf_token()
    return HuggingFaceTokenResponse(token = token, has_token = token is not None)


@router.put("/hugging-face-token", response_model = HuggingFaceTokenResponse)
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


@router.put("/hugging-face-token/migrate", response_model = HuggingFaceTokenResponse)
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


@router.delete("/hugging-face-token", response_model = HuggingFaceTokenResponse)
def clear_hugging_face_token(
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> HuggingFaceTokenResponse:
    require_ui_session(via_api_key)
    with current_credential_write(credential):
        credential_secrets.delete_hf_token()
    return HuggingFaceTokenResponse(token = None, has_token = False)


class HelperPrecachePayload(BaseModel):
    enabled: bool


class HelperPrecacheResponse(BaseModel):
    enabled: bool
    default_enabled: bool = DEFAULT_HELPER_PRECACHE_ENABLED
    disabled_by_env: bool


class XetNoticeReservePayload(BaseModel):
    # A legacy localStorage count from a client that has not reported one before.
    # Can only raise the stored count (see reserve_xet_notice), so a client cannot
    # talk its own way back under the limit with it.
    seen_hint: int = 0


class XetNoticeResponse(BaseModel):
    granted: bool
    shown: int
    limit: int


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
    reload_required: bool
    # Soft RLIMIT_MEMLOCK when finite. mlock cannot exceed it, so the UI warns
    # that residency will not fully pin a model larger than this. None means
    # unlimited (macOS) or not applicable (Windows).
    memlock_limit_bytes: Optional[int] = None


class VramBudgetPayload(BaseModel):
    # None clears the stored budget so env/default applies again; it cannot also
    # mean "leave untouched" as the model-memory switches do, since there is one
    # field. Hence required, not defaulted: with a default, {} would mean "clear it"
    # and a client that dropped the field would silently discard the stored budget.
    fraction: Optional[float] = Field(ge = VRAM_FRACTION_MIN, le = VRAM_FRACTION_MAX)

    @field_validator("fraction", mode = "before")
    @classmethod
    def _reject_bool(cls, value: object) -> object:
        # bool subclasses int, so non-strict parsing turns True into 1.0 and stores
        # the max budget instead of 422; pydantic coerces before the util's guard.
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
    # And so is image/video auto-switch, for the same reason.
    media_auto_switch_model: Optional[bool] = None


class OpenAIAutoSwitchResponse(BaseModel):
    enabled: bool
    auto_unload_idle_seconds: int
    default_enabled: bool = DEFAULT_OPENAI_AUTO_SWITCH_ENABLED
    # True when the idle-unload loop will actually unload (effective TTL > 0). With
    # UNSLOTH_MODEL_IDLE_TTL set and nothing stored, this is true even while enabled
    # is false, so the UI can show idle-unload as active instead of "needs enable".
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


# A quant suffix, as modelOverrideKey builds it. Matched against the loader's quant pattern,
# not a length heuristic: a POSIX path may hold a colon and inherit another model's flags.
_MAX_VARIANT_SUFFIX_LEN = 64

# A local id is a path plus an optional quant suffix, and LoadRequest.model_path is unbounded.
# A limit under PATH_MAX would 422 the server sync while the local save succeeded.
MAX_MODEL_OVERRIDE_KEY_LEN = 4096 + 1 + _MAX_VARIANT_SUFFIX_LEN

# GgufVariantDetail.quant may be a path-qualified variant key, not just a quant suffix.
MAX_GGUF_VARIANT_KEY_LEN = 4096

# A list longer than MAX_GPU_ID cannot name a device the normalizer would store, so bound it
# here and reject an oversized array at the boundary instead of walking it.
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
    # None leaves the stored value alone (the UI has no control for flags); [] clears them.
    llama_extra_args: Optional[list[str]] = None
    # ge=1: the setter drops a falsy value, so reject 0 here instead of discarding it silently.
    max_seq_length: Optional[int] = Field(default = None, ge = 1, le = 1048576)
    custom_context_length: Optional[int] = Field(default = None, ge = 1, le = 1048576)
    kv_cache_dtype: Optional[str] = Field(default = None, max_length = 32)
    # A discrete set, enforced by the normalizer; these bounds only block absurd values.
    mlx_kv_bits: Optional[int] = Field(default = None, ge = 2, le = 8)
    speculative_type: Optional[str] = Field(default = None, max_length = 32)
    spec_draft_n_max: Optional[int] = Field(default = None, ge = 1, le = 16)
    # Parallel decode slots (llama-server --parallel), GGUF-only; None follows the server default.
    n_parallel: Optional[int] = Field(default = None, ge = PARALLEL_SLOTS_MIN, le = PARALLEL_SLOTS_MAX)
    # prompt batch sizes (--batch-size / --ubatch-size), gguf-only; none = llama.cpp defaults
    n_batch: Optional[int] = Field(default = None, ge = BATCH_SIZE_MIN, le = BATCH_SIZE_MAX)
    n_ubatch: Optional[int] = Field(default = None, ge = BATCH_SIZE_MIN, le = BATCH_SIZE_MAX)
    # The remaining llama-server tuning the picker remembers. model_override_load_kwargs
    # already applies all four off a stored row, so a route that drops them leaves the
    # setting reaching a picker load and nothing else, and the panel reads the gap back
    # as unset. Load mode is a discrete set, left to the normalizer like the KV dtype.
    load_mode: Optional[str] = Field(default = None, max_length = 32)
    spec_draft_cache_type: Optional[str] = Field(default = None, max_length = 32)
    # Stored on "is not None", not on truth: 0 checkpoints and a 0 or -1 cache are
    # meaningful values (none kept; cache disabled; no limit). Bounds mirror LoadRequest.
    ctx_checkpoints: Optional[int] = Field(default = None, ge = 0, le = CTX_CHECKPOINTS_MAX)
    cache_ram: Optional[int] = Field(default = None, ge = CACHE_RAM_MIN_MIB, le = CACHE_RAM_MAX_MIB)
    # Does this client know the four above exist? A save REPLACES the entry, so an
    # omission from a build that predates them is indistinguishable from a user
    # clearing them, and during an upgrade -- a cached bundle, or another LAN client
    # still on the old build -- that silently deletes settings it never sent. Only a
    # client that sets this may clear by omission; for anyone else the stored values
    # are carried over. Default False so an old payload, which cannot set it, is the
    # safe case. Not a blanket carry-over: that would make clearing impossible for
    # everyone, trading a mixed-version window for a permanent bug.
    mirrors_server_tuning: bool = False
    tensor_parallel: bool = False
    disable_vision: bool = False
    # Validated in bytes below: pydantic counts characters, so a multi-byte template would pass.
    chat_template_override: Optional[str] = None
    gpu_memory_mode: Optional[Literal["auto", "manual"]] = None
    # -1 is Auto (llama.cpp --fit sizes the offload); the normalizer treats it as unset.
    gpu_layers: Optional[int] = Field(default = None, ge = -1, le = 1024)
    n_cpu_moe: Optional[int] = Field(default = None, ge = 0, le = 1024)
    gpu_ids: Optional[list[int]] = Field(default = None, max_length = MAX_GPU_IDS)
    # An all-default save carries no fields, like a forget; None keeps the legacy contract.
    remove: Optional[bool] = None
    # Fill in, don't replace: the backfill reads the map once then writes each model, so another
    # tab's save was overwritten by this browser's older copy. Field level, not entry level: a
    # legacy entry holds only some fields, and skipping it would strand the rest.
    fill_absent_fields: bool = False

    @field_validator("chat_template_override")
    @classmethod
    def _limit_chat_template_bytes(cls, value: Optional[str]) -> Optional[str]:
        # Mirrors LoadRequest.normalize_blank_chat_template_override.
        if value is None:
            return None
        size = chat_template_byte_length(value)
        if size is None:
            raise ValueError("Chat template contains unpaired surrogate characters.")
        if size > MAX_CHAT_TEMPLATE_BYTES:
            raise ValueError(f"Chat template exceeds the {MAX_CHAT_TEMPLATE_BYTES}-byte limit.")
        return value

    @field_validator(
        "max_seq_length",
        "custom_context_length",
        "spec_draft_n_max",
        "n_parallel",
        "n_batch",
        "n_ubatch",
        "ctx_checkpoints",
        "cache_ram",
        "gpu_layers",
        "n_cpu_moe",
        "gpu_ids",
        mode = "before",
    )
    @classmethod
    def _no_booleans(cls, value: Any) -> Any:
        # bool subclasses int and pydantic parses non-strictly, so `true` arrives as 1: a
        # payload could pin GPU 1 or set a one-token context. _bounded_int rejects bools but
        # never sees one, since coercion happens here first. Only bools, so lax parsing stays.
        if isinstance(value, bool):
            raise ValueError("Expected a number, got a boolean.")
        if isinstance(value, list) and any(isinstance(item, bool) for item in value):
            raise ValueError("Expected numbers, got a boolean.")
        return value


class ModelOverridesResponse(BaseModel):
    overrides: dict[str, dict]
    # Filled only when the caller named a model: the entry ITS load would apply,
    # resolved here rather than in the browser. The folding rules are Python's
    # (casefold is not toLowerCase, and an ambiguous fold matches nothing on
    # purpose), so a client mirroring them can only approximate.
    resolved: Optional[dict] = None
    resolved_key: Optional[str] = None


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


# Distinct from None, which is a real launch this policy does not govern.
_NO_LAUNCH = object()


def _active_launch_placement():
    """``(state, policy_active, mlock_applicable)`` for the running child.

    ``state`` is ``_NO_LAUNCH`` when nothing is running or coming up, so the
    caller can tell "no process" apart from "a process with no load-mode".
    """
    try:
        from routes.inference import get_llama_cpp_backend

        backend = get_llama_cpp_backend()
        pending = bool(getattr(backend, "_memory_launch_pending", False))
        if not backend.is_active and not pending:
            return _NO_LAUNCH, False, True
        return (
            getattr(backend, "_memory_state", None),
            bool(getattr(backend, "_memory_policy_active", False)),
            bool(getattr(backend, "_memory_mlock_applicable", True)),
        )
    except Exception:
        return _NO_LAUNCH, False, True


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
    state, policy_active, mlock_applicable = _active_launch_placement()
    if state is _NO_LAUNCH:
        return False

    # Same predicate the duplicate-load comparator uses, so the reload hint and
    # the reload path can never disagree.
    from core.inference.llama_server_args import memory_state_satisfies_settings

    return not memory_state_satisfies_settings(state, policy_active, mlock_applicable)


def _model_memory_mlock_active(want_mlock: bool) -> bool:
    """Whether page-locking is actually in force, not merely asked for.

    This drives the locked-memory cap warning, so taking it from the toggles
    alone would tell a discrete-GPU user to raise a limit nothing consults.
    With nothing running this is the intent, so the UI reflects the toggle. Once
    a child exists it is what that child got: a full offload to a discrete GPU
    skips the lock, and a diffusion runner has no load-mode at all, so claiming
    otherwise would warn about ulimit -l for a lock nobody took. A user's own
    --mlock counts, since the resolver reads the launched argv.
    """
    if not want_mlock:
        return False
    state, _policy_active, _applicable = _active_launch_placement()
    if state is _NO_LAUNCH:
        return True
    return bool(state and state[0])


def _model_memory_response() -> ModelMemoryResponse:
    keep_resident, no_ram_reserve = get_model_memory_settings()
    mlock_active = _model_memory_mlock_active(should_mlock())
    return ModelMemoryResponse(
        keep_resident = keep_resident,
        no_ram_reserve = no_ram_reserve,
        mlock_active = mlock_active,
        reload_required = _model_memory_reload_required(),
        memlock_limit_bytes = memlock_limit_bytes() if mlock_active else None,
    )


def _vram_budget_reload_required(fraction: float) -> bool:
    """True when a child is running that was sized against a different budget.

    Compares against the fraction the child actually launched with, not merely
    "is something loaded", so re-saving the same value does not nag for a reload.
    Exact equality is fine: both sides come from the same clamp, so a stored 0.97
    and a launched 0.97 are the same float.
    """
    try:
        from routes.inference import get_llama_cpp_backend

        backend = get_llama_cpp_backend()
        # A planned-but-unspawned load has no _process, so is_active is False while
        # the child is already committed to its captured fraction; answer from the
        # pending value there, as _active_launch_placement does for Model Memory.
        pending = getattr(backend, "_vram_fraction_pending", None)
        if pending is not None:
            return float(pending) != float(fraction)
        if not backend.is_active:
            return False
        launched = getattr(backend, "_vram_fraction_launched", None)
        # A child predating this field, or from a path that never set it, cannot be
        # compared; say no rather than nagging on every save.
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


@router.get("/hugging-face-cache", response_model = HuggingFaceCacheResponse)
def get_hugging_face_cache(
    current_subject: str = Depends(get_current_subject),
) -> HuggingFaceCacheResponse:
    return _hugging_face_cache_response()


@router.put("/hugging-face-cache", response_model = HuggingFaceCacheResponse)
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


@router.get("/llama-cpp-path", response_model = LlamaCppPathResponse)
def get_llama_cpp_path(current_subject: str = Depends(get_current_subject)) -> LlamaCppPathResponse:
    return _llama_cpp_path_response()


@router.put("/llama-cpp-path", response_model = LlamaCppPathResponse)
def update_llama_cpp_path(
    payload: LlamaCppPathPayload,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> LlamaCppPathResponse:
    # Only the interactive Studio UI may change this executable setting.
    require_ui_session(via_api_key)
    try:
        set_custom_llama_cpp_path(payload.path)
    except RuntimeError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            # Validator messages are safe to expose to the UI.
            str(exc),
            event = "settings.update_llama_cpp_path_failed",
            log = logger,
        ) from exc
    return _llama_cpp_path_response()


@router.get("/upload-limit", response_model = UploadLimitResponse)
def get_upload_limit(current_subject: str = Depends(get_current_subject)) -> UploadLimitResponse:
    return _upload_limit_response(get_upload_limit_mb())


@router.put("/upload-limit", response_model = UploadLimitResponse)
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


@router.get("/helper-precache", response_model = HelperPrecacheResponse)
def get_helper_precache(
    current_subject: str = Depends(get_current_subject),
) -> HelperPrecacheResponse:
    return _helper_precache_response()


@router.put("/helper-precache", response_model = HelperPrecacheResponse)
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


@router.post("/xet-notice/reserve", response_model = XetNoticeResponse)
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


@router.get("/model-memory", response_model = ModelMemoryResponse)
def get_model_memory(current_subject: str = Depends(get_current_subject)) -> ModelMemoryResponse:
    return _model_memory_response()


@router.put("/model-memory", response_model = ModelMemoryResponse)
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
    # The client clock when the request was sent: the skew (server_now - client_now)
    # translates loaded_at into the server frame. Never persisted.
    client_now: Optional[int] = Field(default = None, ge = 0)


class LastLocalModelResponse(BaseModel):
    id: Optional[str] = None
    kind: Optional[Literal["gguf", "model"]] = None
    gguf_variant: Optional[str] = None
    loaded_at: Optional[int] = None
    # Lets the client translate loaded_at back into its own clock frame.
    server_now: Optional[int] = None


@router.get("/last-local-model", response_model = LastLocalModelResponse)
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


@router.put("/last-local-model", response_model = LastLocalModelResponse)
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


@router.get("/vram-budget", response_model = VramBudgetResponse)
def get_vram_budget(current_subject: str = Depends(get_current_subject)) -> VramBudgetResponse:
    return _vram_budget_response()


@router.put("/vram-budget", response_model = VramBudgetResponse)
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


@router.get("/coding-agents", response_model = CodingAgentsResponse)
def get_coding_agents(current_subject: str = Depends(get_current_subject)) -> CodingAgentsResponse:
    return CodingAgentsResponse(detected = detect_installed_coding_agents())


@router.get("/openai-auto-switch", response_model = OpenAIAutoSwitchResponse)
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


@router.put("/openai-auto-switch", response_model = OpenAIAutoSwitchResponse)
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
        # Drop already-saved chat context too. Configured, not effective: residency
        # zeroes the TTL, and that must not discard KV the user still wants.
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


@router.get("/openai-auto-switch/overrides", response_model = ModelOverridesResponse)
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
    """Whether a load for this model would still pick flags off another entry.

    The carry-over copies a legacy bare ``repo`` row's flags onto the first
    ``repo:QUANT`` save and leaves the bare row in place, and a load reads the
    qualified key first and the bare one after it. So clearing the box for the quant
    is only a clear while the quant keeps a row of its own: an all-default save
    stores nothing, and the next load falls through to a row no page can show.

    Answered rather than repaired. Stripping the flags off the bare row was the first
    fix and it is too broad: that row is the fallback for every quant that has no row,
    so forgetting Q4's flags took Q6's with them, and it did nothing at all when a
    sibling quant had a row of its own.
    """
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


def _other_quants_remain(bare_id: str, removed_ids: list[str]) -> bool:
    """Whether a quant of ``bare_id`` other than the ones being removed still has an entry.

    Such a quant has its own settings and never reads the bare fallback, so this is not
    "is anyone inheriting" but "is this forget the last one for the model". If it is not,
    the bare entry stays: an inheriting quant is exactly what it is there for.
    """
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
    """The stored ``<path>:LABEL`` entry for a bare standalone .gguf path, if any.

    A loose file has no quant to choose between, so it is keyed by the bare path,
    but the label derived from its filename is never empty and that is how the
    picker keyed the same file before, so an upgraded install carries entries
    under it. The auto-switch loader reads that spelling after the bare path
    misses; resolve_model_override_key does not, since folding a POSIX path only
    touches an existing suffix. None for an id that already names a quant, for a
    repo id, and when nothing is stored under the derived key.
    """
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
    """Where a one-time backfill write for ``target_id`` has to land.

    A fill only adds, so unlike a save it cannot retire the other spelling of a cached
    repo. Creating the snapshot-path key while the server already holds the repo id
    would leave two entries for one quant, and the loader reads the load path before the
    advertised id, so an upgraded browser's pre-upgrade copy would shadow the newer
    server config on every API load. Fill into the entry already there instead: nothing
    outranks it, and the fields it lacks still arrive.

    Only in that direction. A repo-id key never outranks an existing path entry, and two
    snapshot paths name two caches, neither of which is knowably the one loaded here.
    """
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


# One override write at a time. A save stores its target key and then reads the map back to
# retire the other spelling of the same cached repo, and a remove clears up to four keys, each
# its own transaction: atomic on their own, but not as a sequence. This route is a plain `def`,
# so FastAPI runs it in a threadpool, and two clients saving one quant under both spellings (the
# repo id the picker sends and the snapshot path an upgraded install still holds) could each
# write before either cleanup ran and then retire the other's row, leaving no override at all
# from two saves that both returned 200. Serialize the whole handler instead: overrides are
# written by a settings edit, never on a hot path, and the server runs one process.
_override_write_lock = threading.Lock()


def _serialized_override_write(func):
    """Run ``func`` under _override_write_lock, keeping the handler body as it reads.

    functools.wraps carries __wrapped__, which inspect.signature follows, so FastAPI still
    sees the endpoint's own parameters and dependencies.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with _override_write_lock:
            return func(*args, **kwargs)

    return wrapper


@router.put("/openai-auto-switch/overrides", response_model = ModelOverridesResponse)
@_serialized_override_write
def update_openai_auto_switch_override(
    payload: ModelOverridePayload, current_subject: str = Depends(get_current_subject)
) -> ModelOverridesResponse:
    from core.inference.llama_server_args import drop_managed_flags, validate_extra_args
    from utils.openai_auto_switch_settings import get_model_override

    try:
        if payload.fill_absent_fields and payload.remove is True:
            # A fill that is also a delete has no meaning; picking one loses or resurrects.
            raise ValueError("fill_absent_fields cannot be combined with remove.")
        # Only model_id is the documented "remove"; otherwise omitted flags carry over.
        requested_extra_args = payload.llama_extra_args
        # fill_absent_fields and mirrors_server_tuning are write modes, not saved fields:
        # leaving either in would make every payload look non-empty (they are bools, so
        # exclude_none does not drop them) and break the legacy "no fields means remove".
        saved_fields = payload.model_dump(
            exclude = {
                "model_id",
                "llama_extra_args",
                "remove",
                "fill_absent_fields",
                "mirrors_server_tuning",
            },
            exclude_none = True,
        )
        if payload.remove is not None:
            is_removal = payload.remove
        else:
            # Both booleans are carried, not just counted: they are stored only when
            # true, so an override whose one setting is either of them has no other
            # saved field and would otherwise read as a removal and be deleted.
            is_removal = (
                not payload.tensor_parallel
                and not payload.disable_vision
                and not {
                    key: value
                    for key, value in saved_fields.items()
                    if key not in ("tensor_parallel", "disable_vision")
                }
            )
        if requested_extra_args is None and not is_removal:
            stored = get_model_override(payload.model_id)
            # A fill keeps the stored flags without echoing them back through validation: one
            # denylisted since it was saved would 400 the migration, which then retries forever.
            if not (payload.fill_absent_fields and stored):
                requested_extra_args = stored.get("llama_extra_args")
                if requested_extra_args is None:
                    # First per-quant save for flags under the bare repo id; carry them over.
                    bare_id = _bare_model_id(payload.model_id)
                    if bare_id:
                        requested_extra_args = get_model_override(bare_id).get("llama_extra_args")
                if requested_extra_args is None:
                    # And for a standalone .gguf upgraded from the build that keyed it by its
                    # filename label: the bare path written here is read before that key, so
                    # its flags would go dark with no page able to show or restore them.
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
        # Not validated on an explicit remove: a 400 would only leave the override in place.
        if payload.remove is True:
            extra_args = []
        elif payload.llama_extra_args is None:
            # Carried over, not sent: the caller is saving some other field and this
            # value predates the request. A flag denylisted since it was written is
            # dropped rather than refused, or an unrelated save fails naming a flag
            # the user may not remember writing (and cannot fix from this payload).
            extra_args, dropped_flags = drop_managed_flags(requested_extra_args)
            if dropped_flags:
                logger.warning(
                    "model_override.dropped_managed_flags model_id=%s flags=%s",
                    payload.model_id,
                    ", ".join(dropped_flags),
                )
        else:
            extra_args = validate_extra_args(requested_extra_args)
        # Same shape as the extra-args carry-over above, for the same reason: a save
        # replaces the entry, so a field the caller never knew about must survive it.
        # A client that declares it mirrors these clears by omission as usual; an
        # older one keeps whatever is stored. On a remove the whole entry goes, so
        # there is nothing to preserve.
        # Gated on is_removal, not on payload.remove: the documented legacy contract is a
        # payload carrying only model_id, which leaves remove None while is_removal is
        # true. Carrying anything over there would rebuild a non-empty row and the clear
        # would silently do nothing.
        _tuning_fields = ("load_mode", "spec_draft_cache_type", "ctx_checkpoints", "cache_ram")
        _kept_tuning = {name: getattr(payload, name) for name in _tuning_fields}
        if not payload.mirrors_server_tuning and not is_removal:
            # The same spellings the extra-args carry-over walks, and in the same order.
            # A cached repo is not an ordinary folded match, so a save under the repo id
            # while the row sits under the snapshot path finds nothing here and then
            # retires that alias below, taking the tuning with it.
            _alias_ids = [payload.model_id]
            for _candidate in (
                _bare_model_id(payload.model_id),
                _legacy_standalone_gguf_key(payload.model_id),
                *cached_repo_alias_keys(payload.model_id),
            ):
                if _candidate and _candidate not in _alias_ids:
                    _alias_ids.append(_candidate)
            # Load order, not the order they were collected in. A lookup reads the
            # concrete load path before the advertised repo id, so on a cache upgraded
            # from a build that keyed rows by path, the snapshot row is the one that
            # applies and the one the retirement block below clears. Reading the repo
            # row first would adopt tuning no load has ever used and drop the tuning
            # that was live. Stable, so every other spelling keeps its position.
            _alias_ids.sort(key = lambda _key: not is_cache_load_path_key(_key))
            # Taken as a unit from the first row that exists, not field by field down
            # the list. A load stops at the first non-empty row (resolve_override_for_load)
            # rather than merging, so tuning in a row that never wins is dormant, and
            # filling a gap in the winner from a loser would switch it on as a side effect
            # of saving something unrelated. Single-valued above, so the distinction only
            # shows up here.
            for _alias_id in _alias_ids:
                _stored_tuning = get_model_override(_alias_id)
                if not _stored_tuning:
                    continue
                for name in _tuning_fields:
                    if _kept_tuning[name] is None:
                        _kept_tuning[name] = _stored_tuning.get(name)
                break
        if payload.remove is True:
            # An explicit remove wins over any other field. Remove the key a load resolves to,
            # not the literal one sent (the browser normalizes casing), and every spelling:
            # clearing one of two leaves the survivor as the sole fold match.
            target_ids = resolve_model_override_keys(payload.model_id) or [
                payload.model_id,
            ]
            for target_id in target_ids:
                set_model_override(target_id, llama_extra_args = [], max_seq_length = None)
            # A standalone .gguf is keyed by its bare path now, but a load also reads the
            # filename-derived <path>:LABEL an upgraded install holds, which would outlive this.
            legacy_id = _legacy_standalone_gguf_key(payload.model_id)
            if legacy_id and legacy_id not in target_ids:
                set_model_override(
                    legacy_id,
                    llama_extra_args = [],
                    max_seq_length = None,
                )
            # The mirror image of the carry-over above: a save under repo:QUANT copies the
            # flags off a legacy bare `repo` entry and leaves it in place, and the loader falls
            # back to it when the qualified key misses, so clearing only the qualified key hands
            # the same flags straight back and the forget does nothing. Nothing in the UI can
            # reach that bare entry. Only once it is nobody else's fallback, though: it backs
            # every quant with no entry of its own, so forgetting Q4 must not strip Q8.
            bare_id = _bare_model_id(payload.model_id)
            if (
                bare_id
                and bare_id not in target_ids
                and not _other_quants_remain(
                    bare_id,
                    target_ids,
                )
            ):
                set_model_override(
                    bare_id,
                    llama_extra_args = [],
                    max_seq_length = None,
                )
            # And the other spelling of a cached repo: the loader reads the load path before
            # the advertised id, so clearing only the id leaves the path entry still applying.
            for alias_id in cached_repo_alias_keys(payload.model_id):
                set_model_override(alias_id, llama_extra_args = [], max_seq_length = None)
        else:
            # Save under the key a load resolves to, as the removal branch does: the literal
            # id would leave two keys for one model, making every other casing ambiguous.
            target_id = resolve_model_override_key(payload.model_id) or payload.model_id
            if payload.fill_absent_fields:
                # A fill retires nothing below, so it must not create the higher-priority
                # spelling of a row the server already holds.
                target_id = _fill_target_id(target_id)
            # An explicit clear keeps a row even when nothing else is set, so long as a
            # fallback would otherwise answer for this model: "no launch flags" and
            # "nothing stored" are the same thing everywhere else, and different here.
            # Written on the quant's own key, so no other quant is touched.
            keep_empty = (
                payload.llama_extra_args == []
                and not payload.fill_absent_fields
                and _fallback_supplies_extra_args(payload.model_id, target_id)
            )
            set_model_override(
                target_id,
                llama_extra_args = extra_args,
                keep_empty_extra_args = keep_empty,
                max_seq_length = payload.max_seq_length,
                custom_context_length = payload.custom_context_length,
                kv_cache_dtype = payload.kv_cache_dtype,
                mlx_kv_bits = payload.mlx_kv_bits,
                speculative_type = payload.speculative_type,
                spec_draft_n_max = payload.spec_draft_n_max,
                n_parallel = payload.n_parallel,
                n_batch = payload.n_batch,
                n_ubatch = payload.n_ubatch,
                load_mode = _kept_tuning["load_mode"],
                spec_draft_cache_type = _kept_tuning["spec_draft_cache_type"],
                ctx_checkpoints = _kept_tuning["ctx_checkpoints"],
                cache_ram = _kept_tuning["cache_ram"],
                tensor_parallel = payload.tensor_parallel,
                disable_vision = payload.disable_vision,
                chat_template_override = payload.chat_template_override,
                gpu_memory_mode = payload.gpu_memory_mode,
                gpu_layers = payload.gpu_layers,
                n_cpu_moe = payload.n_cpu_moe,
                gpu_ids = payload.gpu_ids,
                fill_absent_fields = payload.fill_absent_fields,
            )
            # A repo cached outside the active HF cache is keyed here by its repo id, while the
            # loader reads the snapshot path first and an older release keyed the row by that
            # path, so an upgrade can hold both. Retire the spelling this save supersedes (its
            # flags were carried over above), or the leftover outranks the key just written.
            # After the write, so a rejected save deletes nothing. Not on a fill: that pass only
            # adds, and the migration mirroring both spellings must not delete either.
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
    return ModelOverridesResponse(overrides = get_model_overrides())


class EmbeddingModelPayload(BaseModel):
    embedding_model: str = Field(..., min_length = 1, max_length = MAX_EMBEDDING_MODEL_LENGTH)
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


def _embedding_model_response() -> EmbeddingModelResponse:
    return EmbeddingModelResponse(
        embedding_model = get_rag_embedding_model(),
        embedding_gguf_repo = effective_gguf_repo(),
        default_embedding_model = default_embedding_model(),
        default_embedding_gguf_repo = default_gguf_repo(),
        is_custom = get_stored_embedding_model() is not None,
    )


def _ambient_hf_token() -> Optional[str]:
    """The HF token the loader would use (HF_TOKEN env or the cached login), so a gated
    repo is scanned rather than failing open. None if unavailable."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _llama_backend_active() -> bool:
    """True when this install actually embeds via the llama-server (GGUF) backend.

    Delegates to the embeddings module so a runtime fallback from
    sentence-transformers to llama-server (after a torch/CUDA load or encode
    failure) is honored: in that state the process loads only inert GGUF, so the
    ST pickle gate below must not hard-block a repo whose GGUF companion is clean.
    Before any backend is built this still reflects the resolver."""
    from core.rag import embeddings
    try:
        return embeddings.active_backend_is_llama()
    except Exception:  # noqa: BLE001 - backend probe must never block saving
        return False


def _resolves_as_local_gguf(model: str) -> bool:
    """True when ``model`` is a local .gguf file or a directory holding one, so
    a save on the llama-server backend needs no HF verification (the artifact
    itself is the proof)."""
    from core.rag.embed_llama_server import LlamaServerBackend
    try:
        return LlamaServerBackend._resolve_local_gguf(model) is not None
    except Exception:  # noqa: BLE001 - dir without .gguf, filesystem oddity
        return False


def _local_gguf_backend_error(model: str) -> str | None:
    """409 detail when ``model`` is a local dir without a .gguf but this install
    embeds via llama-server (macOS/CPU default), which needs one. A
    sentence-transformers-only folder would verify fine yet fail at first index.
    None when not applicable. ``force`` skips this check like HF verification."""
    from pathlib import Path

    if not Path(model).expanduser().is_dir():
        return None
    from core.rag.embed_llama_server import LlamaServerBackend

    if not _llama_backend_active():
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
    """409 detail when the llama-server backend would find no .gguf for an HF
    repo: neither the derived companion repo nor the repo itself has one. Saves
    that verify as embedding models would otherwise fail at first index.
    None when not applicable; ``force`` skips this like HF verification."""
    from pathlib import Path

    if Path(model).expanduser().exists():
        return None  # local paths are handled by the local checks
    if not _llama_backend_active():
        return None
    from core.rag import config as rag_config

    candidates = [model] if rag_config._names_gguf(model) else [f"{model}-GGUF", model]
    try:
        from huggingface_hub import list_repo_files
    except Exception:  # noqa: BLE001 - hub client unavailable: don't block saving
        return None
    for candidate in candidates:
        try:
            files = list_repo_files(candidate, token = hf_token)
        except Exception:  # noqa: BLE001 - missing/gated repo: try next candidate
            continue
        if any(f.lower().endswith(".gguf") and "mmproj" not in f.lower() for f in files):
            return None
    checked = " or ".join(repr(c) for c in candidates)
    return (
        f"No GGUF weights found in {checked}, but this install embeds with the "
        "llama-server backend which requires them. Pick a model with a GGUF "
        "companion repo or GGUF files in the repo itself."
    )


@router.get("/embedding-model", response_model = EmbeddingModelResponse)
def get_embedding_model(
    current_subject: str = Depends(get_current_subject),
) -> EmbeddingModelResponse:
    return _embedding_model_response()


@router.put("/embedding-model", response_model = EmbeddingModelResponse)
def update_embedding_model(
    payload: EmbeddingModelPayload, current_subject: str = Depends(get_current_subject)
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
    hf_token = (payload.hf_token or "").strip() or None
    from utils.utils import hf_env_offline

    # Offline, both the Hub malware scan and the is-embedding check are unreachable and degrade
    # to the local cache below; capture the state once.
    local_only_load = hf_env_offline()
    # The env/default model needs no verification; saving it is a no-op override.
    # A local GGUF on the llama-server backend is accepted as-is: it is exactly
    # what the backend loads, and HF metadata cannot verify a local path.
    is_local_gguf = _llama_backend_active() and _resolves_as_local_gguf(model)
    # The pickle gate only matters for the sentence-transformers backend, which is what
    # deserializes pickles. On the llama-server backend the embedder loads GGUF files
    # (inert) from effective_gguf_repo(), so scanning the ST repo's pickle here would
    # wrongly reject a custom repo whose GGUF companion is clean; the GGUF availability
    # checks below cover that path instead.
    scan_st_pickle = (
        model != default_embedding_model() and not is_local_gguf and not _llama_backend_active()
    )
    if scan_st_pickle:
        # Malware/pickle gate before we persist a repo the embedder later loads with
        # SentenceTransformer. Runs even under force (force only skips the is-embedding
        # type check for offline/local repos HF cannot verify); local paths and
        # unreachable scans fail open inside evaluate_file_security.
        from utils.security import evaluate_file_security, security_load_subdirs
        from core.rag.embeddings import _st_module_subdirs

        # Fall back to the loader's own token so a gated/private repo is actually scanned
        # (a token-less scan fails open for exactly the repo that would still load).
        scan_token = hf_token or _ambient_hf_token()
        # Offline: subdir probes would hit the network and hang; the offline gate walks the
        # whole cached snapshot, so no load-subdir hints are needed.
        if local_only_load:
            load_subdirs = ()
        else:
            # Include ST module dirs (0_Transformer/) so a flagged pickle directly under one
            # blocks instead of passing as an unreferenced nested shard.
            load_subdirs = tuple(
                dict.fromkeys(
                    (
                        *security_load_subdirs(model, scan_token),
                        *_st_module_subdirs(model, scan_token),
                    )
                )
            )
        if evaluate_file_security(
            model,
            hf_token = scan_token,
            load_subdirs = load_subdirs,
            local_only_load = local_only_load,
        ).blocked:
            # 403, not 409: the client routes every 409 into the forceable "save anyway"
            # flow, but this block is a hard, non-forceable security refusal.
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

        # A GGUF-named repo on the llama-server backend is loaded from its .gguf
        # files, which rarely carry sentence-transformers metadata; verify the
        # GGUF is available (below) rather than the ST embedding-metadata gate,
        # which would wrongly 409 a valid online GGUF embedder.
        gguf_named = _llama_backend_active() and rag_config._names_gguf(model)
        if not gguf_named and not is_embedding_model(model, hf_token = hf_token):
            # Offline, is_embedding_model can only confirm the ST layout (modules.json); a
            # transformers-native embedder (e.g. gte-modernbert) is unverifiable without Hub
            # metadata. If already cached and loadable, accept it rather than raising a 409 that
            # online would not (ST can load any cached encoder). Uncached -> 409.
            from utils.utils import hf_cache_snapshot_is_loadable

            # Require a genuinely loadable cache (config + weights), not just a resolved refs/main,
            # so a metadata-only partial cache still gets the forceable 409.
            offline_cached = local_only_load and hf_cache_snapshot_is_loadable(model)
            if not offline_cached:
                raise HTTPException(
                    status_code = 409,
                    detail = (
                        f"Could not verify {model!r} as an embedding model on "
                        "Hugging Face (it may be the wrong model type, gated, or "
                        "you may be offline)."
                    ),
                )
        # The Hub GGUF probe (list_repo_files) can hang offline; skip it. Local check stays.
        gguf_error = _local_gguf_backend_error(model)
        if gguf_error is None and not local_only_load:
            gguf_error = _hf_gguf_backend_error(model, hf_token)
        if gguf_error:
            raise HTTPException(status_code = 409, detail = gguf_error)
    set_rag_embedding_model(model)
    logger.info(
        "settings.embedding_model_updated subject=%s model=%s forced=%s",
        current_subject,
        model,
        payload.force,
    )
    return _embedding_model_response()


@router.delete("/embedding-model", response_model = EmbeddingModelResponse)
def reset_embedding_model(
    current_subject: str = Depends(get_current_subject),
) -> EmbeddingModelResponse:
    """Clear the override, returning to the env/default model."""
    reset_rag_embedding_model()
    logger.info("settings.embedding_model_reset subject=%s", current_subject)
    return _embedding_model_response()


class PreviewLinkRotateResponse(BaseModel):
    rotated: bool = True


@router.post("/preview-links/rotate", response_model = PreviewLinkRotateResponse)
def rotate_preview_links(
    current_subject: str = Depends(get_current_subject),
) -> PreviewLinkRotateResponse:
    """Rotate the preview-link signing secret, revoking every previously shared `/p` link."""
    rotate_preview_link_secret()
    logger.info("settings.preview_links_rotated subject=%s", current_subject)
    return PreviewLinkRotateResponse(rotated = True)


class PreviewSharingPayload(BaseModel):
    enabled: bool


class PreviewSharingResponse(BaseModel):
    enabled: bool
    default_enabled: bool = DEFAULT_PREVIEW_SHARING_ENABLED


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


@router.get("/remote-access", response_model = RemoteAccessResponse)
def get_remote_access(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> RemoteAccessResponse:
    return _remote_access_response(request)


@router.post("/remote-access/start", response_model = RemoteAccessResponse)
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


@router.post("/remote-access/stop", response_model = RemoteAccessResponse)
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


@router.put("/remote-access/auto-start", response_model = RemoteAccessResponse)
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


class LanAccessResponse(BaseModel):
    state: Literal["off", "online", "error"]
    urls: list[str] = []
    public_urls: list[str] = []
    error: Optional[str] = None
    auto_start: bool
    managed_by: Optional[Literal["launch", "settings"]] = None
    can_start: bool
    can_stop: bool
    block_reason: Optional[str] = None
    serves_web_ui: bool = True


def _lan_access_response(request: Request) -> LanAccessResponse:
    return LanAccessResponse(**lan_access_status(request.app))


@router.get("/lan-access", response_model = LanAccessResponse)
def get_lan_access(
    request: Request,
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> LanAccessResponse:
    return _lan_access_response(request)


@router.post("/lan-access/start", response_model = LanAccessResponse)
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


@router.post("/lan-access/stop", response_model = LanAccessResponse)
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


@router.put("/lan-access/auto-start", response_model = LanAccessResponse)
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


@router.get("/preview-sharing", response_model = PreviewSharingResponse)
def get_preview_sharing(
    current_subject: str = Depends(get_current_subject),
) -> PreviewSharingResponse:
    return PreviewSharingResponse(enabled = get_preview_sharing_enabled())


@router.put("/preview-sharing", response_model = PreviewSharingResponse)
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
# ~1.5 MB font file as base64; matches MAX_IMPORTED_FONT_DATA_URL_LENGTH in
# the frontend appearance-custom-store.
MAX_FONT_DATA_URL_LENGTH = 2_200_000
# Aggregate cap across all imported fonts; matches
# MAX_TOTAL_IMPORTED_FONT_DATA_URL_LENGTH in the frontend so a synced payload
# always fits the browser's localStorage quota.
MAX_TOTAL_FONT_DATA_URL_LENGTH = 4_400_000

# Characters that could terminate a CSS declaration, escape the quoted
# font-family value (backslash), or smuggle extra fallbacks/comments (comma,
# slash) if a stored name ever reached a stylesheet. The server is the
# authoritative gate; the frontend strips the same set before use.
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
        # fullmatch, not match: re's ``$`` also matches just before a trailing
        # newline, so ``match`` would accept "data:font/woff2;base64,AAAA\n",
        # which the frontend's JS pattern (``$`` = end of string) rejects.
        if not _FONT_DATA_URL_PATTERN.fullmatch(value):
            raise ValueError("dataUrl must be a base64 font data URL.")
        return value


# Optional user-menu items; the boolean is each id's default visibility.
# Settings-tab shortcuts ship hidden.
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

# Navigable sidebar rows the user can pin/reorder; the boolean is each id's default pin state.
# Order and pin state MUST match the frontend's shipped layout (SIDEBAR_NAV_ITEM_IDS /
# SIDEBAR_NAV_DEFAULT_PINNED in features/settings/stores/appearance-custom-store.ts): the client
# sends every id on each save, so a missing id 422s the whole personalization PUT, and a legacy
# record that predates sidebarNav is served this default as if it were an explicit remote choice.
SIDEBAR_NAV_ITEM_DEFAULTS = {
    "hub": True,
    "projects": True,
    "images": True,
    "video": True,
    "audio": False,
    "train": True,
    "recipes": False,
    "export": False,
    "api": False,
}

MAX_SIDEBAR_NAV_INPUT_ITEMS = 4 * len(SIDEBAR_NAV_ITEM_DEFAULTS)

# The sidebarMenu validator below dedupes ids and re-fills any missing ones, so
# the stored list is always exactly one entry per id. Cap the *incoming* list at
# a generous multiple rather than len(defaults): a stale or duplicated payload
# (more items than distinct ids) must reach the validator so it can normalize,
# instead of being rejected by the length constraint before dedupe runs. A
# pathologically long list is still refused.
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


class PersonalizationSidebarNavItem(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    id: Literal[
        "hub",
        "projects",
        "images",
        "video",
        "audio",
        "train",
        "recipes",
        "export",
        "api",
    ]
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
    palette: Literal["standard", "classic", "minimal"] = "standard"
    language: Optional[str] = Field(None, max_length = 20)
    customization: PersonalizationCustomization = Field(
        default_factory = PersonalizationCustomization
    )


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
    paletteSaved: bool = False
    greetingSlothSaved: bool = False


@router.get("/personalization", response_model = PersonalizationResponse)
def get_personalization_settings(
    current_subject: str = Depends(get_current_subject),
) -> PersonalizationResponse:
    stored = get_personalization()
    response = PersonalizationResponse.model_validate(stored or {})
    response.saved = bool(stored)
    appearance = stored.get("appearance") if isinstance(stored, dict) else None
    profile = stored.get("profile") if isinstance(stored, dict) else None
    response.customizationSaved = isinstance(appearance, dict) and "customization" in appearance
    response.paletteSaved = isinstance(appearance, dict) and "palette" in appearance
    response.greetingSlothSaved = isinstance(profile, dict) and "showGreetingSloth" in profile
    return response


def _merge_personalization(base: dict, overlay: dict) -> dict:
    # Recursively overlay only the request's set fields onto the stored record,
    # so a stale client that omits newer keys (palette, customization) does not
    # materialize their defaults and defeat the *Saved legacy detection.
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            merged[key] = _merge_personalization(existing, value)
        else:
            merged[key] = value
    return merged


@router.put("/personalization", response_model = PersonalizationPayload)
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
    # matches storage (and the next GET) for fields the client omitted.
    return PersonalizationPayload.model_validate(merged)


# ── Logs: read the log files from inside the app ─────────────────────────────
# Backs the Settings > Logs tab. The session log always existed, but its
# path was only printed to a console the desktop user never sees.


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
    file_logging_disabled: bool = False


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
    # The reader stopped at the response cap and the rest arrives on the next
    # poll. Without this the caller cannot tell a complete answer from a partial
    # one, which is invisible in manual mode because no next poll is coming.
    more_pending: bool = False
    # File logging is off, so anything readable here is a PREVIOUS session and
    # will never grow. The status stays "ok" because the content is real and
    # worth reading; saying nothing made a stale log look live.
    file_logging_disabled: bool = False
    size_bytes: int = 0


@router.get("/debug/logs/sources", response_model = DebugLogSourcesResponse)
def get_debug_log_sources(
    current_subject: str = Depends(get_current_subject),
    _ui_session: None = Depends(_require_ui_session),
) -> DebugLogSourcesResponse:
    """Every log file the viewer may read, newest first within each family.

    Individual files, not one entry per family: the llama runner writes one file
    per load ATTEMPT, so after a retry the useful one is often not the newest.
    """
    from utils import debug_log_sources

    sources = debug_log_sources.list_sources()
    return DebugLogSourcesResponse(
        sources = [DebugLogSourceModel(**vars(source)) for source in sources],
        default_source_id = debug_log_sources.default_source_id(),
        file_logging_disabled = debug_log_sources.file_logging_disabled(),
    )


@router.get("/debug/logs", response_model = DebugLogResponse)
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
