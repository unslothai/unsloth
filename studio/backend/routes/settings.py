# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import re
from typing import Literal, Optional
from urllib.parse import unquote, urlsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from auth.authentication import get_current_subject
from auth.storage import rotate_preview_link_secret
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
from utils.helper_precache_settings import (
    DEFAULT_HELPER_PRECACHE_ENABLED,
    get_helper_precache_enabled,
    helper_model_disabled_by_env,
    set_helper_precache_enabled,
)
from utils.coding_agents import CODING_AGENTS, detect_installed_coding_agents
from utils.openai_auto_switch_settings import (
    DEFAULT_AUTO_UNLOAD_IDLE_SECONDS,
    DEFAULT_OPENAI_AUTO_SWITCH_ENABLED,
    get_auto_unload_idle_seconds,
    get_model_overrides,
    get_openai_auto_switch_enabled,
    get_stored_auto_unload_idle_seconds,
    set_model_override,
    set_openai_auto_switch,
)
from utils.preview_sharing_settings import (
    DEFAULT_PREVIEW_SHARING_ENABLED,
    get_preview_sharing_enabled,
    set_preview_sharing_enabled,
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

router = APIRouter()

logger = get_logger(__name__)


class UploadLimitPayload(BaseModel):
    max_upload_size_mb: int = Field(..., ge = MIN_UPLOAD_LIMIT_MB, le = MAX_UPLOAD_LIMIT_MB)


class UploadLimitResponse(BaseModel):
    max_upload_size_mb: int
    max_upload_size_bytes: int
    max_upload_size_label: str
    default_upload_size_mb: int
    min_upload_size_mb: int = MIN_UPLOAD_LIMIT_MB
    max_allowed_upload_size_mb: int = MAX_UPLOAD_LIMIT_MB


class HelperPrecachePayload(BaseModel):
    enabled: bool


class HelperPrecacheResponse(BaseModel):
    enabled: bool
    default_enabled: bool = DEFAULT_HELPER_PRECACHE_ENABLED
    disabled_by_env: bool


class OpenAIAutoSwitchPayload(BaseModel):
    enabled: bool
    auto_unload_idle_seconds: int = Field(default = DEFAULT_AUTO_UNLOAD_IDLE_SECONDS, ge = 0)


class OpenAIAutoSwitchResponse(BaseModel):
    enabled: bool
    auto_unload_idle_seconds: int
    default_enabled: bool = DEFAULT_OPENAI_AUTO_SWITCH_ENABLED
    # True when the idle-unload loop will actually unload (effective TTL > 0). With
    # UNSLOTH_MODEL_IDLE_TTL set and nothing stored, this is true even while enabled
    # is false, so the UI can show idle-unload as active instead of "needs enable".
    idle_unload_active: bool = False


class ModelOverridePayload(BaseModel):
    model_id: str = Field(..., min_length = 1)
    llama_extra_args: list[str] = Field(default_factory = list)
    # ge=1: 0 is not a valid sequence length, and the setter drops a falsy value,
    # so reject it at the boundary instead of accepting then silently discarding it.
    max_seq_length: Optional[int] = Field(default = None, ge = 1, le = 1048576)


class ModelOverridesResponse(BaseModel):
    overrides: dict[str, dict]


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
    )


@router.put("/openai-auto-switch", response_model = OpenAIAutoSwitchResponse)
def update_openai_auto_switch(
    payload: OpenAIAutoSwitchPayload, current_subject: str = Depends(get_current_subject)
) -> OpenAIAutoSwitchResponse:
    try:
        enabled, idle_seconds = set_openai_auto_switch(
            payload.enabled, payload.auto_unload_idle_seconds
        )
    except ValueError as exc:
        raise log_and_http_error(
            exc,
            400,
            safe_error_detail(exc, fallback = "Invalid OpenAI auto-switch setting."),
            event = "settings.update_openai_auto_switch_failed",
            log = logger,
        ) from exc
    return OpenAIAutoSwitchResponse(
        enabled = enabled,
        auto_unload_idle_seconds = idle_seconds,
        idle_unload_active = get_auto_unload_idle_seconds() > 0,
    )


@router.get("/openai-auto-switch/overrides", response_model = ModelOverridesResponse)
def get_openai_auto_switch_overrides(
    current_subject: str = Depends(get_current_subject),
) -> ModelOverridesResponse:
    return ModelOverridesResponse(overrides = get_model_overrides())


@router.put("/openai-auto-switch/overrides", response_model = ModelOverridesResponse)
def update_openai_auto_switch_override(
    payload: ModelOverridePayload, current_subject: str = Depends(get_current_subject)
) -> ModelOverridesResponse:
    from core.inference.llama_server_args import validate_extra_args
    try:
        extra_args = validate_extra_args(payload.llama_extra_args)
        set_model_override(
            payload.model_id,
            llama_extra_args = extra_args,
            max_seq_length = payload.max_seq_length,
        )
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
    default_embedding_model: str
    is_custom: bool


def _embedding_model_response() -> EmbeddingModelResponse:
    return EmbeddingModelResponse(
        embedding_model = get_rag_embedding_model(),
        default_embedding_model = default_embedding_model(),
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
        # Include the ST module dirs (0_Transformer/) so a flagged pickle directly under
        # one blocks instead of passing as an unreferenced nested shard.
        load_subdirs = tuple(
            dict.fromkeys(
                (
                    *security_load_subdirs(model, scan_token),
                    *_st_module_subdirs(model, scan_token),
                )
            )
        )
        if evaluate_file_security(model, hf_token = scan_token, load_subdirs = load_subdirs).blocked:
            # 403, not 409: the client routes every 409 into the forceable "save anyway"
            # flow, but this block is a hard, non-forceable security refusal.
            raise HTTPException(
                status_code = 403,
                detail = (
                    f"{model!r} is flagged as unsafe by Hugging Face's security scan and "
                    "cannot be used as the embedding model."
                ),
            )
    if model != default_embedding_model() and not payload.force and not is_local_gguf:
        from core.rag import config as rag_config

        # A GGUF-named repo on the llama-server backend is loaded from its .gguf
        # files, which rarely carry sentence-transformers metadata; verify the
        # GGUF is available (below) rather than the ST embedding-metadata gate,
        # which would wrongly 409 a valid online GGUF embedder.
        gguf_named = _llama_backend_active() and rag_config._names_gguf(model)
        if not gguf_named and not is_embedding_model(model, hf_token = hf_token):
            raise HTTPException(
                status_code = 409,
                detail = (
                    f"Could not verify {model!r} as an embedding model on "
                    "Hugging Face (it may be the wrong model type, gated, or "
                    "you may be offline)."
                ),
            )
        gguf_error = _local_gguf_backend_error(model) or _hf_gguf_backend_error(model, hf_token)
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
