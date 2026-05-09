# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Runtime probe: is the currently loaded model vision-capable, and where
is its OpenAI-compatible endpoint?

Unifies the three Studio inference backends (embedded llama-server for
GGUF, transformers, Unsloth/LoRA) behind a single ``VlmCapability``
dataclass. Read-only — never loads or modifies models.

Why this replaces the old ``VISION_ARCHITECTURES`` allow-list:
- Allow-lists silently exclude legitimately new vision architectures.
- Runtime probing matches the user's actual loaded model.
- The document extractor can caption selected visual references through
  any loaded backend exposing ``/v1/chat/completions`` without
  hard-coding architecture names.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


VlmSource = Literal["gguf", "transformers", "unsloth", "none"]


@dataclass(frozen = True)
class VlmCapability:
    """Immutable snapshot of the loaded model's image-input capability."""

    is_vlm: bool
    endpoint_url: Optional[str]
    model_name: Optional[str]
    source: VlmSource
    reason: Optional[str] = None

    @classmethod
    def none(cls, reason: str = "no model loaded") -> "VlmCapability":
        return cls(
            is_vlm = False,
            endpoint_url = None,
            model_name = None,
            source = "none",
            reason = reason,
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _probe_gguf(llama: Any = None) -> Optional[VlmCapability]:
    if llama is None:
        try:
            from core.inference.llama_cpp import get_llama_cpp_backend
        except Exception:  # pragma: no cover - older embedding paths
            return None

        try:
            llama = get_llama_cpp_backend()
        except Exception:
            return None

    if not getattr(llama, "is_loaded", False):
        return None

    base_url = getattr(llama, "base_url", None)
    model_id = getattr(llama, "model_identifier", None)
    is_vision = bool(getattr(llama, "is_vision", False))

    if not base_url or not model_id:
        # Half-initialised llama-server state — fall through to the
        # transformers probe instead of returning a misleading
        # non-vision GGUF result that suppresses the fallback chain.
        logger.debug(
            "llama-server reports is_loaded=True but base_url / model id missing"
        )
        return None

    return VlmCapability(
        is_vlm = is_vision,
        endpoint_url = base_url,
        model_name = model_id,
        source = "gguf",
        reason = None if is_vision else "gguf: model loaded, is_vision=False (no mmproj clip)",
    )


def _probe_transformers(self_base_url: Optional[str]) -> Optional[VlmCapability]:
    try:
        from core.inference import get_inference_backend
    except ModuleNotFoundError as exc:
        if exc.name == "core.inference" or (
            exc.name and exc.name.startswith("core.inference.")
        ):
            return None
        logger.exception("Failed to import transformers inference backend")
        return None
    except ImportError:
        # A different ImportError variant (e.g. circular import). Treat as
        # backend-unavailable. Anything else (NameError/AttributeError raised
        # by core.inference.__init__) propagates so real bugs aren't masked
        # as "no VLM loaded".
        logger.exception("Failed to import transformers inference backend")
        return None

    try:
        ib = get_inference_backend()
    except Exception:
        return None

    name: Optional[str] = getattr(ib, "active_model_name", None)
    if not name:
        return None

    models: dict = getattr(ib, "models", {}) or {}
    info: dict = models.get(name) or {}
    is_vision = bool(info.get("is_vision", False))
    is_lora = bool(info.get("is_lora", False))
    source: VlmSource = "unsloth" if is_lora else "transformers"

    if not self_base_url:
        return VlmCapability(
            is_vlm = False,
            endpoint_url = None,
            model_name = name,
            source = source,
            reason = f"{source}: self_base_url=None (cannot self-loopback to /v1/chat/completions)",
        )

    return VlmCapability(
        is_vlm = is_vision,
        endpoint_url = self_base_url.rstrip("/"),
        model_name = name,
        source = source,
        reason = None if is_vision else f"{source}: active model not marked is_vision",
    )


def detect_loaded_vlm(
    self_base_url: Optional[str] = None,
    *,
    llama_backend: Any = None,
) -> VlmCapability:
    """Identify the active model and whether it can describe images.

    ``self_base_url`` is only consulted when the active model is served
    by the transformers / Unsloth backend; document image captioning must
    loop back through our own ``/v1/chat/completions``. GGUF models return
    llama-server's own URL and ignore this argument.
    """
    gguf = _probe_gguf(llama_backend)
    if gguf is not None:
        return gguf

    tf = _probe_transformers(self_base_url)
    if tf is not None:
        return tf

    return VlmCapability.none()


def extract_self_base_url(request: Any) -> Optional[str]:
    """Derive a trusted local base URL for the active Studio server.

    The request Host header is attacker-controlled in many deployments,
    so the returned origin always uses ``127.0.0.1``. Only the server
    port is discovered, preferring the port published by ``run.py`` and
    then uvicorn's ASGI scope. ``request.base_url`` is a last-resort
    fallback for tests and non-uvicorn embedding.
    """
    port: Optional[int] = None

    try:
        candidate = getattr(getattr(request, "app", None), "state", None)
        candidate = getattr(candidate, "server_port", None)
        if isinstance(candidate, int) and candidate > 0:
            port = candidate
    except Exception:
        port = None

    if port is None:
        try:
            server = getattr(request, "scope", {}).get("server")
            if (
                isinstance(server, tuple)
                and len(server) >= 2
                and isinstance(server[1], int)
                and server[1] > 0
            ):
                port = server[1]
        except Exception:
            port = None

    if port is None:
        try:
            base = str(getattr(request, "base_url", "") or "")
            if not base:
                return None
            parsed = urlparse(base)
            port = parsed.port if parsed.port is not None else 8888
        except Exception:
            return None

    return f"http://127.0.0.1:{int(port)}"
