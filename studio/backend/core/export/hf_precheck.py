# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fast Hugging Face Hub preflight checks before long export/upload runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)


@dataclass(frozen = True)
class HubPrecheckResult:
    ok: bool
    message: str
    details: Optional[Dict[str, Any]] = None


def _normalize_repo_id(repo_id: str) -> str:
    return repo_id.strip()


def _resolve_writable_namespaces(whoami: dict) -> set[str]:
    namespaces = {whoami.get("name", "")}
    for org in whoami.get("orgs") or []:
        name = org.get("name") if isinstance(org, dict) else None
        if name:
            namespaces.add(name)
    return {name for name in namespaces if name}


def _verify_hub_credentials(
    hf_token: Optional[str] = None,
) -> Tuple[Optional[dict], Optional[HubPrecheckResult]]:
    """Return whoami payload or a failure result."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:
        return None, HubPrecheckResult(
            False,
            f"Hugging Face Hub client is unavailable: {exc}",
        )

    api = HfApi(token = hf_token) if hf_token else HfApi()

    try:
        whoami = api.whoami()
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return None, HubPrecheckResult(
                False,
                "Invalid or expired Hugging Face token. "
                "Check your token or run `huggingface-cli login`.",
            )
        logger.warning("Hub precheck whoami failed: %s", exc)
        return None, HubPrecheckResult(
            False,
            f"Could not verify Hugging Face credentials: {exc}",
        )
    except Exception as exc:
        logger.warning("Hub precheck whoami failed: %s", exc)
        return None, HubPrecheckResult(
            False,
            f"Could not verify Hugging Face credentials: {exc}",
        )

    return whoami, None


def precheck_hub_credentials(
    hf_token: Optional[str] = None,
) -> HubPrecheckResult:
    """Validate Hub credentials and return the authenticated username."""
    whoami, failure = _verify_hub_credentials(hf_token)
    if failure is not None:
        return failure

    username = whoami.get("name") or ""
    writable = _resolve_writable_namespaces(whoami)
    return HubPrecheckResult(
        True,
        f"Logged in to Hugging Face as '{username}'.",
        details = {
            "username": username,
            "writable_namespaces": sorted(writable),
        },
    )


def precheck_hub_upload(
    repo_id: str,
    hf_token: Optional[str] = None,
    private: bool = False,
) -> HubPrecheckResult:
    """Validate token, repo id, and namespace write access before export.

    Returns quickly (network round-trips only) so users do not wait through a
    full model conversion only to fail at upload time.
    """
    repo_id = _normalize_repo_id(repo_id)
    if not repo_id:
        return HubPrecheckResult(False, "Repository ID is required for Hub upload")

    from hub.utils.paths import is_valid_repo_id

    if not is_valid_repo_id(repo_id):
        return HubPrecheckResult(
            False,
            "Invalid repository ID. Use the form username/model-name "
            "(letters, numbers, dots, dashes, underscores only).",
        )

    if "/" not in repo_id:
        return HubPrecheckResult(
            False,
            "Repository ID must include a namespace, e.g. your-username/my-model.",
        )

    namespace, model_name = repo_id.split("/", 1)
    if not model_name:
        return HubPrecheckResult(False, "Repository ID must include a model name.")

    whoami, failure = _verify_hub_credentials(hf_token)
    if failure is not None:
        return failure

    username = whoami.get("name") or ""
    writable = _resolve_writable_namespaces(whoami)
    if namespace not in writable:
        return HubPrecheckResult(
            False,
            f"You do not have write access to Hugging Face namespace '{namespace}'. "
            f"Your account can publish to: {', '.join(sorted(writable)) or username}.",
            details = {
                "username": username,
                "writable_namespaces": sorted(writable),
            },
        )

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    except ImportError as exc:
        return HubPrecheckResult(
            False,
            f"Hugging Face Hub client is unavailable: {exc}",
        )

    api = HfApi(token = hf_token) if hf_token else HfApi()

    repo_exists = False
    try:
        api.repo_info(repo_id, repo_type = "model", token = hf_token)
        repo_exists = True
    except RepositoryNotFoundError:
        repo_exists = False
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403):
            return HubPrecheckResult(
                False,
                f"No read/write access to repository '{repo_id}'. "
                "Check the namespace and your token permissions.",
                details = {"username": username},
            )
        logger.warning("Hub precheck repo_info failed for %s: %s", repo_id, exc)
        return HubPrecheckResult(
            False,
            f"Could not verify repository '{repo_id}': {exc}",
            details = {"username": username},
        )
    except Exception as exc:
        logger.warning("Hub precheck repo_info failed for %s: %s", repo_id, exc)
        return HubPrecheckResult(
            False,
            f"Could not verify repository '{repo_id}': {exc}",
            details = {"username": username},
        )

    # Read-only preflight: do not create or mutate the Hub repo here. Export/upload
    # creates the repository when the artifacts are actually ready to publish.

    if repo_exists:
        message = f"Ready to upload to existing repository '{repo_id}'."
    elif private:
        message = f"Ready to create private repository '{repo_id}'."
    else:
        message = f"Ready to create public repository '{repo_id}'."

    return HubPrecheckResult(
        True,
        message,
        details = {
            "username": username,
            "repo_id": repo_id,
            "repo_exists": repo_exists,
            "private": private,
        },
    )


def precheck_hub_upload_tuple(
    repo_id: str,
    hf_token: Optional[str] = None,
    private: bool = False,
) -> Tuple[bool, str]:
    """Tuple wrapper used by export backends."""
    result = precheck_hub_upload(repo_id, hf_token = hf_token, private = private)
    return result.ok, result.message
