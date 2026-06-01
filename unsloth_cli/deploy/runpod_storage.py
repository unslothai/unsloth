# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RunPod storage: network volumes (REST) and uploading a local model onto one
over the S3-compatible API.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from unsloth_cli.deploy import DeployError

if TYPE_CHECKING:
    from unsloth_cli.deploy.runpod_client import RunPod


REST_BASE = "https://rest.runpod.io/v1"

def create_network_volume(
    client: "RunPod", *, name: str, size_gb: int, datacenter_id: str,
) -> str:
    body = _rest(
        client, "POST", "/networkvolumes",
        {"name": name, "size": size_gb, "dataCenterId": datacenter_id},
    )
    vol_id = body.get("id")
    if not vol_id:
        raise DeployError(f"RunPod network volume create returned no id: {body!r}")
    return vol_id


def delete_network_volume(client: "RunPod", volume_id: str) -> None:
    _rest(client, "DELETE", f"/networkvolumes/{volume_id}", None)


def _rest(client: "RunPod", method: str, path: str, body: Optional[dict]) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        REST_BASE + path,
        data = data,
        method = method,
        headers = {
            "Authorization": f"Bearer {client.api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout = 60) as resp:
            text = resp.read().decode(errors = "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors = "replace")
        raise DeployError(
            f"RunPod REST {method} {path} -> {e.code}: {detail[:400]}"
        ) from e
    except (urllib.error.URLError, OSError) as e:
        raise DeployError(f"RunPod REST {method} {path} failed: {e}") from e
    if not text:
        return {}
    try:
        return json.loads(text)
    except ValueError as e:
        raise DeployError(
            f"RunPod REST {method} {path} returned a non-JSON response: {text[:200]}"
        ) from e


def endpoint_for(datacenter: str) -> str:
    """RunPod's S3 endpoint is per-datacenter, e.g. s3api-eur-is-1.runpod.io."""
    return f"https://s3api-{datacenter.lower()}.runpod.io"


def _iter_files(local_path: Path):
    if local_path.is_dir():
        for p in sorted(local_path.rglob("*")):
            if p.is_file():
                yield p, p.relative_to(local_path).as_posix()
    else:
        yield local_path, local_path.name


def upload_path(
    local_path: Path,
    *,
    volume_id: str,
    datacenter: str,
    access_key: str,
    secret_key: str,
    prefix: str,
    on_file: Optional[Callable[[str, int], None]] = None,
) -> None:
    """Upload `local_path` (file or directory) into the volume under `prefix`."""
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as e:
        raise DeployError(
            "boto3 is required to upload a local model to RunPod storage.\n"
            "Install it with:\n"
            "    pip install unsloth[deploy]"
        ) from e

    s3 = boto3.client(
        "s3",
        aws_access_key_id = access_key,
        aws_secret_access_key = secret_key,
        region_name = datacenter,
        endpoint_url = endpoint_for(datacenter),
        config = Config(signature_version = "s3v4"),
    )

    files = list(_iter_files(local_path))
    if not files:
        raise DeployError(f"Nothing to upload: {local_path} has no files.")

    for path, rel in files:
        key = f"{prefix}/{rel}"
        if on_file is not None:
            on_file(key, path.stat().st_size)
        try:
            s3.upload_file(str(path), volume_id, key)
        except (BotoCoreError, ClientError) as e:
            raise DeployError(f"S3 upload failed for {path} -> {key}: {e}") from e
