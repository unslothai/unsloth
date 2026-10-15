# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD Ryzen AI NPU models from the CLI, which drives a running Studio over HTTP."""

from __future__ import annotations

import json
import os
from typing import Callable, Optional
from urllib.parse import quote

NPU_MODEL_PREFIX = "lemonade:"


class NpuCliError(RuntimeError):
    pass


def npu_model_path(model: Optional[str]) -> Optional[str]:
    """`lemonade:<id>` for an NPU model path or bare id: Lemonade's NPU ids end in `-FLM`, no slash."""
    if not model:
        return None
    text = model.strip()
    if text.startswith(NPU_MODEL_PREFIX):
        return text if len(text) > len(NPU_MODEL_PREFIX) else None
    if "/" in text or "\\" in text or ":" in text or not text.lower().endswith("-flm"):
        return None
    if os.path.exists(text):
        return None
    return NPU_MODEL_PREFIX + text


def npu_needs_studio_message(npu_path: str, *, no_server: bool) -> str:
    """Why `unsloth chat` / `unsloth inference` cannot run `npu_path` without a running Studio."""
    if no_server:
        return "NPU models run inside Unsloth Studio; drop --no-server."
    return (
        f"{npu_path[len(NPU_MODEL_PREFIX) :]} runs on the AMD NPU inside Unsloth Studio. Start it "
        f"with `unsloth studio` (or `unsloth run --model {npu_path}`) and run this again."
    )


def _call(request: Callable[..., object], method: str, path: str, timeout: float):
    """Open `path`, turning an HTTP error into the server's own message."""
    import urllib.error
    try:
        return request(method, path, None, timeout)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read() or b"{}").get("detail")
        except Exception:
            detail = None
        raise NpuCliError(f"{method} {path} failed ({exc.code}): {detail or exc.reason}") from None


def _read_json(response) -> dict:
    with response:
        body = json.loads(response.read() or b"{}")
    return body if isinstance(body, dict) else {}


def prepare_npu_model(
    request: Callable[..., object], model_path: str, echo: Callable[[str], None]
) -> None:
    """Enable the NPU runtime and download `model_path` on the Studio behind `request`.

    `request(method, path, payload, timeout)` returns an open HTTP response for that Studio.
    """
    import http.client
    try:
        _prepare_npu_model(request, model_path, echo)
    except (OSError, http.client.HTTPException) as exc:
        # A refused or dropped connection, or a timeout: Studio stopped or restarted mid-way.
        raise NpuCliError(f"Lost the connection to Unsloth Studio: {exc}") from None


def _prepare_npu_model(
    request: Callable[..., object], model_path: str, echo: Callable[[str], None]
) -> None:
    model_id = model_path[len(NPU_MODEL_PREFIX) :]
    status = _read_json(_call(request, "GET", "/api/npu/status", 60))
    if not status.get("supported"):
        hardware = status.get("hardware") or {}
        if not hardware.get("present"):
            reason = "no AMD Ryzen AI NPU was found on this machine"
        elif hardware.get("family") == "XDNA2" and hardware.get("enabled") is False:
            reason = (
                "the NPU is off by default. Start Unsloth Studio with "
                "UNSLOTH_STUDIO_ENABLE_NPU=1 to use it"
            )
        else:
            reason = (
                f"{hardware.get('name') or 'this NPU'} is not supported (an XDNA2 NPU is needed)"
            )
        raise NpuCliError(f"{model_id} runs on the AMD NPU, but {reason}.")
    if not status.get("ready"):
        echo("Enabling the NPU runtime (Lemonade and FastFlowLM, about 47 MB)...")
        status = _read_json(_call(request, "POST", "/api/npu/enable", 1800))
        if not status.get("ready"):
            problems = (status.get("validation") or {}).get("problems") or []
            detail = status.get("error") or " ".join(problems) or "the NPU driver check failed"
            raise NpuCliError(f"Could not enable the NPU runtime: {detail}")

    models = _read_json(_call(request, "GET", "/api/npu/models", 120)).get("models") or []
    entry = next((m for m in models if isinstance(m, dict) and m.get("id") == model_id), None)
    if entry is None:
        available = sorted(str(m.get("id")) for m in models if isinstance(m, dict))
        listed = ", ".join(available[:12]) + (", ..." if len(available) > 12 else "")
        raise NpuCliError(f"No NPU model named {model_id}. Available: {listed}")
    if entry.get("downloaded"):
        return

    size = entry.get("size_gb")
    echo(
        f"Downloading {model_id}"
        + (f" ({size:.1f} GB)" if isinstance(size, (int, float)) else "")
        + "..."
    )
    last_tenth = -1
    finished = False
    response = _call(
        request, "POST", f"/api/npu/models/{quote(model_id, safe = '')}/download", 7200
    )
    with response:
        for raw in response:
            line = (
                raw.decode("utf-8", "replace").strip()
                if isinstance(raw, bytes)
                else str(raw).strip()
            )
            if not line.startswith("data:"):
                continue
            try:
                event = json.loads(line[len("data:") :])
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("event") == "error" or event.get("error"):
                raise NpuCliError(
                    f"Downloading {model_id} failed: {event.get('error') or 'unknown error'}"
                )
            percent = event.get("percent")
            if isinstance(percent, (int, float)) and int(percent) // 10 > last_tenth:
                last_tenth = int(percent) // 10
                echo(f"  {int(percent)}%")
            if event.get("event") == "complete":
                finished = True
    if not finished:
        raise NpuCliError(
            f"The download of {model_id} ended early. Run the command again to resume it."
        )
