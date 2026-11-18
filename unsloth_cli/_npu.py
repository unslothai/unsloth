# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AMD Ryzen AI NPU models from the CLI, which drives a running Studio over HTTP."""

from __future__ import annotations

import difflib
import json
import os
import re
from typing import Callable, Optional
from urllib.parse import quote

NPU_MODEL_PREFIX = "lemonade:"
# FastFlowLM publishes its NPU builds as FastFlowLM/<name>-NPU2, the name Unsloth Studio shows.
_NPU_REPO_OWNER = "fastflowlm/"
_NPU_NAME_SUFFIX = re.compile(r"-npu\d*$", re.IGNORECASE)


class NpuCliError(RuntimeError):
    pass


def npu_model_path(model: Optional[str]) -> Optional[str]:
    """`lemonade:<name>` for an NPU model path, a bare Lemonade id (`-FLM`), the name Unsloth Studio
    shows (`-NPU2`) or its `FastFlowLM/` repo; `prepare_npu_model` resolves the name to an id."""
    if not model:
        return None
    text = model.strip()
    if text.startswith(NPU_MODEL_PREFIX):
        return text if len(text) > len(NPU_MODEL_PREFIX) else None
    if "\\" in text or ":" in text or os.path.exists(text):
        return None
    if text.lower().startswith(_NPU_REPO_OWNER):
        name = text[len(_NPU_REPO_OWNER) :]
        return NPU_MODEL_PREFIX + text if name and "/" not in name else None
    if "/" in text or not (text.lower().endswith("-flm") or _NPU_NAME_SUFFIX.search(text)):
        return None
    return NPU_MODEL_PREFIX + text


def _resolve_npu_model(models: list, name: str) -> dict:
    """The catalog row whose id, Hugging Face repo or repo name (as Unsloth Studio shows it) is name."""
    rows = [m for m in models if isinstance(m, dict) and m.get("id")]
    wanted = name.lower()
    names: dict[str, str] = {}
    matches = []
    for row in rows:
        repo = str(row.get("hf_repo") or "").strip()
        aliases = [alias for alias in (str(row["id"]), repo, repo.rsplit("/", 1)[-1]) if alias]
        for alias in aliases:
            names.setdefault(alias.lower(), alias)
        if any(alias.lower() == wanted for alias in aliases):
            matches.append(row)
    if len(matches) == 1:
        return matches[0]
    if matches:
        ids = ", ".join(sorted(str(m["id"]) for m in matches))
        raise NpuCliError(f"{name} names more than one NPU model: {ids}. Pick one by its id.")
    close = difflib.get_close_matches(wanted, list(names), n = 3, cutoff = 0.6)
    if close:
        raise NpuCliError(
            f"No NPU model named {name}. Did you mean: {', '.join(names[c] for c in close)}?"
        )
    available = sorted(str(m["id"]) for m in rows)
    listed = ", ".join(available[:12]) + (", ..." if len(available) > 12 else "")
    raise NpuCliError(f"No NPU model named {name}. Available: {listed}")


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
) -> str:
    """Enable the NPU runtime and download model_path on the Studio behind request(method, path,
    payload, timeout), which returns an open HTTP response; returns the `lemonade:<id>` to load."""
    import http.client
    try:
        return _prepare_npu_model(request, model_path, echo)
    except (OSError, http.client.HTTPException) as exc:
        # A refused or dropped connection, or a timeout: Studio stopped or restarted mid-way.
        raise NpuCliError(f"Lost the connection to Unsloth Studio: {exc}") from None


def _prepare_npu_model(
    request: Callable[..., object], model_path: str, echo: Callable[[str], None]
) -> str:
    model_id = model_path[len(NPU_MODEL_PREFIX) :]
    status = _read_json(_call(request, "GET", "/api/npu/status", 60))
    if not status.get("supported"):
        hardware = status.get("hardware") or {}
        if not hardware.get("present"):
            reason = "no AMD Ryzen AI NPU was found on this machine"
        elif hardware.get("family") == "XDNA2" and hardware.get("enabled") is False:
            reason = (
                "the NPU is turned off in this Unsloth Studio. Restart it with "
                "UNSLOTH_STUDIO_ENABLE_NPU=1 to use the NPU"
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
    entry = _resolve_npu_model(models, model_id)
    model_id = str(entry["id"])
    if entry.get("downloaded"):
        return NPU_MODEL_PREFIX + model_id

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
    return NPU_MODEL_PREFIX + model_id
