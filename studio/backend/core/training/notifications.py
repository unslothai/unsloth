# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Outbound webhook notifications for terminal training events."""

from __future__ import annotations

import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

from loggers import get_logger

logger = get_logger(__name__)

_WEBHOOK_TIMEOUT_SEC = 5.0
_WEBHOOK_ATTEMPTS = 2
_EXECUTOR = ThreadPoolExecutor(max_workers = 2, thread_name_prefix = "training-notify")

# Local filesystem paths -> basename; HF repo ids like "unsloth/llama-3" stay.
_LOCAL_PATH_RE = re.compile(r"^(/|~[\\/]|\.{1,2}[\\/]|[A-Za-z]:[\\/]|\\\\)")
_SLACK_HOST_RE = re.compile(r"(^|\.)slack\.com$", re.IGNORECASE)
_DISCORD_HOST_RE = re.compile(r"(^|\.)(discord\.com|discordapp\.com)$", re.IGNORECASE)


@dataclass(frozen = True)
class TrainingTerminalEvent:
    job_id: str
    status: str  # "completed" | "error" | "test"
    model: str
    total_steps: int | None = None
    final_loss: float | None = None
    duration_s: float | None = None
    error: str | None = None


def _safe_model_label(model: str | None) -> str:
    text = (model or "").strip()
    if not text:
        return "your model"
    if _LOCAL_PATH_RE.match(text):
        parts = [p for p in re.split(r"[\\/]", text) if p]
        if parts:
            return parts[-1]
    return text


def _format_duration(seconds: float | None) -> str | None:
    if not seconds or seconds < 0:
        return None
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _summary(event: TrainingTerminalEvent) -> str:
    if event.status == "test":
        return (
            "✅ Unsloth notifications are connected. You'll get a message here "
            "when a training run finishes or fails."
        )
    model = _safe_model_label(event.model)
    head = (
        f"Training complete: {model}"
        if event.status == "completed"
        else f"Training failed: {model}"
    )
    detail: list[str] = []
    if event.total_steps:
        detail.append(f"{event.total_steps} steps")
    if event.final_loss is not None:
        detail.append(f"loss {event.final_loss:.4f}")
    duration = _format_duration(event.duration_s)
    if duration:
        detail.append(duration)

    text = head
    if detail:
        text += "\n" + " · ".join(detail)
    if event.status == "error" and event.error:
        text += f"\n{event.error}"
    return text


def _format_generic(event: TrainingTerminalEvent) -> dict[str, Any]:
    payload = asdict(event)
    payload["model"] = _safe_model_label(event.model)
    payload["message"] = _summary(event)
    return payload


def _format_slack(event: TrainingTerminalEvent) -> dict[str, Any]:
    return {"text": _summary(event)}


def _format_discord(event: TrainingTerminalEvent) -> dict[str, Any]:
    # Discord rejects content longer than 2000 chars; long error text would 400.
    content = _summary(event)
    if len(content) > 1900:
        content = content[:1897] + "..."
    return {"content": content}


FORMATTERS: dict[str, Callable[[TrainingTerminalEvent], dict[str, Any]]] = {
    "generic": _format_generic,
    "slack": _format_slack,
    "discord": _format_discord,
}


def detect_format(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if _SLACK_HOST_RE.search(host):
        return "slack"
    if _DISCORD_HOST_RE.search(host):
        return "discord"
    return "generic"


class WebhookSink:
    def __init__(self, url: str) -> None:
        self.url = url
        self.fmt = detect_format(url)

    def deliver(self, event: TrainingTerminalEvent) -> None:
        body = FORMATTERS[self.fmt](event)
        last_exc: Exception | None = None
        for _ in range(_WEBHOOK_ATTEMPTS):
            try:
                response = httpx.post(self.url, json = body, timeout = _WEBHOOK_TIMEOUT_SEC)
                response.raise_for_status()
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        if last_exc is not None:
            raise last_exc


class TrainingNotifier:
    def __init__(self, sinks: list[WebhookSink]) -> None:
        self._sinks = list(sinks)

    def emit(self, event: TrainingTerminalEvent) -> list[Future]:
        return [_EXECUTOR.submit(self._safe_deliver, sink, event) for sink in self._sinks]

    @staticmethod
    def _safe_deliver(sink: WebhookSink, event: TrainingTerminalEvent) -> None:
        try:
            sink.deliver(event)
        except Exception as exc:  # noqa: BLE001
            # Don't log the exception/URL: webhook URLs embed the secret token.
            logger.warning(
                "Training notification sink failed: %s (%s)",
                type(exc).__name__,
                urlparse(sink.url).hostname or "?",
            )


def get_training_notifier() -> TrainingNotifier:
    try:
        from utils.notification_settings import get_training_webhook
        config = get_training_webhook()
    except Exception:
        config = None

    sinks: list[WebhookSink] = []
    if config and config.get("enabled") and config.get("url"):
        sinks.append(WebhookSink(url = config["url"]))
    return TrainingNotifier(sinks)
