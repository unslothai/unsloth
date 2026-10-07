# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted preference for telling the model today's date."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import json
import re
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

CURRENT_DATE_PROMPT_SETTING_KEY = "include_current_date_in_prompt"
CURRENT_DATE_PROMPT_PREFIX = "The current date is "
# Leads the turn: a trailing sentence made small models answer about the date.
CURRENT_DATE_UPDATE_PREFIX = "[Current date: "
CURRENT_DATE_UPDATE_NOTE_RE = re.compile(
    rf"^\s*{re.escape(CURRENT_DATE_UPDATE_PREFIX)}[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\]\s*"
)
CURRENT_DATE_PROMPT_LINE_RE = re.compile(
    rf"(?m)^{re.escape(CURRENT_DATE_PROMPT_PREFIX)}[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}\.(?=\r?$)"
)
DEFAULT_CURRENT_DATE_PROMPT_ENABLED = True
CURRENT_DATE_TIMEZONE_HEADER = "x-unsloth-timezone"
CURRENT_DATE_TIMEZONE_OFFSET_HEADER = "x-unsloth-timezone-offset-minutes"
MAX_TIMEZONE_OFFSET_MINUTES = 14 * 60


def contains_current_date_prompt_line(text: str) -> bool:
    return CURRENT_DATE_PROMPT_LINE_RE.search(text) is not None


def replace_current_date_prompt_lines(text: str, date_line: str) -> str:
    return CURRENT_DATE_PROMPT_LINE_RE.sub(date_line, text)


def strip_current_date_prompt_lines(text: str) -> str:
    if not contains_current_date_prompt_line(text):
        return text
    return "".join(
        line
        for line in text.splitlines(keepends = True)
        if not CURRENT_DATE_PROMPT_LINE_RE.fullmatch(line.rstrip("\r\n"))
    ).strip()


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    return None


def get_current_date_prompt_enabled() -> bool:
    """Read the persisted preference, defaulting to enabled when it is missing or unreadable."""
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(CURRENT_DATE_PROMPT_SETTING_KEY, None)
    except Exception:
        stored = None
    parsed = _coerce_bool(stored)
    return parsed if parsed is not None else DEFAULT_CURRENT_DATE_PROMPT_ENABLED


def set_current_date_prompt_enabled(value: Any) -> bool:
    """Persist whether prompts should state the current date."""
    parsed = _coerce_bool(value)
    if parsed is None:
        raise ValueError("Include current date in prompt must be true or false.")

    from storage.studio_db import upsert_app_settings

    upsert_app_settings({CURRENT_DATE_PROMPT_SETTING_KEY: parsed})
    return parsed


def _request_local_date(request: Any, now: datetime | None = None) -> date:
    instant = now or datetime.now(timezone.utc)
    try:
        headers = request.headers
    except Exception:
        return instant.astimezone().date()

    timezone_name = str(headers.get(CURRENT_DATE_TIMEZONE_HEADER) or "").strip()
    if timezone_name and len(timezone_name) <= 64:
        try:
            return instant.astimezone(ZoneInfo(timezone_name)).date()
        except (ValueError, ZoneInfoNotFoundError, OSError):
            pass

    try:
        offset_minutes = int(headers.get(CURRENT_DATE_TIMEZONE_OFFSET_HEADER, ""))
    except (TypeError, ValueError):
        return instant.astimezone().date()
    if abs(offset_minutes) > MAX_TIMEZONE_OFFSET_MINUTES:
        return instant.astimezone().date()
    browser_zone = timezone(timedelta(minutes = -offset_minutes))
    return instant.astimezone(browser_zone).date()


def current_date_prompt_line(today: date | None = None, request: Any = None) -> str:
    """Return the shared date sentence, or an empty string when the preference is off."""
    if not get_current_date_prompt_enabled():
        return ""
    resolved_date = today or _request_local_date(request)
    return f"{CURRENT_DATE_PROMPT_PREFIX}{resolved_date.isoformat()}."


_PROBE_SYSTEM = "UNSLOTH_DATE_PROBE_SYSTEM"
_PROBE_USER = "UNSLOTH_DATE_PROBE_USER"
_PROBE_SPECIAL_TOKENS = {
    f"{name}_token": f"UNSLOTH_DATE_PROBE_{name.upper()}"
    for name in ("bos", "eos", "pad", "unk", "sep", "cls", "mask")
}
PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "probe",
            "description": "Probe.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def _render_probe(
    chat_template: str,
    messages: list[dict],
    today: date,
    tools: list | None = None,
    controls: dict | None = None,
) -> str:
    from jinja2.exceptions import TemplateError
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    def tojson(
        value,
        ensure_ascii = False,
        indent = None,
        separators = None,
        sort_keys = False,
    ):
        # Transformers disables HTML escaping, so the probe must match the same bytes.
        return json.dumps(
            value,
            ensure_ascii = ensure_ascii,
            indent = indent,
            separators = separators,
            sort_keys = sort_keys,
        )

    class _GenerationTag(Extension):
        tags = {"generation"}

        def parse(self, parser):
            next(parser.stream)
            return parser.parse_statements(["name:endgeneration"], drop_needle = True)

    env = ImmutableSandboxedEnvironment(
        trim_blocks = True,
        lstrip_blocks = True,
        extensions = [_GenerationTag, "jinja2.ext.loopcontrols"],
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = today.strftime
    return env.from_string(chat_template).render(
        messages = messages,
        tools = tools,
        documents = None,
        add_generation_prompt = False,
        **_PROBE_SPECIAL_TOKENS,
        **(controls or {}),
    )


def _is_whitespace_normalized_substring(value: str, container: str) -> bool:
    """Whether ``value`` remains intact after ignoring formatting whitespace."""
    return " ".join(value.split()) in " ".join(container.split())


@lru_cache(maxsize = 16)
def template_system_turn(
    chat_template: str | None,
    today: date,
    tools: bool = False,
    controls: tuple = (),
) -> tuple[bool, str | None]:
    """How a system turn the chat did not send renders in this template on ``today``.

    Whether one renders at all, and the default system prompt it has to carry so the prompt reads as
    the template's own render: "" when there is none, None when no system turn reproduces it (the
    template rewrites what it is given). A template the probe cannot render takes one carrying nothing.
    """
    if not chat_template:
        return True, ""
    catalog = PROBE_TOOLS if tools else None
    kwargs = dict(controls)
    renders_chat = False
    for content in (lambda text: text, lambda text: [{"type": "text", "text": text}]):
        user = {"role": "user", "content": content(_PROBE_USER)}

        def render(system: str | None) -> str:
            turns = [{"role": "system", "content": content(system)}] if system is not None else []
            return _render_probe(chat_template, [*turns, user], today, catalog, kwargs)

        try:
            bare = render(None)
        except Exception:
            continue
        renders_chat = True
        try:
            with_system = render(_PROBE_SYSTEM)
        except Exception:
            continue
        if with_system.count(_PROBE_SYSTEM) != 1:
            continue
        head, tail = with_system.split(_PROBE_SYSTEM)
        without_system = head + tail
        if _is_whitespace_normalized_substring(bare, without_system):
            return True, ""
        if len(bare) <= len(without_system) or not bare.startswith(head) or not bare.endswith(tail):
            # The explicit-system branch changes more than the text, so do not replay it.
            return True, None
        default = bare[len(head) : len(bare) - len(tail)]
        try:
            replayed = render(default)
        except Exception:
            replayed = None
        carries_token = any(token in default for token in _PROBE_SPECIAL_TOKENS.values())
        return True, (default.strip() if replayed == bare and not carries_token else None)
    return not renders_chat, ("" if not renders_chat else None)


def strip_current_date_update_note(text: str) -> str:
    return CURRENT_DATE_UPDATE_NOTE_RE.sub("", text)
