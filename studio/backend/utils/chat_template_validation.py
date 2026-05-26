# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Optional

MAX_CHAT_TEMPLATE_BYTES = 65_536


def validate_chat_template(template: str) -> Optional[str]:
    if len(template.encode("utf-8")) > MAX_CHAT_TEMPLATE_BYTES:
        return f"Keep it under {MAX_CHAT_TEMPLATE_BYTES:,} bytes."

    try:
        from jinja2 import Environment, TemplateError, nodes
        from jinja2.ext import Extension
    except ImportError as exc:
        return f"Jinja validation is unavailable: {exc}"

    class GenerationExtension(Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(
                ["name:endgeneration"], drop_needle = True
            )
            return nodes.CallBlock(
                self.call_method("_render"), [], [], body
            ).set_lineno(lineno)

        def _render(self, caller):
            return caller()

    try:
        Environment(
            trim_blocks = True,
            lstrip_blocks = True,
            extensions = [GenerationExtension, "jinja2.ext.loopcontrols"],
        ).compile(template)
    except TemplateError as exc:
        return str(exc).strip() or "Invalid Jinja template."

    return None
