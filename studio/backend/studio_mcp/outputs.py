# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Typed tool outputs. Several Studio routes still return host paths to API-key callers, so a tool never passes route JSON through: it copies named fields into a ToolOutput, which refuses anything undeclared. Free text that came from a route (a status message, an error) is declared ``RouteText`` and scrubbed of paths as a backstop; model-written text and Studio URLs are plain ``str`` and left alone."""

from __future__ import annotations

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict

from hub.utils.host_paths import redact_paths_in_text

RouteText = Annotated[str, AfterValidator(redact_paths_in_text)]


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra = "forbid")
