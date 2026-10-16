# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The package logger, named after the module this package replaced."""

from loggers import get_logger

logger = get_logger("core.inference.llama_cpp")
