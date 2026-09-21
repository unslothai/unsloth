# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Intentionally empty: data-designer loads submodules lazily by qualified name in plugin.py, so
# importing this package must not touch data_designer.engine.* during bootstrap (circular import).
