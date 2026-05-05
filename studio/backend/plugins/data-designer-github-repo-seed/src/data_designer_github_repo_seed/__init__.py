# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Intentionally empty. Data-designer loads submodules lazily via qualified names
# (impl_qualified_name / config_qualified_name in plugin.py), so importing this
# package must NOT touch modules that depend on data_designer.engine.* during
# Studio's bootstrap (circular import).
