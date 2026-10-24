# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registers arms/layoutcost.py; load_all imports only siblings, so the register call must live here."""

from . import register_instrument

from ..arms.layoutcost import register as _register

_register(register_instrument)
