# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bring the forced-layout instrument into the registry.

`arms/layoutcost.py` deliberately does not import this package: it was written while this one was
still being built, so it exposes `register(register_instrument)` and takes the decorator as an
argument instead. That left it defined and never called, which is worse than either half alone --
`available()` omitted `layout_cost` while the implementation sat complete one directory away, so
the M3 forced-layout hypothesis read as unmeasured rather than as unwired.

`load_all()` imports siblings of this directory only, so the call has to live here.
"""

from . import register_instrument

from ..arms.layoutcost import register as _register

_register(register_instrument)
