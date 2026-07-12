# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Package marker only, so wheel builds ship this directory. The directory
# itself is placed on the sandbox PYTHONPATH so the Python interpreter's site
# machinery imports the sibling ``sitecustomize`` module at startup; nothing
# in the Studio backend imports this package directly.
