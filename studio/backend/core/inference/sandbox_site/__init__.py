# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# Package marker only, so wheel builds ship this directory. It goes on the
# sandbox PYTHONPATH so site machinery imports the sibling ``sitecustomize`` at
# startup; nothing in the backend imports it directly.
