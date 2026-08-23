# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""studiobench: a real-path performance benchmark for Unsloth Studio.

Layer 1 (this package's runtime, pacer, fixture and scene) drives the SHIPPED app through its own
backend and its own SSE transport. See INTERFACES.md for the contract Layers 2 and 3 plug into.
"""

__version__ = "0.1.0"
