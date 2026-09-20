# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Turning two studiobench payloads into a verdict.

The benchmark answers "how long did this take". These answer the question that actually decides a
pull request, which is "is that difference real". They are separate from `scoring/` because they
operate on a PAIR of runs rather than on one, and separate from `report/` because a verdict needs
something `report/` does not have: a null control to measure the floor against.

    python -m tests.studio.studiobench.sweep.floor_table --floor outputs/null outputs/mine
    python -m tests.studio.studiobench.sweep.ui_parity --null outputs/null outputs/mine
"""
