# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speculative drafter discovery, ranking and budgeting.

One home for the rules a drafter sidecar obeys. Import from this facade rather
than from the submodules, so the internal split can change without touching
every call site.
"""

from utils.models.drafters.common import (
    _drafter_launch_path,
    _drafter_matches_weight,
    _drafter_names_other_weight,
    _drafter_pairing_stem,
    _drafter_split_is_complete,
    _drafter_stem_rank,
    _drafter_total_size,
)
from utils.models.drafters.preference import (
    dflash_precision_rank,
    dflash_preference_key,
    dflash_repo_preference_key,
    dspark_precision_rank,
    dspark_preference_key,
)
from utils.models.drafters.budget import dflash_budget_bytes
from utils.models.drafters.dflash import (
    detect_dflash_file,
    is_dflash_architecture,
)

__all__ = [
    "_drafter_launch_path",
    "_drafter_matches_weight",
    "_drafter_names_other_weight",
    "_drafter_pairing_stem",
    "_drafter_split_is_complete",
    "_drafter_stem_rank",
    "_drafter_total_size",
    "detect_dflash_file",
    "dflash_budget_bytes",
    "dflash_precision_rank",
    "dflash_preference_key",
    "dflash_repo_preference_key",
    "dspark_precision_rank",
    "dspark_preference_key",
    "is_dflash_architecture",
]
