# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speculative drafter discovery, ranking and budgeting facade; import from here, not submodules."""

from utils.models.drafters.common import (
    _drafter_launch_path,
    _drafter_matches_weight,
    _drafter_names_other_weight,
    _drafter_pairing_stem,
    _drafter_split_is_complete,
    split_listing_is_complete,
    _drafter_stem_rank,
    _drafter_total_size,
    is_published_drafter_filename,
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
from utils.models.drafters.dspark_schema import (
    DSPARK_SCHEMA_FUSED,
    DSPARK_SCHEMA_GENERIC,
    DSPARK_SCHEMA_MARKOV2,
    INKLING_DSPARK_SCHEMAS,
    binary_dspark_schemas,
    dspark_drafter_incompatibility,
    dspark_drafter_schema,
    dspark_schema_rank,
    remote_dspark_drafter_schema,
    schema_incompatibility,
    supported_dspark_schemas,
)

__all__ = [
    "_drafter_launch_path",
    "_drafter_matches_weight",
    "_drafter_names_other_weight",
    "_drafter_pairing_stem",
    "_drafter_split_is_complete",
    "split_listing_is_complete",
    "_drafter_stem_rank",
    "_drafter_total_size",
    "is_published_drafter_filename",
    "detect_dflash_file",
    "dflash_budget_bytes",
    "dflash_precision_rank",
    "dflash_preference_key",
    "dflash_repo_preference_key",
    "dspark_precision_rank",
    "dspark_preference_key",
    "is_dflash_architecture",
    "DSPARK_SCHEMA_FUSED",
    "DSPARK_SCHEMA_GENERIC",
    "DSPARK_SCHEMA_MARKOV2",
    "INKLING_DSPARK_SCHEMAS",
    "binary_dspark_schemas",
    "dspark_drafter_incompatibility",
    "dspark_drafter_schema",
    "dspark_schema_rank",
    "remote_dspark_drafter_schema",
    "schema_incompatibility",
    "supported_dspark_schemas",
]
