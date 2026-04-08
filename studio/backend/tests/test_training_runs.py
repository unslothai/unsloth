# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

from storage.studio_db import _extract_project_name_from_config_json
from utils.training_runs import (
    build_default_output_dir_name,
    normalize_project_name,
    slugify_project_name,
)


def test_normalize_project_name_trims_and_collapses_whitespace():
    assert (
        normalize_project_name("  Customer   Support   LoRA  ")
        == "Customer Support LoRA"
    )


def test_normalize_project_name_returns_none_for_empty_or_invalid_values():
    assert normalize_project_name("   ") is None
    assert normalize_project_name(None) is None


def test_slugify_project_name_makes_safe_suffix():
    assert (
        slugify_project_name("Customer Support / LoRA v2") == "customer-support-lora-v2"
    )


def test_slugify_project_name_rejects_path_only_or_separator_only_values():
    assert slugify_project_name("..") is None
    assert slugify_project_name("///") is None


def test_build_default_output_dir_name_appends_project_slug():
    output_dir = build_default_output_dir_name(
        "unsloth/Llama-3.2-3B-Instruct",
        "Customer Support",
        timestamp = 1771227800,
    )

    assert output_dir == "unsloth_Llama-3.2-3B-Instruct_customer-support_1771227800"


def test_build_default_output_dir_name_skips_invalid_project_slug():
    output_dir = build_default_output_dir_name(
        "unsloth/Llama-3.2-3B-Instruct",
        "..",
        timestamp = 1771227800,
    )

    assert output_dir == "unsloth_Llama-3.2-3B-Instruct_1771227800"


def test_extract_project_name_from_config_json_returns_normalized_name():
    config_json = json.dumps({"project_name": "  Sales   Assistant  "})

    assert _extract_project_name_from_config_json(config_json) == "Sales Assistant"


def test_extract_project_name_from_config_json_handles_missing_or_invalid_payload():
    assert _extract_project_name_from_config_json(None) is None
    assert _extract_project_name_from_config_json("not-json") is None
    assert (
        _extract_project_name_from_config_json(json.dumps({"project_name": "   "}))
        is None
    )
