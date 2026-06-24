# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path


def _seed_route_source() -> str:
    return (
        Path(__file__).resolve().parent.parent / "routes" / "data_recipe" / "seed.py"
    ).read_text()


def test_seed_inspect_load_kwargs_disables_remote_code_execution():
    assert '"trust_remote_code": False' in _seed_route_source()


def test_unstructured_upload_names_missing_extractor_dependency():
    # A missing optional extractor (pymupdf4llm/mammoth) must name the package,
    # not collapse into a generic "Text extraction failed".
    source = _seed_route_source()
    assert "except ImportError" in source
    assert "is not installed" in source
