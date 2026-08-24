# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for the datasets/PyArrow warm-up failure."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]


def test_datasets_can_be_reimported_after_a_failed_warm_is_purged():
    probe = textwrap.dedent(
        """
        import importlib
        import sys

        import datasets

        from utils.torch_warmup import purge_partial_import

        sys.modules.pop("datasets", None)
        removed = purge_partial_import("datasets")
        assert "datasets.features.features" in removed

        reimported = importlib.import_module("datasets")
        dataset = reimported.Dataset.from_dict({"text": ["hello"]})
        assert dataset[0]["text"] == "hello"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd = BACKEND,
        text = True,
        capture_output = True,
        timeout = 60,
        check = False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
