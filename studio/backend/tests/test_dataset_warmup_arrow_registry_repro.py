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


def test_datasets_reimport_waits_for_arrow_registry_cleanup():
    probe = textwrap.dedent(
        """
        import importlib
        import sys
        import threading
        import time

        import datasets
        import pyarrow

        from utils.torch_warmup import purge_partial_import

        sys.modules.pop("datasets", None)
        first_unregistered = threading.Event()
        resume_cleanup = threading.Event()
        real_unregister = pyarrow.unregister_extension_type

        def paused_unregister(type_name):
            real_unregister(type_name)
            if type_name.endswith("Array2DExtensionType"):
                first_unregistered.set()
                assert resume_cleanup.wait(5), "cleanup was not resumed"

        pyarrow.unregister_extension_type = paused_unregister
        removed = []
        purge = threading.Thread(
            target=lambda: removed.extend(purge_partial_import("datasets")),
            daemon=True,
        )
        purge.start()
        assert first_unregistered.wait(5), "cleanup did not reach the registry"

        outcome = {}

        def reimport():
            try:
                outcome["module"] = importlib.import_module("datasets")
            except BaseException as exc:
                outcome["error"] = exc

        retry = threading.Thread(target=reimport, daemon=True)
        retry.start()
        time.sleep(0.2)
        retry_waited = retry.is_alive()
        resume_cleanup.set()
        purge.join(5)
        retry.join(5)

        assert not purge.is_alive(), "cleanup did not finish"
        assert not retry.is_alive(), "reimport did not finish"
        assert retry_waited, f"reimport raced cleanup: {outcome.get('error')!r}"
        assert "error" not in outcome, repr(outcome["error"])
        assert "datasets.features.features" in removed
        dataset = outcome["module"].Dataset.from_dict({"text": ["hello"]})
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
