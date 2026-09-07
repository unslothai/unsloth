# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Metadata-only native handle controls; ordinary startup is not LPAC startup."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox.private_catalog import prepare_private_catalog


@pytest.fixture
def authority_case(tmp_path):
    path = os.environ.get("UNSLOTH_AUTHORITY_DRIVER")
    if os.name != "nt" or not path:
        pytest.skip("Set UNSLOTH_AUTHORITY_DRIVER to the freshly built native audit driver")
    driver = Path(path).resolve()
    assert driver.is_file()
    with prepare_private_catalog(tmp_path) as catalog:
        yield driver, catalog


def run_case(case, mode, *args):
    driver, catalog = case
    child = subprocess.run(
        [str(driver), mode, str(catalog.hive_path), *map(str, args)],
        capture_output = True,
        text = True,
        timeout = 20,
    )
    assert child.returncode == 0, (child.returncode, child.stdout, child.stderr)
    lines = child.stdout.splitlines()
    fields = dict(item.split("=", 1) for item in lines[0].split())
    return {key: int(value) for key, value in fields.items()}, lines[1:]


def test_private_hive_is_identified_without_approving_ordinary_startup(authority_case):
    report, types = run_case(authority_case, "clean")
    assert report["native"] == 0 and report["stable"] == 1
    assert report["private"] >= 1 and report["tokens"] == 0
    assert report["error"] in (0, 5)
    if report["foreign"]:
        assert report["error"] == 5  # Ordinary loader can retain host keys.
    assert any(line.startswith("type=Key ") for line in types)
    assert report["retained"] == 1


def test_host_query_handle_is_detected_and_never_closed(authority_case):
    baseline, _ = run_case(authority_case, "clean")
    report, _ = run_case(authority_case, "host-key")
    assert report["error"] == 5 and report["native"] == 0 and report["stable"] == 1
    assert report["foreign"] > baseline["foreign"]
    assert report["positive"] == report["retained"] == 1


def test_other_private_hive_is_not_the_allowed_hive(authority_case, tmp_path):
    baseline, _ = run_case(authority_case, "clean")
    with prepare_private_catalog(tmp_path) as other:
        report, _ = run_case(authority_case, "other-hive", other.hive_path)
    assert report["error"] == 5 and report["foreign"] > baseline["foreign"]
    assert report["positive"] == report["retained"] == 1


def test_retained_token_handle_is_rejected(authority_case):
    report, _ = run_case(authority_case, "token")
    assert report["error"] == 5 and report["tokens"] == 1 and report["stable"] == 1
    assert report["positive"] == 1


def test_foreign_process_handle_is_rejected(authority_case):
    report, _ = run_case(authority_case, "foreign-process", os.getpid())
    assert report["error"] == 5 and report["foreign_processes"] == 1
    assert report["positive"] == 1


def test_invalid_expected_hive_fails_closed(authority_case):
    report, _ = run_case(authority_case, "invalid-root")
    assert report["error"] != 0 and report["native"] != 0 and report["stable"] == 0


def test_handle_budget_is_enforced(authority_case):
    report, _ = run_case(authority_case, "handle-limit")
    assert report["error"] == 111 and report["stable"] == 0
