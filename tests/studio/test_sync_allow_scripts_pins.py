#!/usr/bin/env python3
"""Offline tests for scripts/sync_allow_scripts_pins.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import sync_allow_scripts_pins as sync  # noqa: E402


def write_fixture(tmp: Path, policy: dict, lock_packages: dict) -> None:
    (tmp / "package.json").write_text(
        json.dumps(
            {
                "name": "fixture",
                "dependencies": {"a": "^1.0.0"},
                "allowScripts": policy,
            },
            indent = 2,
        )
        + "\n"
    )
    (tmp / "package-lock.json").write_text(
        json.dumps(
            {
                "lockfileVersion": 3,
                "packages": lock_packages,
            }
        )
        + "\n"
    )


LOCK = {
    "": {"name": "fixture"},
    "node_modules/@biomejs/biome": {"version": "1.9.9", "hasInstallScript": True},
    "node_modules/msw": {"version": "2.15.0", "hasInstallScript": True},
    "node_modules/vite/node_modules/fsevents": {"version": "2.3.3", "hasInstallScript": True},
    "node_modules/clean": {"version": "3.0.0"},
}


def test_split_spec():
    assert sync.split_spec("fsevents") == ("fsevents", None)
    assert sync.split_spec("@biomejs/biome") == ("@biomejs/biome", None)
    assert sync.split_spec("@biomejs/biome@1.9.4") == ("@biomejs/biome", "1.9.4")
    assert sync.split_spec("msw@2.14.3 || 2.15.0") == ("msw", "2.14.3 || 2.15.0")


def test_in_sync_is_clean():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write_fixture(
            tmp,
            {
                "@biomejs/biome@1.9.9": True,
                "msw@2.15.0": True,
                "fsevents": True,
            },
            LOCK,
        )
        assert sync.main(["--check", "--dir", str(tmp)]) == 0


def test_stale_pin_fails_check_and_fix_repairs():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write_fixture(
            tmp,
            {
                "@biomejs/biome@1.9.4": True,  # stale, bumped to 1.9.9
                "msw@2.14.3": False,  # stale denial, bumped to 2.15.0
                "fsevents": True,  # bare: never stale
                "ghost@9.9.9": True,  # not in lockfile: left alone
                "weird@*": True,  # non-exact spec: left alone
            },
            LOCK,
        )
        assert sync.main(["--check", "--dir", str(tmp)]) == 1
        assert sync.main(["--fix", "--dir", str(tmp)]) == 0
        got = json.loads((tmp / "package.json").read_text())["allowScripts"]
        assert got == {
            "@biomejs/biome@1.9.9": True,
            "msw@2.15.0": False,  # value and key order preserved
            "fsevents": True,
            "ghost@9.9.9": True,
            "weird@*": True,
        }
        assert list(got) == [
            "@biomejs/biome@1.9.9",
            "msw@2.15.0",
            "fsevents",
            "ghost@9.9.9",
            "weird@*",
        ]
        assert sync.main(["--check", "--dir", str(tmp)]) == 0


def test_multi_version_disjunction():
    lock = dict(LOCK)
    lock["node_modules/x/node_modules/msw"] = {"version": "2.14.3", "hasInstallScript": True}
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        write_fixture(tmp, {"msw@2.14.3": True}, lock)
        assert sync.main(["--fix", "--dir", str(tmp)]) == 0
        got = json.loads((tmp / "package.json").read_text())["allowScripts"]
        assert got == {"msw@2.14.3 || 2.15.0": True}


def test_no_policy_is_noop():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "package.json").write_text(json.dumps({"name": "fixture"}) + "\n")
        (tmp / "package-lock.json").write_text(json.dumps({"packages": {}}) + "\n")
        before = (tmp / "package.json").read_text()
        assert sync.main(["--fix", "--dir", str(tmp)]) == 0
        assert (tmp / "package.json").read_text() == before


def test_missing_files_is_noop():
    with tempfile.TemporaryDirectory() as d:
        assert sync.main(["--check", "--dir", d]) == 0


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
