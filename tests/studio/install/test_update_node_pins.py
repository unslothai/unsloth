# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""scripts/update_node_pins.py: stays on the pinned LTS major, verifies before it pins."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "update_node_pins", REPO / "scripts" / "update_node_pins.py"
)
U = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(U)

PINS = {
    "schema_version": 1,
    "comment": "c",
    "default_version": "24.18.0",
    "versions": {"24.18.0": {}},
}


def _index(*releases):
    return json.dumps([{"version": v, "lts": lts} for v, lts in releases]).encode()


def _fake(
    index,
    target = "24.22.1",
    wrong = None,
):
    shasums = "".join(
        f"{'0' * 63}{i} {name}\n" for i, name in enumerate(U.archive_names(target))
    ).encode()

    def fetch(url):
        if url.endswith("/index.json"):
            return index
        if url.endswith("SHASUMS256.txt"):
            return shasums
        return b"sig-or-keyring"

    def hasher(url):
        name = url.rsplit("/", 1)[1]
        digest = dict(line.split()[::-1] for line in shasums.decode().splitlines())[name]
        return "f" * 64 if name == wrong else digest

    return fetch, hasher


def test_takes_the_newest_lts_release_of_the_pinned_major():
    fetch, hasher = _fake(
        _index(
            ("v26.12.1", False),
            ("v24.22.1", "Krypton"),
            ("v24.20.0", "Krypton"),
            ("v22.23.4", "Jod"),
        )
    )
    pins, changes = U.update_pins(PINS, fetch, hasher, lambda *a: None)
    assert changes == ["node: 24.18.0 -> 24.22.1"]
    assert pins["default_version"] == "24.22.1"
    assert sorted(pins["versions"]["24.22.1"]) == sorted(U.archive_names("24.22.1"))
    assert pins["comment"] == "c"


def test_a_newer_major_or_a_non_lts_release_is_left_for_review():
    fetch, hasher = _fake(_index(("v26.12.1", False), ("v24.18.0", "Krypton"), ("v28.0.0", "Next")))
    assert U.update_pins(PINS, fetch, hasher, lambda *a: None) == (PINS, [])


def test_a_bad_signature_changes_nothing():
    fetch, hasher = _fake(_index(("v24.22.1", "Krypton")))

    def refuse(*_):
        raise RuntimeError("SHASUMS256.txt is not signed by a Node release key.")

    with pytest.raises(RuntimeError, match = "not signed"):
        U.update_pins(PINS, fetch, hasher, refuse)


def test_an_archive_that_does_not_match_the_signed_digest_is_refused():
    fetch, hasher = _fake(_index(("v24.22.1", "Krypton")), wrong = "node-v24.22.1-win-arm64.zip")
    with pytest.raises(RuntimeError, match = "win-arm64"):
        U.update_pins(PINS, fetch, hasher, lambda *a: None)


def test_the_committed_pins_name_all_six_archives():
    pins = json.loads((REPO / "studio" / "node_prebuilt_pins.json").read_text(encoding = "utf-8"))
    version = pins["default_version"]
    assert sorted(pins["versions"][version]) == sorted(U.archive_names(version))
