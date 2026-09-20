# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import hashlib
import json

from core.inference import mxc_policy


def _config() -> dict:
    return {
        "version": "0.8.0-alpha",
        "containerId": "unsloth-会話",
        "containment": "processcontainer",
        "filesystem": {
            "readwritePaths": [r"D:\会話 work"],
            "readonlyPaths": [r"C:\Python"],
            "deniedPaths": [],
        },
        "fallback": {"allowDaclMutation": False},
        "ui": {"disable": False, "clipboard": "none", "injection": False},
    }


def test_policy_hash_is_the_exact_canonical_config_hash():
    config = _config()
    encoded = mxc_policy.canonical_config_bytes(config)
    assert encoded == json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert mxc_policy.compute_policy_hash(config) == "sha256:" + hashlib.sha256(encoded).hexdigest()


def test_canonical_hash_ignores_json_object_insertion_order():
    config = _config()
    reordered = dict(reversed(list(config.items())))
    assert mxc_policy.compute_policy_hash(reordered) == mxc_policy.compute_policy_hash(config)


def test_one_byte_policy_mutation_changes_the_identity():
    config = _config()
    original = mxc_policy.compute_policy_hash(config)
    config["fallback"]["allowDaclMutation"] = True
    assert mxc_policy.compute_policy_hash(config) != original


def test_ui_policy_mutation_changes_the_identity():
    config = _config()
    original = mxc_policy.compute_policy_hash(config)
    config["ui"]["disable"] = True
    assert mxc_policy.compute_policy_hash(config) != original
