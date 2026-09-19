# SPDX-License-Identifier: AGPL-3.0-only
"""Cross-language canonical MXC policy identity vectors."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from core.inference import mxc_policy, mxc_runtime


_VECTORS = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "native"
        / "mxc-runner"
        / "tests"
        / "policy_hash_vectors.json"
    ).read_text(encoding="utf-8")
)


def _request() -> dict:
    return {
        "protocol": 1,
        "runId": "0123456789abcdef0123456789abcdef",
        "profileId": mxc_policy.PROFILE_ID,
        "profileVersion": 1,
        "schemaVersion": mxc_runtime.MXC_SCHEMA_VERSION,
        "runtimeRevision": mxc_runtime.MXC_REVISION,
        "containerId": "unsloth-0123456789abcdef0123456789abcdef",
        "argv": [r"C:\Program Files\Python\python.exe", "-c", "print('会話')"],
        "executionKind": "python",
        "runtimePath": r"C:\Program Files\Python\python.exe",
        "runtimeIdentity": {"volumeSerialNumber": 111, "fileId": 222},
        "cwd": r"D:\会話\work",
        "workdirIdentity": {"volumeSerialNumber": 333, "fileId": 444},
        "environment": {"Path": r"C:\Windows\System32", "UNICODE": "é"},
        "environmentPolicy": "sanitized-explicit-v1",
        "commandLinePolicy": "windows-createprocess-argv-v1",
        "readwritePaths": [r"D:\会話\work"],
        "readonlyPaths": [r"C:\Program Files\Python", r"C:\Windows"],
        "deniedPaths": [],
        "clearPolicyOnExit": True,
        "networkProfile": "compatibility",
        "allowOutbound": True,
        "allowLocalNetwork": True,
        "uiPolicy": None,
        "timeoutMs": 30_000,
        "allowDaclMutation": False,
        "admission": "atomic-no-dacl-fallback",
    }


def test_python_matches_the_shared_rust_policy_hash_vectors():
    request = _request()
    assert mxc_policy.compute_policy_hash(request) == _VECTORS["python_unicode"]

    terminal = copy.deepcopy(request)
    terminal.update(
        argv=[
            r"C:\Windows\System32\cmd.exe",
            "/d",
            "/s",
            "/c",
            'echo %VAR% && (echo "quoted")',
        ],
        executionKind="terminal",
        runtimePath=r"C:\Windows\System32\cmd.exe",
        environment={},
        timeoutMs=None,
    )
    assert mxc_policy.compute_policy_hash(terminal) == _VECTORS["terminal_empty_optional"]

    mutations = {
        "different_drive": lambda value: value.update(
            cwd=r"E:\alternate\work", readwritePaths=[r"E:\alternate\work"]
        ),
        "network_change": lambda value: value.update(allowLocalNetwork=False),
        "filesystem_change": lambda value: value["readonlyPaths"].append(r"C:\Extra"),
        "dacl_change": lambda value: value.update(allowDaclMutation=True),
        "cwd_change": lambda value: value.update(cwd=r"D:\会話\other"),
        "executable_change": lambda value: value.update(
            runtimePath=r"C:\Python\python.exe",
            argv=[r"C:\Python\python.exe", "-c", "print('会話')"],
        ),
        "environment_change": lambda value: value["environment"].update(EXTRA="1"),
    }
    for name, mutate in mutations.items():
        value = copy.deepcopy(request)
        mutate(value)
        assert mxc_policy.compute_policy_hash(value) == _VECTORS[name]


def test_canonical_hash_ignores_json_object_insertion_order():
    request = _request()
    reordered = dict(reversed(list(request.items())))
    reordered["environment"] = dict(reversed(list(request["environment"].items())))
    assert mxc_policy.compute_policy_hash(reordered) == mxc_policy.compute_policy_hash(request)


def test_one_byte_policy_mutation_changes_the_identity():
    request = _request()
    original = mxc_policy.compute_policy_hash(request)
    request["argv"][-1] = "print('会話!')"
    assert mxc_policy.compute_policy_hash(request) != original
