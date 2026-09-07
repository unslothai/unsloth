# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Strict native startup wire contract. Parsing is not qualification evidence."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import protocol as p
from core.inference.windows_sandbox.profiles import PYTHON_PROFILE, WindowsRuntimeError


@pytest.fixture
def binding():
    return p.LaunchBinding(
        123, p.new_launch_nonce(), bytes.fromhex(PYTHON_PROFILE.digest), b"c" * 32
    )


def packet(binding, **changes):
    values = dict(
        magic = p.STATUS_MAGIC,
        version = p.VERSION,
        phase = p.READY,
        pid = binding.pid,
        checks = p.REQUIRED_GATE_CHECKS,
        error = 0,
        stage = 16,
        nonce = binding.nonce,
        profile = binding.profile_digest,
        content = binding.content_digest,
    )
    values.update(changes)
    return p.STATUS.pack(*values.values())


def test_wire_sizes_and_immutable_launch_binding(binding):
    assert p.STATUS.size == 128
    assert p.ACK.size == 112
    assert p.parse_startup_status(packet(binding), binding) is binding
    assert p.ACK.unpack(p.acknowledgement(binding)) == (
        p.ACK_MAGIC,
        1,
        0,
        binding.nonce,
        binding.profile_digest,
        binding.content_digest,
    )
    with pytest.raises(FrozenInstanceError):
        binding.pid = 124


@pytest.mark.parametrize(
    "change",
    [
        {"magic": b"stdout!!"},
        {"version": 2},
        {"phase": 0},
        {"phase": 3},
        {"pid": 124},
        {"checks": 0},
        {"checks": 0x3FF},
        {"error": 5},
        {"stage": 0},
        {"stage": 15},
        {"stage": 17},
        {"nonce": b"x" * 32},
        {"profile": b"x" * 32},
        {"content": b"x" * 32},
    ],
)
def test_status_rejects_unbound_or_incomplete_gate(binding, change):
    with pytest.raises(WindowsRuntimeError, match = "PROTOCOL_MISMATCH"):
        p.parse_startup_status(packet(binding, **change), binding)


@pytest.mark.parametrize("bit", range(9))
def test_each_missing_native_check_prevents_ready(binding, bit):
    with pytest.raises(WindowsRuntimeError, match = "PROTOCOL_MISMATCH"):
        p.parse_startup_status(packet(binding, checks = p.REQUIRED_GATE_CHECKS ^ (1 << bit)), binding)


@pytest.mark.parametrize("size", [0, 1, 127, 129, 4096])
def test_status_rejects_truncation_and_excess(binding, size):
    with pytest.raises(WindowsRuntimeError, match = "PROTOCOL_MISMATCH"):
        p.parse_startup_status((packet(binding) + b"x" * 4096)[:size], binding)


def test_native_failure_is_bounded_diagnostic_not_launch_record(binding):
    with pytest.raises(WindowsRuntimeError, match = "stage 7.*WinError 5") as error:
        p.parse_startup_status(
            packet(binding, phase = p.FAILED, stage = 7, error = 5, checks = 127), binding
        )
    assert error.value.code == "WINDOWS_SANDBOX_STARTUP_FAILED"


@pytest.mark.parametrize(
    "changes",
    [
        {"pid": True},
        {"pid": 0},
        {"pid": -1},
        {"pid": 2**32},
        {"nonce": b""},
        {"nonce": "a" * 32},
        {"nonce": bytearray(32)},
        {"profile_digest": b"a" * 31},
        {"content_digest": b"a" * 33},
    ],
)
def test_invalid_binding_is_rejected(binding, changes):
    with pytest.raises(WindowsRuntimeError, match = "PROTOCOL_MISMATCH"):
        replace(binding, **changes)


def test_clean_entry_is_not_payload_authorization(binding):
    data = packet(binding, phase = p.CLEAN_ENTRY, checks = 0, stage = 1)
    p.parse_clean_entry(data, binding)
    with pytest.raises(WindowsRuntimeError):
        p.parse_startup_status(data, binding)
    with pytest.raises(WindowsRuntimeError):
        p.parse_clean_entry(packet(binding), binding)


@pytest.mark.parametrize(
    "change", [{"pid": 124}, {"nonce": b"x" * 32}, {"checks": 1}, {"error": 5}, {"stage": 2}]
)
def test_clean_entry_rejects_unbound_or_dirty_start(binding, change):
    values = dict(phase = p.CLEAN_ENTRY, checks = 0, stage = 1)
    values.update(change)
    with pytest.raises(WindowsRuntimeError):
        p.parse_clean_entry(packet(binding, **values), binding)


def test_private_catalog_permission_is_bounded_and_bound(binding):
    path = r"E:\private\hive\winsock.hiv"
    data = p.startup_permission(binding, path, b"m" * 32)
    header = p.STARTUP_GO.unpack_from(data)
    assert p.STARTUP_GO.size == 152 and len(data) < 4096
    assert header == (
        b"USLPGO2\0",
        2,
        0,
        binding.nonce,
        binding.profile_digest,
        binding.content_digest,
        b"m" * 32,
        len(path.encode("utf-16-le")),
        0,
    )
    assert data[p.STARTUP_GO.size :].decode("utf-16-le") == path


@pytest.mark.parametrize(
    "path,marker",
    [
        (r"E:\private\..\host", b"m" * 32),
        (r"E:\private\hive", b"m"),
        ("E:\\" + "x" * 1024, b"m" * 32),
    ],
)
def test_private_catalog_permission_rejects_bad_paths_or_identity(binding, path, marker):
    with pytest.raises(WindowsRuntimeError):
        p.startup_permission(binding, path, marker)
