# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Data-only host configuration checks, not runtime admission evidence."""

from dataclasses import replace
from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox.host_config import HEADER, HostConfiguration, HostPaths
from core.inference.windows_sandbox.profiles import PYTHON_PROFILE, WindowsRuntimeError


@pytest.fixture
def config():
    paths = HostPaths(
        *(
            "E:\\private\\" + name
            for name in (
                "python.dll",
                "stdlib.zip",
                "DLLs",
                "runtime",
                "python.exe",
                "prefix",
                "base",
                "policy.py",
                "shim.py",
                "work",
                "tool.py",
                "temp",
                "aap",
            )
        )
    )
    return HostConfiguration(
        paths,
        "S-1-15-2-1-2-3-4-5-6-7",
        (3, 12, 10),
        b"n" * 32,
        bytes.fromhex(PYTHON_PROFILE.digest),
        b"c" * 32,
    )


def test_expanded_config_binds_activation_plan(config):
    from core.inference.windows_sandbox.activation_plan import ActivationPlan

    plan = ActivationPlan(config.nonce, config.profile_digest, config.content_digest, ()).encode()
    encoded = replace(config, activation_plan = plan).encode()
    assert HEADER.unpack_from(encoded)[1:3] == (2, len(encoded))
    assert encoded.endswith(struct.pack("<I", len(plan)) + plan)
    with pytest.raises(WindowsRuntimeError):
        replace(config, activation_plan = plan, nonce = b"x" * 32).encode()


def test_host_config_is_bounded_data_with_unicode_arguments(config):
    value = replace(
        config,
        arguments = ("--help\n'\"λ😀", ""),
        packages = ("E:\\packages\\λ",),
        native_images = ("E:\\private\\runtime\\native.pyd",),
    )
    data = value.encode()
    magic, version, size, major, minor, patch, packages, args, images, *binding = (
        HEADER.unpack_from(data)
    )
    assert (magic, version, size, major, minor, patch, packages, args, images) == (
        b"USLPCF1\0",
        1,
        len(data),
        3,
        12,
        10,
        1,
        2,
        1,
    )
    assert binding == [value.nonce, value.profile_digest, value.content_digest]
    offset, values = HEADER.size, []
    while offset < len(data):
        (count,) = struct.unpack_from("<I", data, offset)
        offset += 4
        values.append(data[offset : offset + count].decode("utf-16-le"))
        offset += count
    assert offset == len(data) and len(data) <= 65536
    assert values[14:] == [*value.packages, *value.arguments, *value.native_images]
    assert config.arguments == ()


@pytest.mark.parametrize(
    "path",
    [
        "C:\\",
        "\\\\host\\share\\runtime",
        "relative",
        "C:/runtime",
        "C:\\runtime\\..\\escape",
        "C:\\runtime\\.\\alias",
        "C:\\runtime\\nul.txt",
        "C:\\runtime\\LPT¹",
        "C:\\runtime\\end.",
        "C:\\runtime\\end ",
        "C:\\bad\nname",
        "C:\\a\0b",
    ],
)
def test_host_config_rejects_root_network_and_path_aliases(config, path):
    with pytest.raises(WindowsRuntimeError):
        replace(config, paths = replace(config.paths, runtime_dll = path)).encode()


@pytest.mark.parametrize(
    "override",
    [
        {"nonce": b"n"},
        {"profile_digest": bytearray(32)},
        {"content_digest": None},
        {"version": (3, 14, 0)},
        {"version": (3, 12, True)},
        {"version": (3, 12, 2**32)},
        {"packages": []},
        {"arguments": ("a",) * 65},
        {"native_images": ("E:\\outside\\module.pyd",)},
        {"native_images": ("E:\\private\\runtime-other\\module.pyd",)},
        {"arguments": ("\ud800",)},
        {"arguments": ("\0",)},
        {"arguments": ("x" * 16384,)},
        {"arguments": ("x" * 16000,) * 3},
        {"package_sid": "S-1-15-2-1"},
    ],
)
def test_host_config_rejects_malformed_or_unbounded_values(config, override):
    with pytest.raises(WindowsRuntimeError):
        replace(config, **override).encode()
