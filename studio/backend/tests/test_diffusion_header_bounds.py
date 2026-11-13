# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Header readers must not let a downloaded checkpoint name how many bytes Studio buffers."""

from __future__ import annotations

import json
import struct

import pytest

from core.inference import diffusion_comfy_quant as cq
from core.inference import video_minimax_h3_comfy as h3c


def _safetensors(
    path,
    header: dict,
    payload: bytes = b"",
) -> str:
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return str(path)


def test_an_oversized_comfy_quant_declaration_is_refused(tmp_path):
    payload = json.dumps("a" * (2 * 1024 * 1024)).encode("utf-8")
    path = _safetensors(
        tmp_path / "big.safetensors",
        {
            "x.comfy_quant": {
                "dtype": "U8",
                "shape": [len(payload)],
                "data_offsets": [0, len(payload)],
            }
        },
        payload,
    )
    scan = cq.scan_comfy_quant(path)
    assert scan is not None
    assert scan.problems == ["x: unreadable comfy_quant declaration"]


def test_reversed_comfy_quant_offsets_are_refused(tmp_path):
    path = _safetensors(
        tmp_path / "rev.safetensors",
        {"x.comfy_quant": {"dtype": "U8", "shape": [4], "data_offsets": [8, 4]}},
        b"{}      ",
    )
    assert cq.scan_comfy_quant(path).problems == ["x: unreadable comfy_quant declaration"]


def test_a_normal_comfy_quant_declaration_still_reads(tmp_path):
    payload = json.dumps({"format": "int8_tensorwise"}).encode("utf-8")
    path = _safetensors(
        tmp_path / "ok.safetensors",
        {
            "x.comfy_quant": {
                "dtype": "U8",
                "shape": [len(payload)],
                "data_offsets": [0, len(payload)],
            }
        },
        payload,
    )
    scan = cq.scan_comfy_quant(path)
    assert not any("unreadable" in p for p in scan.problems)


def test_h3_curve_metadata_refuses_an_implausible_header_size(tmp_path):
    path = tmp_path / "h3.safetensors"
    path.write_bytes(struct.pack("<Q", 300 * 1024 * 1024) + b"{}")
    with pytest.raises(ValueError, match = "implausible"):
        h3c.h3_comfy_curve_metadata(str(path))


def test_h3_curve_metadata_reads_a_normal_header(tmp_path):
    path = _safetensors(
        tmp_path / "dense.safetensors",
        {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        b"\0" * 4,
    )
    assert h3c.h3_comfy_curve_metadata(path) is None
