# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from picker.service import _find_gguf_in_dir, _iter_ggufs


def test_iter_ggufs_skips_gguf_companions(tmp_path):
    mtp_dir = tmp_path / "MTP"
    mtp_dir.mkdir()
    main = tmp_path / "model-Q8_0.gguf"
    main.write_bytes(b"")
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"")
    (tmp_path / "mtp-model-Q8_0.gguf").write_bytes(b"")
    (mtp_dir / "model-Q8_0-MTP.gguf").write_bytes(b"")
    (tmp_path / "model-Q8_0-be.gguf").write_bytes(b"")

    assert _iter_ggufs(tmp_path) == [main]


def test_find_gguf_in_dir_matches_quant_label(tmp_path):
    mtp_dir = tmp_path / "MTP"
    mtp_dir.mkdir()
    main = tmp_path / "model-Q8_0.gguf"
    main.write_bytes(b"")
    (mtp_dir / "model-Q8_0-MTP.gguf").write_bytes(b"")
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"")

    assert _find_gguf_in_dir(tmp_path, "Q8_0") == main
    assert _find_gguf_in_dir(tmp_path, "Q4_K") is None


def test_find_gguf_in_dir_without_variant_prefers_largest_model(tmp_path):
    smaller = tmp_path / "a-model-Q4_K_M.gguf"
    larger = tmp_path / "z-model-Q8_0.gguf"
    smaller.write_bytes(b"0")
    larger.write_bytes(b"00")

    assert _find_gguf_in_dir(tmp_path, None) == larger


def test_find_gguf_in_dir_matches_bpw_variant_base_label(tmp_path):
    target = tmp_path / "model-IQ4_XS-3.53bpw.gguf"
    target.write_bytes(b"")
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"")

    assert _find_gguf_in_dir(tmp_path, "IQ4_XS") == target
    assert _find_gguf_in_dir(tmp_path, "IQ4_XS-3.53bpw") == target
    assert _find_gguf_in_dir(tmp_path, "Q4_K") is None
