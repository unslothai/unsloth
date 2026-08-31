# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import unsloth.save as save_mod


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "50GB"),
        ("", "0"),
        ("  ", "0"),
        ("0", "0"),
        ("none", "0"),
        ("NONE", "0"),
        ("1MB", "1MB"),
        ("256m", "256MB"),
        ("512 MB", "512MB"),
        ("4G", "4GB"),
        (" 8gb ", "8GB"),
    ],
)
def test_resolve_gguf_shard_size(value, expected):
    assert save_mod._resolve_gguf_shard_size(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "0MB",
        "0GB",
        "-1GB",
        "1.5GB",
        "512",
        "64KB",
        "GB",
        "2TB",
        "2GBx",
    ],
)
def test_invalid_gguf_shard_sizes_fail_before_conversion(value):
    with pytest.raises(ValueError, match = "gguf_shard_size"):
        save_mod._resolve_gguf_shard_size(value)


@pytest.mark.parametrize("value", [0, False, 512, object()])
def test_non_string_gguf_shard_sizes_are_rejected(value):
    with pytest.raises(TypeError, match = "string or None"):
        save_mod._resolve_gguf_shard_size(value)


def test_oversized_gguf_shard_size_is_rejected(monkeypatch):
    monkeypatch.setattr(save_mod.sys, "maxsize", 2_147_483_647)
    with pytest.raises(ValueError, match = "too large"):
        save_mod._resolve_gguf_shard_size("3GB")


def test_public_save_rejects_invalid_size_before_model_work():
    with pytest.raises(ValueError, match = "gguf_shard_size"):
        save_mod.unsloth_save_pretrained_gguf(
            object(),
            "unused",
            tokenizer = object(),
            gguf_shard_size = "1.5GB",
        )


@pytest.mark.parametrize(
    "first_conversion, methods, is_vlm, expected",
    [
        ("f16", ["f16"], False, "256MB"),
        ("bf16", ["bf16", "q4_k_m"], False, "256MB"),
        ("f32", ["f32"], False, "256MB"),
        ("q8_0", ["q8_0"], False, "0"),
        ("bf16", ["q4_k_m"], False, "0"),
        ("f16", ["f16"], True, "0"),
    ],
)
def test_converter_only_shards_final_full_precision_outputs(
    first_conversion, methods, is_vlm, expected
):
    assert (
        save_mod._converter_gguf_shard_size(
            "256MB",
            first_conversion,
            methods,
            is_vlm,
        )
        == expected
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("model.F16-mmproj.gguf", True),
        ("mmproj-model.F16.gguf", True),
        ("mtp-model.Q8_0.gguf", True),
        ("model.Q8_0-MTP.gguf", True),
        ("ordinary-mmprojector-model.gguf", False),
        ("ordinary-mtp-model.gguf", False),
        ("attempt.gguf", False),
        ("model-00001-of-00002.gguf", False),
    ],
)
def test_gguf_companion_detection_uses_exact_name_shapes(name, expected):
    assert save_mod._is_gguf_companion(name) is expected


def test_vlm_split_only_replaces_main_model(tmp_path, monkeypatch):
    main = tmp_path / "vision model ü.F16.gguf"
    mmproj = tmp_path / "vision model ü.BF16-mmproj.gguf"
    mtp = tmp_path / "mtp-vision model ü.Q8_0.gguf"
    main.write_bytes(b"m" * 2_000_001)
    mmproj.write_bytes(b"projector")
    mtp.write_bytes(b"drafter")

    monkeypatch.setattr(save_mod, "_find_llama_gguf_split", lambda _: "splitter")

    def fake_run(args, **kwargs):
        assert args[2:4] == ["--split-max-size", "1M"]
        prefix = Path(args[-1])
        (prefix.parent / f"{prefix.name}-00001-of-00002.gguf").write_bytes(b"one")
        (prefix.parent / f"{prefix.name}-00002-of-00002.gguf").write_bytes(b"two")
        return subprocess.CompletedProcess(args, 0, stdout = "ok", stderr = "")

    monkeypatch.setattr(save_mod.subprocess, "run", fake_run)
    output = save_mod._split_main_gguf(
        [str(main), str(mmproj), str(mtp)],
        "1MB",
        "quantizer",
    )

    assert [Path(path).name for path in output] == [
        "vision model ü.F16-00001-of-00002.gguf",
        "vision model ü.F16-00002-of-00002.gguf",
        mmproj.name,
        mtp.name,
    ]
    assert not main.exists()
    assert mmproj.read_bytes() == b"projector"
    assert mtp.read_bytes() == b"drafter"


def test_vlm_main_below_boundary_stays_single_file(tmp_path, monkeypatch):
    main = tmp_path / "model.F16.gguf"
    mmproj = tmp_path / "model.BF16-mmproj.gguf"
    main.write_bytes(b"main")
    mmproj.write_bytes(b"projector")
    monkeypatch.setattr(
        save_mod,
        "_find_llama_gguf_split",
        lambda _: pytest.fail("splitter should not be required"),
    )

    output = save_mod._split_main_gguf([str(main), str(mmproj)], "1MB", "quantizer")

    assert output == [str(main), str(mmproj)]


def test_vlm_split_failure_keeps_single_file(tmp_path, monkeypatch):
    main = tmp_path / "model.F16.gguf"
    mmproj = tmp_path / "model.BF16-mmproj.gguf"
    main.write_bytes(b"m" * 2_000_001)
    mmproj.write_bytes(b"projector")
    monkeypatch.setattr(save_mod, "_find_llama_gguf_split", lambda _: "splitter")

    def fail_run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr = "invalid gguf")

    monkeypatch.setattr(save_mod.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match = "invalid gguf"):
        save_mod._split_main_gguf([str(main), str(mmproj)], "1MB", "quantizer")
    assert main.exists()
    assert mmproj.exists()


def test_vlm_incomplete_split_keeps_single_file(tmp_path, monkeypatch):
    main = tmp_path / "model.F16.gguf"
    main.write_bytes(b"m" * 2_000_001)
    monkeypatch.setattr(save_mod, "_find_llama_gguf_split", lambda _: "splitter")

    def incomplete_run(args, **kwargs):
        prefix = Path(args[-1])
        (prefix.parent / f"{prefix.name}-00001-of-00002.gguf").write_bytes(b"one")
        return subprocess.CompletedProcess(args, 0, stdout = "ok", stderr = "")

    monkeypatch.setattr(save_mod.subprocess, "run", incomplete_run)

    with pytest.raises(RuntimeError, match = "incomplete shard set"):
        save_mod._split_main_gguf([str(main)], "1MB", "quantizer")
    assert main.exists()


def test_vlm_split_does_not_overwrite_existing_shards(tmp_path, monkeypatch):
    main = tmp_path / "model.F16.gguf"
    existing = tmp_path / "model.F16-00001-of-00002.gguf"
    main.write_bytes(b"m" * 2_000_001)
    existing.write_bytes(b"existing")
    monkeypatch.setattr(save_mod, "_find_llama_gguf_split", lambda _: "splitter")

    def fake_run(args, **kwargs):
        prefix = Path(args[-1])
        (prefix.parent / f"{prefix.name}-00001-of-00002.gguf").write_bytes(b"one")
        (prefix.parent / f"{prefix.name}-00002-of-00002.gguf").write_bytes(b"two")
        return subprocess.CompletedProcess(args, 0, stdout = "ok", stderr = "")

    monkeypatch.setattr(save_mod.subprocess, "run", fake_run)

    with pytest.raises(FileExistsError, match = "refusing to overwrite"):
        save_mod._split_main_gguf([str(main)], "1MB", "quantizer")
    assert main.exists()
    assert existing.read_bytes() == b"existing"
