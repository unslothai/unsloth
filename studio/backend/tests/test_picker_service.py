# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from types import SimpleNamespace

from picker.service import (
    MAX_TEMPLATE_METADATA_BYTES,
    _chat_template_from_dir,
    _chat_template_from_processor_json,
    _chat_template_from_tokenizer_config,
    _chat_template_from_tokenizer_dir,
    _find_gguf_in_dir,
    _iter_ggufs,
    read_default_chat_template,
    validate_chat_template,
)


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


def test_find_gguf_in_dir_without_variant_prefers_first_split(tmp_path):
    first = tmp_path / "model-Q4_K_M-00001-of-00003.gguf"
    second = tmp_path / "model-Q4_K_M-00002-of-00003.gguf"
    third = tmp_path / "model-Q4_K_M-00003-of-00003.gguf"
    first.write_bytes(b"0")
    second.write_bytes(b"000")
    third.write_bytes(b"00")

    assert _find_gguf_in_dir(tmp_path, None) == first

    first.unlink()
    assert _find_gguf_in_dir(tmp_path, None) == second


def test_find_gguf_in_dir_matches_bpw_variant_base_label(tmp_path):
    target = tmp_path / "model-IQ4_XS-3.53bpw.gguf"
    target.write_bytes(b"")
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"")

    assert _find_gguf_in_dir(tmp_path, "IQ4_XS") == target
    assert _find_gguf_in_dir(tmp_path, "IQ4_XS-3.53bpw") == target
    assert _find_gguf_in_dir(tmp_path, "Q4_K") is None


def test_validate_chat_template_accepts_valid_and_empty():
    assert validate_chat_template("{{ messages[0].content }}").valid is True
    assert validate_chat_template("").valid is True
    assert validate_chat_template("   ").valid is True


def test_validate_chat_template_reports_syntax_error_with_line():
    result = validate_chat_template("{% if %}{% endif %}")
    assert result.valid is False
    assert result.error is not None
    assert result.error.startswith("Line ")


def test_chat_template_from_tokenizer_config_reads_string():
    assert _chat_template_from_tokenizer_config({"chat_template": "HELLO"}) == "HELLO"
    assert _chat_template_from_tokenizer_config({"chat_template": "   "}) is None
    assert _chat_template_from_tokenizer_config({}) is None


def test_chat_template_from_tokenizer_config_prefers_named_default():
    config = {
        "chat_template": [
            {"name": "tool_use", "template": "TOOL"},
            {"name": "default", "template": "DEFAULT"},
        ]
    }
    assert _chat_template_from_tokenizer_config(config) == "DEFAULT"


def test_chat_template_from_tokenizer_config_falls_back_to_first_entry():
    config = {
        "chat_template": [
            {"name": "tool_use", "template": "TOOL"},
            {"name": "other", "template": "OTHER"},
        ]
    }
    assert _chat_template_from_tokenizer_config(config) == "TOOL"


def test_chat_template_from_tokenizer_dir_prefers_jinja_file(tmp_path):
    (tmp_path / "chat_template.jinja").write_text("FROM_JINJA", encoding = "utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "FROM_CONFIG"}), encoding = "utf-8"
    )
    assert _chat_template_from_tokenizer_dir(tmp_path) == "FROM_JINJA"


def test_chat_template_from_tokenizer_dir_reads_tokenizer_config(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "FROM_CONFIG"}), encoding = "utf-8"
    )
    assert _chat_template_from_tokenizer_dir(tmp_path) == "FROM_CONFIG"


def test_chat_template_from_dir_without_variant_prefers_tokenizer(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "FROM_CONFIG"}), encoding = "utf-8"
    )
    assert _chat_template_from_dir(tmp_path) == "FROM_CONFIG"


def test_chat_template_from_dir_with_variant_still_prefers_tokenizer(tmp_path, monkeypatch):
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "FROM_CONFIG"}), encoding = "utf-8"
    )
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"")
    monkeypatch.setattr("picker.service.read_gguf_chat_template", lambda _path: "FROM_GGUF")
    # Selecting a variant must not flip precedence to the embedded GGUF template.
    assert _chat_template_from_dir(tmp_path, "Q4_K_M") == "FROM_CONFIG"


def test_chat_template_from_dir_with_variant_falls_back_to_gguf(tmp_path, monkeypatch):
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"")
    monkeypatch.setattr("picker.service.read_gguf_chat_template", lambda _path: "FROM_GGUF")
    # With no tokenizer sidecar, the embedded GGUF template is still the fallback.
    assert _chat_template_from_dir(tmp_path, "Q4_K_M") == "FROM_GGUF"


def test_chat_template_from_dir_returns_none_when_absent(tmp_path):
    assert _chat_template_from_dir(tmp_path) is None


def test_read_default_chat_template_direct_gguf_prefers_sidecar(tmp_path, monkeypatch):
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "FROM_CONFIG"}), encoding = "utf-8"
    )
    monkeypatch.setattr("picker.service._build_browse_allowlist", lambda: [tmp_path])
    monkeypatch.setattr("picker.service.read_gguf_chat_template", lambda _path: "FROM_GGUF")
    # A directly selected .gguf must prefer a maintained sidecar over its embedded copy.
    assert read_default_chat_template(str(gguf)) == "FROM_CONFIG"


def test_read_default_chat_template_direct_gguf_falls_back_to_embedded(tmp_path, monkeypatch):
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"")
    monkeypatch.setattr("picker.service._build_browse_allowlist", lambda: [tmp_path])
    monkeypatch.setattr("picker.service.read_gguf_chat_template", lambda _path: "FROM_GGUF")
    # With no sidecar next to the file, the embedded GGUF template is the fallback.
    assert read_default_chat_template(str(gguf)) == "FROM_GGUF"


def test_tokenizer_config_over_size_limit_is_skipped_not_parsed(tmp_path):
    # An oversized tokenizer_config.json must be skipped before json.loads so a
    # hostile sidecar cannot exhaust memory.
    padding = "x" * (MAX_TEMPLATE_METADATA_BYTES + 1024)
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "HELLO", "_pad": padding}), encoding = "utf-8"
    )
    assert _chat_template_from_tokenizer_dir(tmp_path) is None


def test_processor_json_over_size_limit_is_skipped_not_parsed(tmp_path):
    padding = "x" * (MAX_TEMPLATE_METADATA_BYTES + 1024)
    (tmp_path / "chat_template.json").write_text(
        json.dumps({"default": "HELLO", "_pad": padding}), encoding = "utf-8"
    )
    assert _chat_template_from_processor_json(tmp_path) is None


def test_tokenizer_config_at_size_limit_is_still_read(tmp_path):
    # A normal-sized config is unaffected by the bound (regression guard).
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "FROM_CONFIG"}), encoding = "utf-8"
    )
    assert _chat_template_from_tokenizer_dir(tmp_path) == "FROM_CONFIG"


def test_remote_template_over_size_limit_is_skipped_before_download(monkeypatch):
    # An uncached Hub repo whose template exceeds the cap must be skipped via the
    # remote size pre-check, never downloaded.
    import huggingface_hub

    monkeypatch.setattr("picker.service.resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr("picker.service.iter_hf_cache_snapshots", lambda resolved: [])

    def _fail_download(*args, **kwargs):
        raise AssertionError("oversized remote template must not be downloaded")

    def _fake_get_paths_info(self, repo_id, paths, **kwargs):
        return [SimpleNamespace(path = p, size = MAX_TEMPLATE_METADATA_BYTES + 1) for p in paths]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fail_download)
    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", _fake_get_paths_info)

    assert read_default_chat_template("org/oversized-model") is None


def test_remote_oversized_jinja_falls_through_to_tokenizer_template(tmp_path, monkeypatch):
    # A raw chat_template.jinja between the response cap (MAX_CHAT_TEMPLATE_BYTES)
    # and the download bound (MAX_TEMPLATE_METADATA_BYTES) must not be returned: the
    # route drops it, so the remote path must skip the oversized Jinja and fall
    # through to the smaller tokenizer_config.json.
    import huggingface_hub
    from picker.schemas import MAX_CHAT_TEMPLATE_BYTES

    big_jinja = tmp_path / "chat_template.jinja"
    big_jinja.write_text("{{ x }}" * (MAX_CHAT_TEMPLATE_BYTES // 4), encoding = "utf-8")
    assert MAX_CHAT_TEMPLATE_BYTES < big_jinja.stat().st_size < MAX_TEMPLATE_METADATA_BYTES
    tokenizer_config = tmp_path / "tokenizer_config.json"
    tokenizer_config.write_text(json.dumps({"chat_template": "SMALL_TEMPLATE"}), encoding = "utf-8")
    files = {
        "chat_template.jinja": big_jinja,
        "tokenizer_config.json": tokenizer_config,
    }

    monkeypatch.setattr("picker.service.resolve_cached_repo_id_case", lambda name: name)
    monkeypatch.setattr("picker.service.iter_hf_cache_snapshots", lambda resolved: [])

    def _fake_download(repo_id, rel, **kwargs):
        target = files.get(rel)
        if target is None:
            raise FileNotFoundError(rel)
        return str(target)

    def _fake_get_paths_info(self, repo_id, paths, **kwargs):
        return [
            SimpleNamespace(
                path = p,
                size = files[p].stat().st_size if p in files else 0,
            )
            for p in paths
        ]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    monkeypatch.setattr(huggingface_hub.HfApi, "get_paths_info", _fake_get_paths_info)

    assert read_default_chat_template("org/big-jinja-model") == "SMALL_TEMPLATE"
