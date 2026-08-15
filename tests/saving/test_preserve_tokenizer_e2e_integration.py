# SPDX-License-Identifier: AGPL-3.0-only
"""Integration e2e for #8444 — real HF tokenizer export roundtrip."""

import json
import tempfile
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from unsloth.save import _preserve_tokenizer_class


@pytest.mark.integration
def test_real_bge_tokenizer_export_roundtrip():
    base = "unsloth/bge-small-en-v1.5"
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code = True)
    orig_class = tok.__class__.__name__

    with tempfile.TemporaryDirectory() as td:
        export = Path(td) / "export"
        export.mkdir()
        tok.save_pretrained(export)
        cfg_path = export / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding = "utf-8"))
        cfg["tokenizer_class"] = "TokenizersBackend"
        cfg_path.write_text(json.dumps(cfg, indent = 2), encoding = "utf-8")

        _preserve_tokenizer_class(tok, export)
        fixed = json.loads(cfg_path.read_text(encoding = "utf-8"))
        assert fixed["tokenizer_class"] != "TokenizersBackend"
        assert fixed["tokenizer_class"] in {orig_class, "BertTokenizer", "BertTokenizerFast"}

        reloaded = AutoTokenizer.from_pretrained(export, trust_remote_code = True)
        ids = reloaded("hello world")["input_ids"]
        assert len(ids) > 0
