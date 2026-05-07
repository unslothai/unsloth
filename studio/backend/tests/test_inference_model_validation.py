# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from models.inference import LoadRequest


def _base_load_request(**overrides):
    data = {
        "model_path": "unsloth/test-model-GGUF",
        "hf_token": None,
        "max_seq_length": 4096,
        "load_in_4bit": True,
        "is_lora": False,
        "gguf_variant": "Q4_K_M",
    }
    data.update(overrides)
    return LoadRequest.model_validate(data)


def test_blank_chat_template_override_normalizes_to_none():
    req = _base_load_request(chat_template_override = "   \n\t")

    assert req.chat_template_override is None


def test_nonblank_chat_template_override_is_preserved_verbatim():
    template = "  {{ messages }}  "
    req = _base_load_request(chat_template_override = template)

    assert req.chat_template_override == template
