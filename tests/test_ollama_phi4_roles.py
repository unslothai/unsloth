# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Phi-4 Ollama template writes roles as plain text, like the HF chat template and ollama.com/library/phi4."""

import os
import re
import runpy

MAPPERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "ollama_template_mappers.py",
)
OLLAMA_TEMPLATES = runpy.run_path(MAPPERS_PATH)["OLLAMA_TEMPLATES"]


def _template_body(modelfile):
    return re.search(r'TEMPLATE """(.*?)"""', modelfile, re.S).group(1)


def test_phi4_roles_are_plain_text():
    body = _template_body(OLLAMA_TEMPLATES["phi-4"])
    # Phi-4's vocabulary has no <|system|> / <|user|> / <|assistant|> tokens, so these would
    # be split into several ordinary tokens on every turn.
    for role in ("system", "user", "assistant"):
        assert f"<|{role}|>" not in body
        assert f"<|im_start|>{role}<|im_sep|>" in body
