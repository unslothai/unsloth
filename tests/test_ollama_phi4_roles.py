# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import os
import re
import runpy

MAPPERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "ollama_template_mappers.py",
)
OLLAMA_TEMPLATES = runpy.run_path(MAPPERS_PATH)["OLLAMA_TEMPLATES"]


def test_phi4_roles_are_plain_text():
    body = re.search(r'TEMPLATE """(.*?)"""', OLLAMA_TEMPLATES["phi-4"], re.S).group(1)
    # Phi-4's vocab has no <|user|>-style role tokens; they would tokenize as plain text.
    for role in ("system", "user", "assistant"):
        assert f"<|{role}|>" not in body
        assert f"<|im_start|>{role}<|im_sep|>" in body
