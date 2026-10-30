# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import importlib.util
import random
import re
from pathlib import Path

_CHUNKING = (
    Path(__file__).resolve().parent.parent
    / "plugins"
    / "data-designer-unstructured-seed"
    / "src"
    / "data_designer_unstructured_seed"
    / "chunking.py"
)


def _load_chunking():
    spec = importlib.util.spec_from_file_location("_uss_blank_lines_under_test", _CHUNKING)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chunking = _load_chunking()


def _previous_normalize(text):
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def test_normalize_matches_previous_output():
    rng = random.Random(0)
    alphabet = ["\n", "\n", "\r", "\r\n", "a", " ", "\t", "\x85", " ", "é", "\n\n\n"]
    cases = ["", "\n", "\n\n", "\n\n\n", "a\n\n\n\n\nb", "\r\r\r", " \n\n\n ", "x" * 50]
    cases += [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))) for _ in range(20000)
    ]
    old = re.compile(r"\n{3,}")
    for text in cases:
        assert chunking.normalize_unstructured_text(text) == _previous_normalize(text), repr(text)
        assert [m.span() for m in chunking._BLANK_LINE_RUN.finditer(text)] == [
            m.span() for m in old.finditer(text)
        ], repr(text)
