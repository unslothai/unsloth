# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The decision-model GGUF converter never imports a checkpoint's auto_map code.

The converter loads several architectures' tokenizers with trust_remote_code=True; decision
backbones are built-in classes, so the converter child gets no usable modules cache.
"""

from __future__ import annotations

import os

import pytest

from unsloth.models import decision_gguf

_KEY = "HF_MODULES_CACHE"


def test_the_decision_converter_gets_no_usable_modules_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(_KEY, str(tmp_path / "modules"))
    outfile = tmp_path / "model.gguf"
    seen = {}

    def _run(command, what, print_output, env):
        seen.update(env)
        outfile.write_bytes(b"GGUF")

    monkeypatch.setattr(decision_gguf, "_run", _run)
    decision_gguf._convert(tmp_path, tmp_path / "checkpoint", "f16", outfile, False, False)
    assert seen[_KEY] != str(tmp_path / "modules")
    with pytest.raises(OSError):
        os.makedirs(seen[_KEY], exist_ok = True)
    assert os.environ[_KEY] == str(tmp_path / "modules")
