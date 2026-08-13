# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The training worker installs the soundfile decoder before it reads any row.

Deliberately dependency-free, unlike test_audio_dataset_decode.py: that module
importorskips soundfile and librosa, so on a host with neither it skips, and a
host with neither is exactly where this ordering is load-bearing.
"""

from __future__ import annotations

import ast
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_TRAINER = _BACKEND / "core/training/trainer.py"


def _load_and_format_dataset_body() -> str:
    text = _TRAINER.read_text(encoding = "utf-8")
    return text[text.index("    def load_and_format_dataset(") :]


def _calls_the_shim(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ensure_audio_decoding"
        for n in ast.walk(node)
    )


def test_the_shim_is_installed_before_the_first_row_is_read():
    # The format-check endpoint installs it in the API process. This worker is a
    # separate process and starts without it, so an Audio column decoding inside
    # load_dataset() raised `datasets`' own "please install 'torchcodec'" before
    # the audio branches further down could install anything.
    body = _load_and_format_dataset_body()
    assert body.index("ensure_audio_decoding()") < body.index("= load_dataset(")


def test_the_audio_branches_are_still_covered():
    # They call it themselves and report the FFmpeg-naming failure, which stays the
    # message a caller sees when neither backend works. This only has to precede them.
    body = _load_and_format_dataset_body()
    assert body.index("ensure_audio_decoding()") < body.index(
        "# ========== AUDIO MODELS: custom preprocessing =========="
    )


def test_the_import_is_module_level():
    # A local import inside the audio branch would leave the early call a NameError.
    text = _TRAINER.read_text(encoding = "utf-8")
    assert "\nfrom utils.datasets.audio_decode import ensure_audio_decoding\n" in text


def test_the_early_call_cannot_stop_a_text_run():
    # It sits above the method's own try, so anything ensure_audio_decoding() does not
    # catch (`import librosa` raises more than ImportError) would escape the method and
    # fail every run, audio or not. The audio branches below re-run it and report.
    fn = next(
        node
        for node in ast.walk(ast.parse(_TRAINER.read_text(encoding = "utf-8")))
        if isinstance(node, ast.FunctionDef) and node.name == "load_and_format_dataset"
    )
    first = next(stmt for stmt in fn.body if _calls_the_shim(stmt))
    assert isinstance(first, ast.Try), "the early call is not wrapped"
    # Directly in the try body, not merely somewhere inside a larger block.
    assert any(isinstance(b, ast.Expr) and _calls_the_shim(b) for b in first.body)
    assert any(getattr(h.type, "id", "") == "Exception" for h in first.handlers)


def test_a_datasets_without_the_torchcodec_flag_returns_a_bool():
    # `datasets` < 4 has no config.TORCHCODEC_AVAILABLE, and pyproject still allows
    # datasets>=3.4.1. Reading the attribute raised AttributeError, which the function
    # catches nowhere and its callers do not expect: the call in the audio branch is
    # unguarded, so an audio run on such an install died reporting a missing config
    # attribute rather than training. Those versions decode through soundfile already,
    # so the answer is True and nothing is patched.
    import sys
    import types

    from utils.datasets import audio_decode

    fake_config = types.SimpleNamespace()  # no TORCHCODEC_AVAILABLE, as on datasets 3.x
    fake_audio = types.ModuleType("datasets.features.audio")
    fake_audio.Audio = type("Audio", (), {"decode_example": None, "encode_example": None})
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.config = fake_config
    fake_features = types.ModuleType("datasets.features")

    saved = {
        k: sys.modules.get(k) for k in ("datasets", "datasets.features", "datasets.features.audio")
    }
    sys.modules["datasets"] = fake_datasets
    sys.modules["datasets.features"] = fake_features
    sys.modules["datasets.features.audio"] = fake_audio
    installed_before = audio_decode._installed
    try:
        assert audio_decode.ensure_audio_decoding() is True
        assert audio_decode._installed == installed_before, "patched a version that works"
        assert fake_audio.Audio.decode_example is None, "patched datasets<4"
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
