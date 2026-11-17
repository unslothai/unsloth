# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The load requests' speed_mode text states what an unset speed_mode resolves to."""

import ast
from pathlib import Path

from core.inference.diffusion_speed import SPEED_DEFAULT, SPEED_OFF, resolve_speed_mode

_BACKEND = Path(__file__).resolve().parents[1]


def _speed_mode_description(class_name: str) -> str:
    tree = ast.parse((_BACKEND / "models" / "inference.py").read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    field = next(
        n
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) == "speed_mode"
    )
    desc = next(k.value for k in field.value.keywords if k.arg == "description")
    return ast.literal_eval(desc)


def _video_unset() -> tuple[str, str]:
    # video.py resolves an unset speed_mode with dense_default = SPEED_DEFAULT for every kind.
    src = (_BACKEND / "core" / "inference" / "video.py").read_text(encoding = "utf-8")
    assert 'speed_mode, is_gguf = kind == "gguf", dense_default = SPEED_DEFAULT' in src
    return (
        resolve_speed_mode(None, is_gguf = False, dense_default = SPEED_DEFAULT),
        resolve_speed_mode(None, is_gguf = True, dense_default = SPEED_DEFAULT),
    )


def test_video_speed_mode_text_says_unset_is_default():
    assert _video_unset() == (SPEED_DEFAULT, SPEED_DEFAULT)
    text = _speed_mode_description("VideoLoadRequest")
    assert "Unset: default for every video load" in text
    assert "default off" not in text


def test_image_speed_mode_text_says_unset_is_off_for_dense_and_default_for_gguf():
    assert resolve_speed_mode(None, is_gguf = False) == SPEED_OFF
    assert resolve_speed_mode(None, is_gguf = True) == SPEED_DEFAULT
    text = _speed_mode_description("DiffusionLoadRequest")
    assert "Unset: off for a dense load, default for a GGUF load" in text
    assert "default off" not in text
