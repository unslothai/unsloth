# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Old _supports_flash_attn_2 flag counts only where the installed dispatch still reads it (< 5.4.0)."""

import ast
import functools
import inspect
import pathlib
from unittest import mock

import pytest
import transformers
from transformers import PreTrainedModel

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
NAME = "_transformers_honors_legacy_flash_attn_2_flag"


def _load():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == NAME]
    assert len(nodes) == 1
    namespace = {"functools": functools, "inspect": inspect}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE), "exec"), namespace)
    return namespace[NAME]


def _probe_matches_dispatch():
    check = getattr(PreTrainedModel, "_flash_attn_can_dispatch", None) or getattr(
        PreTrainedModel, "_flash_attn_2_can_dispatch"
    )
    return "_supports_flash_attn_2" in inspect.getsource(check)


def test_probe_agrees_with_installed_transformers():
    assert _load()() == _probe_matches_dispatch()


def _version():
    return tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())


@pytest.mark.skipif(_version() < (5, 4), reason = "the dispatch reads the old flag before 5.4.0")
def test_transformers_5_4_does_not_honor_the_legacy_flag():
    assert _load()() is False


@pytest.mark.skipif(_version() >= (5, 4), reason = "5.4.0 stopped reading the old flag")
def test_transformers_before_5_4_honors_the_legacy_flag():
    assert _load()() is True


def test_class_level_legacy_attribute_means_honored():
    class Old:
        _supports_flash_attn_2 = False

    with mock.patch.object(transformers, "PreTrainedModel", Old):
        assert _load()() is True


def test_dispatch_that_reads_the_legacy_flag_means_honored():
    class Mid:
        _supports_flash_attn = False

        def _flash_attn_2_can_dispatch(self):
            return self._supports_flash_attn or getattr(self, "_supports_flash_attn_2", False)

    with mock.patch.object(transformers, "PreTrainedModel", Mid):
        assert _load()() is True


def test_dispatch_that_ignores_the_legacy_flag_means_not_honored():
    class New:
        _supports_flash_attn = False

        def _flash_attn_can_dispatch(self):
            return self._supports_flash_attn

    with mock.patch.object(transformers, "PreTrainedModel", New):
        assert _load()() is False


def test_unknown_layout_keeps_previous_behavior():
    class Unknown:
        pass

    with mock.patch.object(transformers, "PreTrainedModel", Unknown):
        assert _load()() is True
