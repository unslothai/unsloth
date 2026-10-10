# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""diffusers 0.40 LTX-2 pipelines must import on the pinned transformers 5.5.

diffusers 0.40.0 and main import ``Gemma4UnifiedForConditionalGeneration`` at module level in every
``diffusers.pipelines.ltx2.pipeline_ltx2*`` module, and that class only exists from transformers 5.10. On the Studio
pin every LTX-2 / LTX-2.3 load failed with ``cannot import name 'Gemma4UnifiedForConditionalGeneration' from
'transformers'``. These tests build a fake transformers and a fake lazy diffusers with the same shape (including
transformers replacing its own ``sys.modules`` entry when ``processing_utils`` is first imported) so they run
hermetically on CPU without either package installed.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

from core.inference import ltx2_import_compat as compat

BACKEND = Path(__file__).resolve().parents[1]
NAME = "Gemma4UnifiedForConditionalGeneration"

_TRANSFORMERS_INIT = """
import sys
import types


class Gemma3ForConditionalGeneration:
    pass


class ProcessorMixin:
    pass

{extra}
"""

# Mirrors transformers: processing_utils import replaces sys.modules['transformers'].
_PROCESSING_UTILS = """
import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "transformers", os.path.join(_here, "__init__.py"), submodule_search_locations=[_here]
)
_fresh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fresh)
sys.modules["transformers"] = _fresh
"""

_DIFFUSERS_INIT = """
import importlib

__version__ = "0.40.0"
_LAZY = {
    "LTX2Pipeline": "diffusers.pipelines.ltx2.pipeline_ltx2",
    "LTX2ImageToVideoPipeline": "diffusers.pipelines.ltx2.pipeline_ltx2_image2video",
    "LTX2VideoTransformer3DModel": "diffusers.models_stub",
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    try:
        module = importlib.import_module(_LAZY[name])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import {_LAZY[name]} because of the following error (look up to see its traceback):\\n{exc}"
        ) from exc
    return getattr(module, name)
"""

# Real transformers swaps while resolving Gemma3, before Gemma4Unified is looked up.
_PIPELINE = """
from transformers import processing_utils  # noqa: F401
from transformers import (
    Gemma3ForConditionalGeneration,
    Gemma4UnifiedForConditionalGeneration,
    ProcessorMixin,
)


class {cls}:
    def __init__(self, text_encoder: Gemma3ForConditionalGeneration | Gemma4UnifiedForConditionalGeneration):
        self.text_encoder = text_encoder
"""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(textwrap.dedent(text), encoding = "utf-8")


@pytest.fixture
def fake_stack(tmp_path, monkeypatch):
    """Build fake transformers + diffusers under tmp_path and isolate sys.modules for them."""

    def build(*, transformers_has_class: bool) -> Path:
        root = tmp_path / "site"
        extra = f"class {NAME}:\n    real = True\n" if transformers_has_class else ""
        _write(root / "transformers" / "__init__.py", _TRANSFORMERS_INIT.format(extra = extra))
        _write(root / "transformers" / "processing_utils.py", _PROCESSING_UTILS)
        _write(root / "diffusers" / "__init__.py", _DIFFUSERS_INIT)
        _write(
            root / "diffusers" / "models_stub.py", "class LTX2VideoTransformer3DModel:\n    pass\n"
        )
        _write(root / "diffusers" / "pipelines" / "__init__.py", "")
        _write(root / "diffusers" / "pipelines" / "ltx2" / "__init__.py", "")
        _write(
            root / "diffusers" / "pipelines" / "ltx2" / "connectors.py",
            "class LTX2TextConnectors:\n    pass\n",
        )
        _write(
            root / "diffusers" / "pipelines" / "ltx2" / "pipeline_ltx2.py",
            _PIPELINE.format(cls = "LTX2Pipeline"),
        )
        _write(
            root / "diffusers" / "pipelines" / "ltx2" / "pipeline_ltx2_image2video.py",
            _PIPELINE.format(cls = "LTX2ImageToVideoPipeline"),
        )
        monkeypatch.syspath_prepend(str(root))
        return root

    saved = {
        k: v
        for k, v in sys.modules.items()
        if k in ("transformers", "diffusers") or k.startswith(("transformers.", "diffusers."))
    }
    for key in saved:
        del sys.modules[key]
    compat._reset_for_tests()
    try:
        yield build
    finally:
        for key in [
            k
            for k in sys.modules
            if k in ("transformers", "diffusers") or k.startswith(("transformers.", "diffusers."))
        ]:
            del sys.modules[key]
        sys.modules.update(saved)
        compat._reset_for_tests()


def test_fake_stack_reproduces_the_failure(fake_stack):
    """The unpatched import fails the way the Colab run did, so the tests below prove something."""
    fake_stack(transformers_has_class = False)
    import diffusers

    with pytest.raises(RuntimeError, match = NAME):
        diffusers.LTX2Pipeline  # noqa: B018


def test_ensure_makes_every_ltx2_pipeline_importable(fake_stack):
    fake_stack(transformers_has_class = False)
    original = __import__("transformers")

    assert compat.ensure_ltx2_pipelines_importable() is True

    import diffusers

    assert diffusers.LTX2Pipeline.__name__ == "LTX2Pipeline"
    assert diffusers.LTX2ImageToVideoPipeline.__name__ == "LTX2ImageToVideoPipeline"
    assert sys.modules["transformers"] is not original
    assert NAME not in sys.modules["transformers"].__dict__
    assert NAME not in original.__dict__
    pipe = diffusers.LTX2Pipeline(
        text_encoder = sys.modules["transformers"].Gemma3ForConditionalGeneration()
    )
    assert pipe.text_encoder is not None
    assert compat.ensure_ltx2_pipelines_importable() is True


def test_slow_import_mode_package_import_is_shimmed_too(fake_stack):
    """With DIFFUSERS_SLOW_IMPORT the ltx2 package __init__ imports the pipelines eagerly, before any target list."""
    import importlib

    root = fake_stack(transformers_has_class = False)
    _write(
        root / "diffusers" / "pipelines" / "ltx2" / "__init__.py",
        "from .pipeline_ltx2 import LTX2Pipeline  # noqa: F401\n",
    )
    assert compat.ensure_ltx2_pipelines_importable() is True
    module = importlib.import_module("diffusers.pipelines.ltx2.pipeline_ltx2")
    assert module.LTX2Pipeline is not None
    assert NAME not in sys.modules["transformers"].__dict__


def test_placeholder_refuses_to_load(fake_stack):
    fake_stack(transformers_has_class = False)
    assert compat.ensure_ltx2_pipelines_importable() is True
    import diffusers.pipelines.ltx2.pipeline_ltx2 as mod

    stand_in = getattr(mod, NAME)
    with pytest.raises(ImportError, match = r"transformers >= 5\.10"):
        stand_in.from_pretrained("any/repo")
    with pytest.raises(ImportError, match = r"transformers >= 5\.10"):
        stand_in()


def test_new_transformers_is_left_alone(fake_stack):
    fake_stack(transformers_has_class = True)
    assert compat.ensure_ltx2_pipelines_importable() is True
    assert "diffusers.pipelines.ltx2.pipeline_ltx2" not in sys.modules
    import diffusers.pipelines.ltx2.pipeline_ltx2 as mod

    assert getattr(getattr(mod, NAME), "real", False) is True


def test_no_diffusers_is_not_an_error(fake_stack):
    """A host without diffusers (sd.cpp-only) gets False, never an exception."""
    fake_stack(transformers_has_class = False)
    sys.modules["diffusers"] = None  # blocks the import; the fixture restores sys.modules
    assert compat.ensure_ltx2_pipelines_importable() is False


def test_studio_availability_check_accepts_ltx2_on_old_transformers(fake_stack):
    """The training preflight (strict) used to refuse LTX-2 outright with 'cannot import'."""
    fake_stack(transformers_has_class = False)
    from core.inference.diffusion_families import assert_pipeline_class_available

    assert_pipeline_class_available("LTX2Pipeline", "ltx-2", strict = True)


def test_is_ltx2_pipeline_class():
    assert compat.is_ltx2_pipeline_class("LTX2Pipeline")
    assert compat.is_ltx2_pipeline_class("LTX2ImageToVideoPipeline")
    assert not compat.is_ltx2_pipeline_class("LTX2VideoTransformer3DModel")
    assert not compat.is_ltx2_pipeline_class("WanPipeline")
    assert not compat.is_ltx2_pipeline_class(None)


def _functions_importing_ltx2_pipeline(path: Path) -> dict[str, list[int]]:
    """Function name -> line numbers of ``from diffusers import LTX2*Pipeline`` in it."""
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    hits: dict[str, list[int]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "diffusers"
                and any(compat.is_ltx2_pipeline_class(a.name) for a in node.names)
            ):
                hits.setdefault(fn.name, []).append(node.lineno)
    return hits


def _calls_ensure_before(fn_src: str) -> bool:
    ensure = fn_src.find("ensure_ltx2_pipelines_importable(")
    imp = fn_src.find("from diffusers import LTX2")
    return ensure != -1 and (imp == -1 or ensure < imp)


@pytest.mark.parametrize(
    "rel",
    [
        "core/inference/video_ltx2.py",
        "core/inference/video.py",
        "core/training/diffusion_dit_trainer.py",
    ],
)
def test_every_ltx2_pipeline_import_goes_through_the_shim(rel):
    path = BACKEND / rel
    source = path.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    funcs = {
        n.name: ast.get_source_segment(source, n)
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in _functions_importing_ltx2_pipeline(path):
        assert _calls_ensure_before(
            funcs[name]
        ), f"{rel}:{name} imports an LTX-2 pipeline without the shim"


def test_video_backend_pipeline_getattr_goes_through_the_shim():
    source = (BACKEND / "core/inference/video.py").read_text(encoding = "utf-8")
    anchor = source.index("pipeline_cls = getattr(diffusers, fam.pipeline_class)")
    window = source[max(0, anchor - 400) : anchor]
    assert "ensure_ltx2_pipelines_importable(" in window
