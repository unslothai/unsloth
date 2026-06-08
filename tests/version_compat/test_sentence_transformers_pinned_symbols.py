# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check across sentence-transformers PyPI minor
versions. unsloth has a custom integration in
unsloth/models/sentence_transformer.py that:

  - Imports SentenceTransformer / SentenceTransformerTrainer at the
    top of the public surface (lines 1467, 1798, 1947, 2154).
  - Walks `sentence_transformers.models` for Transformer / Pooling /
    Normalize (lines 1016, 1206, 1467).
  - Calls `sentence_transformers.util.import_from_string` and
    `load_dir_path` (lines 1177, 1205).
  - Tolerates two alternate base-class paths
    (sentence_transformers.base.modules.transformer.Transformer vs
    sentence_transformers.models.transformer.Transformer; lines
    1169-1171) — at least ONE must resolve.

Strategy: GitHub raw fetch + symbol grep (no pip install, runs CPU-only
on every PR + daily cron). Versioning policy: ST is unpinned in
unsloth/pyproject.toml; cover the most recent minors (5.x line) plus
`main`.
"""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def


# Policy: unsloth/pyproject.toml does NOT pin sentence-transformers. We
# track the last few minors plus main. Add a row when a new minor lands.
ST_TAGS = [
    "v5.0.0",
    "v5.1.2",
    "v5.2.3",
    "v5.3.0",
    "v5.4.1",
    "master",
]


# -------------------------------------------------------------------------
# Top-level public surface: SentenceTransformer + SentenceTransformerTrainer
# must be importable as `from sentence_transformers import X`.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ST_TAGS)
def test_st_top_level_exports(tag: str):
    src = fetch_text(
        "UKPLab/sentence-transformers", tag, "sentence_transformers/__init__.py"
    )
    assert src is not None, f"{tag}: sentence_transformers/__init__.py missing"
    needed = ("SentenceTransformer", "SentenceTransformerTrainer")
    missing = [n for n in needed if n not in src]
    assert not missing, (
        f"{tag}: sentence_transformers top-level missing {missing}; "
        f"unsloth.models.sentence_transformer:1467,2154 will ImportError"
    )


# -------------------------------------------------------------------------
# Sub-modules: Transformer / Pooling / Normalize. unsloth walks
# `sentence_transformers.models` to introspect these (line 1016, 1206).
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ST_TAGS)
def test_st_models_re_exports(tag: str):
    """Transformer / Pooling / Normalize must be reachable through
    `sentence_transformers.models`. ST 5.4 reorganised the package
    (no more top-level `models/` dir; modules live under
    `sentence_transformer/` and `base/modules/`), but the public
    re-export at `sentence_transformers/__init__.py` still has to
    surface these three so user code (and unsloth/models/sentence_transformer.py:1016,1206,1467)
    can `from sentence_transformers.models import Transformer` (or
    equivalently `from sentence_transformers import models`)."""
    # Layout 1 (legacy &lt; 5.4): sentence_transformers/models[.py|/__init__.py].
    # Layout 2 (&gt;= 5.4): top-level __init__.py re-exports the symbols
    # plus the modules live under base/modules and sentence_transformer/.
    legacy_candidates = [
        "sentence_transformers/models/__init__.py",
        "sentence_transformers/models.py",
    ]
    legacy_hit = first_match("UKPLab/sentence-transformers", tag, legacy_candidates)
    needed = ("Transformer", "Pooling", "Normalize")
    if legacy_hit is not None:
        _path, src = legacy_hit
        missing = [n for n in needed if n not in src]
        assert not missing, (
            f"{tag}: legacy sentence_transformers/models layout missing "
            f"{missing}; unsloth.models.sentence_transformer:1016,1206,1467 "
            f"ImportError"
        )
        return

    # ST 5.4+ modular layout: classes moved under
    #   - sentence_transformers/base/modules/transformer.py             (Transformer)
    #   - sentence_transformers/sentence_transformer/modules/pooling.py (Pooling)
    #   - sentence_transformers/sentence_transformer/modules/normalize.py (Normalize)
    # Backward compatibility for `from sentence_transformers.models
    # import X` is set up at import time via
    # `sentence_transformers.util.deprecated_import.setup_deprecated_module_imports`
    # called from sentence_transformers/__init__.py.
    expected_paths = {
        "Transformer": [
            "sentence_transformers/base/modules/transformer.py",
            "sentence_transformers/sentence_transformer/Transformer.py",
            "sentence_transformers/sentence_transformer/transformer.py",
        ],
        "Pooling": [
            "sentence_transformers/sentence_transformer/modules/pooling.py",
            "sentence_transformers/sentence_transformer/Pooling.py",
        ],
        "Normalize": [
            "sentence_transformers/sentence_transformer/modules/normalize.py",
            "sentence_transformers/sentence_transformer/Normalize.py",
        ],
    }
    for cls, paths in expected_paths.items():
        for p in paths:
            src = fetch_text("UKPLab/sentence-transformers", tag, p)
            if src and has_def(src, cls, "class"):
                break
        else:
            pytest.fail(
                f"{tag}: ST 5.4+ layout: class {cls} not found in any of {paths}"
            )

    # The backward-compat shim must be wired up so user code doing
    # `from sentence_transformers.models import Pooling` keeps working.
    top = fetch_text(
        "UKPLab/sentence-transformers", tag, "sentence_transformers/__init__.py"
    )
    assert top is not None, f"{tag}: sentence_transformers/__init__.py missing"
    has_shim = bool(
        re.search(r"setup_deprecated_module_imports\s*\(", top)
        or "import_from_string" in top  # fallback signal
    )
    assert has_shim, (
        f"{tag}: ST 5.4+ layout: deprecated-module shim NOT wired in "
        f"sentence_transformers/__init__.py; `from "
        f"sentence_transformers.models import Pooling` will ImportError "
        f"on real install"
    )


# -------------------------------------------------------------------------
# Transformer base class: unsloth checks two alternate paths at
# sentence_transformer.py:1169-1171. At least ONE must resolve.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ST_TAGS)
def test_st_transformer_base_class_either_path(tag: str):
    candidates = [
        "sentence_transformers/models/Transformer.py",
        "sentence_transformers/models/transformer.py",
        "sentence_transformers/models/transformer/__init__.py",
        "sentence_transformers/base/modules/transformer.py",
    ]
    for p in candidates:
        src = fetch_text("UKPLab/sentence-transformers", tag, p)
        if src is not None and has_def(src, "Transformer", "class"):
            return
    pytest.fail(
        f"{tag}: class Transformer not in any of {candidates} — "
        f"unsloth's three-path probe in sentence_transformer.py:1169-1171 "
        f"will ImportError on every fallback"
    )


# -------------------------------------------------------------------------
# sentence_transformers.util: import_from_string + load_dir_path are the
# two helpers unsloth.models.sentence_transformer:1177,1205 calls.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("tag", ST_TAGS)
def test_st_util_helpers(tag: str):
    """`sentence_transformers.util.{import_from_string, load_dir_path}` —
    used by unsloth.models.sentence_transformer:1177,1205. ST 5.4+ moved
    util into a package; we accept either layout. We also accept the
    function being defined in any submodule of the util package, since
    `from sentence_transformers.util import import_from_string` works
    when util/__init__.py re-exports."""
    candidates = [
        "sentence_transformers/util.py",
        "sentence_transformers/util/__init__.py",
    ]
    hit = first_match("UKPLab/sentence-transformers", tag, candidates)
    assert (
        hit is not None
    ), f"{tag}: sentence_transformers/util[.py|/__init__.py] both missing"
    _path, src = hit
    for fn in ("import_from_string", "load_dir_path"):
        defined_here = has_def(src, fn, "func")
        reexported = bool(re.search(rf"\b{re.escape(fn)}\b", src))
        if not (defined_here or reexported):
            # Try common subfiles for the modular layout.
            subpaths = [
                "sentence_transformers/util/import_utils.py",
                "sentence_transformers/util/file_utils.py",
                "sentence_transformers/util/_helpers.py",
                "sentence_transformers/util/_utils.py",
            ]
            found = False
            for sp in subpaths:
                sub = fetch_text("UKPLab/sentence-transformers", tag, sp)
                if sub and (has_def(sub, fn, "func") or fn in sub):
                    found = True
                    break
            assert found, (
                f"{tag}: sentence_transformers.util.{fn} not found in "
                f"util[.py|/__init__.py] or any of {subpaths}"
            )
