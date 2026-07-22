# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pinned-symbol compat check for the symbols unsloth's sentence_transformer
integration relies on, across ST PyPI minors (GitHub raw fetch + symbol grep)."""

from __future__ import annotations

import re

import pytest

from tests.version_compat._fetch import fetch_text, first_match, has_def


# ST is unpinned in pyproject.toml; track the last few minors plus main.
ST_TAGS = [
    "v5.0.0",
    "v5.1.2",
    "v5.2.3",
    "v5.3.0",
    "v5.4.1",
    "v5.5.1",
    "v5.6.0",
    "master",
]


# Top-level: SentenceTransformer + SentenceTransformerTrainer must be importable.
@pytest.mark.parametrize("tag", ST_TAGS)
def test_st_top_level_exports(tag: str):
    src = fetch_text("UKPLab/sentence-transformers", tag, "sentence_transformers/__init__.py")
    assert src is not None, f"{tag}: sentence_transformers/__init__.py missing"
    needed = ("SentenceTransformer", "SentenceTransformerTrainer")
    missing = [n for n in needed if n not in src]
    assert not missing, (
        f"{tag}: sentence_transformers top-level missing {missing}; "
        f"unsloth.models.sentence_transformer:1467,2154 will ImportError"
    )


# Sub-modules: unsloth walks `sentence_transformers.models` for these classes.
@pytest.mark.parametrize("tag", ST_TAGS)
def test_st_models_re_exports(tag: str):
    """Transformer / Pooling / Normalize must stay reachable via
    `sentence_transformers.models` despite the ST 5.4 package reorg."""
    # Layout 1 (legacy < 5.4): sentence_transformers/models[.py|/__init__.py].
    # Layout 2 (>= 5.4): top-level re-exports; modules under base/modules + sentence_transformer/.
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

    # ST 5.4+ modular layout: classes moved under base/modules and
    # sentence_transformer/modules; backward compat wired via
    # setup_deprecated_module_imports in __init__.py.
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
            pytest.fail(f"{tag}: ST 5.4+ layout: class {cls} not found in any of {paths}")

    # The backward-compat shim must be wired so `from ...models import Pooling` keeps working.
    top = fetch_text("UKPLab/sentence-transformers", tag, "sentence_transformers/__init__.py")
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


# Transformer base class: unsloth probes alternate paths; at least ONE must resolve.
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


# Transformer.load classmethod: unsloth builds saved-ST modules through it (#6881).
@pytest.mark.parametrize("tag", ST_TAGS)
def test_st_transformer_load_accepts_unsloth_kwargs(tag: str):
    """unsloth builds saved ST models via Transformer.load(...) so the saved
    modality_config is honored (#6881). If .load stops accepting the hub kwargs it
    passes (and has no **kwargs), update the fix before it silently regresses. Not
    locating .load is a SKIP (may be inherited); the live test guards the install."""
    candidates = [
        "sentence_transformers/models/Transformer.py",
        "sentence_transformers/models/transformer.py",
        "sentence_transformers/base/modules/transformer.py",
        "sentence_transformers/base/modules/module.py",
    ]
    for p in candidates:
        src = fetch_text("UKPLab/sentence-transformers", tag, p)
        if src is None or not has_def(src, "load", "func"):
            continue
        m = re.search(r"def\s+load\s*\((.*?)\)\s*(?:->[^:]*)?:", src, re.S)
        if m is None:
            continue
        sig = m.group(1)
        accepts_var_kw = "**" in sig
        missing = [
            kw
            for kw in ("token", "cache_folder", "revision", "trust_remote_code")
            if not (accepts_var_kw or re.search(rf"\b{re.escape(kw)}\b", sig))
        ]
        assert not missing, (
            f"{tag}: Transformer.load in {p} no longer accepts {missing} and has no "
            f"**kwargs; update unsloth.models.sentence_transformer._create_transformer_module "
            f"(#6881) before it silently falls back to Transformer(...)."
        )
        return
    pytest.skip(f"{tag}: Transformer.load not locatable in {candidates} (may be inherited)")


# sentence_transformers.util: import_from_string + load_dir_path helpers unsloth calls.
@pytest.mark.parametrize("tag", ST_TAGS)
def test_st_util_helpers(tag: str):
    """util.{import_from_string, load_dir_path} must resolve; accept either the
    flat or the ST 5.4+ package layout, or a re-export from a util submodule."""
    candidates = [
        "sentence_transformers/util.py",
        "sentence_transformers/util/__init__.py",
    ]
    hit = first_match("UKPLab/sentence-transformers", tag, candidates)
    assert hit is not None, f"{tag}: sentence_transformers/util[.py|/__init__.py] both missing"
    _path, src = hit
    for fn in ("import_from_string", "load_dir_path"):
        defined_here = has_def(src, fn, "func")
        reexported = bool(re.search(rf"\b{re.escape(fn)}\b", src))
        if not (defined_here or reexported):
            # Modular-layout subfiles.
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
