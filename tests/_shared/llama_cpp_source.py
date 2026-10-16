# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The source of ``core.inference.llama_cpp`` for tests that read it as text or as an AST.

The backend used to be one module, and dozens of tests read ``llama_cpp.py`` off disk or
through ``inspect.getsource`` to pin an ordering, a literal or a call shape. It is a package
now, so those tests read the whole package through here: every module's text in one string,
in a fixed order, so a check that held on the single file holds on the same bytes.

Shared by ``studio/backend/tests`` and ``tests/studio``; both conftests put this directory
on ``sys.path``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path


def repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "studio" / "backend" / "core" / "inference").is_dir():
            return parent
    raise RuntimeError("the unsloth checkout root was not found above tests/_shared")


PACKAGE_DIR = repo_root() / "studio" / "backend" / "core" / "inference" / "llama_cpp"
# Relative to studio/backend, the form the backend tests spell paths in.
PACKAGE_RELATIVE = "core/inference/llama_cpp"


def module_paths() -> list[Path]:
    """Every module of the package, ``__init__.py`` first, then by name."""
    return sorted(PACKAGE_DIR.glob("*.py"))


def module_relative_paths() -> list[str]:
    """``core/inference/llama_cpp/<module>.py`` for every module, in ``module_paths`` order."""
    return [f"{PACKAGE_RELATIVE}/{path.name}" for path in module_paths()]


def module_source(name: str) -> str:
    """One module's text, by its bare name (``backend_load``)."""
    return (PACKAGE_DIR / f"{name}.py").read_text(encoding = "utf-8")


def package_source() -> str:
    """Every module's text joined in ``module_paths`` order, without ``__init__``.

    A string check that held on the old single file holds here, and ``ast.parse`` of it yields
    one tree whose ``FunctionDef`` nodes are the package's functions and methods; line numbers
    are relative to this concatenation, so only compare them within one function. ``__init__``
    is left out because it only re-exports: its list of every name would be the first hit for
    a scan that looks a definition up by name.
    """
    return "\n".join(
        path.read_text(encoding = "utf-8") for path in module_paths() if path.name != "__init__.py"
    )


def package_tree() -> ast.Module:
    return ast.parse(package_source())


def class_source(cls) -> str:
    """The source of ``cls`` and every mixin it is composed from, in MRO order.

    ``inspect.getsource(LlamaCppBackend)`` returns only the composing class body; the methods
    live on the mixins, so a check over "the class" has to read all of them.
    """
    return "\n".join(inspect.getsource(klass) for klass in cls.__mro__ if klass is not object)


def class_tree(cls) -> ast.Module:
    return ast.parse(class_source(cls))
