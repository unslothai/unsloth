# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The compressed-tensors drift guard, with nothing but transformers installed.

`test_compressed_tensors_load_error_message.py` re-derives transformers' own refusals and
matches them, which is the right check. It can only run where the whole Studio backend is
importable, because it imports fastapi and `models.inference` at module scope and then
execs `routes/inference.py`. That is one environment, and `studio-backend-ci.yml` pins it
to `transformers>=4.51,<5.5`:

    .github/workflows/studio-backend-ci.yml:275   pip install 'transformers>=4.51,<5.5'
    .github/workflows/studio-backend-ci.yml:568   pip install 'transformers>=4.51,<5.5'

The rewording this matcher exists to survive landed in transformers 5.10, outside that
window. So the guard, as installed, never sees the wording it was built for: it is green
on the half of the range that was never in danger. That is the same shape as the defect
the matcher fixes, one level up.

This file closes it by needing only `transformers`. The signatures and `_diagnosis_text`
are read out of `routes/inference.py` with `ast`, so no fastapi, no PyJWT, no route
import, no event loop. It therefore runs in any environment that has transformers at all,
including the 4.57.6 and >=5,<6 legs of `consolidated-tests-ci.yml`, and can be pointed at
a newer transformers with a bare `pip install -U transformers` and nothing else.

Measured on 4.57.6, 5.0.0, 5.5.4, 5.9.0, 5.10.4, 5.13.1 and 5.17.0: both sites matched at
every one, and 20 to 24 quantizer modules were scanned for over-acceptance with none found.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import patch

import pytest

_ROUTE = Path(__file__).resolve().parent.parent / "routes" / "inference.py"


def _route_pieces():
    """`_MISSING_COMPRESSED_TENSORS_SIGNATURES` and `_diagnosis_text`, without importing.

    The matcher is two module-level constants and one small pure function, none of which
    needs the FastAPI app that surrounds them. Lifting exactly those out with `ast` is what
    lets this run beside a transformers the backend environment cannot hold.
    """
    tree = ast.parse(_ROUTE.read_text(encoding = "utf-8"))
    literals = {}
    diagnosis = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "_MISSING_COMPRESSED_TENSORS_SIGNATURES",
                    "_COMPRESSED_TENSORS_INFERENCE_UNSUPPORTED_MESSAGE",
                ):
                    literals[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name == "_diagnosis_text":
            node.decorator_list = []
            diagnosis = node
    assert "_MISSING_COMPRESSED_TENSORS_SIGNATURES" in literals, _ROUTE
    assert diagnosis is not None, "_diagnosis_text moved; this guard reads the wrong text"
    namespace = {"re": re}
    exec(  # noqa: S102 - the source is this repository's own, read from disk above
        compile(ast.Module(body = [diagnosis], type_ignores = []), str(_ROUTE), "exec"),
        namespace,
    )
    return literals["_MISSING_COMPRESSED_TENSORS_SIGNATURES"], namespace["_diagnosis_text"]


def _matches(message: str) -> bool:
    signatures, diagnosis_text = _route_pieces()
    lowered = diagnosis_text(message).lower()
    return any(signature in lowered for signature in signatures)


def _refusals_from_installed_transformers():
    """The two refusals the INSTALLED transformers raises with the library absent.

    Reached through `is_compressed_tensors_available`, the gate both sites consult, so
    these are the real `raise` statements and the real strings rather than a copy.
    """
    pytest.importorskip("transformers")
    from transformers.quantizers import quantizer_compressed_tensors as quantizer_module
    from transformers.utils import quantization_config as config_module

    messages = {}
    with patch.object(config_module, "is_compressed_tensors_available", return_value = False):
        with pytest.raises(ImportError) as raised:
            config_module.CompressedTensorsConfig()
        messages["config"] = str(raised.value)
    with patch.object(quantizer_module, "is_compressed_tensors_available", return_value = False):
        quantizer = quantizer_module.CompressedTensorsHfQuantizer.__new__(
            quantizer_module.CompressedTensorsHfQuantizer
        )
        with pytest.raises(ImportError) as raised:
            quantizer.validate_environment()
        messages["quantizer"] = str(raised.value)
    return messages


def test_the_installed_transformers_refusals_are_both_recognised():
    """The drift guard proper, runnable against any transformers on any host."""
    import transformers
    for site, message in _refusals_from_installed_transformers().items():
        assert "pip install" in message, (site, message)
        assert _matches(message), (
            f"transformers {transformers.__version__} reworded the {site} "
            f"compressed-tensors refusal: {message!r}. Add the part that survived to "
            "_MISSING_COMPRESSED_TENSORS_SIGNATURES in studio/backend/routes/inference.py."
        )


def test_no_signature_pins_a_version_number():
    """The version token is what moved in 5.10, so no signature may contain one."""
    signatures, _ = _route_pieces()
    assert signatures
    for signature in signatures:
        assert not re.search(r"\d+\.\d+", signature), signature
        # Long enough to be this refusal and not an arbitrary sentence saying the words.
        assert len(signature) >= 30, signature


def test_no_other_quantizer_family_is_claimed():
    """Every other quantizer's "you are missing a library" message must not match.

    Parsed out of the installed transformers rather than listed, so a family added
    upstream is covered the day it lands. A parse that finds too little would pass
    without checking anything, so the module count is asserted.
    """
    pytest.importorskip("transformers")
    import transformers.quantizers as quantizers_package

    package_dir = Path(quantizers_package.__file__).parent
    scanned, claimed = set(), []
    for source_file in sorted(package_dir.glob("quantizer_*.py")):
        tree = ast.parse(source_file.read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            for literal in ast.walk(node.exc):
                if not isinstance(literal, ast.Constant) or not isinstance(literal.value, str):
                    continue
                if not re.search(r"requires|install", literal.value, re.IGNORECASE):
                    continue
                scanned.add(source_file.stem)
                if source_file.stem != "quantizer_compressed_tensors" and _matches(literal.value):
                    claimed.append((source_file.stem, literal.value))
    assert len(scanned) >= 8, sorted(scanned)
    assert not claimed, claimed


def test_this_guard_needs_no_studio_backend_dependency():
    """The property that makes it runnable outside the backend's pinned environment.

    If this file grows an import of the route module, fastapi, or anything under
    `models.`, it goes back to running only where `transformers>=4.51,<5.5` is installed
    and stops covering the range it exists for.
    """
    tree = ast.parse(Path(__file__).read_text(encoding = "utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    forbidden = {"fastapi", "models", "routes", "core", "utils", "starlette", "jwt"}
    assert not (imported & forbidden), sorted(imported & forbidden)
