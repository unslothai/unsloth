# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The compressed-tensors drift guard, with nothing but transformers installed.

`test_compressed_tensors_load_error_message.py` runs the same check, but only where the
whole backend imports, and `studio-backend-ci.yml` pins that environment to
`transformers>=4.51,<5.5` -- outside the 5.10 rewording the matcher exists to survive.
This file reads the signatures and `_diagnosis_text` out of `routes/inference.py` with
`ast` instead, so it needs no fastapi and runs on any transformers.
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

    Lifting them out with `ast` is what lets this run beside a transformers the backend
    environment cannot hold.
    """
    tree = ast.parse(_ROUTE.read_text(encoding="utf-8"))
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
        compile(ast.Module(body=[diagnosis], type_ignores=[]), str(_ROUTE), "exec"),
        namespace,
    )
    return literals["_MISSING_COMPRESSED_TENSORS_SIGNATURES"], namespace["_diagnosis_text"]


def _matches(message: str) -> bool:
    signatures, diagnosis_text = _route_pieces()
    lowered = diagnosis_text(message).lower()
    return any(signature in lowered for signature in signatures)


def _refusals_from_installed_transformers():
    """The two refusals the INSTALLED transformers raises with the library absent.

    Through `is_compressed_tensors_available`, so these are the real strings, not a copy.
    """
    pytest.importorskip("transformers")
    from transformers.quantizers import quantizer_compressed_tensors as quantizer_module
    from transformers.utils import quantization_config as config_module

    messages = {}
    with patch.object(config_module, "is_compressed_tensors_available", return_value=False):
        with pytest.raises(ImportError) as raised:
            config_module.CompressedTensorsConfig()
        messages["config"] = str(raised.value)
    with patch.object(quantizer_module, "is_compressed_tensors_available", return_value=False):
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

    Parsed from the installed transformers, so a family added upstream is covered the day
    it lands. The module count is asserted: a parse finding too little would pass empty.
    """
    pytest.importorskip("transformers")
    import transformers.quantizers as quantizers_package

    package_dir = Path(quantizers_package.__file__).parent
    scanned, claimed = set(), []
    for source_file in sorted(package_dir.glob("quantizer_*.py")):
        tree = ast.parse(source_file.read_text(encoding="utf-8"))
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

    An import of fastapi or the route module puts it back inside the `<5.5` pin.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    forbidden = {"fastapi", "models", "routes", "core", "utils", "starlette", "jwt"}
    assert not (imported & forbidden), sorted(imported & forbidden)


def test_a_workflow_that_installs_a_modern_transformers_actually_collects_this_file():
    """The guard above is worth nothing in an environment that never runs it.

    Every auto-discovering job pins `transformers>=4.51,<5.5`; the jobs that install a
    modern one collect listed paths, not trees, so an unlisted file never runs.
    """
    workflow = (
        Path(__file__).resolve().parents[3] / ".github" / "workflows" / "consolidated-tests-ci.yml"
    )
    text = workflow.read_text(encoding="utf-8")
    assert (
        "studio/backend/tests/test_compressed_tensors_signature_drift.py" in text
    ), "the modern-transformers matrix does not collect this file"
    assert 'transformers_spec: "transformers>=5,<6"' in text
