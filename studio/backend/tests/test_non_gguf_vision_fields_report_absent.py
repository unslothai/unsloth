# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``disable_vision`` on an MLX / safetensors load: the response says ``False``,
and that is the answer, not a dropped echo.

The observation this pins down is real: those load paths never route through
``_llama_runtime_fields``, so ``disable_vision``, ``vision_disabled_by_user``
and ``vision_on_cpu`` come out ``False`` however the request was set. The
conclusion that this is a bug is not, because the request field and the response
field are different propositions:

  * request ``disable_vision`` (``models/inference.py``) = "please load this
    vision GGUF without its projector", documented "Ignored ... for non-GGUF
    models" and in fact never read outside ``core/inference/llama_cpp.py``;
  * response ``disable_vision`` = "the vision projector WAS deliberately left
    unloaded", ``vision_on_cpu`` = "the projector is running on the CPU
    (``--no-mmproj-offload``)", ``vision_disabled_by_user`` = ``is_vision and
    disable_vision`` as the llama.cpp backend computed it.

On an MLX load the request is ignored, so the projector was not left unloaded
and there is no ``--no-mmproj-offload`` placement to report. ``False`` is the
true answer to the question the response field asks. It is the same contract the
whole family already follows -- the non-GGUF response sites pass none of the
GGUF-only runtime fields, ``tensor_parallel`` included -- and the exact mirror of
``_llama_runtime_fields`` forcing the MLX fields to ``None`` on the GGUF path
("Not MLX, so the MLX runtime fields report as absent").

Echoing the request faithfully would be a regression, not a fix: the attach
menu's refusal ("Vision is turned off for X. Turn it back on in the model's
Advanced Settings") is driven by ``vision_disabled_by_user``, and the Vision
switch lives only in the GGUF half of Advanced Settings, so an MLX model
reporting ``True`` would send the user to a control that is not rendered.
``tests/frontend`` has no place for that half; the companion assertion lives in
``studio/frontend/tests/mlx-advanced-has-no-vision-row.test.ts``.

These are structural assertions over ``routes/inference.py`` rather than route
calls: what is being pinned is that the non-GGUF sites keep passing NOTHING for
these fields, which is a property of the call sites themselves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROUTES = Path(__file__).resolve().parent.parent / "routes" / "inference.py"

# Response-only. None of the three is a request field except disable_vision,
# which is a different proposition under the same name (see the module docstring).
_VISION_FIELDS = ("disable_vision", "vision_disabled_by_user", "vision_on_cpu")
_RUNTIME_FIELDS_CALL = "_llama_runtime_fields(llama_backend)"


def _load_response_calls() -> dict[int, tuple[set[str], list[str]]]:
    """``{lineno: (explicit kwarg names, ``**`` expressions)}`` for every
    ``LoadResponse(...)`` in ``routes/inference.py``."""
    tree = ast.parse(_ROUTES.read_text(encoding = "utf-8"))
    out: dict[int, tuple[set[str], list[str]]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LoadResponse"
        ):
            out[node.lineno] = (
                {kw.arg for kw in node.keywords if kw.arg is not None},
                [ast.unparse(kw.value) for kw in node.keywords if kw.arg is None],
            )
    return out


def _non_gguf_sites() -> list[int]:
    """Line numbers of the ``LoadResponse`` sites that are NOT the GGUF one.

    Found by which site splats the llama.cpp runtime fields, so adding or moving
    a call site does not silently empty this list the way hardcoded line numbers
    would.
    """
    calls = _load_response_calls()
    return sorted(
        lineno for lineno, (_kwargs, splats) in calls.items() if _RUNTIME_FIELDS_CALL not in splats
    )


def test_exactly_one_site_carries_the_llama_runtime_fields():
    calls = _load_response_calls()
    assert calls, "no LoadResponse call sites found -- the AST walk is stale"
    carriers = [l for l, (_k, splats) in calls.items() if _RUNTIME_FIELDS_CALL in splats]
    assert len(carriers) == 1, f"expected one GGUF response site, found {carriers}"
    # And there is more than one site, or the rest of this file proves nothing.
    assert len(calls) > 1


def test_the_non_gguf_sites_pass_none_of_the_vision_fields():
    """Not merely "they are False": the routes do not mention them at all.

    A site that passed one of the three explicitly would be making a claim about
    a runtime it did not launch.
    """
    sites = _non_gguf_sites()
    assert sites, "no non-GGUF LoadResponse site found"
    for lineno in sites:
        kwargs, _splats = _load_response_calls()[lineno]
        assert not (set(_VISION_FIELDS) & kwargs), (
            f"LoadResponse at line {lineno} passes vision runtime fields "
            f"{sorted(set(_VISION_FIELDS) & kwargs)}"
        )


def test_the_non_gguf_response_serializes_all_three_as_false():
    """Rebuilt from each site's OWN kwarg set, so the defaults are what a real
    MLX load returns and not a hand-written payload."""
    from models.inference import LoadResponse
    for lineno in _non_gguf_sites():
        kwargs, _splats = _load_response_calls()[lineno]
        payload = {name: None for name in kwargs}
        payload.update(
            status = "loaded",
            model = "mlx-community/Qwen3-VL-8B-4bit",
            display_name = "Qwen3-VL-8B",
            is_gguf = False,
            is_mlx = True,
            is_vision = True,
            is_lora = False,
            is_local_model = False,
            is_audio = False,
            has_audio_input = False,
            requires_trust_remote_code = False,
            supports_reasoning = False,
            reasoning_always_on = False,
            supports_preserve_thinking = False,
            supports_tools = False,
            reasoning_effort_levels = [],
            reasoning_style = "enable_thinking",
            inference = {},
        )
        dumped = LoadResponse(**payload).model_dump()
        assert {name: dumped[name] for name in _VISION_FIELDS} == {
            name: False for name in _VISION_FIELDS
        }, f"LoadResponse at line {lineno}"


@pytest.mark.parametrize("field", _VISION_FIELDS)
def test_the_response_fields_default_to_false(field):
    """``False``, never ``None``: the client reads them as booleans (the store
    seeds ``disableVision`` straight off ``disable_vision``), so a null would
    become an "unknown" state no consumer handles."""
    from models.inference import LoadResponse
    assert LoadResponse.model_fields[field].default is False


def test_the_request_field_says_it_is_ignored_for_non_gguf():
    """The contract is documented, not merely implemented. If someone widens
    ``disable_vision`` to MLX, this is the line that has to change with it."""
    from models.inference import LoadRequest

    description = (LoadRequest.model_fields["disable_vision"].description or "").lower()
    assert "non-gguf" in description
    assert "ignored" in description


def test_only_the_llama_cpp_backend_reads_the_request_field():
    """The claim behind "False is correct": no non-GGUF loader can act on it.

    Scoped to ``core/inference`` -- ``routes/inference.py`` reads it to size a
    GGUF VRAM estimate, which is gated on ``is_gguf`` at the call site.
    """
    inference_dir = Path(__file__).resolve().parent.parent / "core" / "inference"
    readers = sorted(
        path.name
        for path in inference_dir.rglob("*.py")
        if "disable_vision" in path.read_text(encoding = "utf-8", errors = "replace")
    )
    assert readers == ["llama_cpp.py"], readers
