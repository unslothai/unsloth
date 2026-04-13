"""Test structural integrity of all Studio mapping dicts."""

import os, sys

sys.path.insert(0, os.path.dirname(__file__))

MAPPINGS_FILE = os.path.join(
    os.path.dirname(__file__),
    "studio",
    "backend",
    "utils",
    "datasets",
    "model_mappings.py",
)


def _load():
    g = {}
    exec(open(MAPPINGS_FILE).read(), g)
    return g


def test_all_template_keys_are_strings():
    g = _load()
    for key in g["TEMPLATE_TO_MODEL_MAPPER"]:
        assert isinstance(key, str), f"Non-string key: {key!r}"
        assert len(key) > 0, "Empty template key"


def test_all_model_ids_are_strings():
    g = _load()
    for key, values in g["TEMPLATE_TO_MODEL_MAPPER"].items():
        assert isinstance(
            values, tuple
        ), f"{key}: values is {type(values)}, expected tuple"
        for v in values:
            assert isinstance(v, str), f"{key}: non-string model ID {v!r}"
            assert "/" in v, f"{key}: model ID {v!r} missing org/ prefix"


def test_all_response_markers_have_instruction_and_response():
    g = _load()
    for key, markers in g["TEMPLATE_TO_RESPONSES_MAPPER"].items():
        assert "instruction" in markers, f"{key}: missing 'instruction' key"
        assert "response" in markers, f"{key}: missing 'response' key"
        assert len(markers["instruction"]) > 0, f"{key}: empty instruction"
        assert len(markers["response"]) > 0, f"{key}: empty response"


def test_no_model_id_appears_in_multiple_templates():
    """Each model ID should map to exactly one template (last-write-wins in the dict build)."""
    g = _load()
    seen = {}
    conflicts = []
    for template, models in g["TEMPLATE_TO_MODEL_MAPPER"].items():
        for model in models:
            if model in seen and seen[model] != template:
                conflicts.append(f"{model}: {seen[model]} vs {template}")
            seen[model] = template
    assert not conflicts, f"Model ID conflicts:\n" + "\n".join(conflicts)
