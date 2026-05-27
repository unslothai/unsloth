"""Text-only FastLanguageModel routing for vision-capable configs."""

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LOADER_PATH = REPO_ROOT / "unsloth" / "models" / "loader.py"
VISION_PATH = REPO_ROOT / "unsloth" / "models" / "vision.py"


def _source(path):
    return path.read_text()


def _class_method(tree, class_name, method_name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _load_text_only_helper():
    source = _source(LOADER_PATH)
    tree = ast.parse(source)
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_get_text_only_config":
            exec(ast.get_source_segment(source, node), ns)
            return ns[node.name]
    raise AssertionError("_get_text_only_config not found")


def test_gemma3_vision_config_resolves_to_text_config():
    transformers = pytest.importorskip("transformers")
    helper = _load_text_only_helper()

    config = transformers.Gemma3Config()
    text_config = helper(config, "google/gemma-3-27b-it")

    assert isinstance(text_config, transformers.Gemma3TextConfig)
    assert text_config.model_type == "gemma3_text"
    model_class = transformers.AutoModelForCausalLM._model_mapping[type(text_config)]
    assert model_class.__name__ == "Gemma3ForCausalLM"


def test_text_only_helper_rejects_configs_without_text_submodel():
    helper = _load_text_only_helper()

    class VisionOnlyConfig:
        vision_config = object()

    with pytest.raises(ValueError, match = "Cannot load vision-only as text-only"):
        helper(VisionOnlyConfig(), "vision-only")


def test_fast_language_model_forces_text_only_when_delegating_to_fast_model():
    source = _source(LOADER_PATH)
    method = _class_method(ast.parse(source), "FastLanguageModel", "from_pretrained")
    fast_model_calls = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "from_pretrained"
            and isinstance(func.value, ast.Name)
            and func.value.id == "FastModel"
        ):
            fast_model_calls.append(node)

    assert len(fast_model_calls) == 2
    for call in fast_model_calls:
        force_kwarg = [
            kw for kw in call.keywords if kw.arg == "_force_text_only"
        ]
        assert len(force_kwarg) == 1
        assert isinstance(force_kwarg[0].value, ast.Constant)
        assert force_kwarg[0].value.value is True


def test_fast_model_text_only_does_not_override_explicit_auto_model():
    source = _source(LOADER_PATH)
    method_source = ast.get_source_segment(
        source, _class_method(ast.parse(source), "FastModel", "from_pretrained")
    )

    assert "_force_text_only = kwargs.pop(\"_force_text_only\", False)" in method_source
    assert "load_text_only = _force_text_only and auto_model is None" in method_source
    assert "model_config = _get_text_only_config(model_config, old_model_name)" in method_source
    assert "_force_text_only = load_text_only" in method_source


def test_fast_base_model_text_only_bypasses_vision_auto_model():
    source = _source(VISION_PATH)
    method_source = ast.get_source_segment(
        source, _class_method(ast.parse(source), "FastBaseModel", "from_pretrained")
    )

    assert "_force_text_only = False" in method_source
    assert "auto_model = AutoModelForCausalLM" in method_source
    assert "auto_config = _get_text_only_config(auto_config, model_name)" in method_source
