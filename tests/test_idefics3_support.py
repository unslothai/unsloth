import ast
from pathlib import Path


VISION_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"


def _vision_tree_and_source():
    source = VISION_PATH.read_text(encoding = "utf-8")
    return ast.parse(source), source


def _literal_assignment(tree, name):
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Could not find assignment for {name}")


def test_idefics3_is_enabled_for_vllm_vision_loading():
    tree, _ = _vision_tree_and_source()
    supported = _literal_assignment(tree, "VLLM_SUPPORTED_VLM")

    assert supported.count("idefics3") == 1


def test_fast_inference_error_uses_the_vlm_allowlist():
    tree, source = _vision_tree_and_source()
    supported = _literal_assignment(tree, "VLLM_SUPPORTED_VLM")

    assert len(supported) == len(set(supported))
    assert 'supported_vlms = ", ".join(VLLM_SUPPORTED_VLM)' in source
    assert (
        "Fast inference is only supported for Language models and Qwen2.5-VL, "
        "Gemma3 among vision models"
    ) not in source
