import ast
import types
from pathlib import Path


def _load_is_gpt_oss():
    # Extract _is_gpt_oss without importing unsloth (needs unsloth_zoo / a GPU).
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_is_gpt_oss"
    ]
    module = ast.Module(body = helpers, type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_is_gpt_oss"]


def _model(architectures = None, model_type = None):
    config = types.SimpleNamespace()
    if architectures is not None:
        config.architectures = architectures
    if model_type is not None:
        config.model_type = model_type
    return types.SimpleNamespace(config = config)


def test_detects_gpt_oss_by_architecture():
    # architectures is a list, so detection must use membership, not ==.
    is_gpt_oss = _load_is_gpt_oss()
    assert is_gpt_oss(_model(architectures = ["GptOssForCausalLM"])) is True
    assert is_gpt_oss(_model(architectures = ["GptOssForCausalLM"], model_type = "gpt_oss")) is True


def test_detects_gpt_oss_by_model_type():
    is_gpt_oss = _load_is_gpt_oss()
    assert is_gpt_oss(_model(architectures = ["SomethingElse"], model_type = "gpt-oss")) is True
    assert is_gpt_oss(_model(architectures = ["SomethingElse"], model_type = "gpt_oss")) is True


def test_non_gpt_oss_is_false():
    is_gpt_oss = _load_is_gpt_oss()
    assert is_gpt_oss(_model(architectures = ["LlamaForCausalLM"], model_type = "llama")) is False
    assert is_gpt_oss(_model()) is False
    assert is_gpt_oss(types.SimpleNamespace()) is False
