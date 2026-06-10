import ast
import importlib
from pathlib import Path


_UTILS_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"


def _load_package_available_bool():
    source = _UTILS_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_is_package_available_bool"
    )
    module = ast.Module(body = [function], type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = {"importlib": importlib}
    exec(compile(module, str(_UTILS_PATH), "exec"), namespace)
    return namespace["_is_package_available_bool"]


def test_package_available_bool_returns_false_for_absent_package():
    is_package_available = _load_package_available_bool()

    assert is_package_available("_definitely_missing_unsloth_dependency_") is False


def test_utils_does_not_call_transformers_private_package_helper():
    source = _UTILS_PATH.read_text(encoding = "utf-8")

    assert "from transformers.utils.import_utils import _is_package_available" not in source
    assert '_is_package_available("flash_attn")' not in source
    assert 'return _is_package_available("vllm")' not in source
