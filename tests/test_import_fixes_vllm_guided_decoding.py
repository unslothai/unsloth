import ast
import builtins
import importlib
from pathlib import Path
import sys
import types

import pytest


def _load_function(function_name):
    source_path = Path(__file__).resolve().parents[1] / "unsloth" / "import_fixes.py"
    tree = ast.parse(source_path.read_text(encoding = "utf-8"), filename = str(source_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected {function_name} in import_fixes.py")

    module_ast = ast.Module(body = selected, type_ignores = [])
    namespace = {
        "importlib": importlib,
        "os": __import__("os"),
        "sys": sys,
    }
    exec(compile(module_ast, str(source_path), "exec"), namespace)
    return namespace[function_name]


def _load_fix_vllm_guided_decoding_params():
    return _load_function("fix_vllm_guided_decoding_params")


def test_fix_vllm_guided_decoding_params_skips_broken_binary_import(
    monkeypatch,
):
    fix_vllm_guided_decoding_params = _load_fix_vllm_guided_decoding_params()

    logged_messages = []

    class Logger:
        def warning(self, message):
            logged_messages.append(("warning", message))

    fix_vllm_guided_decoding_params.__globals__["logger"] = Logger()
    fix_vllm_guided_decoding_params.__globals__["importlib_version"] = lambda _: "0.13.1"

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "vllm":
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    original_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "vllm.sampling_params":
            raise ImportError(
                "vllm/_C.abi3.so: undefined symbol: "
                "_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib"
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Should not raise: broken vLLM extension import must be downgraded to warning.
    fix_vllm_guided_decoding_params()

    warning_messages = [message for level, message in logged_messages if level == "warning"]
    assert any("broken native extension" in message for message in warning_messages)


def test_fix_vllm_guided_decoding_params_keeps_transformers_mismatch_raise(
    monkeypatch,
):
    fix_vllm_guided_decoding_params = _load_fix_vllm_guided_decoding_params()

    class Logger:
        def warning(self, message):
            del message

    fix_vllm_guided_decoding_params.__globals__["logger"] = Logger()
    fix_vllm_guided_decoding_params.__globals__["importlib_version"] = lambda _: "0.13.1"

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "vllm":
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    original_import = builtins.__import__

    def fake_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name == "vllm":
            raise ImportError(
                "cannot import name 'ALLOWED_LAYER_TYPES' "
                "from transformers.configuration_utils"
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        fix_vllm_guided_decoding_params()

    assert "does not yet support transformers>=5.0.0" in str(excinfo.value)


def test_fix_vllm_pdl_blackwell_handles_broken_vllm_spec_checks(monkeypatch):
    fix_vllm_pdl_blackwell = _load_function("fix_vllm_pdl_blackwell")

    class Logger:
        def info(self, message):
            del message

        def debug(self, message):
            del message

    fix_vllm_pdl_blackwell.__globals__["logger"] = Logger()

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        get_device_capability = lambda _: (10, 0),
        get_device_name = lambda _: "Fake SM100",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "vllm":
            return object()
        if name.startswith("vllm.lora.ops.triton_ops"):
            raise ImportError(
                "vllm/_C.abi3.so: undefined symbol: "
                "_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib"
            )
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.delenv("TRITON_DISABLE_PDL", raising=False)

    # Should not raise when vLLM submodule spec checks fail due broken extension.
    fix_vllm_pdl_blackwell()
