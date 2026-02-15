import ast
import importlib
import os
from pathlib import Path
import sys
import types


def _load_fix_vllm_pdl_blackwell():
    source_path = Path(__file__).resolve().parents[1] / "unsloth" / "import_fixes.py"
    tree = ast.parse(source_path.read_text(encoding = "utf-8"), filename = str(source_path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fix_vllm_pdl_blackwell"
    ]
    if len(selected) != 1:
        raise RuntimeError("Expected fix_vllm_pdl_blackwell in import_fixes.py")

    module_ast = ast.Module(body = selected, type_ignores = [])
    namespace = {
        "importlib": importlib,
        "os": os,
        "sys": sys,
    }
    exec(compile(module_ast, str(source_path), "exec"), namespace)
    return namespace["fix_vllm_pdl_blackwell"]


def test_fix_vllm_pdl_blackwell_patches_known_consumers(monkeypatch):
    fix_vllm_pdl_blackwell = _load_fix_vllm_pdl_blackwell()

    logged_messages = []

    class Logger:
        def info(self, message):
            logged_messages.append(("info", message))

        def debug(self, message):
            logged_messages.append(("debug", message))

    fix_vllm_pdl_blackwell.__globals__["logger"] = Logger()
    fix_vllm_pdl_blackwell.__globals__["importlib_version"] = lambda _: "0.13.1"
    fix_vllm_pdl_blackwell.__globals__["Version"] = lambda version: version

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        get_device_capability = lambda _: (10, 0),
        get_device_name = lambda _: "Fake SM100",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    vllm_lora = types.ModuleType("vllm.lora")
    vllm_lora.__path__ = []
    vllm_ops = types.ModuleType("vllm.lora.ops")
    vllm_ops.__path__ = []
    triton_ops = types.ModuleType("vllm.lora.ops.triton_ops")
    triton_ops.__path__ = []

    cache_state = {"cleared": False}

    def utils_supports_pdl(*args, **kwargs):
        del args, kwargs
        return True

    def cache_clear():
        cache_state["cleared"] = True

    utils_supports_pdl.cache_clear = cache_clear

    utils_mod = types.ModuleType("vllm.lora.ops.triton_ops.utils")
    utils_mod.supports_pdl = utils_supports_pdl

    expand_mod = types.ModuleType("vllm.lora.ops.triton_ops.lora_expand_op")
    expand_mod.supports_pdl = lambda *args, **kwargs: True

    shrink_mod = types.ModuleType("vllm.lora.ops.triton_ops.lora_shrink_op")
    shrink_mod.supports_pdl = lambda *args, **kwargs: True

    fused_mod = types.ModuleType("vllm.lora.ops.triton_ops.fused_moe_lora_op")
    fused_mod.supports_pdl = lambda *args, **kwargs: True

    extra_mod = types.ModuleType("vllm.lora.ops.triton_ops.extra_consumer")
    extra_mod.supports_pdl = lambda *args, **kwargs: True

    triton_ops.utils = utils_mod
    triton_ops.lora_expand_op = expand_mod
    triton_ops.lora_shrink_op = shrink_mod
    triton_ops.fused_moe_lora_op = fused_mod
    triton_ops.extra_consumer = extra_mod

    module_map = {
        "vllm": vllm,
        "vllm.lora": vllm_lora,
        "vllm.lora.ops": vllm_ops,
        "vllm.lora.ops.triton_ops": triton_ops,
        "vllm.lora.ops.triton_ops.utils": utils_mod,
        "vllm.lora.ops.triton_ops.lora_expand_op": expand_mod,
        "vllm.lora.ops.triton_ops.lora_shrink_op": shrink_mod,
        "vllm.lora.ops.triton_ops.fused_moe_lora_op": fused_mod,
        "vllm.lora.ops.triton_ops.extra_consumer": extra_mod,
    }
    for name, module in module_map.items():
        monkeypatch.setitem(sys.modules, name, module)

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name.startswith("vllm"):
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setenv("TRITON_DISABLE_PDL", "0")

    fix_vllm_pdl_blackwell()

    assert os.environ["TRITON_DISABLE_PDL"] == "1"
    assert cache_state["cleared"] is True
    assert utils_mod.supports_pdl() is False
    assert expand_mod.supports_pdl() is False
    assert shrink_mod.supports_pdl() is False
    assert fused_mod.supports_pdl() is False
    assert extra_mod.supports_pdl() is False

    info_messages = [message for level, message in logged_messages if level == "info"]
    assert any("lora_expand_op" in message for message in info_messages)
    assert any("lora_shrink_op" in message for message in info_messages)
    assert any("fused_moe_lora_op" in message for message in info_messages)
    assert any("extra_consumer" in message for message in info_messages)


def test_fix_vllm_pdl_blackwell_skips_at_fixed_version(monkeypatch):
    fix_vllm_pdl_blackwell = _load_fix_vllm_pdl_blackwell()

    logged_messages = []

    class Logger:
        def info(self, message):
            logged_messages.append(("info", message))

        def debug(self, message):
            logged_messages.append(("debug", message))

    fix_vllm_pdl_blackwell.__globals__["logger"] = Logger()
    fix_vllm_pdl_blackwell.__globals__["importlib_version"] = lambda _: "0.13.2"
    fix_vllm_pdl_blackwell.__globals__["Version"] = lambda version: version

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
        if name.startswith("vllm"):
            return object()
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setenv("TRITON_DISABLE_PDL", "0")

    fix_vllm_pdl_blackwell()

    assert os.environ["TRITON_DISABLE_PDL"] == "0"
    info_messages = [message for level, message in logged_messages if level == "info"]
    assert any("skipping workaround" in message for message in info_messages)
