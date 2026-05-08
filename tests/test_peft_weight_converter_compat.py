import importlib.util
import inspect
import sys
import threading
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_FIXES = REPO_ROOT / "unsloth" / "import_fixes.py"


def _load_patch_function():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_import_fixes_under_test", IMPORT_FIXES
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.patch_peft_weight_converter_compatibility


def _install_fake_peft(twc_namespace):
    peft_pkg = types.ModuleType("peft")
    peft_pkg.__path__ = []
    peft_utils = types.ModuleType("peft.utils")
    peft_utils.__path__ = []
    twc = types.ModuleType("peft.utils.transformers_weight_conversion")
    for k, v in twc_namespace.items():
        setattr(twc, k, v)
    peft_utils.transformers_weight_conversion = twc
    sys.modules["peft"] = peft_pkg
    sys.modules["peft.utils"] = peft_utils
    sys.modules["peft.utils.transformers_weight_conversion"] = twc
    return twc


@pytest.fixture(autouse = True)
def _restore_peft_modules():
    saved = {
        k: sys.modules.get(k)
        for k in (
            "peft",
            "peft.utils",
            "peft.utils.transformers_weight_conversion",
        )
    }
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


class _LegacyConverter:
    def __init__(self, source_patterns, target_patterns, operations):
        self.source_patterns = source_patterns
        self.target_patterns = target_patterns
        self.operations = operations
        self.distributed_operation = None
        self.quantization_operation = None


class _ModernConverter:
    def __init__(
        self,
        source_patterns,
        target_patterns,
        operations,
        distributed_operation = None,
        quantization_operation = None,
    ):
        self.source_patterns = source_patterns
        self.target_patterns = target_patterns
        self.operations = operations
        self.distributed_operation = distributed_operation
        self.quantization_operation = quantization_operation


def _make_legacy_converter():
    return _LegacyConverter(["src.*"], ["tgt.*"], [])


def _make_modern_converter():
    return _ModernConverter(["src.*"], ["tgt.*"], [])


def _build_that_calls_init(weight_conversions, adapter_name, peft_config = None):
    out = []
    for c in weight_conversions or []:
        out.append(
            c.__class__(
                source_patterns = c.source_patterns,
                target_patterns = c.target_patterns,
                operations = c.operations,
                distributed_operation = "dist-x",
                quantization_operation = "quant-y",
            )
        )
    return out


class _FakePeftConfig:
    def __init__(self, target_modules = None, target_parameters = None):
        self.target_modules = target_modules
        self.target_parameters = target_parameters
        self.convert_calls = []


def _fake_convert_moe(peft_config, model_type):
    peft_config.convert_calls.append((model_type, set(peft_config.target_modules or [])))
    target_modules = set(peft_config.target_modules or [])
    converted = set()
    remaining = set()
    for target in target_modules:
        if target in {"gate_proj", "up_proj", "down_proj"}:
            converted.add("gate_up_proj" if target in {"gate_proj", "up_proj"} else "down_proj")
        else:
            remaining.add(target)
    peft_config.target_parameters = set(peft_config.target_parameters or []) | converted
    peft_config.target_modules = remaining


def test_two_arg_call_preserves_upstream_signature():
    twc = _install_fake_peft({"build_peft_weight_mapping": _build_that_calls_init})
    patch = _load_patch_function()
    patch()

    sig = inspect.signature(twc.build_peft_weight_mapping)
    assert "peft_config" in sig.parameters
    assert sig.parameters["peft_config"].default is None

    out = twc.build_peft_weight_mapping([_make_legacy_converter()], "default")
    assert len(out) == 1
    assert out[0].distributed_operation == "dist-x"
    assert out[0].quantization_operation == "quant-y"


def test_legacy_init_succeeds_after_patch():
    twc = _install_fake_peft({"build_peft_weight_mapping": _build_that_calls_init})
    patch = _load_patch_function()
    patch()

    out = twc.build_peft_weight_mapping([_make_legacy_converter()], "default", None)
    assert len(out) == 1
    assert out[0].distributed_operation == "dist-x"
    assert out[0].quantization_operation == "quant-y"


def test_modern_init_not_patched():
    twc = _install_fake_peft({"build_peft_weight_mapping": _build_that_calls_init})
    pre_init = _ModernConverter.__init__
    patch = _load_patch_function()
    patch()

    twc.build_peft_weight_mapping([_make_modern_converter()], "default", None)
    assert _ModernConverter.__init__ is pre_init


def test_class_init_restored_after_call():
    twc = _install_fake_peft({"build_peft_weight_mapping": _build_that_calls_init})
    pre_init = _LegacyConverter.__init__
    patch = _load_patch_function()
    patch()

    twc.build_peft_weight_mapping([_make_legacy_converter()], "default", None)
    assert _LegacyConverter.__init__ is pre_init


def test_class_init_restored_after_original_build_raises():
    def _raise(weight_conversions, adapter_name, peft_config = None):
        raise RuntimeError("simulated PEFT failure")

    twc = _install_fake_peft({"build_peft_weight_mapping": _raise})
    pre_init = _LegacyConverter.__init__
    patch = _load_patch_function()
    patch()

    with pytest.raises(RuntimeError):
        twc.build_peft_weight_mapping([_make_legacy_converter()], "default", None)
    assert _LegacyConverter.__init__ is pre_init


def test_partial_patch_restored_when_inspect_signature_raises_mid_loop():
    twc = _install_fake_peft({"build_peft_weight_mapping": _build_that_calls_init})
    pre_legacy = _LegacyConverter.__init__

    class _BadInitConverter:
        def __init__(self, source_patterns, target_patterns, operations):
            self.source_patterns = source_patterns
            self.target_patterns = target_patterns
            self.operations = operations

    pre_bad = _BadInitConverter.__init__
    patch = _load_patch_function()
    patch()

    real_signature = inspect.signature

    def _fake_signature(callable_):
        if callable_ is _BadInitConverter.__init__:
            raise ValueError("inspect.signature failed mid-loop")
        return real_signature(callable_)

    inspect.signature = _fake_signature
    try:
        legacy = _LegacyConverter(["src.*"], ["tgt.*"], [])
        bad = _BadInitConverter.__new__(_BadInitConverter)
        bad.source_patterns = ["src.*"]
        bad.target_patterns = ["tgt.*"]
        bad.operations = []
        with pytest.raises(ValueError):
            twc.build_peft_weight_mapping([legacy, bad], "default", None)
    finally:
        inspect.signature = real_signature

    assert _LegacyConverter.__init__ is pre_legacy
    assert _BadInitConverter.__init__ is pre_bad


def test_idempotent_install_does_not_double_wrap():
    twc = _install_fake_peft({"build_peft_weight_mapping": _build_that_calls_init})
    patch = _load_patch_function()
    patch()
    first_wrapped = twc.build_peft_weight_mapping
    patch()
    assert twc.build_peft_weight_mapping is first_wrapped


def test_moe_patch_installs_when_weight_converter_patch_already_present():
    twc = _install_fake_peft(
        {
            "build_peft_weight_mapping": _build_that_calls_init,
            "_convert_peft_config_moe": _fake_convert_moe,
            "_unsloth_weight_converter_compat_patch": True,
        }
    )
    patch = _load_patch_function()
    patch()

    config = _FakePeftConfig(
        target_modules = {"shared_experts.down_proj", "down_proj"},
        target_parameters = None,
    )
    twc._convert_peft_config_moe(config, "glm4_moe")

    assert config.convert_calls == [("glm4_moe", {"down_proj"})]
    assert config.target_modules == {"shared_experts.down_proj"}
    assert config.target_parameters == {"down_proj"}


def test_concurrent_legacy_calls_no_typeerror():
    import time

    def _slow_build(weight_conversions, adapter_name, peft_config = None):
        time.sleep(0.05)
        return _build_that_calls_init(weight_conversions, adapter_name, peft_config)

    twc = _install_fake_peft({"build_peft_weight_mapping": _slow_build})
    patch = _load_patch_function()
    patch()

    errors = []
    results = []
    start = threading.Event()

    def _worker():
        start.wait(timeout = 10)
        try:
            out = twc.build_peft_weight_mapping(
                [_make_legacy_converter()], "default", None
            )
            results.append(out)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target = _worker) for _ in range(8)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join(timeout = 15)

    assert errors == []
    assert len(results) == 8
    for out in results:
        assert out[0].distributed_operation == "dist-x"
        assert out[0].quantization_operation == "quant-y"
    assert _LegacyConverter.__init__.__qualname__.startswith("_LegacyConverter")


def test_empty_conversions_short_circuits_without_patching():
    twc = _install_fake_peft({"build_peft_weight_mapping": _build_that_calls_init})
    pre_init = _LegacyConverter.__init__
    patch = _load_patch_function()
    patch()

    out = twc.build_peft_weight_mapping([], "default", None)
    assert out == []
    assert _LegacyConverter.__init__ is pre_init


def test_moe_conversion_skips_when_target_parameters_already_set():
    twc = _install_fake_peft(
        {
            "build_peft_weight_mapping": _build_that_calls_init,
            "_convert_peft_config_moe": _fake_convert_moe,
        }
    )
    patch = _load_patch_function()
    patch()

    config = _FakePeftConfig(
        target_modules = {"gate_proj", "up_proj"},
        target_parameters = {"mlp.experts.gate_up_proj"},
    )
    twc._convert_peft_config_moe(config, "qwen3_moe")

    assert config.convert_calls == []
    assert config.target_modules == {"gate_proj", "up_proj"}
    assert config.target_parameters == {"mlp.experts.gate_up_proj"}


def test_moe_conversion_preserves_qualified_targets_as_explicit():
    twc = _install_fake_peft(
        {
            "build_peft_weight_mapping": _build_that_calls_init,
            "_convert_peft_config_moe": _fake_convert_moe,
        }
    )
    patch = _load_patch_function()
    patch()

    config = _FakePeftConfig(
        target_modules = {"shared_experts.down_proj", "down_proj"},
        target_parameters = None,
    )
    twc._convert_peft_config_moe(config, "glm4_moe")

    assert config.convert_calls == [("glm4_moe", {"down_proj"})]
    assert config.target_modules == {"shared_experts.down_proj"}
    assert config.target_parameters == {"down_proj"}


def test_moe_conversion_leaves_only_qualified_targets_unconverted():
    twc = _install_fake_peft(
        {
            "build_peft_weight_mapping": _build_that_calls_init,
            "_convert_peft_config_moe": _fake_convert_moe,
        }
    )
    patch = _load_patch_function()
    patch()

    config = _FakePeftConfig(
        target_modules = {"shared_expert.up_proj"},
        target_parameters = None,
    )
    twc._convert_peft_config_moe(config, "qwen3_next")

    assert config.convert_calls == []
    assert config.target_modules == {"shared_expert.up_proj"}
    assert config.target_parameters is None
