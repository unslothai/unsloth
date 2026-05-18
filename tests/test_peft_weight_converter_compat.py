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
    attr_missing = object()
    saved = {
        k: sys.modules.get(k)
        for k in (
            "peft",
            "peft.utils",
            "peft.utils.transformers_weight_conversion",
        )
    }
    saved_twc = saved["peft.utils.transformers_weight_conversion"]
    saved_twc_attrs = {}
    if saved_twc is not None:
        for attr in (
            "build_peft_weight_mapping",
            "convert_peft_config_for_transformers",
            "_unsloth_weight_converter_compat_patch",
            "_unsloth_mixtral_unfused_target_patch",
        ):
            saved_twc_attrs[attr] = getattr(saved_twc, attr, attr_missing)

    yield

    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    if saved_twc is not None:
        for attr, value in saved_twc_attrs.items():
            if value is attr_missing:
                if hasattr(saved_twc, attr):
                    delattr(saved_twc, attr)
            else:
                setattr(saved_twc, attr, value)


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
    def __init__(self, target_modules):
        self.target_modules = target_modules
        self.target_parameters = None
        self.convert_calls = []


class _FakeModelConfig:
    model_type = "mixtral"


class _FakeNonMixtralConfig:
    model_type = "qwen3_moe"


class _FakeUnfusedMixtralModel:
    config = _FakeModelConfig()

    def named_parameters(self):
        for name in (
            "model.layers.0.mlp.experts.0.w1.weight",
            "model.layers.0.mlp.experts.0.w2.weight",
            "model.layers.0.mlp.experts.0.w3.weight",
        ):
            yield name, object()


class _FakeFusedMixtralModel:
    config = _FakeModelConfig()

    def named_parameters(self):
        for name in (
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
        ):
            yield name, object()


class _FakeNonMixtralModel(_FakeUnfusedMixtralModel):
    config = _FakeNonMixtralConfig()


def _convert_that_fuses_mixtral(peft_config, model, conversions):
    peft_config.convert_calls.append(
        (
            getattr(model.config, "model_type", None),
            set(peft_config.target_modules or ()),
            conversions,
        )
    )
    if getattr(model.config, "model_type", None) == "mixtral":
        target_modules = set(peft_config.target_modules or ())
        if {"w1", "w3"}.intersection(target_modules):
            target_modules.difference_update({"w1", "w2", "w3"})
            peft_config.target_modules = target_modules
            peft_config.target_parameters = {"gate_up_proj", "down_proj"}


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


def test_mixtral_unfused_targets_skip_peft_moe_conversion():
    twc = _install_fake_peft(
        {
            "build_peft_weight_mapping": _build_that_calls_init,
            "convert_peft_config_for_transformers": _convert_that_fuses_mixtral,
        }
    )
    patch = _load_patch_function()
    patch()

    config = _FakePeftConfig(
        target_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"}
    )
    twc.convert_peft_config_for_transformers(
        config, _FakeUnfusedMixtralModel(), conversions = ["conversion"]
    )

    assert config.convert_calls == []
    assert config.target_modules == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "w1",
        "w2",
        "w3",
    }
    assert config.target_parameters is None


def test_mixtral_fused_layout_still_delegates_to_peft_conversion():
    twc = _install_fake_peft(
        {
            "build_peft_weight_mapping": _build_that_calls_init,
            "convert_peft_config_for_transformers": _convert_that_fuses_mixtral,
        }
    )
    patch = _load_patch_function()
    patch()

    config = _FakePeftConfig(target_modules = {"w1", "w2", "w3"})
    twc.convert_peft_config_for_transformers(
        config, _FakeFusedMixtralModel(), conversions = ["conversion"]
    )

    assert config.convert_calls == [("mixtral", {"w1", "w2", "w3"}, ["conversion"])]
    assert config.target_modules == set()
    assert config.target_parameters == {"gate_up_proj", "down_proj"}


def test_non_mixtral_layout_still_delegates_to_peft_conversion():
    twc = _install_fake_peft(
        {
            "build_peft_weight_mapping": _build_that_calls_init,
            "convert_peft_config_for_transformers": _convert_that_fuses_mixtral,
        }
    )
    patch = _load_patch_function()
    patch()

    config = _FakePeftConfig(target_modules = {"w1", "w2", "w3"})
    twc.convert_peft_config_for_transformers(
        config, _FakeNonMixtralModel(), conversions = ["conversion"]
    )

    assert config.convert_calls == [("qwen3_moe", {"w1", "w2", "w3"}, ["conversion"])]
    assert config.target_modules == {"w1", "w2", "w3"}
    assert config.target_parameters is None


def test_real_peft_keeps_lora_on_unfused_mixtral_w_modules():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    patch = _load_patch_function()
    patch()

    class Config:
        model_type = "mixtral"
        num_local_experts = 2
        tie_word_embeddings = False

        def get(self, key, default = None):
            return getattr(self, key, default)

        def to_dict(self):
            return {
                "model_type": self.model_type,
                "num_local_experts": self.num_local_experts,
                "tie_word_embeddings": self.tie_word_embeddings,
            }

    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Linear(8, 4, bias = False)
            self.w2 = torch.nn.Linear(4, 8, bias = False)
            self.w3 = torch.nn.Linear(8, 4, bias = False)

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([Expert(), Expert()])

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias = False)
            self.k_proj = torch.nn.Linear(8, 8, bias = False)
            self.v_proj = torch.nn.Linear(8, 8, bias = False)
            self.o_proj = torch.nn.Linear(8, 8, bias = False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()
            self.mlp = Mlp()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Config()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer()])

        def forward(self, input_ids = None, **kwargs):
            return None

    config = LoraConfig(
        r = 2,
        lora_alpha = 2,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
    )
    model = get_peft_model(Model(), config)
    peft_config = model.peft_config["default"]
    lora_names = [name for name, _ in model.named_parameters() if "lora_" in name]

    assert peft.__version__
    assert {"w1", "w2", "w3"}.issubset(peft_config.target_modules)
    assert peft_config.target_parameters is None
    assert any(".w1.lora_A." in name for name in lora_names)
    assert any(".w2.lora_A." in name for name in lora_names)
    assert any(".w3.lora_A." in name for name in lora_names)
    assert not any("gate_up_proj" in name for name in lora_names)
