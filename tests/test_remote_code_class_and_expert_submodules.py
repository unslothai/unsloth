# SPDX-License-Identifier: AGPL-3.0-only
"""Remote code whose config shares a native class name (Nemotron-H hub checkpoints), and
LoRA targets on per-expert submodules (`mixer.experts.<i>.up_proj`)."""

import re
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _utils():
    from unsloth.models import _utils
    return _utils


# --------------------------------------------------------------------------------------
# flash attention class flag
# --------------------------------------------------------------------------------------


def test_old_flag_alone_is_not_flash_support_on_new_transformers():
    U = _utils()
    from transformers.modeling_utils import PreTrainedModel

    class OldRemote:
        _supports_flash_attn_2 = True

    class NewNative:
        _supports_flash_attn = True

    class Neither:
        pass

    # 5.0 to 5.3 define the new flag but their dispatch check still accepts the old one.
    legacy_ok = not hasattr(PreTrainedModel, "_supports_flash_attn") or (
        U._flash_dispatch_reads_legacy_flag(PreTrainedModel)
    )
    assert U._model_class_supports_flash_attention(OldRemote) is legacy_ok
    assert U._model_class_supports_flash_attention(NewNative) is True
    assert U._model_class_supports_flash_attention(Neither) is False
    assert U._model_class_supports_flash_attention(None) is False


def test_resolver_does_not_request_flash_for_old_flag_remote_class(monkeypatch):
    U = _utils()
    # Without flash-attn installed the ladder never reaches flash, and this would pass either way.
    monkeypatch.setattr(U, "HAS_FLASH_ATTENTION", True)
    from transformers.modeling_utils import PreTrainedModel

    if not hasattr(PreTrainedModel, "_supports_flash_attn") or (
        U._flash_dispatch_reads_legacy_flag(PreTrainedModel)
    ):
        pytest.skip("transformers still dispatches on _supports_flash_attn_2")

    class OldRemote:
        _supports_flash_attn_2 = True
        _supports_sdpa = True

    class NewRemote:
        _supports_flash_attn = True
        _supports_sdpa = True

    config = SimpleNamespace(model_type = "nemotron_h", _attn_implementation = None)
    impl = U.resolve_attention_implementation(OldRemote, config, dtype = torch.bfloat16)
    assert "flash" not in str(impl)
    # Control: the same config does reach flash for a class carrying the dispatched flag.
    config = SimpleNamespace(model_type = "nemotron_h", _attn_implementation = None)
    impl = U.resolve_attention_implementation(NewRemote, config, dtype = torch.bfloat16)
    assert impl == "flash_attention_2"


# --------------------------------------------------------------------------------------
# remote class resolution
# --------------------------------------------------------------------------------------


def _install_fake_remote_modules(monkeypatch, package = "transformers_modules.fake_repo.abc123"):
    """A remote config/model pair whose config class name collides with a native one."""
    from transformers import PretrainedConfig
    from transformers.models.llama.configuration_llama import LlamaConfig  # noqa: F401  (the collision target)

    pkg = types.ModuleType(package)
    cfg_mod = types.ModuleType(package + ".configuration_llama")
    model_mod = types.ModuleType(package + ".modeling_llama")

    class LlamaConfig(PretrainedConfig):  # same name as the native class on purpose
        model_type = "llama"

    class LlamaForCausalLM:
        _supports_flash_attn_2 = True

    LlamaConfig.__module__ = cfg_mod.__name__
    LlamaForCausalLM.__module__ = model_mod.__name__
    cfg_mod.LlamaConfig = LlamaConfig
    model_mod.LlamaForCausalLM = LlamaForCausalLM
    for m in (pkg, cfg_mod, model_mod):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    config = LlamaConfig()
    config.auto_map = {
        "AutoConfig": "configuration_llama.LlamaConfig",
        "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM",
    }
    config._name_or_path = "fake/repo"
    return config, LlamaForCausalLM


def test_remote_config_resolves_to_remote_model_class(monkeypatch):
    U = _utils()
    from transformers import AutoModelForCausalLM

    config, remote_cls = _install_fake_remote_modules(monkeypatch)
    assert U.resolve_model_class(AutoModelForCausalLM, config) is remote_cls


def test_fetch_of_a_missing_modeling_module_uses_the_load_options(monkeypatch):
    """A cold-cache fetch uses the load's revision, token and offline flag."""
    U = _utils()
    from transformers import AutoModelForCausalLM
    import transformers.dynamic_module_utils as dmu

    config, _ = _install_fake_remote_modules(monkeypatch)
    monkeypatch.delitem(sys.modules, "transformers_modules.fake_repo.abc123.modeling_llama")
    seen = {}

    class Fetched:
        pass

    def fake_get(class_ref, repo_id, **kw):
        seen.update(class_ref = class_ref, repo_id = repo_id, **kw)
        return Fetched

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", fake_get)
    got = U.resolve_model_class(
        AutoModelForCausalLM,
        config,
        trust_remote_code = True,
        revision = "deadbeef",
        code_revision = "cafe",
        token = "tok",
        cache_dir = "/c",
        local_files_only = True,
    )
    assert got is Fetched
    assert seen == dict(
        class_ref = "modeling_llama.LlamaForCausalLM",
        repo_id = "fake/repo",
        revision = "deadbeef",
        code_revision = "cafe",
        token = "tok",
        cache_dir = "/c",
        local_files_only = True,
    )


def test_cross_repository_auto_map_skips_the_local_sibling(monkeypatch):
    """An `other/repo--module.Class` reference must not resolve to a same-named sibling module."""
    U = _utils()
    from transformers import AutoModelForCausalLM
    import transformers.dynamic_module_utils as dmu

    config, local_cls = _install_fake_remote_modules(monkeypatch)
    config.auto_map["AutoModelForCausalLM"] = "other/repo--modeling_llama.LlamaForCausalLM"
    seen = {}

    class Remote:
        pass

    def fake_get(class_ref, repo_id, **kw):
        seen.update(class_ref = class_ref, repo_id = repo_id)
        return Remote

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", fake_get)
    got = U.resolve_model_class(AutoModelForCausalLM, config, trust_remote_code = True)
    assert got is Remote and got is not local_cls
    # Unsplit, with the model path, as from_pretrained calls it.
    assert seen == dict(
        class_ref = "other/repo--modeling_llama.LlamaForCausalLM", repo_id = "fake/repo"
    )


def test_native_config_keeps_native_resolution():
    U = _utils()
    from transformers import AutoModelForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig()
    config.auto_map = {
        "AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM"
    }  # ignored: not remote
    assert U.resolve_model_class(AutoModelForCausalLM, config) is LlamaForCausalLM


def test_remote_config_without_auto_class_entry_stays_native(monkeypatch):
    U = _utils()
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    config, _ = _install_fake_remote_modules(monkeypatch)
    config.auto_map = {"AutoConfig": "configuration_llama.LlamaConfig"}
    assert U.resolve_model_class(AutoModelForCausalLM, config) is LlamaForCausalLM


# --------------------------------------------------------------------------------------
# per-expert submodule LoRA targets
# --------------------------------------------------------------------------------------


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(8, 16, bias = False)
        self.down_proj = torch.nn.Linear(16, 8, bias = False)


class _MoE(torch.nn.Module):
    def __init__(self, n = 4):
        super().__init__()
        self.experts = torch.nn.ModuleList([_Expert() for _ in range(n)])
        self.shared_experts = _Expert()
        self.fc1_latent_proj = torch.nn.Identity()
        self.gate = _Router(n)


# Nemotron-Labs-Teacher's expert classes come from the checkpoint's own modeling file.
_Expert.__module__ = "transformers_modules.fake_teacher.modeling_nemotron_h"
sys.modules.setdefault(_Expert.__module__, sys.modules[__name__])


class _Router(torch.nn.Module):  # a Parameter-backed router, as in Nemotron-H
    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(n, 8))


class _Mamba(torch.nn.Module):  # a mixer with a Linear directly under it, like the Mamba layers
    def __init__(self):
        super().__init__()
        self.in_proj = torch.nn.Linear(8, 32, bias = False)


class _Layer(torch.nn.Module):
    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer


class _Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_Layer(_Mamba()), _Layer(_MoE()), _Layer(_Mamba()), _Layer(_MoE())]
        )


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_routed_experts = 4, model_type = "nemotron_h")
        self.model = _Inner()


def _text_only_regex():
    """The default FastModel.get_peft_model regex misses nested experts in a text-only model."""
    import importlib

    stub = sys.modules.get("unsloth_zoo.peft_utils")
    if stub is not None and getattr(stub, "__file__", None) is None:
        # Another test file leaves a stub in sys.modules; the real module is wanted here.
        del sys.modules["unsloth_zoo.peft_utils"]
    peft_utils = importlib.import_module("unsloth_zoo.peft_utils")
    return peft_utils.get_peft_regex(_Model())


def test_text_only_regex_misses_nested_experts_on_its_own():
    regex = _text_only_regex()
    names = [n for n, m in _Model().named_modules() if isinstance(m, torch.nn.Linear)]
    assert not any(re.fullmatch(regex, n) for n in names if ".experts." in n)


def test_expert_submodule_leaves_and_regex_reach_every_expert():
    U = _utils()
    model = _Model()
    regex = _text_only_regex()
    leaves = U.get_moe_expert_submodule_leaves(model, regex)
    assert leaves == ["down_proj", "up_proj"]
    extended = f"(?:{regex})|(?:{U.moe_expert_submodule_regex(leaves)})"
    matched = {n for n, m in model.named_modules() if re.fullmatch(extended, n)}
    linears = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    experts = {n for n in linears if ".experts." in n or ".shared_experts." in n}
    assert experts <= matched
    assert "model.layers.1.mixer.gate" not in matched  # the router is not a Linear
    assert "model.layers.1.mixer.fc1_latent_proj" not in matched  # Identity, not a Linear
    assert "model.layers.0.mixer.in_proj" in matched  # the block-level leaves stay
    assert all(isinstance(dict(model.named_modules())[n], torch.nn.Linear) for n in matched)


def test_expert_submodule_leaves_follow_the_request():
    U = _utils()
    model = _Model()
    assert U.get_moe_expert_submodule_leaves(model, ["down_proj"]) == ["down_proj"]
    assert U.get_moe_expert_submodule_leaves(model, ["q_proj", "k_proj"]) == []
    assert U.get_moe_expert_submodule_leaves(model, None) == []


def test_non_moe_model_adds_nothing():
    U = _utils()
    model = _Model()
    model.config = SimpleNamespace(model_type = "llama")
    assert U.get_moe_expert_submodule_leaves(model, ["up_proj", "down_proj"]) == []


def test_only_a_generated_regex_is_widened_to_the_routed_experts():
    """A caller-written regex is never widened; the generated text-only regex is."""
    U = _utils()
    model = _Model()
    caller_regex = r".*\.shared_experts\.down_proj"
    kept, detect, leaves = U.widen_target_regex_to_expert_submodules(
        model, caller_regex, caller_regex, auto_regex = False
    )
    assert kept == caller_regex and detect == caller_regex and leaves == []
    matched = {n for n, _ in model.named_modules() if re.fullmatch(kept, n)}
    assert matched and all(".shared_experts." in n for n in matched)

    generated = _text_only_regex()
    widened, detect, leaves = U.widen_target_regex_to_expert_submodules(
        model, generated, generated, auto_regex = True
    )
    assert leaves == ["down_proj", "up_proj"]
    assert detect == widened != generated
    matched = {n for n, _ in model.named_modules() if re.fullmatch(widened, n)}
    assert any(".experts." in n for n in matched)

    # A leaf list as the detection target keeps its own identity through the widening.
    widened, detect, leaves = U.widen_target_regex_to_expert_submodules(
        model, generated, ["down_proj"], auto_regex = True
    )
    assert leaves == ["down_proj"] and detect == ["down_proj"] and widened != generated


def test_gate_and_up_expert_leaves_stay_separate():
    U = _utils()
    model = _Model()
    assert U.get_moe_expert_submodule_leaves(model, ["up_proj"]) == ["up_proj"]
    assert (
        U.get_moe_expert_submodule_leaves(model, ["gate_proj"]) == []
    )  # the fixture has no gate leaf
    assert U.get_moe_expert_submodule_leaves(model, ["gate_up_proj"]) == ["up_proj"]


def test_legacy_flash_flag_follows_the_installed_dispatch_check():
    """The helper reads the installed dispatch check for the legacy flag."""
    U = _utils()

    class OldDispatch:
        _supports_flash_attn = False

        def _flash_attn_can_dispatch(self):
            if not (self._supports_flash_attn or getattr(self, "_supports_flash_attn_2", False)):
                raise ValueError("no")

    class NewDispatch:
        _supports_flash_attn = False

        def _flash_attn_can_dispatch(self):
            if not self._supports_flash_attn:
                message = "x"
                if self._supports_flash_attn or getattr(self, "_supports_flash_attn_2", False):
                    message += ", "
                raise ValueError(message)

    assert U._flash_dispatch_reads_legacy_flag(OldDispatch) is True
    assert U._flash_dispatch_reads_legacy_flag(NewDispatch) is False


def test_force_download_bypasses_the_imported_sibling(monkeypatch):
    U = _utils()
    calls = []

    def fake_get_class(class_ref, repo_id, **kw):
        calls.append((class_ref, repo_id, kw))
        return type("Built", (), {})

    import transformers.dynamic_module_utils as dmu

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", fake_get_class)
    import sys, types

    module = types.ModuleType("transformers_modules.fake.configuration_fake")
    sys.modules["transformers_modules.fake.configuration_fake"] = module
    sibling = types.ModuleType("transformers_modules.fake.modeling_fake")
    sibling.FakeForCausalLM = type("FakeForCausalLM", (), {})
    sys.modules["transformers_modules.fake.modeling_fake"] = sibling
    try:
        cfg_cls = type("FakeConfig", (), {})
        cfg_cls.__module__ = "transformers_modules.fake.configuration_fake"
        cfg = cfg_cls()
        cfg.auto_map = {"AutoModelForCausalLM": "modeling_fake.FakeForCausalLM"}
        cfg._name_or_path = "fake/repo"
        auto = type("AutoModelForCausalLM", (), {})
        assert U._resolve_remote_model_class(auto, cfg) is sibling.FakeForCausalLM
        assert calls == []
        built = U._resolve_remote_model_class(
            auto, cfg, trust_remote_code = True, force_download = True
        )
        assert (
            built is not sibling.FakeForCausalLM
            and calls
            and calls[0][2].get("force_download") is True
        )
        assert U._resolve_remote_model_class(auto, cfg, trust_remote_code = False) is None
    finally:
        sys.modules.pop("transformers_modules.fake.configuration_fake", None)
        sys.modules.pop("transformers_modules.fake.modeling_fake", None)


def test_every_resolver_probe_forwards_the_trust_decision():
    """Class probes pass trust_remote_code and the load's hub kwargs."""
    import ast, inspect
    from unsloth.models import llama, loader, loader_utils, vision

    probes = ("resolve_model_class", "_resolve_omni_auto_model")
    planner = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(loader_utils)))
        if isinstance(node, ast.FunctionDef) and node.name == "planner_model_class"
    )
    for module, tree in (
        (loader, ast.parse(inspect.getsource(loader))),
        (vision, ast.parse(inspect.getsource(vision))),
        (llama, ast.parse(inspect.getsource(llama))),
        (loader_utils, planner),
    ):
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) in probes):
                continue
            names = {k.arg for k in node.keywords}
            splats = [getattr(k.value, "id", "") for k in node.keywords if k.arg is None]
            assert "trust_remote_code" in names or "_probe_hub_kwargs" in splats, (
                module.__name__,
                node.lineno,
            )
            if module in (loader, vision) and not splats:
                assert {"revision", "token", "local_files_only"} <= names, (
                    module.__name__,
                    node.lineno,
                )


def test_a_code_revision_skips_the_materialised_sibling(monkeypatch):
    """With a code_revision the resolver must ask transformers, not the sibling module."""
    import importlib
    import types

    from unsloth.models import _utils

    sibling = types.ModuleType("transformers_modules.tiny_rev.modeling_tiny")

    class FromConfigRevision:
        pass

    class FromCodeRevision:
        pass

    sibling.TinyForCausalLM = FromConfigRevision
    config_cls = type(
        "TinyConfig", (), {"__module__": "transformers_modules.tiny_rev.configuration_tiny"}
    )
    config = config_cls()
    config.auto_map = {"AutoModelForCausalLM": "modeling_tiny.TinyForCausalLM"}
    config._name_or_path = "someone/tiny"
    monkeypatch.setitem(__import__("sys").modules, sibling.__name__, sibling)
    import transformers.dynamic_module_utils as dmu

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", lambda *a, **k: FromCodeRevision)
    auto = type("AutoModelForCausalLM", (), {})
    assert _utils._resolve_remote_model_class(auto, config) is FromConfigRevision
    assert (
        _utils._resolve_remote_model_class(
            auto, config, trust_remote_code = True, code_revision = "abc123"
        )
        is FromCodeRevision
    )
    # A load revision is the code revision for same-repo code, so it skips the sibling too.
    assert (
        _utils._resolve_remote_model_class(auto, config, trust_remote_code = True, revision = "b")
        is FromCodeRevision
    )


def test_native_per_expert_layouts_are_not_widened():
    """Qwen3-MoE on transformers 4.x nests native per-expert Linears the same way; those
    keep main's targets instead of gaining LoRA on every routed expert."""
    U = _utils()

    class _NativeExpert(_Expert):
        pass

    _NativeExpert.__module__ = "transformers.models.qwen3_moe.modeling_qwen3_moe"
    model = _Model()
    for layer in model.model.layers:
        mixer = layer.mixer
        if isinstance(mixer, _MoE):
            mixer.experts = torch.nn.ModuleList([_NativeExpert() for _ in mixer.experts])
            mixer.shared_experts = _NativeExpert()
    generated = _text_only_regex()
    kept, detect, leaves = U.widen_target_regex_to_expert_submodules(
        model, generated, generated, auto_regex = True
    )
    assert kept == generated and detect == generated and leaves == []


def test_a_native_expert_tower_next_to_remote_code_is_not_widened():
    """Only the remote expert blocks' own parents gain the nested alternative."""
    U = _utils()

    class _NativeExpert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = torch.nn.Linear(8, 16, bias = False)
            self.down_proj = torch.nn.Linear(16, 8, bias = False)

    class _NativeTower(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([_NativeExpert() for _ in range(4)])

    _NativeExpert.__module__ = "transformers.models.some_vlm.modeling_some_vlm"
    model = _Model()
    model.visual = torch.nn.ModuleList([_NativeTower(), _NativeTower()])
    generated = _text_only_regex()
    widened, _, leaves = U.widen_target_regex_to_expert_submodules(
        model, generated, generated, auto_regex = True
    )
    assert leaves == ["down_proj", "up_proj"]
    matched = {n for n, _ in model.named_modules() if re.fullmatch(widened, n)}
    assert any(n.startswith("model.layers.") and ".experts." in n for n in matched)
    assert not any(n.startswith("visual.") for n in matched)


def test_a_load_that_did_not_ask_for_remote_code_never_fetches(monkeypatch):
    """Without trust_remote_code the resolver may reuse an imported sibling, never the Hub."""
    import transformers.dynamic_module_utils as dmu
    from unsloth.models import _utils

    fetched = []
    monkeypatch.setattr(
        dmu, "get_class_from_dynamic_module", lambda *a, **k: fetched.append(k) or object
    )
    config_cls = type(
        "TinyConfig", (), {"__module__": "transformers_modules.not_imported.configuration_tiny"}
    )
    config = config_cls()
    config.auto_map = {"AutoModel": "modeling_tiny.TinyModel"}
    config._name_or_path = "someone/tiny"
    auto = type("AutoModel", (), {})
    for trust in (None, False):
        assert _utils._resolve_remote_model_class(auto, config, trust_remote_code = trust) is None
    assert fetched == []


def test_sentence_transformer_probes_use_the_loads_options(monkeypatch):
    """Both FastSentenceTransformer class probes see the load's trust, revision, token,
    cache and offline mode."""
    import ast
    import inspect

    from unsloth.models import _utils, sentence_transformer

    seen = []
    monkeypatch.setattr(
        _utils, "resolve_model_class", lambda auto, config, **kw: seen.append(kw) or None
    )
    options = dict(trust_remote_code = True, revision = "abc", token = "t", local_files_only = True)
    _utils.resolve_encoder_attention_implementation(object, object(), **options)
    monkeypatch.setattr(
        sentence_transformer, "resolve_model_class", lambda auto, config, **kw: seen.append(kw)
    )
    sentence_transformer.FastSentenceTransformer._has_add_pooling_layer(object(), object, **options)
    assert seen == [options, options]
    probes = {"resolve_encoder_attention_implementation", "_has_add_pooling_layer"}
    calls = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(sentence_transformer)))
        if isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) in probes or getattr(node.func, "attr", None) in probes)
    ]
    assert len(calls) == 2
    for call in calls:
        splats = [getattr(k.value, "id", "") for k in call.keywords if k.arg is None]
        assert "_remote_class_probe_kwargs" in splats, call.lineno
