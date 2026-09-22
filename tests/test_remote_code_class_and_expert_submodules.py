# SPDX-License-Identifier: AGPL-3.0-only
"""Remote code whose config shares a native class name, and per-expert submodule LoRA targets.

The Nemotron-H hub checkpoints (NVIDIA-Nemotron-Labs-Teacher) ship `NemotronHConfig` and
`NemotronHForCausalLM` as remote code. transformers' auto mapping is keyed by config class
name, so `resolve_model_class` returned the native `NemotronHForCausalLM`, whose flags said
flash attention is fine; the remote class that is actually built only carries the old
`_supports_flash_attn_2` flag, which transformers 5 no longer dispatches on, and the load
stopped with "does not support Flash Attention 2 yet". Their routed experts also live one
level below the block (`mixer.experts.<i>.up_proj`), where the text-only LoRA regex never
looked, so the experts trained inside the Omni wrapper but not in the standalone model.
"""
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

    if hasattr(PreTrainedModel, "_supports_flash_attn"):
        assert U._model_class_supports_flash_attention(OldRemote) is False
    else:
        assert U._model_class_supports_flash_attention(OldRemote) is True
    assert U._model_class_supports_flash_attention(NewNative) is True
    assert U._model_class_supports_flash_attention(Neither) is False
    assert U._model_class_supports_flash_attention(None) is False


def test_resolver_does_not_request_flash_for_old_flag_remote_class():
    U = _utils()
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "_supports_flash_attn"):
        pytest.skip("transformers still dispatches on _supports_flash_attn_2")

    class OldRemote:
        _supports_flash_attn_2 = True
        _supports_sdpa = True

    config = SimpleNamespace(model_type = "nemotron_h", _attn_implementation = None)
    impl = U.resolve_attention_implementation(OldRemote, config, dtype = torch.bfloat16)
    assert "flash" not in str(impl)


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
    """On a cold cache only the configuration module exists; the modeling module is fetched
    with the same revision, credentials and offline flag the load will use."""
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
        AutoModelForCausalLM, config,
        revision = "deadbeef", code_revision = "cafe", token = "tok", cache_dir = "/c",
        local_files_only = True,
    )
    assert got is Fetched
    assert seen == dict(
        class_ref = "modeling_llama.LlamaForCausalLM", repo_id = "fake/repo",
        revision = "deadbeef", code_revision = "cafe", token = "tok", cache_dir = "/c",
        local_files_only = True,
    )


def test_cross_repository_auto_map_skips_the_local_sibling(monkeypatch):
    """`other/repo--module.Class` names a class in another repository: a same-named module
    next to the config must not answer for it."""
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
    got = U.resolve_model_class(AutoModelForCausalLM, config)
    assert got is Remote and got is not local_cls
    assert seen == dict(class_ref = "modeling_llama.LlamaForCausalLM", repo_id = "other/repo")


def test_native_config_keeps_native_resolution():
    U = _utils()
    from transformers import AutoModelForCausalLM, LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    config = LlamaConfig()
    config.auto_map = {"AutoModelForCausalLM": "modeling_llama.LlamaForCausalLM"}  # ignored: not remote
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
        self.layers = torch.nn.ModuleList([_Layer(_Mamba()), _Layer(_MoE()), _Layer(_Mamba()), _Layer(_MoE())])


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_routed_experts = 4, model_type = "nemotron_h")
        self.model = _Inner()


def _text_only_regex():
    """The regex FastModel.get_peft_model builds with its defaults (every family on): the
    tagged branch needs a `language` / `text` component in the name and the untagged branch
    stops one level under the block, so a standalone text model's nested experts miss both."""
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
    assert "model.layers.1.mixer.gate" not in matched            # the router is not a Linear
    assert "model.layers.1.mixer.fc1_latent_proj" not in matched  # Identity, not a Linear
    assert "model.layers.0.mixer.in_proj" in matched              # the block-level leaves stay
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
