# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""text_only on repo-code composites: load only the language model when the checkpoint stores it whole under one prefix."""

import ast
import copy
import functools
import json
import re
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REPO_ROOT = Path(__file__).resolve().parents[2]

from packaging.version import Version as _V

# The remote text-only plan needs transformers 5 key_mapping semantics; on 4.x it must decline.
needs_tf5 = pytest.mark.skipif(
    _V(transformers.__version__) < _V("5.0.0"), reason = "plan declines on transformers 4.x"
)
UTILS_PATH = REPO_ROOT / "unsloth" / "models" / "_utils.py"
LOADER_PATH = REPO_ROOT / "unsloth" / "models" / "loader.py"
VISION_PATH = REPO_ROOT / "unsloth" / "models" / "vision.py"

_HELPERS = (
    "_resolve_remote_model_class",
    "resolve_model_class",
    "_is_family_text_decoder",
    "_remap_text_only_skip_modules",
    "_get_text_only_config",
    "_is_remote_code_config",
    "_checkpoint_weight_names",
    "_infer_text_submodel_prefix",
    "_resolve_text_causal_lm_class",
    "_meta_parameter_names",
    "_strip_skip_module_prefix",
    "_get_remote_composite_text_only",
    "_merge_key_mapping",
    "_rebase_user_quantization_config",
    "_drop_text_only_key_mapping",
    "_trusted_remote_code_commit",
    "_adapter_fits_text_model",
)

_CONSTANTS = ("_REMOTE_CODE_HUB_KWARGS",)


def _ns():
    from packaging.version import Version

    source = UTILS_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    funcs = {
        n.name: ast.get_source_segment(source, n)
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    ns = {
        "copy": copy,
        "re": re,
        "Version": Version,
        "transformers_version": transformers.__version__,
    }
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in _CONSTANTS for t in n.targets
        ):
            exec(ast.get_source_segment(source, n), ns)
    for name in _HELPERS:
        if name in funcs:
            exec(funcs[name], ns)
    return ns


# Nemotron-3-Nano-Omni shape: llm_config (stock text decoder), vision config, whole causal LM kept as `language_model`.
_CONFIGURATION = """
from transformers import PretrainedConfig, LlamaConfig


class TinyVisionConfig(PretrainedConfig):
    model_type = "tiny_vision"

    def __init__(self, hidden_size = 8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size


class TinyOmniConfig(PretrainedConfig):
    model_type = "tiny_omni_remote"
    is_composition = True

    def __init__(self, llm_config = None, vision_config = None, **kwargs):
        super().__init__(**kwargs)
        self.llm_config = LlamaConfig(**(llm_config or {}))
        self.vision_config = TinyVisionConfig(**(vision_config or {}))

    @property
    def text_config(self):
        return self.llm_config
"""

_MODELING = """
import torch
from torch import nn
from transformers import PreTrainedModel, LlamaForCausalLM
from .configuration_tiny_omni import TinyOmniConfig


class TinyOmni(PreTrainedModel):
    config_class = TinyOmniConfig
    _no_split_modules = []

    def __init__(self, config):
        super().__init__(config)
        self.language_model = LlamaForCausalLM(config.llm_config)
        self.vision_model = nn.Linear(config.vision_config.hidden_size, config.vision_config.hidden_size)
        self.mlp1 = nn.Linear(config.vision_config.hidden_size, config.llm_config.hidden_size)
        self.post_init()

    def forward(self, pixel_values, input_ids = None, **kwargs):
        return self.language_model(input_ids = input_ids, **kwargs)


class TinyTextLM(LlamaForCausalLM):
    # A repo-code text decoder, named by the nested llm_config auto_map (Nemotron-Omni's NemotronHForCausalLM shape).
    pass
"""


def _llama_kwargs(**extra):
    kw = dict(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 64,
        max_position_embeddings = 64,
        tie_word_embeddings = False,
    )
    kw.update(extra)
    return kw


def _write_repo(
    tmp_path,
    *,
    prefix = "language_model.",
    drop = (),
    llm_extra = None,
    name = "tiny_omni",
    alias = True,
    text_auto_map = None,
):
    from safetensors.torch import save_file

    repo = tmp_path / name
    repo.mkdir()
    configuration = _CONFIGURATION
    if not alias:
        # InternVL / Nemotron-Nano-VL shape: llm_config only, no text_config alias for get_text_config() to find.
        configuration = configuration.replace(
            "    @property\n    def text_config(self):\n        return self.llm_config\n", ""
        )
        assert "def text_config" not in configuration
    (repo / "configuration_tiny_omni.py").write_text(configuration)
    (repo / "modeling_tiny_omni.py").write_text(_MODELING)
    llm = LlamaConfigDict = _llama_kwargs(**(llm_extra or {}))
    cfg = {
        "model_type": "tiny_omni_remote",
        "architectures": ["TinyOmni"],
        "auto_map": {
            "AutoConfig": "configuration_tiny_omni.TinyOmniConfig",
            "AutoModel": "modeling_tiny_omni.TinyOmni",
            "AutoModelForCausalLM": "modeling_tiny_omni.TinyOmni",
        },
        "llm_config": {"model_type": "llama", "architectures": ["LlamaForCausalLM"], **llm},
        "vision_config": {"hidden_size": 8},
    }
    if text_auto_map is not None:
        cfg["llm_config"]["auto_map"] = {"AutoModelForCausalLM": text_auto_map}
    (repo / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    text = transformers.LlamaForCausalLM(transformers.LlamaConfig(**llm))
    weights = {}
    for i, (k, v) in enumerate(text.state_dict().items()):
        if k in drop:
            continue
        weights[prefix + k] = torch.full_like(v, 0.01 * (i + 1)).contiguous()
    weights["vision_model.weight"] = torch.zeros(8, 8)
    weights["vision_model.bias"] = torch.zeros(8)
    weights["mlp1.weight"] = torch.zeros(llm["hidden_size"], 8)
    weights["mlp1.bias"] = torch.zeros(llm["hidden_size"])
    save_file(weights, str(repo / "model.safetensors"))
    return repo, weights


def _load_parent_config(repo):
    return transformers.AutoConfig.from_pretrained(repo, trust_remote_code = True)




def test_prefix_inference_finds_single_full_cover():
    ns = _ns()
    expected = ["model.embed_tokens.weight", "model.layers.0.w", "lm_head.weight"]
    ckpt = {"language_model." + e for e in expected} | {"vision_model.x.weight"}
    assert ns["_infer_text_submodel_prefix"](expected, ckpt) == "language_model."


def test_prefix_inference_nested_prefix():
    ns = _ns()
    expected = ["backbone.embeddings.weight", "lm_head.weight"]
    ckpt = {"model.llm." + e for e in expected}
    assert ns["_infer_text_submodel_prefix"](expected, ckpt) == "model.llm."


def test_prefix_inference_rejects_partial_cover():
    ns = _ns()
    expected = ["model.embed_tokens.weight", "model.layers.0.w", "lm_head.weight"]
    ckpt = {"language_model.model.embed_tokens.weight", "language_model.lm_head.weight"}
    assert ns["_infer_text_submodel_prefix"](expected, ckpt) is None


def test_prefix_inference_rejects_unprefixed_and_ambiguous():
    ns = _ns()
    expected = ["a.weight", "b.weight"]
    assert ns["_infer_text_submodel_prefix"](expected, set(expected)) is None
    ambiguous = {"x." + e for e in expected} | {"y." + e for e in expected}
    assert ns["_infer_text_submodel_prefix"](expected, ambiguous) is None
    assert ns["_infer_text_submodel_prefix"](expected, None) is None
    assert ns["_infer_text_submodel_prefix"]([], {"x.a.weight"}) is None


def test_prefix_inference_gemma_style_split_layout_is_rejected():
    # Gemma 3 / LLaVA split decoder and head across prefixes, so no single prefix covers a CausalLM.
    ns = _ns()
    expected = ["model.embed_tokens.weight", "model.layers.0.w", "lm_head.weight"]
    ckpt = {
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.w",
        "lm_head.weight",
    }
    assert ns["_infer_text_submodel_prefix"](expected, ckpt) is None




@needs_tf5
def test_remote_composite_resolves_text_config_and_mapping(tmp_path):
    ns = _ns()
    repo, _ = _write_repo(tmp_path)
    parent = _load_parent_config(repo)
    assert ns["_is_remote_code_config"](parent)
    plan = ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True)
    assert plan is not None
    text_config, mapping, _ = plan
    assert text_config.model_type == "llama"
    assert not hasattr(text_config, "vision_config")
    assert mapping == {r"^language_model\.": ""}
    assert text_config is not parent.llm_config


@needs_tf5
def test_llm_config_without_text_config_alias(tmp_path):
    ns = _ns()
    repo, weights = _write_repo(tmp_path, alias = False, name = "no_alias")
    parent = _load_parent_config(repo)
    assert parent.get_text_config() is parent
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    assert text_config.model_type == "llama"
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        repo,
        config = text_config,
        key_mapping = mapping,
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
        output_loading_info = True,
    )
    assert not info["missing_keys"]
    assert torch.equal(model.lm_head.weight, weights["language_model.lm_head.weight"])


def test_remote_composite_needs_trust_remote_code(tmp_path):
    ns = _ns()
    repo, _ = _write_repo(tmp_path)
    parent = _load_parent_config(repo)
    assert ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = False) is None


def test_remote_composite_missing_text_weight_keeps_full_model(tmp_path):
    ns = _ns()
    repo, _ = _write_repo(tmp_path, drop = ("model.layers.1.mlp.down_proj.weight",))
    parent = _load_parent_config(repo)
    assert ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True) is None


def test_native_composites_never_take_the_branch():
    ns = _ns()
    for cfg in (
        transformers.Gemma3Config(),
        transformers.LlavaConfig(),
        transformers.Qwen2VLConfig(),
        transformers.MllamaConfig(),
    ):
        assert not ns["_is_remote_code_config"](cfg), type(cfg)
        assert ns["_get_remote_composite_text_only"](cfg, "x/y", trust_remote_code = True) is None


def test_skip_modules_are_rebased_on_the_text_model(tmp_path):
    ns = _ns()
    qc = {"llm_int8_skip_modules": ["llm.lm_head", "vision_model", "llm.model.layers.0.mlp"]}
    out = ns["_strip_skip_module_prefix"](qc, "llm.")
    assert out["llm_int8_skip_modules"] == ["lm_head", "vision_model", "model.layers.0.mlp"]
    assert qc["llm_int8_skip_modules"][0] == "llm.lm_head"  # input not mutated


def test_merge_key_mapping_keeps_user_entries_on_top():
    ns = _ns()
    kw = {"key_mapping": {r"^foo\.": "bar."}}
    ns["_merge_key_mapping"](kw, {r"^language_model\.": ""})
    assert kw["key_mapping"] == {r"^language_model\.": "", r"^foo\.": "bar."}
    kw = {}
    ns["_merge_key_mapping"](kw, {r"^language_model\.": ""})
    assert kw["key_mapping"] == {r"^language_model\.": ""}


@needs_tf5
def test_hardcoded_flash_attention_does_not_block_the_meta_build(tmp_path):
    # Nemotron-Omni's config __init__ forces llm_config._attn_implementation = "flash_attention_2".
    ns = _ns()
    repo, _ = _write_repo(tmp_path)
    parent = _load_parent_config(repo)
    parent.llm_config._attn_implementation = "flash_attention_2"
    assert (
        ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True) is not None
    )
    assert parent.llm_config._attn_implementation == "flash_attention_2"




@needs_tf5
def test_plan_loads_only_the_language_model_with_real_weights(tmp_path):
    ns = _ns()
    repo, weights = _write_repo(tmp_path)
    parent = _load_parent_config(repo)
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        repo,
        config = text_config,
        key_mapping = mapping,
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
        output_loading_info = True,
    )
    assert type(model).__name__ == "LlamaForCausalLM"
    assert not info["missing_keys"], info["missing_keys"]
    loaded = model.state_dict()
    for k, v in weights.items():
        if k.startswith("language_model."):
            assert torch.equal(loaded[k[len("language_model.") :]], v), k
    assert not any("vision" in n or "mlp1" in n for n, _ in model.named_modules())
    out = model(input_ids = torch.tensor([[1, 2, 3, 4]]), logits_to_keep = 2, use_cache = False)
    assert out.logits.shape == (1, 2, 64)


def test_wrapper_forward_needs_pixel_values_which_is_the_bug(tmp_path):
    # Base behaviour this branch replaces: the wrapper is what AutoModelForCausalLM builds, and it takes no text batch.
    repo, _ = _write_repo(tmp_path, name = "tiny_omni_wrapper")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        repo, trust_remote_code = True, dtype = torch.float32, local_files_only = True
    )
    assert type(model).__name__ == "TinyOmni"
    with pytest.raises(TypeError, match = "pixel_values"):
        model(input_ids = torch.tensor([[1, 2, 3]]))


@needs_tf5
def test_tied_embeddings_do_not_need_a_stored_head(tmp_path):
    ns = _ns()
    repo, _ = _write_repo(
        tmp_path, llm_extra = {"tie_word_embeddings": True}, drop = ("lm_head.weight",), name = "tied"
    )
    parent = _load_parent_config(repo)
    plan = ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True)
    assert plan is not None




def test_loader_and_vision_call_the_remote_branch_only_after_the_family_gate():
    loader = LOADER_PATH.read_text(encoding = "utf-8")
    vision = VISION_PATH.read_text(encoding = "utf-8")
    assert "_get_remote_composite_text_only(" in loader
    assert "_get_remote_composite_text_only(" in vision
    i_family = loader.index("family_decoder = text_class is not None and _is_family_text_decoder(")
    i_remote = loader.index("_get_remote_composite_text_only(")
    assert i_family < i_remote
    assert "if not family_decoder:" in loader


def test_transformers_4_keeps_the_full_model(tmp_path, monkeypatch):
    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "tf4")
    parent = _load_parent_config(repo)
    ns["transformers_version"] = "4.57.6"
    assert ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True) is None


def test_trusted_load_records_the_commit_its_repo_code_ran_at():
    # Nested text config has no commit, so the composite's is taken before the remap.
    vision = VISION_PATH.read_text(encoding = "utf-8")
    i_parent = vision.index("parent_config = auto_config\n")
    i_commit = vision.index('_trusted_code_commit = getattr(parent_config, "_commit_hash", None)')
    i_remap = vision.index("text_config = _get_text_only_config(parent_config, model_name)")
    assert i_parent < i_commit < i_remap
    i_trust = vision.index("model._unsloth_trust_remote_code = trust_remote_code")
    i_stamp = vision.index("model._unsloth_trust_remote_code_commit = (")
    assert i_trust < i_stamp
    stamp = vision[i_stamp : vision.index("\n        )\n", i_stamp)]
    assert "if trust_remote_code" in stamp and '"_commit_hash"' in stamp




def _cache_as_hub_repo(
    tmp_path,
    monkeypatch,
    repo,
    repo_id = "fake-org/tiny-omni",
    with_index = False,
):
    import shutil
    import huggingface_hub.constants as hub_constants

    sha = "0123456789abcdef0123456789abcdef01234567"
    cache = tmp_path / "hub_cache"
    root = cache / ("models--" + repo_id.replace("/", "--"))
    (root / "refs").mkdir(parents = True)
    (root / "refs" / "main").write_text(sha)
    snapshot = root / "snapshots" / sha
    shutil.copytree(repo, snapshot)
    if with_index:
        from safetensors import safe_open
        with safe_open(str(snapshot / "model.safetensors"), framework = "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    return repo_id, sha


@needs_tf5
def test_text_config_keeps_the_parent_commit(tmp_path, monkeypatch):
    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "commit")
    repo_id, sha = _cache_as_hub_repo(tmp_path, monkeypatch, repo, with_index = True)
    parent = transformers.AutoConfig.from_pretrained(
        repo_id, trust_remote_code = True, local_files_only = True
    )
    assert parent._commit_hash == sha
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, repo_id, trust_remote_code = True, local_files_only = True
    )
    assert text_config._commit_hash == sha
    assert getattr(parent.llm_config, "_commit_hash", None) is None  # parent left untouched
    model = transformers.AutoModelForCausalLM.from_pretrained(
        repo_id,
        config = text_config,
        key_mapping = mapping,
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
    )
    assert model.config._commit_hash == sha


@needs_tf5
def test_cached_single_file_checkpoint_is_read_offline(tmp_path, monkeypatch):
    ns = _ns()
    repo, weights = _write_repo(tmp_path, name = "single")
    repo_id, _ = _cache_as_hub_repo(tmp_path, monkeypatch, repo, repo_id = "fake-org/single")
    names = ns["_checkpoint_weight_names"](repo_id, local_files_only = True)
    assert names == set(weights)
    parent = transformers.AutoConfig.from_pretrained(
        repo_id, trust_remote_code = True, local_files_only = True
    )
    plan = ns["_get_remote_composite_text_only"](
        parent, repo_id, trust_remote_code = True, local_files_only = True
    )
    assert plan is not None
    assert plan[1] == {r"^language_model\.": ""}
    assert ns["_checkpoint_weight_names"]("fake-org/absent", local_files_only = True) is None




@needs_tf5
def test_adapter_trained_on_the_wrapper_keeps_the_full_model(tmp_path):
    # Tensors under language_model. would not reach the standalone decoder.
    peft = pytest.importorskip("peft")
    from safetensors.torch import load_file

    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "peft_base")
    parent = _load_parent_config(repo)
    text_config, mapping, text_names = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    lora = dict(r = 4, target_modules = ["q_proj", "v_proj"], init_lora_weights = False)

    wrapper = transformers.AutoModelForCausalLM.from_pretrained(
        repo, trust_remote_code = True, dtype = torch.float32, local_files_only = True
    )
    wrapper_adapter = tmp_path / "wrapper_adapter"
    peft.get_peft_model(wrapper, peft.LoraConfig(**lora)).save_pretrained(wrapper_adapter)

    def text_model():
        return transformers.AutoModelForCausalLM.from_pretrained(
            repo,
            config = copy.deepcopy(text_config),
            key_mapping = mapping,
            trust_remote_code = True,
            dtype = torch.float32,
            local_files_only = True,
        )

    text_adapter = tmp_path / "text_adapter"
    peft.get_peft_model(text_model(), peft.LoraConfig(**lora)).save_pretrained(text_adapter)

    saved = list(load_file(str(wrapper_adapter / "adapter_model.safetensors")).values())
    from real_accelerator import has_real_accelerator

    loaded = peft.PeftModel.from_pretrained(
        text_model(),
        wrapper_adapter,
        torch_device = "cuda" if has_real_accelerator() else "cpu",
    )
    got = [v for k, v in loaded.state_dict().items() if "lora_" in k]
    assert got and not any(any(torch.equal(v, s) for s in saved) for v in got)

    assert ns["_adapter_fits_text_model"](str(wrapper_adapter), mapping) is False
    assert ns["_adapter_fits_text_model"](str(text_adapter), mapping) is True
    assert ns["_adapter_fits_text_model"](str(text_adapter), mapping, text_names = text_names) is True
    assert (
        ns["_adapter_fits_text_model"](str(wrapper_adapter), mapping, text_names = text_names)
        is False
    )
    assert ns["_adapter_fits_text_model"](str(tmp_path / "missing"), mapping) is False


def test_loader_checks_the_adapter_before_taking_the_text_only_branch():
    loader = LOADER_PATH.read_text(encoding = "utf-8")
    i_plan = loader.index("remote_text_only = _get_remote_composite_text_only(")
    i_gate = loader.index("and not _adapter_fits_text_model(", i_plan)
    assert (
        "text_names = remote_text_only[2],"
        in loader[i_gate : loader.index("\n                ):", i_gate)]
    )
    i_take = loader.index("text_config, _text_key_mapping = remote_text_only", i_plan)
    assert i_plan < i_gate < i_take
    gate = loader[loader.rindex("if (", 0, i_gate) : i_take]
    assert "and is_peft" in gate and "old_model_name" in gate and "remote_text_only = None" in gate




@needs_tf5
def test_fast_inference_keeps_the_full_model_path(tmp_path):
    # vLLM loads the composite by repo name; a prefix key_mapping means nothing there.
    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "vllm")
    parent = _load_parent_config(repo)
    assert ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True)
    assert (
        ns["_get_remote_composite_text_only"](
            parent, str(repo), trust_remote_code = True, fast_inference = True
        )
        is None
    )


def test_loader_and_vision_forward_fast_inference_to_the_plan():
    for path in (LOADER_PATH, VISION_PATH):
        src = path.read_text(encoding = "utf-8")
        i = src.index("_get_remote_composite_text_only(\n")
        depth, j = 0, i
        while True:
            depth += {"(": 1, ")": -1}.get(src[j], 0)
            if src[j] == ")" and depth == 0:
                break
            j += 1
        call = src[i : j + 1]
        assert "fast_inference = fast_inference," in call, path.name


@needs_tf5
def test_sequence_valued_auto_map_entry(tmp_path):
    # A [slow, fast] AutoTokenizer entry (a shape transformers accepts) must not make the plan decline.
    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "tok_pair")
    cfg = json.loads((repo / "config.json").read_text())
    cfg["auto_map"]["AutoTokenizer"] = ["tokenization_tiny.TinyTokenizer", None]
    (repo / "config.json").write_text(json.dumps(cfg))
    parent = _load_parent_config(repo)
    assert isinstance(parent.auto_map["AutoTokenizer"], list)
    assert ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True)


@needs_tf5
@pytest.mark.parametrize(
    "prefix", ["language_model.", "model.language_model.", "language_model.model."]
)
@pytest.mark.parametrize("alias", [True, False])
def test_skip_modules_rebased_once_for_nested_prefixes(tmp_path, prefix, alias):
    # Prefix strip must run once, before the Gemma-layout remap could leave model.lm_head.
    ns = _ns()
    repo, _ = _write_repo(tmp_path, prefix = prefix, alias = alias, name = "nested")
    parent = _load_parent_config(repo)
    parent.quantization_config = {
        "llm_int8_skip_modules": [prefix + "lm_head", prefix + "model.layers.0.mlp", "vision_model"]
    }
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    assert mapping == {"^" + re.escape(prefix): ""}
    skip = text_config.quantization_config["llm_int8_skip_modules"]
    assert skip[:2] == ["lm_head", "model.layers.0.mlp"]
    model = transformers.LlamaForCausalLM(text_config)
    modules = dict(model.named_modules())
    assert all(name in modules for name in skip[:2])
    assert parent.quantization_config["llm_int8_skip_modules"][0] == prefix + "lm_head"


@needs_tf5
@pytest.mark.parametrize("prefix", ["language_model.", "model.language_model."])
@pytest.mark.parametrize("as_dict", [True, False])
def test_caller_quantization_config_skip_modules_are_rebased(tmp_path, prefix, as_dict):
    # Caller's quantization_config wins over the config's, so its skip names need rebasing too.
    ns = _ns()
    repo, _ = _write_repo(tmp_path, prefix = prefix, name = "user_qc")
    parent = _load_parent_config(repo)
    _, mapping, _ = ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True)
    names = [prefix + "lm_head", prefix + "model.layers.0.mlp", "vision_model", "lm_head"]
    if as_dict:
        user_qc = {"load_in_4bit": True, "llm_int8_skip_modules": list(names)}
    else:
        user_qc = transformers.BitsAndBytesConfig(
            load_in_4bit = True, llm_int8_skip_modules = list(names)
        )
    kw = {"quantization_config": user_qc}
    ns["_merge_key_mapping"](kw, mapping)
    ns["_rebase_user_quantization_config"](kw, mapping)
    out = kw["quantization_config"]
    skip = out["llm_int8_skip_modules"] if as_dict else out.llm_int8_skip_modules
    assert skip == ["lm_head", "model.layers.0.mlp", "vision_model"]
    assert type(out) is type(user_qc) and out is not user_qc
    if not as_dict:
        assert out.load_in_4bit and out.quant_method == user_qc.quant_method
    orig = user_qc["llm_int8_skip_modules"] if as_dict else user_qc.llm_int8_skip_modules
    assert orig == names
    kw = {}
    ns["_rebase_user_quantization_config"](kw, mapping)
    assert "quantization_config" not in kw
    plain = transformers.BitsAndBytesConfig(load_in_4bit = True)
    kw = {"quantization_config": plain}
    ns["_rebase_user_quantization_config"](kw, mapping)
    assert kw["quantization_config"] is plain


def test_loader_and_vision_rebase_the_caller_quantization_config():
    for path in (LOADER_PATH, VISION_PATH):
        src = path.read_text(encoding = "utf-8")
        assert re.search(
            r"_merge_key_mapping\(kwargs, _text_key_mapping\)\n\s+"
            r"_rebase_user_quantization_config\(kwargs, _text_key_mapping\)\n",
            src,
        ), path.name


def _from_pretrained_node(path, cls):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(
        n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
    )


def _qc_reads_after_rebase(fn, name):
    rebase = min(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "_rebase_user_quantization_config"
    )
    reads = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in n.targets
        ):
            if (
                n.lineno > rebase
                and "quantization_config" in ast.unparse(n.value)
                and "kwargs" in ast.unparse(n.value)
            ):
                reads.append(n.lineno)
    return rebase, reads


def test_post_load_quantization_config_stamp_uses_the_rebased_config():
    # FastModel's local quantization_config must be re-read after the rebase, else a saved decoder keeps composite skip names.
    fn = _from_pretrained_node(LOADER_PATH, "FastModel")
    rebase, reads = _qc_reads_after_rebase(fn, "quantization_config")
    assert reads, "FastModel keeps the pre-rebase quantization_config for the post-load stamp"
    stamp = max(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and ast.unparse(n).startswith(
            "model.config.update({'quantization_config': quantization_config"
        )
    )
    assert rebase < min(reads) < stamp
    fn = _from_pretrained_node(VISION_PATH, "FastBaseModel")
    rebase, reads = _qc_reads_after_rebase(fn, "user_quantization_config")
    assert reads


def _saved_keys(directory):
    from safetensors import safe_open

    keys = []
    for f in sorted(Path(directory).glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            keys += list(fh.keys())
    return sorted(keys)


@needs_tf5
def test_saved_text_only_decoder_keeps_standalone_names(tmp_path):
    # transformers 5 reverses key_mapping in save_pretrained, which wrote language_model.* and reloaded random.
    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "save_names")
    parent = _load_parent_config(repo)
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    kw = {}
    ns["_merge_key_mapping"](kw, mapping)

    def load():
        return transformers.AutoModelForCausalLM.from_pretrained(
            repo,
            config = text_config,
            key_mapping = kw["key_mapping"],
            trust_remote_code = True,
            dtype = torch.float32,
            local_files_only = True,
        )

    model = load()
    model.save_pretrained(tmp_path / "raw")
    assert "model.embed_tokens.weight" not in _saved_keys(tmp_path / "raw")

    model = load()
    ns["_drop_text_only_key_mapping"](model, mapping)
    model.save_pretrained(tmp_path / "saved")
    expected = sorted(model.state_dict())
    assert _saved_keys(tmp_path / "saved") == expected
    reloaded, info = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path / "saved", dtype = torch.float32, local_files_only = True, output_loading_info = True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"]
    state = reloaded.state_dict()
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())

    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(model, LoraConfig(r = 2, target_modules = ["q_proj", "v_proj"]))
    peft_model.save_pretrained(tmp_path / "adapter")
    assert all(k.startswith("base_model.model.model.") for k in _saved_keys(tmp_path / "adapter"))
    peft_model.merge_and_unload().save_pretrained(tmp_path / "merged")
    assert _saved_keys(tmp_path / "merged") == expected


@needs_tf5
def test_drop_keeps_every_other_weight_conversion():
    # Only the plan's own prefix strip goes; a caller's key_mapping and model conversions still reverse on save.
    from transformers.core_model_loading import WeightRenaming

    ns = _ns()
    plan = WeightRenaming(source_patterns = r"^language_model\.", target_patterns = "")
    user = WeightRenaming(source_patterns = r"^old\.", target_patterns = "new.")
    other = WeightRenaming(source_patterns = "LayerNorm.gamma", target_patterns = "LayerNorm.weight")

    class Holder:
        pass

    model = Holder()
    model._weight_conversions = [plan, user, other]
    ns["_drop_text_only_key_mapping"](model, {r"^language_model\.": ""})
    assert model._weight_conversions == [user, other]
    ns["_drop_text_only_key_mapping"](model, None)
    assert model._weight_conversions == [user, other]
    bare = Holder()
    ns["_drop_text_only_key_mapping"](bare, {r"^language_model\.": ""})
    assert not hasattr(bare, "_weight_conversions")


def test_loader_and_vision_drop_the_plan_mapping_after_the_load():
    loader = LOADER_PATH.read_text(encoding = "utf-8")
    vision = VISION_PATH.read_text(encoding = "utf-8")
    assert re.search(
        r"text_only_decoder = text_only_decoder,\n\s+\*args,\n\s+\*\*kwargs,\n\s+\)\n"
        r"\s+_drop_text_only_key_mapping\(model, _text_key_mapping\)\n",
        loader,
    )
    i_load = vision.index("model = auto_model.from_pretrained(")
    i_drop = vision.index("_drop_text_only_key_mapping(model, _text_key_mapping)")
    assert (
        i_load < i_drop < vision.index("restore_remote_code_non_persistent_buffers(model)", i_load)
    )




def _convert_to_sharded_bin(repo):
    from safetensors.torch import load_file

    weights = load_file(str(repo / "model.safetensors"))
    (repo / "model.safetensors").unlink()
    shard = "pytorch_model-00001-of-00001.bin"
    torch.save(weights, str(repo / shard))
    (repo / "pytorch_model.bin.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {k: shard for k in weights}})
    )
    return weights


@needs_tf5
def test_sharded_bin_checkpoint_is_probed_from_its_index(tmp_path, monkeypatch):
    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "bin_sharded")
    weights = _convert_to_sharded_bin(repo)
    assert ns["_checkpoint_weight_names"](str(repo)) == set(weights)
    parent = _load_parent_config(repo)
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        repo,
        config = text_config,
        key_mapping = mapping,
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
        output_loading_info = True,
    )
    assert not info["missing_keys"]
    assert torch.equal(model.lm_head.weight, weights["language_model.lm_head.weight"])
    repo_id, _ = _cache_as_hub_repo(tmp_path, monkeypatch, repo, repo_id = "fake-org/bin-sharded")
    assert ns["_checkpoint_weight_names"](repo_id, local_files_only = True) == set(weights)


def test_unsharded_bin_is_never_unpickled_to_probe(tmp_path, monkeypatch):
    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "bin_single")
    weights = _convert_to_sharded_bin(repo)
    (repo / "pytorch_model.bin.index.json").unlink()
    (repo / "pytorch_model-00001-of-00001.bin").rename(repo / "pytorch_model.bin")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("torch.load called"))
    assert ns["_checkpoint_weight_names"](str(repo)) is None




@needs_tf5
def test_nested_repo_code_class_lookup_honours_local_files_only(tmp_path, monkeypatch):
    # The nested llm_config names repo code; its lookup must not reach the Hub under local_files_only.
    import threading
    import http.server
    import huggingface_hub.constants as hub_constants

    ns = _ns()
    repo, _ = _write_repo(
        tmp_path, name = "nested_code", text_auto_map = "modeling_tiny_omni.TinyTextLM"
    )
    repo_id, _ = _cache_as_hub_repo(tmp_path, monkeypatch, repo, repo_id = "fake-org/nested-code")
    parent = transformers.AutoConfig.from_pretrained(
        repo_id, trust_remote_code = True, local_files_only = True
    )
    seen = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_HEAD(self):
            seen.append(self.path)
            self.send_response(503)
            self.end_headers()

        do_GET = do_HEAD

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
        monkeypatch.setattr(hub_constants, "ENDPOINT", f"http://127.0.0.1:{server.server_port}")
        monkeypatch.setattr(
            hub_constants,
            "HUGGINGFACE_CO_URL_TEMPLATE",
            f"http://127.0.0.1:{server.server_port}" + "/{repo_id}/resolve/{revision}/{filename}",
        )
        plan = ns["_get_remote_composite_text_only"](
            parent, repo_id, trust_remote_code = True, local_files_only = True
        )
    finally:
        server.shutdown()
        server.server_close()
    assert plan is not None
    assert seen == []


@needs_tf5
def test_bin_adapter_is_read_like_a_safetensors_one(tmp_path):
    # safe_serialization = False writes adapter_model.bin; PEFT loads it, so the gate must read it too.
    peft = pytest.importorskip("peft")

    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "peft_bin_base")
    parent = _load_parent_config(repo)
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    lora = dict(r = 4, target_modules = ["q_proj", "v_proj"])
    text = transformers.AutoModelForCausalLM.from_pretrained(
        repo,
        config = text_config,
        key_mapping = mapping,
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
    )
    text_adapter = tmp_path / "text_adapter_bin"
    peft.get_peft_model(text, peft.LoraConfig(**lora)).save_pretrained(
        text_adapter, safe_serialization = False
    )
    wrapper = transformers.AutoModelForCausalLM.from_pretrained(
        repo, trust_remote_code = True, dtype = torch.float32, local_files_only = True
    )
    wrapper_adapter = tmp_path / "wrapper_adapter_bin"
    peft.get_peft_model(wrapper, peft.LoraConfig(**lora)).save_pretrained(
        wrapper_adapter, safe_serialization = False
    )
    assert (text_adapter / "adapter_model.bin").is_file()
    assert not (text_adapter / "adapter_model.safetensors").exists()
    assert ns["_adapter_fits_text_model"](str(text_adapter), mapping) is True
    assert ns["_adapter_fits_text_model"](str(wrapper_adapter), mapping) is False


@needs_tf5
def test_probes_read_the_commit_the_config_was_resolved_at(tmp_path, monkeypatch):
    # main has moved to a checkpoint missing text weights; the probe must read the pinned commit.
    import shutil

    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "pinned")
    repo_id, sha = _cache_as_hub_repo(tmp_path, monkeypatch, repo, repo_id = "fake-org/pinned")
    parent = transformers.AutoConfig.from_pretrained(
        repo_id, trust_remote_code = True, local_files_only = True
    )
    assert parent._commit_hash == sha
    moved, _ = _write_repo(tmp_path, name = "moved", drop = ("model.layers.1.mlp.down_proj.weight",))
    root = tmp_path / "hub_cache" / "models--fake-org--pinned"
    new_sha = "f" * 40
    shutil.copytree(moved, root / "snapshots" / new_sha)
    (root / "refs" / "main").write_text(new_sha)
    plan = ns["_get_remote_composite_text_only"](
        parent, repo_id, trust_remote_code = True, local_files_only = True
    )
    assert plan is not None
    assert plan[0]._commit_hash == sha


@needs_tf5
def test_nested_cross_repo_code_is_never_imported(tmp_path, monkeypatch):
    # owner/repo--module.Class in llm_config.auto_map points at another repository's code.
    import transformers.dynamic_module_utils as dmu

    ns = _ns()
    repo, _ = _write_repo(
        tmp_path, name = "cross_repo", text_auto_map = "other-org/other-repo--modeling_x.TinyTextLM"
    )
    parent = _load_parent_config(repo)

    imported = []

    def _record(class_ref, *args, **kwargs):
        imported.append(class_ref)
        raise OSError("not fetched in this test")

    monkeypatch.setattr(dmu, "get_class_from_dynamic_module", _record)
    assert ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True) is None
    assert imported == []


@needs_tf5
def test_nested_own_repo_code_still_takes_the_plan(tmp_path):
    ns = _ns()
    repo, weights = _write_repo(
        tmp_path, name = "own_repo_code", text_auto_map = "modeling_tiny_omni.TinyTextLM"
    )
    parent = _load_parent_config(repo)
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        repo,
        config = text_config,
        key_mapping = mapping,
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
        output_loading_info = True,
    )
    assert type(model).__name__ == "TinyTextLM"
    assert not info["missing_keys"]




@needs_tf5
def test_subfolder_checkpoint_is_probed_where_the_load_reads_it(tmp_path, monkeypatch):
    import shutil

    ns = _ns()
    repo, weights = _write_repo(tmp_path, name = "sub")
    (repo / "weights").mkdir()
    shutil.move(str(repo / "model.safetensors"), str(repo / "weights" / "model.safetensors"))
    parent = _load_parent_config(repo)
    assert ns["_checkpoint_weight_names"](str(repo)) is None
    assert ns["_checkpoint_weight_names"](str(repo), subfolder = "weights") == set(weights)
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True, subfolder = "weights"
    )
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        repo,
        config = text_config,
        key_mapping = mapping,
        subfolder = "weights",
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
        output_loading_info = True,
    )
    assert not info["missing_keys"]
    assert torch.equal(model.lm_head.weight, weights["language_model.lm_head.weight"])
    repo_id, _ = _cache_as_hub_repo(tmp_path, monkeypatch, repo, repo_id = "fake-org/sub")
    assert ns["_checkpoint_weight_names"](
        repo_id, local_files_only = True, subfolder = "weights"
    ) == set(weights)


def test_loader_and_vision_forward_subfolder_to_the_plan():
    for path in (LOADER_PATH, VISION_PATH):
        src = path.read_text(encoding = "utf-8")
        i = src.index("_get_remote_composite_text_only(\n")
        depth, j = 0, i
        while True:
            depth += {"(": 1, ")": -1}.get(src[j], 0)
            if src[j] == ")" and depth == 0:
                break
            j += 1
        assert 'subfolder = kwargs.get("subfolder"),' in src[i : j + 1], path.name




@needs_tf5
def test_module_keyed_device_map_keeps_the_full_composite(tmp_path):
    # A composite device map names wrapper modules; from_pretrained refuses it on the standalone decoder.
    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "dmap")
    parent = _load_parent_config(repo)
    composite_map = {"language_model": "cpu", "vision_model": "cpu", "mlp1": "cpu"}
    plan = ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True)
    text_config, mapping, _ = plan
    model = transformers.LlamaForCausalLM(text_config)
    assert not any(
        name == key or name.startswith(key + ".")
        for name, _ in model.named_parameters()
        for key in composite_map
    )
    assert (
        ns["_get_remote_composite_text_only"](
            parent, str(repo), trust_remote_code = True, device_map = composite_map
        )
        is None
    )
    for device_map in ({"": "cpu"}, "cpu", "auto", "sequential", None):
        assert ns["_get_remote_composite_text_only"](
            parent, str(repo), trust_remote_code = True, device_map = device_map
        ), device_map


def test_loader_and_vision_forward_device_map_to_the_plan():
    for path in (LOADER_PATH, VISION_PATH):
        src = path.read_text(encoding = "utf-8")
        i = src.index("_get_remote_composite_text_only(\n")
        depth, j = 0, i
        while True:
            depth += {"(": 1, ")": -1}.get(src[j], 0)
            if src[j] == ")" and depth == 0:
                break
            j += 1
        assert "device_map = device_map," in src[i : j + 1], path.name




def _as_variant(repo, variant, layout):
    weights_file = repo / "model.safetensors"
    if layout == "single":
        weights_file.rename(repo / f"model.{variant}.safetensors")
    elif layout == "index":
        from safetensors import safe_open

        shard = f"model-00001-of-00001.{variant}.safetensors"
        with safe_open(str(weights_file), framework = "pt") as f:
            weight_map = {k: shard for k in f.keys()}
        weights_file.rename(repo / shard)
        (repo / f"model.safetensors.index.{variant}.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
    else:
        _convert_to_sharded_bin(repo)
        index = repo / "pytorch_model.bin.index.json"
        shard = "pytorch_model-00001-of-00001.bin"
        variant_shard = f"pytorch_model-00001-of-00001.{variant}.bin"
        (repo / shard).rename(repo / variant_shard)
        weight_map = {k: variant_shard for k in json.loads(index.read_text())["weight_map"]}
        index.unlink()
        (repo / f"pytorch_model.bin.index.{variant}.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )


@needs_tf5
@pytest.mark.parametrize("layout", ["single", "index", "bin_index"])
def test_variant_checkpoint_is_probed_where_the_load_reads_it(tmp_path, monkeypatch, layout):
    ns = _ns()
    repo, weights = _write_repo(tmp_path, name = f"variant_{layout}")
    _as_variant(repo, "fp16", layout)
    parent = _load_parent_config(repo)
    assert ns["_checkpoint_weight_names"](str(repo)) is None
    assert ns["_checkpoint_weight_names"](str(repo), variant = "fp16") == set(weights)
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True, variant = "fp16"
    )
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        repo,
        config = text_config,
        key_mapping = mapping,
        variant = "fp16",
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
        output_loading_info = True,
    )
    assert not info["missing_keys"]
    assert torch.equal(model.lm_head.weight, weights["language_model.lm_head.weight"])
    repo_id, _ = _cache_as_hub_repo(tmp_path, monkeypatch, repo, repo_id = f"fake-org/v-{layout}")
    assert ns["_checkpoint_weight_names"](repo_id, local_files_only = True, variant = "fp16") == set(
        weights
    )


@needs_tf5
def test_custom_cache_dir_is_probed(tmp_path, monkeypatch):
    import huggingface_hub.constants as hub_constants

    ns = _ns()
    repo, weights = _write_repo(
        tmp_path, name = "cache_dir", text_auto_map = "modeling_tiny_omni.TinyTextLM"
    )
    repo_id, _ = _cache_as_hub_repo(tmp_path, monkeypatch, repo, repo_id = "fake-org/cache-dir")
    cache_dir = str(tmp_path / "hub_cache")
    empty = tmp_path / "default_cache"
    empty.mkdir()
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(empty))
    parent = transformers.AutoConfig.from_pretrained(
        repo_id, trust_remote_code = True, local_files_only = True, cache_dir = cache_dir
    )
    assert ns["_checkpoint_weight_names"](repo_id, local_files_only = True) is None
    assert ns["_checkpoint_weight_names"](
        repo_id, local_files_only = True, cache_dir = cache_dir
    ) == set(weights)
    plan = ns["_get_remote_composite_text_only"](
        parent, repo_id, trust_remote_code = True, local_files_only = True, cache_dir = cache_dir
    )
    assert plan is not None and plan[1] == {r"^language_model\.": ""}


def test_loader_and_vision_forward_variant_and_cache_dir_to_the_plan():
    for path in (LOADER_PATH, VISION_PATH):
        src = path.read_text(encoding = "utf-8")
        i = src.index("_get_remote_composite_text_only(\n")
        depth, j = 0, i
        while True:
            depth += {"(": 1, ")": -1}.get(src[j], 0)
            if src[j] == ")" and depth == 0:
                break
            j += 1
        for arg in ('variant = kwargs.get("variant"),', 'cache_dir = kwargs.get("cache_dir"),'):
            assert arg in src[i : j + 1], (path.name, arg)




@needs_tf5
def test_nested_text_class_is_resolved_at_code_revision(tmp_path, monkeypatch):
    # Weights from main, code from code_revision; main's modeling file lacks the nested text class.
    import shutil

    ns = _ns()
    repo, weights = _write_repo(
        tmp_path, name = "code_rev", text_auto_map = "modeling_tiny_omni.TinyTextLM"
    )
    code = tmp_path / "code_rev_snapshot"
    shutil.copytree(repo, code)
    modeling = repo / "modeling_tiny_omni.py"
    modeling.write_text(modeling.read_text().replace("class TinyTextLM(", "class OtherTextLM("))
    repo_id, _ = _cache_as_hub_repo(tmp_path, monkeypatch, repo, repo_id = "fake-org/code-rev")
    root = tmp_path / "hub_cache" / "models--fake-org--code-rev"
    code_sha = "c" * 40
    shutil.copytree(code, root / "snapshots" / code_sha)
    (root / "refs" / "code-branch").write_text(code_sha)
    parent = transformers.AutoConfig.from_pretrained(
        repo_id, trust_remote_code = True, local_files_only = True
    )
    assert (
        ns["_get_remote_composite_text_only"](
            parent, repo_id, trust_remote_code = True, local_files_only = True
        )
        is None
    )
    text_config, mapping, _ = ns["_get_remote_composite_text_only"](
        parent,
        repo_id,
        trust_remote_code = True,
        local_files_only = True,
        code_revision = "code-branch",
    )
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        repo_id,
        config = text_config,
        key_mapping = mapping,
        code_revision = "code-branch",
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
        output_loading_info = True,
    )
    assert type(model).__name__ == "TinyTextLM"
    assert not info["missing_keys"]


def test_loader_and_vision_forward_code_revision_to_the_plan():
    for path in (LOADER_PATH, VISION_PATH):
        src = path.read_text(encoding = "utf-8")
        i = src.index("_get_remote_composite_text_only(\n")
        depth, j = 0, i
        while True:
            depth += {"(": 1, ")": -1}.get(src[j], 0)
            if src[j] == ")" and depth == 0:
                break
            j += 1
        assert 'code_revision = kwargs.get("code_revision"),' in src[i : j + 1], path.name


def test_trusted_commit_names_the_code_revision_commit(tmp_path, monkeypatch):
    # Export re-read pins to the code_revision commit, never the weights' commit.
    import shutil

    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "stamp")
    repo_id, weights_sha = _cache_as_hub_repo(tmp_path, monkeypatch, repo, repo_id = "fake-org/stamp")
    root = tmp_path / "hub_cache" / "models--fake-org--stamp"
    code_sha = "c" * 40
    shutil.copytree(repo, root / "snapshots" / code_sha)
    (root / "refs" / "code-branch").write_text(code_sha)
    # transformers 4.x reads its default cache path at import time, so name the cache explicitly.
    stamp = functools.partial(
        ns["_trusted_remote_code_commit"], cache_dir = str(tmp_path / "hub_cache")
    )
    assert stamp(repo_id, weights_sha) == weights_sha
    assert (
        stamp(repo_id, weights_sha, code_revision = "code-branch", local_files_only = True) == code_sha
    )
    assert stamp(repo_id, weights_sha, code_revision = "d" * 40, local_files_only = True) == "d" * 40
    # Unresolvable: "" (unsloth-zoo refuses), not None (it would fall back to the weights' commit).
    assert stamp(repo_id, weights_sha, code_revision = "gone", local_files_only = True) == ""
    assert stamp(str(repo), None, code_revision = "code-branch") is None


def test_vision_stamp_uses_the_code_revision_commit():
    vision = VISION_PATH.read_text(encoding = "utf-8")
    i_stamp = vision.index("model._unsloth_trust_remote_code_commit = (")
    stamp = vision[i_stamp : vision.index("\n        )\n", i_stamp)]
    assert "_trusted_remote_code_commit(" in stamp
    assert 'code_revision = kwargs.get("code_revision"),' in stamp




@needs_tf5
def test_adapter_on_wrapper_only_modules_keeps_the_full_model(tmp_path):
    # LoRA on vision_model / mlp1 only: the standalone decoder has no target for it.
    peft = pytest.importorskip("peft")

    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "peft_vision")
    parent = _load_parent_config(repo)
    text_config, mapping, text_names = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    wrapper = transformers.AutoModelForCausalLM.from_pretrained(
        repo, trust_remote_code = True, dtype = torch.float32, local_files_only = True
    )
    adapter = tmp_path / "vision_adapter"
    peft.get_peft_model(
        wrapper, peft.LoraConfig(r = 4, target_modules = ["vision_model", "mlp1"])
    ).save_pretrained(adapter)
    text = transformers.AutoModelForCausalLM.from_pretrained(
        repo,
        config = text_config,
        key_mapping = mapping,
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
    )
    with pytest.raises(ValueError):
        peft.PeftModel.from_pretrained(text, adapter, torch_device = "cpu")
    assert ns["_adapter_fits_text_model"](str(adapter), mapping, text_names = text_names) is False


@needs_tf5
def test_adapter_in_text_model_naming_still_fits(tmp_path):
    # LoRA on attention plus a saved lm_head / embed_tokens with tied embeddings (lm_head.weight is a tied alias).
    peft = pytest.importorskip("peft")

    ns = _ns()
    repo, _ = _write_repo(
        tmp_path,
        llm_extra = {"tie_word_embeddings": True},
        drop = ("lm_head.weight",),
        name = "peft_tied",
    )
    parent = _load_parent_config(repo)
    text_config, mapping, text_names = ns["_get_remote_composite_text_only"](
        parent, str(repo), trust_remote_code = True
    )
    text = transformers.AutoModelForCausalLM.from_pretrained(
        repo,
        config = text_config,
        key_mapping = mapping,
        trust_remote_code = True,
        dtype = torch.float32,
        local_files_only = True,
    )
    adapter = tmp_path / "tied_adapter"
    peft.get_peft_model(
        text,
        peft.LoraConfig(
            r = 4,
            target_modules = ["q_proj", "v_proj", "lm_head"],
            modules_to_save = ["embed_tokens"],
        ),
    ).save_pretrained(adapter)
    assert ns["_adapter_fits_text_model"](str(adapter), mapping, text_names = text_names) is True
