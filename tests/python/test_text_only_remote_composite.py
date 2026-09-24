"""text_only on repo-code composites: load only the language model when the checkpoint stores it whole under one prefix."""

import ast
import copy
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
needs_tf5 = pytest.mark.skipif(_V(transformers.__version__) < _V("5.0.0"), reason = "plan declines on transformers 4.x")
UTILS_PATH = REPO_ROOT / "unsloth" / "models" / "_utils.py"
LOADER_PATH = REPO_ROOT / "unsloth" / "models" / "loader.py"
VISION_PATH = REPO_ROOT / "unsloth" / "models" / "vision.py"

_HELPERS = (
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
)


def _ns():
    # Exec the helpers without importing unsloth (which needs a GPU).
    from packaging.version import Version

    source = UTILS_PATH.read_text(encoding = "utf-8")
    funcs = {
        n.name: ast.get_source_segment(source, n)
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef)
    }
    ns = {
        "copy": copy,
        "re": re,
        "Version": Version,
        "transformers_version": transformers.__version__,
    }
    for name in _HELPERS:
        exec(funcs[name], ns)
    return ns


# A repo-code composite shaped like Nemotron-3-Nano-Omni: parent config with llm_config (a stock text decoder
# config), a vision config, and a wrapper that keeps the whole causal LM as `language_model` next to vision parts.
_CONFIGURATION = '''
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
'''

_MODELING = '''
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
'''


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
    tmp_path, *, prefix = "language_model.", drop = (), llm_extra = None, name = "tiny_omni", alias = True
):
    # Save a remote-code composite checkpoint whose text weights sit under `prefix` (sentinel-filled).
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


# ---------------------------------------------------------------- prefix inference (pure)


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
    # A missing text weight would be randomly initialised, so no prefix is returned.
    ns = _ns()
    expected = ["model.embed_tokens.weight", "model.layers.0.w", "lm_head.weight"]
    ckpt = {"language_model.model.embed_tokens.weight", "language_model.lm_head.weight"}
    assert ns["_infer_text_submodel_prefix"](expected, ckpt) is None


def test_prefix_inference_rejects_unprefixed_and_ambiguous():
    ns = _ns()
    expected = ["a.weight", "b.weight"]
    # Stored at the root: no wrapper prefix to strip, the full-model path is left alone.
    assert ns["_infer_text_submodel_prefix"](expected, set(expected)) is None
    ambiguous = {"x." + e for e in expected} | {"y." + e for e in expected}
    assert ns["_infer_text_submodel_prefix"](expected, ambiguous) is None
    assert ns["_infer_text_submodel_prefix"](expected, None) is None
    assert ns["_infer_text_submodel_prefix"]([], {"x.a.weight"}) is None


def test_prefix_inference_gemma_style_split_layout_is_rejected():
    # Gemma 3 / LLaVA split the decoder (language_model.model.*) from the head (language_model.lm_head / lm_head):
    # a CausalLM's names are NOT all under one prefix in the tf5 layout, so the new branch cannot claim it.
    ns = _ns()
    expected = ["model.embed_tokens.weight", "model.layers.0.w", "lm_head.weight"]
    ckpt = {"model.language_model.embed_tokens.weight", "model.language_model.layers.0.w", "lm_head.weight"}
    assert ns["_infer_text_submodel_prefix"](expected, ckpt) is None


# ---------------------------------------------------------------- gate on real configs


@needs_tf5
def test_remote_composite_resolves_text_config_and_mapping(tmp_path):
    ns = _ns()
    repo, _ = _write_repo(tmp_path)
    parent = _load_parent_config(repo)
    assert ns["_is_remote_code_config"](parent)
    plan = ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True)
    assert plan is not None
    text_config, mapping = plan
    assert text_config.model_type == "llama"
    assert not hasattr(text_config, "vision_config")
    assert mapping == {r"^language_model\.": ""}
    # The parent keeps its own llm_config object (the plan copies it).
    assert text_config is not parent.llm_config


@needs_tf5
def test_llm_config_without_text_config_alias(tmp_path):
    ns = _ns()
    repo, weights = _write_repo(tmp_path, alias = False, name = "no_alias")
    parent = _load_parent_config(repo)
    # get_text_config() does not see llm_config here, so the family path had nothing to offer.
    assert parent.get_text_config() is parent
    text_config, mapping = ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True)
    assert text_config.model_type == "llama"
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        repo, config = text_config, key_mapping = mapping, trust_remote_code = True,
        dtype = torch.float32, local_files_only = True, output_loading_info = True,
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
    # Stock transformers configs (Gemma 3, LLaVA, Qwen2-VL, Mllama) are not repo code: unchanged behaviour.
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
    assert ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True) is not None
    assert parent.llm_config._attn_implementation == "flash_attention_2"


# ---------------------------------------------------------------- end to end: the plan loads real weights


@needs_tf5
def test_plan_loads_only_the_language_model_with_real_weights(tmp_path):
    ns = _ns()
    repo, weights = _write_repo(tmp_path)
    parent = _load_parent_config(repo)
    text_config, mapping = ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True)
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
    # And it is a plain causal LM: text-only forward with logits_to_keep works.
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


# ---------------------------------------------------------------- wiring


def test_loader_and_vision_call_the_remote_branch_only_after_the_family_gate():
    loader = LOADER_PATH.read_text(encoding = "utf-8")
    vision = VISION_PATH.read_text(encoding = "utf-8")
    assert "_get_remote_composite_text_only(" in loader
    assert "_get_remote_composite_text_only(" in vision
    # The family decoder test runs first; the new branch only fills the case that used to fall back to the full model.
    i_family = loader.index("family_decoder = text_class is not None and _is_family_text_decoder(")
    i_remote = loader.index("_get_remote_composite_text_only(")
    assert i_family < i_remote
    assert "if not family_decoder:" in loader


def test_transformers_4_keeps_the_full_model(tmp_path, monkeypatch):
    # On transformers 4.x the prefix strip cannot be expressed with key_mapping; the plan must decline.
    ns = _ns()
    repo, _ = _write_repo(tmp_path, name = "tf4")
    parent = _load_parent_config(repo)
    ns["transformers_version"] = "4.57.6"
    assert ns["_get_remote_composite_text_only"](parent, str(repo), trust_remote_code = True) is None
