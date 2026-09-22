# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Load checkpoints published only in Mistral's own format through transformers.

Mistral-Large-3 (and its Base, BF16 and FP8 uploads) ships `params.json` and
`consolidated*.safetensors` and no `config.json`. Its text decoder is the same
architecture transformers implements as `mistral4` (MLA attention, softmax routed
MoE with a shared expert, YaRN with the llama 4 attention scale): Mistral publishes
Mistral-Small-4 in both formats and its tensors are byte for byte the same under the
two naming schemes, the per-expert `w1` / `w3` / `w2` being the halves of the fused
`gate_up_proj` and the rows of `down_proj`.

So instead of converting 700 GB on disk, a small directory is written next to the
cache holding a translated `config.json`, an index pointing at the original shards
and the tokenizer files. The key renames and the expert merges are handed to
transformers as weight conversions for that one load only, and dropped afterwards so
`save_pretrained` writes ordinary transformers names.

Only the text decoder is loaded; the vision encoder in the same shards is skipped.
Anything this module does not recognise returns None and the loader keeps its
previous error message.
"""

import contextlib
import functools
import hashlib
import json
import os
from typing import Optional

__all__ = [
    "MistralFormatRedirect",
    "mistral_params_to_mistral4_config",
    "mistral_format_weight_conversions",
    "prepare_mistral_format_checkpoint",
    "mistral_format_redirect",
]

# Keys of the multimodal parts that share the shards with the text decoder.
_NON_TEXT_PREFIXES = (
    "vision_encoder.",
    "patch_merger.",
    "pre_mm_projector_norm.",
    "vision_language_adapter.",
    "mm_audio_embeddings.",
    "audio_",
)
_TOKENIZER_FILES = (
    "tekken.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "SYSTEM_PROMPT.txt",
)
_VIEW_MARKER = "unsloth_mistral_format.json"


class MistralFormatRedirect(Exception):
    """Raised by the loader when a Mistral-format checkpoint has a transformers view;
    `mistral_format_redirect` catches it and loads the view instead."""

    def __init__(self, path, source):
        super().__init__(f"Unsloth: loading {source} through its transformers view at {path}")
        self.path = path
        self.source = source


def _fp8_block_quantization(quant) -> Optional[dict]:
    """compressed-tensors FP8 128x128 block weights with dynamic activations (what
    Mistral-Large-3 ships) as a transformers fine-grained FP8 config, else None."""
    if not isinstance(quant, dict):
        return None
    if str(quant.get("quant_method", "")).lower().replace("_", "-") != "compressed-tensors":
        return None
    if quant.get("format") != "float-quantized":
        return None
    block = None
    for group in (quant.get("config_groups") or {}).values():
        weights = (group or {}).get("weights") or {}
        acts = (group or {}).get("input_activations")
        if weights.get("type") != "float" or int(weights.get("num_bits", 0)) != 8:
            return None
        if weights.get("strategy") != "block" or not weights.get("block_structure"):
            return None
        if acts is not None and not (acts.get("dynamic") and acts.get("type") == "float"):
            return None
        if group.get("output_activations") is not None:
            return None
        this_block = list(weights["block_structure"])
        if block is not None and this_block != block:
            return None
        block = this_block
    if block is None:
        return None
    # The ignore list names the Mistral modules through their transformers names already
    # (q_a_proj, kv_a_proj_with_mqa, gate, lm_head, embed_tokens); vision entries do not apply.
    not_convert = []
    for entry in quant.get("ignore") or []:
        name = str(entry)
        if name.startswith("re:"):
            name = name[3:].strip("^$").replace(".*", "").replace("\\", "").strip(".")
        if any(p.rstrip(".") in name for p in _NON_TEXT_PREFIXES) or "patch_merger" in name:
            continue
        if name:
            not_convert.append(name.split(".")[-1] if name.startswith("model.") else name)
    return {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": block,
        "modules_to_not_convert": sorted(set(not_convert)),
    }


def mistral_params_to_mistral4_config(params: dict) -> Optional[dict]:
    """Translate a Mistral `params.json` into `Mistral4Config` keyword arguments, the way
    Mistral itself translated Mistral-Small-4's. None when the checkpoint is not an MLA
    mixture of experts with a shared expert, or uses something `mistral4` cannot express."""
    if not isinstance(params, dict):
        return None
    moe = params.get("moe")
    if not params.get("qk_nope_head_dim") or not isinstance(moe, dict):
        return None
    if not params.get("q_lora_rank"):
        return None  # a plain `wq` has no published Mistral-format checkpoint to map from
    if params.get("quantization"):
        return None  # per-tensor FP8 (Small-4 style); that family also ships config.json
    if int(moe.get("num_shared_experts") or 0) < 1:
        return None
    if int(moe.get("route_every_n", 1)) != 1:
        return None
    if str(moe.get("renorm_strategy", "WEIGHTS")).upper() != "WEIGHTS":
        return None
    if moe.get("use_load_balancing_bias"):
        return None  # sigmoid-style correction bias; mistral4 routes with a plain softmax
    if params.get("sliding_window"):
        return None

    rope_theta = float(params.get("rope_theta", 10000.0))
    llama4 = params.get("llama_4_scaling") or {}
    yarn = params.get("yarn") or {}
    if yarn:
        original = int(yarn.get("original_max_position_embeddings", 8192))
        if llama4 and int(llama4.get("original_max_position_embeddings", original)) != original:
            return None  # mistral4 reads one original length for both
        rope_parameters = {
            "rope_type": "yarn",
            "type": "yarn",
            "rope_theta": rope_theta,
            "factor": float(yarn["factor"]),
            "original_max_position_embeddings": original,
            "beta_fast": float(yarn.get("beta", 32)),
            "beta_slow": float(yarn.get("alpha", 1)),
            # Same values Mistral wrote into Mistral-Small-4's config.json for the same
            # `apply_scale: false` yarn block.
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": float(llama4.get("beta", 0.0)),
        }
    else:
        rope_parameters = {
            "rope_type": "default",
            "rope_theta": rope_theta,
            "original_max_position_embeddings": int(
                llama4.get("original_max_position_embeddings", 8192)
            ),
            "llama_4_scaling_beta": float(llama4.get("beta", 0.0)),
        }

    config = {
        "model_type": "mistral4",
        "architectures": ["Mistral4ForCausalLM"],
        "hidden_size": int(params["dim"]),
        "intermediate_size": int(params["hidden_dim"]),
        "num_hidden_layers": int(params["n_layers"]),
        "num_attention_heads": int(params["n_heads"]),
        "num_key_value_heads": int(params.get("n_kv_heads", params["n_heads"])),
        "rms_norm_eps": float(params.get("norm_eps", 1e-6)),
        "vocab_size": int(params["vocab_size"]),
        "tie_word_embeddings": bool(params.get("tied_embeddings", False)),
        "max_position_embeddings": int(params.get("max_position_embeddings", 128_000)),
        "hidden_act": params.get("activation", "silu"),
        "q_lora_rank": int(params["q_lora_rank"]),
        "kv_lora_rank": int(params["kv_lora_rank"]),
        "qk_nope_head_dim": int(params["qk_nope_head_dim"]),
        "qk_rope_head_dim": int(params["qk_rope_head_dim"]),
        "v_head_dim": int(params["v_head_dim"]),
        "first_k_dense_replace": int(moe.get("first_k_dense_replace", 0)),
        "n_routed_experts": int(moe["num_experts"]),
        "num_experts_per_tok": int(moe["num_experts_per_tok"]),
        "moe_intermediate_size": int(moe["expert_hidden_dim"]),
        "n_shared_experts": int(moe["num_shared_experts"]),
        "routed_scaling_factor": float(moe.get("routed_scale", 1.0)),
        "n_group": int(moe.get("num_expert_groups", 1)),
        "topk_group": int(moe.get("num_expert_groups_per_tok", 1)),
        "norm_topk_prob": True,
        "rope_interleave": True,
        "rope_parameters": rope_parameters,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 11,
        "attention_bias": False,
        "mlp_bias": False,
    }
    quant = params.get("quantization_config")
    if quant is not None:
        fp8 = _fp8_block_quantization(quant)
        if fp8 is None:
            return None
        config["quantization_config"] = fp8
    return config


def mistral_format_weight_conversions():
    """Renames and merges from Mistral's tensor names to `Mistral4ForCausalLM`'s.

    Renames chain in order; the expert merges run last and name their own targets. The
    name correspondence is the one Mistral-Small-4's two uploads share byte for byte."""
    from transformers.core_model_loading import (
        Concatenate,
        MergeModulelist,
        WeightConverter,
        WeightRenaming,
    )

    renames = [
        (r"^tok_embeddings\.weight$", "model.embed_tokens.weight"),
        (r"^norm\.weight$", "model.norm.weight"),
        (r"^output\.weight$", "lm_head.weight"),
        (r"^layers\.", "model.layers."),
        (r"\.attention_norm\.", ".input_layernorm."),
        (r"\.ffn_norm\.", ".post_attention_layernorm."),
        (r"\.attention\.wq_a\.", ".self_attn.q_a_proj."),
        (r"\.attention\.q_a_norm\.", ".self_attn.q_a_layernorm."),
        (r"\.attention\.wq_b\.", ".self_attn.q_b_proj."),
        (r"\.attention\.wkv_a_with_mqa\.", ".self_attn.kv_a_proj_with_mqa."),
        (r"\.attention\.kv_a_norm\.", ".self_attn.kv_a_layernorm."),
        (r"\.attention\.wkv_b\.", ".self_attn.kv_b_proj."),
        (r"\.attention\.wo\.", ".self_attn.o_proj."),
        (r"\.feed_forward\.w1\.", ".mlp.gate_proj."),
        (r"\.feed_forward\.w2\.", ".mlp.down_proj."),
        (r"\.feed_forward\.w3\.", ".mlp.up_proj."),
        (r"\.gate\.weight$", ".mlp.gate.weight"),
        (r"\.shared_experts\.w1\.", ".mlp.shared_experts.gate_proj."),
        (r"\.shared_experts\.w2\.", ".mlp.shared_experts.down_proj."),
        (r"\.shared_experts\.w3\.", ".mlp.shared_experts.up_proj."),
        (r"\.experts\.(\d+)\.w1\.", r".mlp.experts.\1.gate_proj."),
        (r"\.experts\.(\d+)\.w2\.", r".mlp.experts.\1.down_proj."),
        (r"\.experts\.(\d+)\.w3\.", r".mlp.experts.\1.up_proj."),
        # compressed-tensors' block scale is the dequantization multiplier, which is what
        # transformers' fine-grained FP8 calls weight_scale_inv.
        (r"\.weight_scale$", ".weight_scale_inv"),
    ]
    conversions = [WeightRenaming(source_patterns = s, target_patterns = t) for s, t in renames]
    # Scales first: transformers stops at the first converter that matches, and a `.weight`
    # source also matches the `.weight_scale_inv` key next to it. These only apply to a load
    # that keeps FP8; see `_mistral_format_conversions` for the dequantizing one.
    for suffix, merge in (
        (".weight_scale_inv", True),
        (".weight_scale_inv", False),
        (".weight", True),
        (".weight", False),
    ):
        target_suffix = "_scale_inv" if suffix != ".weight" else ""
        if merge:
            converter = WeightConverter(
                source_patterns = [
                    rf"mlp.experts.*.gate_proj{suffix}$"
                    if target_suffix
                    else f"mlp.experts.*.gate_proj{suffix}",
                    rf"mlp.experts.*.up_proj{suffix}$"
                    if target_suffix
                    else f"mlp.experts.*.up_proj{suffix}",
                ],
                target_patterns = f"mlp.experts.gate_up_proj{target_suffix}",
                operations = [MergeModulelist(dim = 0), Concatenate(dim = 1)],
            )
        else:
            converter = WeightConverter(
                source_patterns = (
                    rf"mlp.experts.*.down_proj{suffix}$"
                    if target_suffix
                    else f"mlp.experts.*.down_proj{suffix}"
                ),
                target_patterns = f"mlp.experts.down_proj{target_suffix}",
                operations = [MergeModulelist(dim = 0)],
            )
        conversions.append(converter)
    return conversions


def _mistral4_available() -> bool:
    try:
        from transformers import Mistral4Config, Mistral4ForCausalLM  # noqa: F401
        from transformers.conversion_mapping import register_checkpoint_conversion_mapping  # noqa: F401
        return True
    except Exception:
        return False


def _fetch(model_name, filename, token, revision, local_files_only):
    if os.path.isdir(model_name):
        path = os.path.join(model_name, filename)
        return path if os.path.isfile(path) else None
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(
            model_name,
            filename,
            token = token,
            revision = revision,
            local_files_only = local_files_only,
        )
    except Exception:
        return None


def _view_root():
    from huggingface_hub import constants
    return os.path.join(os.path.dirname(constants.HF_HUB_CACHE), "unsloth_mistral_format")


def prepare_mistral_format_checkpoint(
    model_name,
    token = None,
    revision = None,
    local_files_only = False,
) -> Optional[str]:
    """Write (or reuse) a transformers view of a Mistral-format checkpoint and return its
    directory, or None when the architecture is not one `mistral4` implements, this
    transformers has no `mistral4`, or the files cannot be read."""
    if not _mistral4_available():
        return None
    params_path = _fetch(model_name, "params.json", token, revision, local_files_only)
    if params_path is None:
        return None
    with open(params_path, encoding = "utf-8") as f:
        params = json.load(f)
    config = mistral_params_to_mistral4_config(params)
    if config is None:
        return None

    index_path = _fetch(
        model_name, "consolidated.safetensors.index.json", token, revision, local_files_only
    )
    if index_path is not None:
        with open(index_path, encoding = "utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        single = _fetch(model_name, "consolidated.safetensors", token, revision, local_files_only)
        if single is None:
            return None
        from safetensors import safe_open

        with safe_open(single, framework = "pt") as f:
            weight_map = {k: "consolidated.safetensors" for k in f.keys()}
    weight_map = {k: v for k, v in weight_map.items() if not k.startswith(_NON_TEXT_PREFIXES)}
    shards = sorted(set(weight_map.values()))

    # The view names the shards by absolute path, so nothing is linked or copied.
    shard_paths = {}
    if os.path.isdir(model_name):
        for shard in shards:
            shard_paths[shard] = os.path.abspath(os.path.join(model_name, shard))
    else:
        from huggingface_hub import snapshot_download
        snapshot = snapshot_download(
            model_name,
            revision = revision,
            token = token,
            local_files_only = local_files_only,
            allow_patterns = shards + ["params.json", *list(_TOKENIZER_FILES)],
        )
        for shard in shards:
            shard_paths[shard] = os.path.join(snapshot, shard)
    if not all(os.path.isfile(p) for p in shard_paths.values()):
        return None

    source_dir = os.path.dirname(shard_paths[shards[0]])
    digest = hashlib.sha256(
        (os.path.realpath(source_dir) + json.dumps(config, sort_keys = True)).encode()
    ).hexdigest()[:16]
    safe_name = str(model_name).strip("/").replace("/", "--")[-80:]
    view = os.path.join(_view_root(), f"{safe_name}-{digest}")
    marker = os.path.join(view, _VIEW_MARKER)
    if os.path.isfile(marker):
        return view

    os.makedirs(view, exist_ok = True)
    tmp = lambda name: os.path.join(view, name + ".tmp")
    with open(tmp("config.json"), "w", encoding = "utf-8") as f:
        json.dump(config, f, indent = 2)
    with open(tmp("model.safetensors.index.json"), "w", encoding = "utf-8") as f:
        json.dump(
            {"metadata": {}, "weight_map": {k: shard_paths[v] for k, v in weight_map.items()}},
            f,
        )
    for name in ("config.json", "model.safetensors.index.json"):
        os.replace(tmp(name), os.path.join(view, name))
    for name in _TOKENIZER_FILES:
        src = os.path.join(source_dir, name)
        if not os.path.isfile(src):
            src = _fetch(model_name, name, token, revision, local_files_only)
        if src and os.path.isfile(src):
            import shutil
            shutil.copyfile(src, os.path.join(view, name))
    with open(marker, "w", encoding = "utf-8") as f:
        json.dump({"source": str(model_name), "revision": revision}, f)
    return view


def is_mistral_format_view(path) -> bool:
    return isinstance(path, str) and os.path.isfile(os.path.join(path, _VIEW_MARKER))


def _is_expert_scale_merge(conversion):
    from transformers.core_model_loading import WeightConverter
    targets = getattr(conversion, "target_patterns", None) or []
    return isinstance(conversion, WeightConverter) and any(
        str(t).startswith("mlp.experts.") and str(t).endswith("_scale_inv") for t in targets
    )


def _register_mistral4_causal_lm():
    # transformers only reaches Mistral4 through Mistral3's vision wrapper, so AutoModelForCausalLM
    # does not know the text-only config the view carries. The pairing is transformers' own;
    # `AutoModelForCausalLM.register` skips transformers' native configs, so add it the same way
    # register does for everything else.
    from transformers import Mistral4Config, Mistral4ForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
    if MODEL_FOR_CAUSAL_LM_MAPPING.get(Mistral4Config, None) is None:
        MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral4Config] = Mistral4ForCausalLM


@contextlib.contextmanager
def _mistral_format_conversions():
    """Register the Mistral-name conversions for Mistral4ForCausalLM for one load, then put
    back whatever was registered before."""
    from transformers import conversion_mapping as cm

    _register_mistral4_causal_lm()
    cm.get_checkpoint_conversion_mapping("mistral4")  # builds the cache
    key = "Mistral4ForCausalLM"
    cache = cm._checkpoint_conversion_mapping_cache
    had_entry, previous = key in cache, cache.get(key)
    was_user = key in cm.USER_REGISTERED_MAPPINGS
    cm.register_checkpoint_conversion_mapping(
        key, mistral_format_weight_conversions(), overwrite = True
    )
    # A dequantizing FP8 load folds each expert's scale into its weight inside transformers'
    # own `.weight` converters, so the separate scale merges must not claim those keys first.
    fp8_quantizer, original_update = None, None
    try:
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
        fp8_quantizer = FineGrainedFP8HfQuantizer
        original_update = FineGrainedFP8HfQuantizer.__dict__.get("update_weight_conversions")
    except Exception:
        pass
    if original_update is not None:

        def update_weight_conversions(self, weight_conversions):
            if getattr(self.quantization_config, "dequantize", False):
                weight_conversions = [
                    c for c in weight_conversions if not _is_expert_scale_merge(c)
                ]
            return original_update(self, weight_conversions)

        fp8_quantizer.update_weight_conversions = update_weight_conversions
    try:
        yield
    finally:
        if original_update is not None:
            fp8_quantizer.update_weight_conversions = original_update
        if had_entry:
            cache[key] = previous
        else:
            cache.pop(key, None)
        if not was_user:
            cm.USER_REGISTERED_MAPPINGS.discard(key)


def _forget_load_conversions(model):
    # The model was read under Mistral's names; saving must write transformers' own.
    seen = set()
    for module in (model, getattr(model, "model", None), getattr(model, "base_model", None)):
        if module is not None and id(module) not in seen:
            seen.add(id(module))
            try:
                module._weight_conversions = []
            except Exception:
                pass


def mistral_format_redirect(fn):
    """Retry a `from_pretrained` against the transformers view the loader found for a
    Mistral-format checkpoint."""

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except MistralFormatRedirect as redirect:
            view, source = redirect.path, redirect.source
        if "model_name" in kwargs or not args:
            kwargs["model_name"] = view
        else:
            args = (view,) + tuple(args[1:])
        # The view pins the snapshot it was built from.
        kwargs.pop("revision", None)
        print(
            f"Unsloth: `{source}` is in Mistral's own format. Loading its text decoder "
            f"as transformers' Mistral4 through {view} (no weights are copied)."
        )
        with _mistral_format_conversions():
            result = fn(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        _forget_load_conversions(model)
        return result

    return _wrapper
