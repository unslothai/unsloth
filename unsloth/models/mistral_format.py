# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Load Mistral-format checkpoints (params.json + consolidated*.safetensors, no config.json,
e.g. Mistral-Large-3) as transformers `mistral4` via a translated view of the same shards.

Mistral-Small-4 ships in both formats with byte-identical tensors, which fixes the name
mapping. Conversions are registered for one load only so `save_pretrained` writes
transformers names. Only the text decoder loads; unrecognised checkpoints return None.
"""

import contextlib
import functools
import hashlib
import json
import re
import uuid
import os
from typing import Optional

__all__ = [
    "MistralFormatRedirect",
    "mistral_params_to_mistral4_config",
    "mistral_format_weight_conversions",
    "prepare_mistral_format_checkpoint",
    "mistral_format_redirect",
]

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
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "SYSTEM_PROMPT.txt",
)
_VIEW_MARKER = "unsloth_mistral_format.json"


class MistralFormatRedirect(Exception):
    """Raised by the loader; `mistral_format_redirect` catches it and loads the view."""

    def __init__(self, path, source):
        super().__init__(f"Unsloth: loading {source} through its transformers view at {path}")
        self.path = path
        self.source = source


def _fp8_block_quantization(quant) -> Optional[dict]:
    """compressed-tensors FP8 128x128 block (Mistral-Large-3) as fine-grained FP8 config, else None."""
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
        if acts is not None and not (
            acts.get("dynamic") and acts.get("type") == "float" and int(acts.get("num_bits", 8)) == 8
        ):
            return None
        if group.get("output_activations") is not None:
            return None
        # Fine-grained FP8 has no targets: a narrower group would claim unscaled bf16 Linears.
        if [str(t) for t in group.get("targets") or ["Linear"]] != ["Linear"]:
            return None
        this_block = list(weights["block_structure"])
        if block is not None and this_block != block:
            return None
        block = this_block
    if block is None:
        return None
    # The ignore list already uses transformers module names.
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
    """`params.json` -> `Mistral4Config` kwargs as Mistral did for Small-4; None if not expressible."""
    if not isinstance(params, dict):
        return None
    moe = params.get("moe")
    if not params.get("qk_nope_head_dim") or not isinstance(moe, dict):
        return None
    if not params.get("q_lora_rank"):
        return None
    if params.get("quantization"):
        return None  # per-tensor FP8: Small-4 ships config.json
    if int(moe.get("num_shared_experts") or 0) < 1:
        return None
    if int(moe.get("route_every_n", 1)) != 1:
        return None
    if str(moe.get("renorm_strategy", "WEIGHTS")).upper() != "WEIGHTS":
        return None
    if moe.get("use_load_balancing_bias"):
        return None  # mistral4 routes with plain softmax
    if params.get("sliding_window"):
        return None

    rope_theta = float(params.get("rope_theta", 10000.0))
    llama4 = params.get("llama_4_scaling") or {}
    yarn = params.get("yarn") or {}
    if yarn:
        if yarn.get("apply_scale"):
            return None  # the mscale below is the `apply_scale: false` translation
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
            # Matches Mistral-Small-4's config.json for `apply_scale: false`.
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
    """Mistral -> `Mistral4ForCausalLM` renames (chained in order), then expert merges."""
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
        # compressed-tensors' block scale is the multiplier transformers calls weight_scale_inv.
        (r"\.weight_scale$", ".weight_scale_inv"),
    ]
    conversions = [WeightRenaming(source_patterns = s, target_patterns = t) for s, t in renames]
    # Scales first: transformers stops at the first match and `.weight` also matches `.weight_scale_inv`.
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


def _fp8_dequantize_on_load_available() -> bool:
    try:
        from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer
    except Exception:
        return False
    return callable(getattr(FineGrainedFP8HfQuantizer, "update_weight_conversions", None))


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
    """Write (or reuse) a transformers view of a Mistral-format checkpoint; None if unsupported."""
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
    if config.get("quantization_config") and not _fp8_dequantize_on_load_available():
        print(
            f"Unsloth: `{model_name}` stores fp8 weights; loading them through transformers needs "
            "transformers >= 5.8. Please upgrade transformers."
        )
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
    for shard in shards:
        # Lexical, not realpath: hub snapshot files are symlinks into the blob store.
        norm = os.path.normpath(str(shard))
        if os.path.isabs(norm) or os.path.splitdrive(norm)[0] or norm.split(os.sep)[0] == os.pardir:
            return None

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
    # Key on what the view copies or points at, so a checkpoint edited in place gets a fresh view.
    source_files = {}
    for name in _TOKENIZER_FILES + ("params.json",):
        try:
            stat = os.stat(os.path.join(source_dir, name))
            source_files[name] = [stat.st_size, stat.st_mtime_ns]
        except OSError:
            pass
    digest = hashlib.sha256(
        json.dumps(
            [os.path.realpath(source_dir), config, weight_map, source_files], sort_keys = True
        ).encode()
    ).hexdigest()[:16]
    safe_name = re.sub(r"[\\/:]+", "--", str(model_name).strip("/\\"))[-80:]
    view = os.path.join(_view_root(), f"{safe_name}-{digest}")
    marker = os.path.join(view, _VIEW_MARKER)
    if os.path.isfile(marker):
        return view

    os.makedirs(view, exist_ok = True)
    # Per-process temp names: ranks of one launch can build the same view at once.
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    tmp = lambda name: os.path.join(view, name + suffix)
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
            shutil.copyfile(src, tmp(name))
            os.replace(tmp(name), os.path.join(view, name))
    tekken = os.path.join(view, "tekken.json")
    if os.path.isfile(tekken) and not os.path.isfile(os.path.join(view, "tokenizer.json")):
        # transformers converts a lone tekken.json without BOS (5.17) or with shifted ids (5.5).
        _write_tokenizer_from_tekken(tekken, view, tmp)
    with open(tmp(_VIEW_MARKER), "w", encoding = "utf-8") as f:
        json.dump({"source": str(model_name), "revision": revision}, f)
    os.replace(tmp(_VIEW_MARKER), marker)
    return view


def _bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _write_tokenizer_from_tekken(tekken_path, view, tmp):
    """tokenizer.json (+ tokenizer_config.json) from tekken.json, as tiktoken-style byte-level BPE:
    ids match mistral-common's Tekkenizer, and the tokenizer.json Mistral ships beside it."""
    import base64

    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import BPE

    with open(tekken_path, encoding = "utf-8") as f:
        tekken = json.load(f)
    config = tekken["config"]
    n_special = int(config["default_num_special_tokens"])
    specials = {int(t["rank"]): t["token_str"] for t in tekken.get("special_tokens", [])}
    special_strs = [specials.get(i, f"<SPECIAL_{i}>") for i in range(n_special)]
    enc = _bytes_to_unicode()
    ranks = {}
    for entry in tekken["vocab"][: int(config["default_vocab_size"]) - n_special]:
        ranks["".join(enc[b] for b in base64.b64decode(entry["token_bytes"]))] = len(ranks)
    merges = []
    for token, rank in ranks.items():
        local = [
            (token[:i], token[i:])
            for i in range(1, len(token))
            if token[:i] in ranks and token[i:] in ranks
        ]
        local.sort(key = lambda pair: (ranks[pair[0]], ranks[pair[1]]))
        merges.extend((a, b, rank) for a, b in local)
    merges.sort(key = lambda merge: merge[2])
    vocab = {token: i for i, token in enumerate(special_strs)}
    vocab.update({token: rank + n_special for token, rank in ranks.items()})
    tokenizer = Tokenizer(BPE(vocab, [(a, b) for a, b, _ in merges], ignore_merges = True))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(config["pattern"]), behavior = "isolated", invert = False),
            pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_strs)
    bos = special_strs[1]
    tokenizer.post_processor = processors.TemplateProcessing(
        single = f"{bos} $A", pair = f"{bos} $A $B", special_tokens = [(bos, 1)]
    )
    tokenizer.save(tmp("tokenizer.json"))
    os.replace(tmp("tokenizer.json"), os.path.join(view, "tokenizer.json"))
    if not os.path.isfile(os.path.join(view, "tokenizer_config.json")):
        named = {"<s>": "bos_token", "</s>": "eos_token", "<unk>": "unk_token", "<pad>": "pad_token"}
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": False,
            **{named[t]: t for t in special_strs if t in named},
        }
        with open(tmp("tokenizer_config.json"), "w", encoding = "utf-8") as f:
            json.dump(tokenizer_config, f, indent = 2)
        os.replace(tmp("tokenizer_config.json"), os.path.join(view, "tokenizer_config.json"))


def is_mistral_format_view(path) -> bool:
    return isinstance(path, str) and os.path.isfile(os.path.join(path, _VIEW_MARKER))


def _record_source(model):
    """Adapters record `model.name_or_path` as their base: name the source, not this host's view."""
    for module in (model, getattr(model, "model", None), getattr(model, "base_model", None)):
        view = getattr(getattr(module, "config", None), "_name_or_path", None)
        if module is None or not is_mistral_format_view(view):
            continue
        try:
            with open(os.path.join(view, _VIEW_MARKER), encoding = "utf-8") as f:
                source = json.load(f).get("source")
        except Exception:
            source = None
        if source:
            try:
                module.name_or_path = source
            except Exception:
                pass


def raise_if_merging_mistral_format_view(model, save_method):
    """Merged saves re-read the base shards, which a view names in Mistral's own layout."""
    view = getattr(getattr(model, "config", None), "_name_or_path", None)
    if str(save_method).strip().lower() == "lora" or not is_mistral_format_view(view):
        return
    try:
        with open(os.path.join(view, _VIEW_MARKER), encoding = "utf-8") as f:
            source = json.load(f).get("source") or view
    except Exception:
        source = view
    raise NotImplementedError(
        f"Unsloth: `{source}` was loaded from Mistral's own checkpoint format, and "
        f'`save_method = "{save_method}"` (merged and GGUF exports) cannot read those '
        f"shards yet. Save the LoRA adapter instead (`model.save_pretrained(...)` or "
        f'`save_method = "lora"`); it reloads through the same path.'
    )


def _is_expert_scale_merge(conversion):
    from transformers.core_model_loading import WeightConverter
    targets = getattr(conversion, "target_patterns", None) or []
    return isinstance(conversion, WeightConverter) and any(
        str(t).startswith("mlp.experts.") and str(t).endswith("_scale_inv") for t in targets
    )


def _register_mistral4_causal_lm():
    # AutoModelForCausalLM lacks Mistral4Config and `register` refuses native configs.
    from transformers import Mistral4Config, Mistral4ForCausalLM
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING
    if MODEL_FOR_CAUSAL_LM_MAPPING.get(Mistral4Config, None) is None:
        MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[Mistral4Config] = Mistral4ForCausalLM


_active_conversions = 0


def mistral_format_conversions_active() -> bool:
    return _active_conversions > 0


@contextlib.contextmanager
def _mistral_format_conversions():
    """Register Mistral-name conversions for one load, then restore the previous ones."""
    global _active_conversions
    from transformers import conversion_mapping as cm

    _register_mistral4_causal_lm()
    cm.get_checkpoint_conversion_mapping("mistral4")
    # Class-name lookup arrived with USER_REGISTERED_MAPPINGS; older releases look up model_type only.
    # Never both: 5.17 would also apply a "mistral4" entry to the inner Mistral4Model.
    user_registered = getattr(cm, "USER_REGISTERED_MAPPINGS", None)
    keys = ("Mistral4ForCausalLM",) if user_registered is not None else ("mistral4",)
    cache = cm._checkpoint_conversion_mapping_cache
    previous = {key: (key in cache, cache.get(key)) for key in keys}
    was_user = {key: user_registered is not None and key in user_registered for key in keys}
    conversions = mistral_format_weight_conversions()
    for key in keys:
        cm.register_checkpoint_conversion_mapping(key, conversions, overwrite = True)
    # Dequantizing FP8 folds scales in transformers' `.weight` converters; scale merges must not claim them.
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
    _active_conversions += 1
    try:
        yield
    finally:
        _active_conversions -= 1
        if original_update is not None:
            fp8_quantizer.update_weight_conversions = original_update
        for key in keys:
            had_entry, entry = previous[key]
            if had_entry:
                cache[key] = entry
            else:
                cache.pop(key, None)
            if user_registered is not None and not was_user[key]:
                user_registered.discard(key)


def _forget_load_conversions(model):
    seen = set()
    for module in (model, getattr(model, "model", None), getattr(model, "base_model", None)):
        if module is not None and id(module) not in seen:
            seen.add(id(module))
            try:
                module._weight_conversions = []
            except Exception:
                pass


def mistral_format_redirect(fn):
    """Retry `from_pretrained` against the Mistral-format view the loader found."""

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except MistralFormatRedirect as redirect:
            view, source = redirect.path, redirect.source
        if view is not None:
            if "model_name" in kwargs or not args:
                kwargs["model_name"] = view
            else:
                args = (view,) + tuple(args[1:])
            kwargs.pop("revision", None)
            print(
                f"Unsloth: `{source}` is in Mistral's own format. Loading its text decoder "
                f"as transformers' Mistral4 through {view} (no weights are copied)."
            )
        with _mistral_format_conversions():
            result = fn(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        _forget_load_conversions(model)
        _record_source(model)
        return result

    return _wrapper
