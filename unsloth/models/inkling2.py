# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Thinking Machines Inkling 2 (``inkling2``: Inkling-2-Small, ``inkling2_loop``: Inkling-2-Mini).

transformers >= 5.24 ships both natively. Two things need Unsloth's help:

- Attention projections are wrapper modules (``q_proj`` = ``Inkling2QueryProjection`` holding ``.linear`` plus a
  scaled L2 clip; ``o_proj`` holds ``.linear`` and the output ``gate_proj``), so the usual LoRA names would land on
  modules PEFT cannot wrap. ``unsloth_lora_target_modules`` maps them to the real Linears.
- The released checkpoints declare ``quantization_config = {"quant_method": "modelopt", "quant_algo": "NVFP4"}``,
  but their layout is Inkling's own (``w13_weight`` + ``_scale`` / ``_amax`` / ``_outer_scale_denominator``), which
  transformers' Inkling conversion mapping decodes itself. The generic ModelOpt NVFP4 rewrite must not touch it.
  transformers 5.24 dequantizes those experts for ``inkling2_loop`` only; ``inkling2`` (Small) maps them as if they
  were plain / FP8 tensors, so ``register_inkling2_expert_dequantization`` gives it the loop's transforms.
"""

import re

__all__ = [
    "INKLING2_MODEL_TYPES",
    "is_inkling2_config",
    "inkling2_owns_modelopt_quantization",
    "drop_inkling2_quantization_config",
    "register_inkling2_expert_dequantization",
    "patch_inkling2_model",
    "inkling2_lora_target_modules",
    "raise_if_merging_inkling2",
    "save_inkling2_merged_16bit",
]

INKLING2_MODEL_TYPES = frozenset(
    {
        "inkling2",
        "inkling2_text",
        "inkling2_loop",
        "inkling2_loop_text",
    }
)

# Usual LoRA alias -> Linear leaves under one language-model attention block.
_ATTENTION_TARGETS = {
    "q_proj": ("self_attn.q_proj.linear",),
    "k_proj": ("self_attn.k_proj.linear",),
    "v_proj": ("self_attn.v_proj.linear",),
    # Inkling2GatedOutputProjection: sigmoid(gate_proj(x)) * attn, then linear.
    "o_proj": ("self_attn.o_proj.linear", "self_attn.o_proj.gate_proj"),
}
# Routed experts are 3D parameters, trained through target_parameters by get_peft_model.
_EXPERT_ALIASES = frozenset({"gate_proj", "up_proj", "down_proj", "gate_up_proj"})
# Vision tower leaves (Inkling2VisionAttention / Inkling2VisionMLP are plain Linears).
_VISION_ATTENTION = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "out_proj",
    "out_proj": "out_proj",
}
_VISION_MLP = {"fc1": "fc1", "fc2": "fc2", "gate_proj": "fc1", "up_proj": "fc1", "down_proj": "fc2"}
_DEFAULT = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _model_types(config):
    """model_type of a config (object or dict) and of its text_config."""
    out = set()
    for cfg in (config, _get(config, "text_config")):
        model_type = _get(cfg, "model_type")
        if model_type:
            out.add(str(model_type).lower())
    return out


def _get(cfg, key):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(key)
    try:
        return getattr(cfg, key, None)
    except Exception:
        return None


def is_inkling2_config(config) -> bool:
    return bool(_model_types(config) & INKLING2_MODEL_TYPES)


def raise_if_merging_inkling2(model, save_method):
    """Only ``lora``, ``merged_16bit`` and GGUF exports are supported for Inkling 2: the merged 16-bit save is
    ``save_inkling2_merged_16bit``, which rewrites the checkpoint in its own layout. ``merged_4bit`` and the other
    merged formats re-read the base checkpoint through the generic merge, which cannot place Inkling's LoRA weights."""
    method = str(save_method).strip().lower().replace("-", "_").replace(" ", "_")
    if method in ("lora", "merged_16bit", "gguf"):
        return
    if not is_inkling2_config(getattr(model, "config", None)):
        return
    raise NotImplementedError(
        f'Unsloth: `save_method = "{save_method}"` does not support Inkling 2 yet. Use `save_method = "merged_16bit"` '
        "(also what `save_pretrained_gguf` uses) or save the LoRA adapter with `model.save_pretrained(...)`."
    )


_EXPERT_COMPANIONS = ("_scale", "_amax", "_outer_scale_denominator")
_EXPERT_KEY = re.compile(r"\.experts\.w(?:13|2)_weight$")


def _peft_name(name):
    return re.sub(r"^base_model\.model\.", "", name).replace(".base_layer", "")


def _inkling2_lora_deltas(model):
    """{transformers parameter name: callable(expert_slice or None) -> float32 CPU delta} for every active LoRA.

    Linear and embedding deltas come from PEFT's ``get_delta_weight`` and must be evaluated inside
    ``lora_relative_to_original_base``, so residual inits (PiSSA, OLoRA, ...) are measured against the checkpoint
    the writer rereads rather than the residual base they were trained on. Expert deltas subtract the snapshot
    themselves."""
    import torch
    from peft.tuners.lora import LoraLayer

    try:
        from peft.tuners.lora.layer import ParamWrapper
    except ImportError:
        ParamWrapper = ()
    from peft.utils import ModulesToSaveWrapper

    deltas = {}
    for name, module in model.named_modules():
        if isinstance(module, ModulesToSaveWrapper):
            raise NotImplementedError(
                "Unsloth: merged Inkling 2 exports do not support `modules_to_save` yet; save the LoRA adapter."
            )
        if not isinstance(module, LoraLayer):
            continue
        # Embedding LoRA keeps its factors in lora_embedding_A / lora_embedding_B, not lora_A / lora_B.
        factors = set(module.lora_A) | set(getattr(module, "lora_embedding_A", {}))
        adapters = [a for a in getattr(module, "active_adapters", []) if a in factors]
        if not adapters:
            continue
        if any(module.use_dora.get(a, False) for a in adapters) or any(
            getattr(module, "lora_bias", {}).get(a, False) for a in adapters
        ):
            raise NotImplementedError(
                "Unsloth: merged Inkling 2 exports do not support DoRA or LoRA bias."
            )
        initial = getattr(module, "_unsloth_initial_lora", None) or {}
        path = _peft_name(name)
        if ParamWrapper and isinstance(module, ParamWrapper):
            key = f"{path}.{module.parameter_name}"
            # Bound now: lora_relative_to_original_base swaps these factors while the writer runs.
            terms = []
            for a in adapters:
                terms.append((module.lora_A[a].weight, module.lora_B[a].weight, module.scaling[a]))
                if a in initial and initial[a][2]:
                    A0, B0, scaling0 = initial[a]
                    terms.append((A0, B0, -scaling0))

            def expert_delta(
                experts = None,
                module = module,
                terms = terms,
            ):
                total = None
                E = module.num_experts
                # Same contraction as peft's ParamWrapper.get_delta_weight, one expert slice at a time.
                pattern = (
                    "o r e, e r i -> e o i"
                    if module._did_swap_in_out_features
                    else "o r e, e r i -> e i o"
                )
                for A, B, scaling in terms:
                    A = A.detach().float().cpu().reshape(E, -1, A.shape[-1])
                    B = B.detach().float().cpu()
                    B = B.reshape(B.shape[0], -1, E)
                    if experts is not None:
                        A, B = A[experts], B[..., experts]
                    d = torch.einsum(pattern, B, A) * scaling
                    total = d if total is None else total + d
                return total

            if key in deltas:
                raise RuntimeError(f"Unsloth: two LoRA adapters target {key}.")
            deltas[key] = expert_delta
        else:
            if initial and not isinstance(
                module.get_base_layer(), (torch.nn.Linear, torch.nn.Embedding)
            ):
                raise NotImplementedError(
                    "Unsloth: merged Inkling 2 exports support residual LoRA inits (PiSSA, OLoRA, ...) only on "
                    "Linear and Embedding layers; save the LoRA adapter."
                )
            key = f"{path}.weight"

            def linear_delta(
                experts = None,
                module = module,
                adapters = adapters,
            ):
                total = None
                for a in adapters:
                    d = module.get_delta_weight(a).detach().float().cpu()
                    total = d if total is None else total + d
                return total

            deltas[key] = linear_delta
    return deltas


def _inkling2_trained_biases(model):
    """{transformers parameter name: live tensor} for the biases a LoRA config trains (``bias = "all"`` or
    ``"lora_only"``), which the writer must take from the model instead of the checkpoint."""
    from peft.tuners.lora import LoraLayer

    configs = getattr(model, "peft_config", None) or {}
    active = getattr(model, "active_adapters", None) or list(configs)
    if callable(active):
        active = active()
    modes = {str(getattr(configs[a], "bias", "none")) for a in active if a in configs}
    lora_only = set()
    if "lora_only" in modes:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                bias = getattr(module.get_base_layer(), "bias", None)
                if bias is not None:
                    lora_only.add(id(bias))
    biases = {}
    for name, param in model.named_parameters():
        if "lora_" in name or not name.endswith(".bias"):
            continue
        if "all" in modes or id(param) in lora_only or param.requires_grad:
            biases[_peft_name(name)] = param
    return biases


def save_inkling2_merged_16bit(
    model,
    tokenizer,
    save_directory,
    push_to_hub = False,
    token = None,
    private = None,
):
    """Merge LoRA into Inkling 2 and write the checkpoint in its original layout (``model.llm.*``, ``attn.wq_du``,
    ``controllers.*``) with the MoE experts dequantized to bfloat16, which transformers, vLLM's loader and llama.cpp's
    converter all read. Every other tensor is copied unchanged; the LoRA deltas are added in float32."""
    import gc
    import json
    import os
    import shutil
    import tempfile

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.core_model_loading import (
        Inkling2LoopDequantizeExperts,
        Interleave,
        WeightConverter,
        WeightRenaming,
        rename_source_key,
    )

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    conversions = getattr(base, "_weight_conversions", None)
    if not conversions:
        from transformers.conversion_mapping import get_model_conversion_mapping
        conversions = get_model_conversion_mapping(base, add_legacy = False)
    renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    pattern_to_converter = {p: c for c in converters for p in c.source_patterns}

    source = getattr(base.config, "_name_or_path", None)
    if not source:
        raise RuntimeError("Unsloth: cannot find the Inkling 2 base checkpoint to merge into.")
    if not os.path.isdir(source):
        from huggingface_hub import snapshot_download
        source = snapshot_download(
            source,
            allow_patterns = ["*.safetensors", "*.json", "*.jinja"],
            token = token,
            revision = getattr(base.config, "_commit_hash", None),
        )
    index_path = os.path.join(source, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        with safe_open(os.path.join(source, "model.safetensors"), "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    deltas = _inkling2_lora_deltas(model)
    biases = _inkling2_trained_biases(model)
    used = set()
    print(f"Unsloth: Merging {len(deltas)} LoRA weights into Inkling 2 ({source}).")

    final_directory = save_directory
    if push_to_hub:
        save_directory = tempfile.mkdtemp(prefix = "unsloth_inkling2_merge_")
    os.makedirs(save_directory, exist_ok = True)

    handles = {}

    def tensor(key):
        filename = weight_map[key]
        if filename not in handles:
            handles[filename] = safe_open(os.path.join(source, filename), "pt", device = "cpu")
        return handles[filename].get_tensor(key)

    def hf_name(key):
        renamed, pattern = rename_source_key(key, renamings, converters)
        return renamed, pattern_to_converter.get(pattern)

    dequant = Inkling2LoopDequantizeExperts()
    patterns = ["weight", "weight_scale", "weight_amax", "weight_outer_scale_denominator"]

    def merged_experts(key):
        name, converter = hf_name(key)
        found = {
            p: tensor(key + p[len("weight") :])
            for p in patterns
            if key + p[len("weight") :] in weight_map
        }
        W = dequant.convert(found, patterns, ["out"], model = None)["out"].to(torch.bfloat16)
        delta = deltas.get(name)
        if delta is None:
            return W
        used.add(name)
        interleaved = converter is not None and any(
            isinstance(op, Interleave) for op in converter.operations
        )
        step = 8
        for start in range(0, W.shape[0], step):
            experts = slice(start, start + step)
            d = delta(experts)
            block = W[experts]
            if (
                interleaved
            ):  # the checkpoint interleaves gate / up rows, transformers stores them halved
                block = Interleave(dim = 1).convert({"x": block}, ["x"], ["x"])["x"]
            if tuple(d.shape) != tuple(block.shape):
                raise RuntimeError(
                    f"Unsloth: LoRA delta {tuple(d.shape)} does not fit {name} {tuple(block.shape)}."
                )
            block = (block.float() + d).to(torch.bfloat16)
            if interleaved:
                block = Interleave(dim = 1, inverse = True).convert({"x": block}, ["x"], ["x"])["x"]
            W[experts] = block
        return W

    companions = {
        key + suffix
        for key in weight_map
        if _EXPERT_KEY.search(key)
        for suffix in _EXPERT_COMPANIONS
        if key + suffix in weight_map
    }
    shard_limit = 4 * 1024**3
    shards, current, current_size = [], {}, 0

    def flush():
        nonlocal current, current_size
        if not current:
            return
        filename = f"model-{len(shards) + 1:05d}.safetensors"
        save_file(current, os.path.join(save_directory, filename), metadata = {"format": "pt"})
        shards.append((filename, list(current)))
        current, current_size = {}, 0
        gc.collect()

    from unsloth.models._utils import lora_relative_to_original_base

    # Residual LoRA inits trained on a base with their initial adapter subtracted; the writer rereads the
    # original checkpoint, so measure every delta against it.
    with lora_relative_to_original_base(model):
        for key in sorted(weight_map, key = lambda k: (weight_map[k], k)):
            if key in companions:
                continue
            if _EXPERT_KEY.search(key):
                W = merged_experts(key)
            else:
                W = tensor(key)
                name, _ = hf_name(key)
                if name in biases:
                    live = biases[name].detach().cpu()
                    if tuple(live.shape) != tuple(W.shape):
                        raise RuntimeError(
                            f"Unsloth: trained bias {tuple(live.shape)} does not fit {name} {tuple(W.shape)}."
                        )
                    W = live.to(W.dtype)
                    used.add(name)
                delta = deltas.get(name)
                if delta is not None:
                    d = delta()
                    if tuple(d.shape) != tuple(W.shape):
                        raise RuntimeError(
                            f"Unsloth: LoRA delta {tuple(d.shape)} does not fit {name} {tuple(W.shape)}."
                        )
                    W = (W.float() + d).to(W.dtype)
                    used.add(name)
            current[key] = W.contiguous()
            current_size += W.numel() * W.element_size()
            if current_size >= shard_limit:
                flush()
    flush()
    handles.clear()

    missing = sorted((set(deltas) | set(biases)) - used)
    if missing:
        raise RuntimeError(
            f"Unsloth: {len(missing)} trained weights have no tensor in the Inkling 2 checkpoint, e.g. {missing[:3]}."
        )

    total = len(shards)
    new_map = {}
    for i, (filename, keys) in enumerate(shards, start = 1):
        final = f"model-{i:05d}-of-{total:05d}.safetensors"
        os.replace(os.path.join(save_directory, filename), os.path.join(save_directory, final))
        new_map.update({k: final for k in keys})
    with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": new_map}, f, indent = 2)

    with open(os.path.join(source, "config.json")) as f:
        config = json.load(f)
    for cfg in (config, config.get("text_config") or {}):
        cfg.pop("quantization_config", None)
    with open(os.path.join(save_directory, "config.json"), "w") as f:
        json.dump(config, f, indent = 2)
    for filename in os.listdir(source):
        if filename in ("config.json", "hf_quant_config.json", "model.safetensors.index.json"):
            continue
        if filename.endswith((".json", ".jinja")):
            shutil.copy(os.path.join(source, filename), os.path.join(save_directory, filename))
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory)
    print(
        f"Unsloth: Merged {len(used)} LoRA weights; saved Inkling 2 in 16-bit to '{save_directory}'."
    )

    if push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi(token = token)
        api.create_repo(final_directory, private = private, exist_ok = True)
        api.upload_folder(folder_path = save_directory, repo_id = final_directory)
        shutil.rmtree(save_directory, ignore_errors = True)
        print(f"Unsloth: Uploaded the merged model to https://huggingface.co/{final_directory}")


def inkling2_owns_modelopt_quantization(config) -> bool:
    """True for an Inkling 2 config whose ModelOpt quantization_config transformers decodes in its own mapping."""
    if not is_inkling2_config(config):
        return False
    quant = _get(config, "quantization_config")
    if quant is not None and not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else vars(quant)
    return isinstance(quant, dict) and str(quant.get("quant_method", "")).lower() == "modelopt"


def drop_inkling2_quantization_config(
    config,
    verbose = True,
    load_in_4bit = False,
) -> None:
    """Remove the released checkpoint's ModelOpt marker from the config (and its text_config)."""
    for cfg in (config, _get(config, "text_config")):
        if cfg is None or _get(cfg, "quantization_config") is None:
            continue
        try:
            delattr(cfg, "quantization_config")
        except Exception:
            pass
        if _get(cfg, "quantization_config") is not None:
            try:
                cfg.quantization_config = None
            except Exception:
                pass
    if verbose:
        print(
            "Unsloth: Inkling 2 stores its MoE experts in NVFP4 / MXFP8; transformers dequantizes them while "
            "loading" + (", then bitsandbytes quantizes them to 4bit." if load_in_4bit else ".")
        )


def _is_expert_transform(transform) -> bool:
    patterns = getattr(transform, "_original_source_patterns", None) or ()
    return any(str(p).startswith(r"mlp\.experts\.w") for p in patterns)


def _has_dequant_op(transform) -> bool:
    return any(
        type(op).__name__ == "Inkling2LoopDequantizeExperts"
        for op in getattr(transform, "operations", ())
    )


def register_inkling2_expert_dequantization() -> bool:
    """Load the released NVFP4 / MXFP8 experts of ``inkling2`` checkpoints the way transformers loads ``inkling2_loop``.

    Upstream builds ``inkling2_loop_text`` as its two dequantizing expert converters plus ``inkling2_text`` without
    its expert transforms; this registers ``inkling2_text`` the same way. Returns False (and changes nothing) when
    either mapping no longer has that shape."""
    try:
        from transformers import conversion_mapping as cm
    except Exception:
        return False
    get = getattr(cm, "get_checkpoint_conversion_mapping", None)
    register = getattr(cm, "register_checkpoint_conversion_mapping", None)
    if get is None or register is None:
        return False
    try:
        loop_text, text = get("inkling2_loop_text"), get("inkling2_text")
    except Exception:
        return False
    if not loop_text or not text:
        return False
    dequant = [t for t in loop_text if _is_expert_transform(t)]
    if len(dequant) != 2 or not all(_has_dequant_op(t) for t in dequant):
        return False
    if any(_has_dequant_op(t) for t in text):
        return True  # already registered
    rest = [t for t in text if not _is_expert_transform(t)]
    if len(rest) == len(text):
        return False  # upstream no longer maps the experts here: nothing to replace
    register("inkling2_text", dequant + rest, overwrite = True)
    return True


def inkling2_lora_target_modules(
    model,
    target_modules,
    attention = True,
    mlp = True,
    vision = False,
    language = True,
):
    """Exact module names for the usual LoRA aliases. Unknown names pass through as PEFT suffixes."""
    if isinstance(target_modules, str) and target_modules != "all-linear":
        return target_modules
    if target_modules is None or target_modules == "all-linear":
        wanted = list(_DEFAULT)
    else:
        wanted = [str(x) for x in target_modules]

    language_on = language
    language, vision_layers = [], []
    for name, _ in model.named_modules():
        if re.search(r"(?:^|\.)vision_tower\.layers\.\d+$", name):
            vision_layers.append(name)
        elif (
            re.search(r"(?:^|\.)layers\.\d+$", name)
            and "vision_tower" not in name
            and ".mtp" not in name
        ):
            language.append(name)
    modules = dict(model.named_modules())

    out, passthrough = [], []
    for alias in wanted:
        known = False
        if alias in _ATTENTION_TARGETS:
            known = True
            if attention and language_on:
                for layer in language:
                    for leaf in _ATTENTION_TARGETS[alias]:
                        if f"{layer}.{leaf}" in modules:
                            out.append(f"{layer}.{leaf}")
        if alias in _EXPERT_ALIASES:
            known = True
        if vision and alias in _VISION_ATTENTION and attention:
            known = True
            for layer in vision_layers:
                name = f"{layer}.self_attn.{_VISION_ATTENTION[alias]}"
                if name in modules:
                    out.append(name)
        if vision and alias in _VISION_MLP and mlp:
            known = True
            for layer in vision_layers:
                name = f"{layer}.mlp.{_VISION_MLP[alias]}"
                if name in modules:
                    out.append(name)
        if alias in ("out_proj", "fc1", "fc2"):
            known = True
        if not known:
            passthrough.append(alias)
    result = []
    for name in out + passthrough:
        if name not in result:
            result.append(name)
    return result


def _unsloth_lora_target_modules(
    self,
    target_modules,
    attention = True,
    mlp = True,
    vision = False,
    language = True,
):
    return inkling2_lora_target_modules(
        self, target_modules, attention = attention, mlp = mlp, vision = vision, language = language
    )


def patch_inkling2_model(model) -> bool:
    """Give a loaded Inkling 2 model Unsloth's LoRA target mapping. Returns True when the model is Inkling 2."""
    config = getattr(model, "config", None)
    if not is_inkling2_config(config):
        return False
    cls = type(model)
    if getattr(cls, "unsloth_lora_target_modules", None) is not _unsloth_lora_target_modules:
        cls.unsloth_lora_target_modules = _unsloth_lora_target_modules
    return True
