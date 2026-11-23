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
    "checkpoint_inkling2_message_inputs",
    "inkling2_lora_target_modules",
    "raise_if_merging_inkling2",
    "save_inkling2_merged_16bit",
    "raise_if_inkling2_parallel_unsupported",
    "INKLING2_TP_REASON",
    "INKLING2_PP_REASON",
    "INKLING2_PARALLEL_ALTERNATIVE",
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


def _inkling2_trainable_tokens(model):
    """{transformers parameter name: [(row indices, live rows), ...]} for LoRA ``trainable_token_indices``.

    PEFT keeps those rows in a TrainableTokensLayer and merges them like ``get_merged_weights``: each active adapter's
    ``trainable_tokens_delta`` replaces its rows (``index_copy``), in adapter order. A tied layer shares its partner's
    rows and tensor, so only the owning layer is written."""
    try:
        from peft.tuners.trainable_tokens import TrainableTokensLayer
    except ImportError:
        return {}
    import torch

    tokens = {}
    for name, module in model.named_modules():
        if not isinstance(module, TrainableTokensLayer) or module.tied_adapter is not None:
            continue
        adapters = [a for a in module.active_adapters if a in module.trainable_tokens_delta]
        if not adapters:
            continue
        key = _peft_name(re.sub(r"\.token_adapter$", "", name)) + ".weight"
        tokens[key] = [
            (
                torch.tensor(module.token_indices[a], dtype = torch.long),
                module.trainable_tokens_delta[a].detach().float().cpu(),
            )
            for a in adapters
        ]
    return tokens


_STALE_WEIGHT_FILE = re.compile(
    r"(?:model\.safetensors|pytorch_model\.bin|(?:model|pytorch_model)-\d{5}(?:-of-\d{5})?\.(?:safetensors|bin))"
    r"|pytorch_model\.bin\.index\.json"
    r"|adapter_config\.json|adapter_model\.(?:safetensors|bin)"
)


def _remove_stale_inkling2_weights(save_directory, kept):
    """Drop weight files an earlier save left that the new index does not list.

    transformers loads model.safetensors ahead of an index, so a leftover unsharded file would
    silently shadow the merged shards, and a leftover LoRA adapter (adapter_config.json plus its
    weights) would be loaded onto the merged model, applying it twice. Runs only after the new
    shards and index are in place.
    """
    import os
    for filename in os.listdir(save_directory):
        if filename in kept or not _STALE_WEIGHT_FILE.fullmatch(filename):
            continue
        path = os.path.join(save_directory, filename)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)


def save_inkling2_merged_16bit(
    model,
    tokenizer,
    save_directory,
    push_to_hub = False,
    token = None,
    private = None,
    dtype = None,
):
    """Merge LoRA into Inkling 2 and write the checkpoint in its original layout (``model.llm.*``, ``attn.wq_du``,
    ``controllers.*``) with the MoE experts dequantized to bfloat16, which transformers, vLLM's loader and llama.cpp's
    converter all read. Every other tensor is copied unchanged; the LoRA deltas are added in float32.

    ``dtype = torch.float32`` (or ``"float32"``) instead writes every floating tensor, experts included, in float32
    without rounding the merged deltas; any other non-None ``dtype`` is refused."""
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

    if dtype is None:
        float32 = False
    elif (
        isinstance(dtype, str)
        and dtype.strip().lower().removeprefix("torch.") in ("float32", "fp32", "f32")
    ) or dtype is torch.float32:
        float32 = True
    else:
        raise NotImplementedError(
            f"Unsloth: merged Inkling 2 exports support `dtype = None` (checkpoint dtype, experts in bfloat16) "
            f"or `torch.float32`, not {dtype!r}."
        )
    expert_dtype = torch.float32 if float32 else torch.bfloat16

    def out_dtype(W):
        return torch.float32 if float32 and W.is_floating_point() else W.dtype

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
    tokens = _inkling2_trainable_tokens(model)
    both = sorted(set(tokens) & set(deltas))
    if both:
        raise NotImplementedError(
            f"Unsloth: merged Inkling 2 exports do not support LoRA and `trainable_token_indices` on the same "
            f"weight ({both[0]}); save the LoRA adapter."
        )
    used = set()
    print(f"Unsloth: Merging {len(deltas)} LoRA weights into Inkling 2 ({source}).")

    final_directory = save_directory
    if push_to_hub:
        from huggingface_hub import HfApi

        from unsloth.save import _refuse_private_push_to_public_repo

        # create_repo(private = True) leaves an existing public repo public: refuse before merging or uploading.
        api = HfApi(token = token)
        final_directory = getattr(
            api.create_repo(final_directory, private = private, exist_ok = True),
            "repo_id",
            final_directory,
        )
        _refuse_private_push_to_public_repo(final_directory, private, api = api)
        save_directory = tempfile.mkdtemp(prefix = "unsloth_inkling2_merge_")
    os.makedirs(save_directory, exist_ok = True)
    if os.path.samefile(save_directory, source):
        raise RuntimeError(
            "Unsloth: the merged Inkling 2 export would overwrite the checkpoint it reads; pick another directory."
        )

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
        W = dequant.convert(found, patterns, ["out"], model = None)["out"].to(expert_dtype)
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
            block = (block.float() + d).to(expert_dtype)
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
                    W = live.to(out_dtype(W))
                    used.add(name)
                delta = deltas.get(name)
                if delta is not None:
                    d = delta()
                    if tuple(d.shape) != tuple(W.shape):
                        raise RuntimeError(
                            f"Unsloth: LoRA delta {tuple(d.shape)} does not fit {name} {tuple(W.shape)}."
                        )
                    W = (W.float() + d).to(out_dtype(W))
                    used.add(name)
                if name in tokens:
                    W = W.to(out_dtype(W))
                    for index, rows in tokens[name]:
                        if rows.shape[1:] != W.shape[1:] or int(index.max()) >= W.shape[0]:
                            raise RuntimeError(
                                f"Unsloth: trainable tokens {tuple(rows.shape)} do not fit {name} {tuple(W.shape)}."
                            )
                        W = W.index_copy(0, index, rows.to(W.dtype))
                    used.add(name)
                W = W.to(out_dtype(W))
            current[key] = W.contiguous()
            current_size += W.numel() * W.element_size()
            if current_size >= shard_limit:
                flush()
    flush()
    handles.clear()

    missing = sorted((set(deltas) | set(biases) | set(tokens)) - used)
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
    index_file = os.path.join(save_directory, "model.safetensors.index.json")
    with open(index_file + ".tmp", "w") as f:
        json.dump({"metadata": {}, "weight_map": new_map}, f, indent = 2)
    os.replace(index_file + ".tmp", index_file)
    _remove_stale_inkling2_weights(save_directory, set(new_map.values()))

    with open(os.path.join(source, "config.json")) as f:
        config = json.load(f)
    for cfg in (config, config.get("text_config") or {}):
        cfg.pop("quantization_config", None)
        if float32:
            for field in ("dtype", "torch_dtype"):
                if field in cfg:
                    cfg[field] = "float32"
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
        f"Unsloth: Merged {len(used)} LoRA weights; saved Inkling 2 in {'float32' if float32 else '16-bit'} to '{save_directory}'."
    )

    if push_to_hub:
        api.upload_folder(folder_path = save_directory, repo_id = final_directory)
        shutil.rmtree(save_directory, ignore_errors = True)
        print(f"Unsloth: Uploaded the merged model to https://huggingface.co/{final_directory}")


INKLING2_TP_REASON = (
    "its attention projections are wrapper modules (q / k / v / o hold the Linear beside scaled L2 clips, short "
    "convolutions over the key / value channels and a sigmoid output gate), next to per-head attention sink and "
    "hidden-state-conditioned relative position bias projections, and tensor parallel has no sharding rule for them"
)
INKLING2_PP_REASON = (
    "its decoder carries a stack of block states (block attention residuals: every layer and the final norm read "
    "learned queries over the block outputs) beside hidden_states, and Inkling-2-Mini also runs layers 6-17 three "
    "times per forward with message gates feeding pass k-1 into pass k, so a pipeline stage boundary would cut "
    "state the pipeline does not carry"
)
INKLING2_PARALLEL_ALTERNATIVE = (
    'train data parallel instead: UNSLOTH_PARALLEL="dp=N" (DDP, the whole model on every GPU) or '
    'UNSLOTH_PARALLEL="fsdp=1" (native FSDP2) under torchrun, or split the model over the GPUs of one process '
    'with device_map = "balanced"'
)


def raise_if_inkling2_parallel_unsupported(config, model_types = ()) -> None:
    """Tensor / pipeline parallel (UNSLOTH_TP_SIZE / UNSLOTH_PP_SIZE, which UNSLOTH_PARALLEL tp= / pp= and
    unsloth.parallelize set) on Inkling 2, refused from the config before any weight loads. Left alone, tensor
    parallel stops at the first wrapper projection and pipeline parallel at the residual modules, both after a full
    load on every rank when quantizing on load (the Mini checkpoint is 14.4 GB in 4-bit)."""
    types = {str(t).lower() for t in (model_types or ())}
    if not (is_inkling2_config(config) or types & INKLING2_MODEL_TYPES):
        return
    import os

    def size(name):
        try:
            return int(str(os.environ.get(name) or "1").strip())
        except ValueError:
            return 1  # parallel_env reports the malformed value itself

    tp, pp = size("UNSLOTH_TP_SIZE"), size("UNSLOTH_PP_SIZE")
    asked = [
        (f"tensor parallel (UNSLOTH_TP_SIZE={tp})", INKLING2_TP_REASON) if tp > 1 else None,
        (f"pipeline parallel (UNSLOTH_PP_SIZE={pp})", INKLING2_PP_REASON) if pp > 1 else None,
    ]
    asked = [x for x in asked if x is not None]
    if not asked:
        return
    raise NotImplementedError(
        "Unsloth: "
        + "; ".join(f"{what} does not support Inkling 2 yet: {why}" for what, why in asked)
        + ". Expert parallel is not supported either. Unset "
        + " / ".join(what.split("(")[1].split("=")[0] for what, _ in asked)
        + " (or drop tp / pp from UNSLOTH_PARALLEL) and "
        + INKLING2_PARALLEL_ALTERNATIVE
        + "."
    )


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


def _message_input_math(query, norm, scaling, block_states, partial_block_states, loop_layer_idx):
    """`Inkling2LoopMessageAttentionResidual.forward` (transformers 5.24) as plain tensor ops: the eager path the module
    takes, op for op, so values and gradients are the module's bit for bit. ``norm`` is the stock unweighted RMSNorm's
    eps (inlined, no module call in the recompute), or the residual's own `norm` module when it has hooks or a forward
    other than the stock one (see _message_norm), so those apply to the original forward and the recomputation alike."""
    import torch

    stacked = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim = -2).float()
    if isinstance(norm, float):
        stacked = stacked * torch.rsqrt(
            stacked.square().mean(-1, keepdim = True) + norm
        )  # unweighted RMSNorm, fp32
    else:
        stacked = norm(stacked)
    query_states = query[loop_layer_idx].float().expand(*stacked.shape[:-2], -1, -1)
    attn_weights = torch.matmul(query_states, stacked.transpose(2, 3)) * scaling
    attn_weights = torch.nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, stacked).transpose(1, 2).contiguous()
    return attn_output.transpose(1, 2)


def _checkpoint_use_reentrant(text_model):
    """The ``use_reentrant`` the text model's layers checkpoint with (transformers keeps it on the partial it installs
    as ``_gradient_checkpointing_func``; 5.24 defaults to False). Unknown: torch's default, True."""
    import functools

    function = getattr(text_model, "_gradient_checkpointing_func", None)
    while isinstance(function, functools.partial):
        if "use_reentrant" in function.keywords:
            return bool(function.keywords["use_reentrant"])
        function = function.func
    return True


def _weights_offloaded(module):
    """True when accelerate keeps the module's weights elsewhere until its hook runs (CPU / disk offload: the
    parameter sits on ``meta``) or the hook executes on another device than the parameter's."""
    query = getattr(module, "query", None)
    if query is None or query.device.type == "meta":
        return True
    hook = getattr(module, "_hf_hook", None)
    for h in getattr(hook, "hooks", None) or ([hook] if hook is not None else []):
        if getattr(h, "offload", False):
            return True
        execution_device = getattr(h, "execution_device", None)
        if execution_device is not None:
            try:
                import torch
                if torch.device(execution_device) != query.device:
                    return True
            except Exception:
                return True
    return False


def _message_norm(norm):
    """What _message_input_math normalizes with: the stock unweighted RMSNorm (no hooks, no instance or class forward
    override, no parameters) as its eps, inlined so the recompute stays plain tensor ops; anything else as the module,
    called like transformers does so its hooks and forward apply."""
    from torch.nn.modules import module as torch_module

    stock = type(norm).__name__ in (
        "Inkling2LoopUnweightedRMSNorm",
        "Inkling2UnweightedRMSNorm",
    ) and getattr(type(norm).forward, "__module__", "").startswith("transformers.")
    if (
        stock
        and "forward" not in norm.__dict__
        and not norm._forward_hooks
        and not norm._forward_pre_hooks
        and not torch_module._global_forward_hooks
        and not torch_module._global_forward_pre_hooks
        and not any(True for _ in norm.parameters())
        and isinstance(getattr(norm, "eps", None), (int, float))
    ):
        return float(norm.eps)
    return norm


def _weak_or_dead(obj):
    import weakref
    return (lambda: None) if obj is None else weakref.ref(obj)


class _CheckpointedMessageInput:
    """Inkling-2-Mini's loop message residual runs in the text model's forward, between the checkpointed decoder
    layers, so everything its fp32 attention saves for backward stays on the GPU for the whole step: about 50 MB per
    call at 1024 tokens, 24 calls (12 loop layers x 2 later passes), so 1.2 GB. Under gradient checkpointing run it
    checkpointed: only its inputs (block states the layers keep anyway) are held and backward recomputes the small
    attention, with the same values and gradients.

    The recompute is the module's math as plain tensor ops (_message_input_math), not a module call: recomputing
    through the module (its accelerate device hook and compiled-module wrapper) from the autograd engine gave
    run-to-run different first-step gradients on a model split over two GPUs, where the engine runs each device's
    backward on its own thread. The checkpoint is reentrant or not as the model's layers are (``autograd.grad`` needs
    non-reentrant), and a module whose weights accelerate offloads keeps its hooked forward (the hook loads them).
    UNSLOTH_INKLING2_MESSAGE_CHECKPOINT=0 keeps the stock residual.

    Installed as the residual's instance ``forward``. ``copy.deepcopy`` of the residual (or the model) binds the copy to
    the copied residual and owner, and the residual pickles: the residual and owner are weak references at run time,
    remapped through the deepcopy memo and pickled as strong references."""

    _unsloth_checkpointed_message_input = True

    def __init__(self, text_model, message_input, forward):
        from torch.utils.checkpoint import checkpoint

        self._checkpoint = checkpoint  # bound at install, as the layers bind theirs at enable
        self._owner = _weak_or_dead(text_model)
        self._module = _weak_or_dead(message_input)
        # The stock bound forward is kept as its importable function, anything else (accelerate's device hook) as is.
        if getattr(forward, "__self__", None) is message_input and hasattr(forward, "__func__"):
            self._function, self._forward = forward.__func__, None
        else:
            self._function, self._forward = None, forward

    def _call_original(self, mi, *args, **kwargs):
        if self._forward is not None:
            return self._forward(*args, **kwargs)
        return self._function(mi, *args, **kwargs)

    def __deepcopy__(self, memo):
        import copy

        new = object.__new__(type(self))
        memo[id(self)] = new
        owner, module = self._owner(), self._module()
        new._owner = _weak_or_dead(memo.get(id(owner), owner))
        new._module = _weak_or_dead(memo.get(id(module), module))
        new._function = self._function
        new._checkpoint = self._checkpoint
        new._forward = copy.deepcopy(self._forward, memo)
        return new

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_owner"], state["_module"] = self._owner(), self._module()
        return state

    def __setstate__(self, state):
        state = dict(state)
        self._owner = _weak_or_dead(state.pop("_owner"))
        self._module = _weak_or_dead(state.pop("_module"))
        self.__dict__.update(state)

    def __call__(self, *args, **kwargs):
        import os

        import torch

        tm, mi = self._owner(), self._module()
        if mi is None:
            raise RuntimeError(
                "Unsloth: the Inkling 2 message residual this forward belongs to was freed."
            )
        if (
            tm is None
            or kwargs
            or len(args) != 3
            or not getattr(tm, "gradient_checkpointing", False)
            or not tm.training
            or not torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or getattr(getattr(mi, "config", None), "_attn_implementation", "eager") != "eager"
            or os.environ.get("UNSLOTH_INKLING2_MESSAGE_CHECKPOINT", "1") == "0"
            or _weights_offloaded(mi)
        ):
            return self._call_original(mi, *args, **kwargs)
        block_states, partial_block_states, loop_layer_idx = args
        device = mi.query.device
        # Where accelerate's hook would put the inputs (a split model); autograd follows the moves.
        block_states = block_states.to(device)
        partial_block_states = partial_block_states.to(device)
        scaling = block_states.shape[-1] ** -0.5
        return self._checkpoint(
            _message_input_math,
            mi.query,
            _message_norm(mi.norm),
            scaling,
            block_states,
            partial_block_states,
            int(loop_layer_idx),
            use_reentrant = _checkpoint_use_reentrant(tm),
            preserve_rng_state = False,
        )


def _checkpointed_message_input(text_model, message_input, forward):
    """See _CheckpointedMessageInput."""
    return _CheckpointedMessageInput(text_model, message_input, forward)


def checkpoint_inkling2_message_inputs(model) -> int:
    """Wrap every loop text model's `message_input` call (see _checkpointed_message_input). Returns the count."""
    n = 0
    for module in model.modules():
        message_input = getattr(module, "message_input", None)
        if message_input is None or not type(message_input).__name__.endswith(
            "LoopMessageAttentionResidual"
        ):
            continue
        if getattr(message_input.forward, "_unsloth_checkpointed_message_input", False):
            continue
        # The instance forward (accelerate's device hook when the model is split) stays inside the checkpoint.
        message_input.forward = _checkpointed_message_input(
            module, message_input, message_input.forward
        )
        n += 1
    return n


def patch_inkling2_model(model) -> bool:
    """Give a loaded Inkling 2 model Unsloth's LoRA target mapping. Returns True when the model is Inkling 2."""
    config = getattr(model, "config", None)
    if not is_inkling2_config(config):
        return False
    cls = type(model)
    if getattr(cls, "unsloth_lora_target_modules", None) is not _unsloth_lora_target_modules:
        cls.unsloth_lora_target_modules = _unsloth_lora_target_modules
    checkpoint_inkling2_message_inputs(model)
    return True
