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
    "inkling2_lora_gguf_adapter",
    "raise_if_merging_inkling2",
    "save_inkling2_merged_16bit",
    "raise_if_inkling2_parallel_unsupported",
    "INKLING2_TP_REASON",
    "INKLING2_PARALLEL_ALTERNATIVE",
    "inkling2_pipeline_adapter",
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
                    if getattr(module, "_did_swap_in_out_features", False)
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


def _checkpoint_quant_method(
    source,
    revision,
    token = None,
):
    """(quant_method, folder) of the checkpoint config at `source` (local folder or Hub id): "" when it has none;
    `folder` holds its config.json (a Hub id's snapshot), None when there is none."""
    import json
    import os

    if os.path.isdir(source):
        path = os.path.join(source, "config.json")
        if not os.path.isfile(path):
            return "", None
    else:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            path = hf_hub_download(source, "config.json", revision = revision, token = token)
        except EntryNotFoundError:
            return "", None
    with open(path, encoding = "utf-8") as f:
        config = json.load(f)
    folder = os.path.dirname(path)
    for cfg in (config, config.get("text_config") or {}):
        quant = cfg.get("quantization_config") if isinstance(cfg, dict) else None
        if isinstance(quant, dict) and quant.get("quant_method"):
            return str(quant["quant_method"]).lower(), folder
    return "", folder


def inkling2_merge_source(
    source,
    revision = None,
    token = None,
    save_directory = None,
):
    """(checkpoint, revision) a merged Inkling 2 export reads its base tensors from.

    A pre-quantized bitsandbytes repo (``unsloth/Inkling-2-Mini-unsloth-bnb-4bit``) stores packed 4-bit attention and
    expert weights, which the merge cannot add LoRA deltas to or write as 16-bit: merge into the original checkpoint
    it was quantized from (the mapper's 16-bit name, at its default revision). Its 4-bit weights are the ones a
    4-bit load of that checkpoint quantizes, so the deltas apply unchanged. A hub cache snapshot of that repo (what
    a run started from a cached model records) maps the same way. A pre-quantized folder with no known original is
    refused, and so is any other local folder (whatever its name: its tensors may not be the Hub repo's).
    `save_directory` (the merged export's folder) must not be the pre-quantized checkpoint's own folder."""
    import os

    method, folder = _checkpoint_quant_method(source, revision, token = token)
    if method != "bitsandbytes":
        return source, revision
    from .loader_utils import prequantized_snapshot_original
    from .mapper import INT_TO_FLOAT_MAPPER

    if os.path.isdir(source):
        original = prequantized_snapshot_original(source)
    else:
        original = INT_TO_FLOAT_MAPPER.get(str(source).lower())
    if original is None:
        raise NotImplementedError(
            f"Unsloth: {source} is a pre-quantized bitsandbytes 4-bit Inkling 2 checkpoint, and a merged export "
            "needs the original weights. Load it by its Hub id (for example unsloth/Inkling-2-Mini-unsloth-bnb-4bit), "
            "or load the original checkpoint (for example unsloth/Inkling-2-Mini) with "
            "`load_in_4bit = True, use_exact_model_name = True` to merge, or save the LoRA adapter with "
            "`model.save_pretrained(...)`."
        )
    if (
        save_directory is not None
        and folder is not None
        and os.path.isdir(save_directory)
        and os.path.samefile(save_directory, folder)
    ):
        raise RuntimeError(
            f"Unsloth: the merged Inkling 2 export would overwrite the checkpoint it was loaded from ({folder}); "
            "pick another directory."
        )
    print(f"Unsloth: {source} is pre-quantized; merging into its original checkpoint {original}.")
    return original, None


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
    source = getattr(base.config, "_name_or_path", None)
    if not source:
        raise RuntimeError("Unsloth: cannot find the Inkling 2 base checkpoint to merge into.")
    loaded_from = source
    source, revision = inkling2_merge_source(
        source,
        getattr(base.config, "_commit_hash", None),
        token = token,
        save_directory = None if push_to_hub else save_directory,
    )
    # A pre-quantized load's converters map that repo's layout (packed stacks under the model's names); the
    # original checkpoint read below needs the model's own mapping.
    conversions = getattr(base, "_weight_conversions", None) if source == loaded_from else None
    if not conversions:
        from transformers.conversion_mapping import get_model_conversion_mapping
        conversions = get_model_conversion_mapping(base, add_legacy = False)
    renamings = [c for c in conversions if isinstance(c, WeightRenaming)]
    converters = [c for c in conversions if isinstance(c, WeightConverter)]
    pattern_to_converter = {p: c for c in converters for p in c.source_patterns}

    if not os.path.isdir(source):
        from huggingface_hub import snapshot_download
        source = snapshot_download(
            source,
            allow_patterns = ["*.safetensors", "*.json", "*.jinja"],
            token = token,
            revision = revision,
        )
    index_path = os.path.join(source, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding = "utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        with safe_open(os.path.join(source, "model.safetensors"), "pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}
    from unsloth_zoo.saving_utils import _shard_name_stays_inside

    # The index comes from the checkpoint's publisher: a shard outside its folder would copy another checkpoint's tensors into the export.
    escaping = sorted({f for f in weight_map.values() if not _shard_name_stays_inside(f)})
    if escaping:
        raise RuntimeError(
            f"Unsloth: the Inkling 2 checkpoint's weight index names a file outside its folder ({escaping[0]!r})."
        )

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
    with open(index_file + ".tmp", "w", encoding = "utf-8") as f:
        json.dump({"metadata": {}, "weight_map": new_map}, f, indent = 2)
    os.replace(index_file + ".tmp", index_file)
    _remove_stale_inkling2_weights(save_directory, set(new_map.values()))

    with open(os.path.join(source, "config.json"), encoding = "utf-8") as f:
        config = json.load(f)
    for cfg in (config, config.get("text_config") or {}):
        cfg.pop("quantization_config", None)
        if float32:
            for field in ("dtype", "torch_dtype"):
                if field in cfg:
                    cfg[field] = "float32"
    with open(os.path.join(save_directory, "config.json"), "w", encoding = "utf-8") as f:
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
        # The upload only adds files, so an existing repo keeps an older model.safetensors or LoRA adapter that would
        # shadow the merged shards on reload: delete them in the same commit, as the local save does.
        stale = sorted(
            name
            for name in api.list_repo_files(final_directory)
            if name not in new_map.values() and _STALE_WEIGHT_FILE.fullmatch(name)
        )
        api.upload_folder(
            folder_path = save_directory,
            repo_id = final_directory,
            delete_patterns = stale or None,
        )
        shutil.rmtree(save_directory, ignore_errors = True)
        print(f"Unsloth: Uploaded the merged model to https://huggingface.co/{final_directory}")


INKLING2_TP_REASON = (
    "tensor parallel shards its attention by heads (query and key / value heads, with their short convolutions, "
    "sinks and relative position bias), so the tp size must divide every attention's query and key / value head "
    "counts"
)
INKLING2_PARALLEL_ALTERNATIVE = (
    'train data parallel instead: UNSLOTH_PARALLEL="dp=N" (DDP, the whole model on every GPU) or '
    'UNSLOTH_PARALLEL="fsdp=1" (native FSDP2) under torchrun, or split the model over the GPUs of one process '
    'with device_map = "balanced"'
)


def _inkling2_tp_head_problem(config, tp):
    """Why tensor parallel over `tp` ranks cannot shard this Inkling 2 config's attention heads, or None. The
    model's own attention (models/tp_native.py, _inkling2_attention_region) shards query and key / value heads
    alike, so tp must divide both counts of the full and the sliding window attention."""
    text = _get(config, "text_config") or config
    bad = []
    for key in (
        "num_attention_heads",
        "num_key_value_heads",
        "swa_num_attention_heads",
        "swa_num_key_value_heads",
    ):
        value = _get(text, key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0 and value % tp:
            bad.append(f"{key} = {value}")
    return ", ".join(bad) or None


def raise_if_inkling2_parallel_unsupported(config, model_types = ()) -> None:
    """Tensor / pipeline parallel (UNSLOTH_TP_SIZE / UNSLOTH_PP_SIZE, which UNSLOTH_PARALLEL tp= / pp= and
    unsloth.parallelize set) on Inkling 2, checked from the config before any weight loads. Tensor parallel is
    refused only for a tp size that does not divide the attention head counts; pipeline parallel runs through
    Inkling2PipelineAdapter (models/pp_native.py); the two together are refused."""
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
    if tp > 1 and pp > 1:
        raise NotImplementedError(
            f"Unsloth: pipeline parallel x tensor parallel (UNSLOTH_PP_SIZE={pp}, UNSLOTH_TP_SIZE={tp}) does not "
            "support Inkling 2 yet: each runs alone (pipeline stages keep Inkling-2-Mini's looped layers on one "
            "stage; tensor parallel shards the attention heads), never combined. Unset UNSLOTH_PP_SIZE / "
            "UNSLOTH_TP_SIZE (or drop pp / tp from UNSLOTH_PARALLEL) for one of them, or "
            + INKLING2_PARALLEL_ALTERNATIVE
            + "."
        )
    heads = _inkling2_tp_head_problem(config, tp) if tp > 1 and config is not None else None
    if heads:
        raise NotImplementedError(
            f"Unsloth: tensor parallel (UNSLOTH_TP_SIZE={tp}) cannot shard this Inkling 2 model: "
            f"{INKLING2_TP_REASON}, and {tp} does not divide {heads}. Choose a tp size that divides them, or "
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
        from .inkling2_packed import INKLING2_PACKED_EXPERTS_ENV, inkling2_packed_experts_enabled
        if inkling2_packed_experts_enabled():
            print(
                f"Unsloth: Inkling 2 stores its MoE experts in NVFP4 / MXFP8; {INKLING2_PACKED_EXPERTS_ENV}=1 keeps "
                "them packed as released."
            )
            return
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


# transformers Linear leaf -> the original checkpoint leaf llama.cpp's Inkling 2 converter maps (transformers'
# own conversion mapping, reversed). Leaves the converter already maps under their transformers name (sink_proj,
# the router, the loop controllers) are absent.
_GGUF_LORA_LEAVES = {
    "self_attn.q_proj.linear": "attn.wq_du",
    "self_attn.k_proj.linear": "attn.wk_dv",
    "self_attn.v_proj.linear": "attn.wv_dv",
    "self_attn.o_proj.linear": "attn.wo_ud",
    "self_attn.o_proj.gate_proj": "attn.gate_attn",
    "self_attn.relative_position_bias.linear": "attn.wr_du",
    "self_attn.k_proj.sconv.weight_proj": "attn.k_sconv.weight_proj",
    "self_attn.v_proj.sconv.weight_proj": "attn.v_sconv.weight_proj",
}
_LORA_KEY = re.compile(r"^(.*)\.(lora_A|lora_B)\.weight$")


def _expert_lora_b_layout(lora_config, parameter):
    """How the saved fused expert lora_B packs its experts: the adapter's own record, else PEFT's rank_major."""
    record = lora_config.get("unsloth_fused_expert_lora") or {}
    if not isinstance(record, dict):
        record = {}
    if record.get("lora_A_layout", "grouped_by_expert") != "grouped_by_expert":
        raise NotImplementedError(
            f"Unsloth: Inkling 2 LoRA GGUF export does not know the expert lora_A layout "
            f"{record.get('lora_A_layout')!r}; export the merged model as GGUF instead."
        )
    per_parameter = (record.get("parameters") or {}).get(parameter) or {}
    layout = (
        per_parameter.get("lora_B_layout")
        or record.get("lora_B_layout")
        or lora_config.get("lora_B_layout")
        or "rank_major"
    )
    if layout not in ("rank_major", "grouped_by_expert"):
        raise NotImplementedError(
            f"Unsloth: Inkling 2 LoRA GGUF export does not know the expert lora_B layout {layout!r}; "
            "export the merged model as GGUF instead."
        )
    return layout


def inkling2_lora_gguf_adapter(
    model,
    adapter_dir,
    converted = (),
):
    """Copy of the adapter saved in ``adapter_dir`` that llama.cpp's convert_lora_to_gguf.py can map, in a new
    temporary directory the caller removes.

    The converter maps Inkling 2's original checkpoint names, while PEFT saves transformers module names:
    attention Linears are renamed to their checkpoint leaves (``self_attn.q_proj.linear`` -> ``attn.wq_du``), and
    each fused expert ParamWrapper (``mlp.experts`` / ``mlp.experts.base_layer``, factors flattened to
    ``(E * r, in)`` and ``(out, E * r)``) becomes ``mlp.experts.gate_up_proj`` / ``down_proj`` with per-expert
    ``(E, r, in)`` / ``(E, out, r)`` factors, the layout llama.cpp applies with mul_mat_id (PEFT 0.18, which does not
    swap a 3D parameter's features, trains the transposed pair, so its factors swap roles). transformers keeps
    gate_up halved (gate rows, then up rows), which is the order the converter writes the base expert stack in.
    A LoRA scaling other than ``lora_alpha / r`` (rsLoRA, alpha_pattern, rank_pattern) is folded into lora_B, since
    the GGUF carries one alpha and llama.cpp scales every tensor by ``alpha / rank``. Modules in ``converted`` were
    saved relative to the original base with the live scale already folded, so they keep the configured one."""
    import json
    import os
    import shutil
    import tempfile

    import torch
    from safetensors.torch import load_file, save_file

    try:
        from peft.tuners.lora.layer import ParamWrapper
    except ImportError:
        ParamWrapper = ()
    from peft.tuners.lora import LoraLayer

    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding = "utf-8") as f:
        lora_config = json.load(f)
    alpha = float(lora_config.get("lora_alpha") or 0)

    # Converter reads the default adapter; fold its live scaling even if another is active.
    if "default" in (getattr(model, "peft_config", None) or {}):
        adapter = "default"
    else:
        adapter = getattr(model, "active_adapter", None)
        if callable(adapter):
            adapter = adapter()
        if isinstance(adapter, (list, tuple)):
            adapter = adapter[0] if adapter else None
        adapter = adapter or "default"

    # Saved keys are the live module names without the adapter name.
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and adapter in getattr(module, "lora_A", {}):
            layers[name] = module

    tensors = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"), device = "cpu")
    out = {}
    swapped_roles = []
    for key, tensor in tensors.items():
        match = _LORA_KEY.match(key)
        if match is None:
            out[key] = tensor  # the converter reports anything it cannot place
            continue
        prefix, kind = match.groups()
        live_name = prefix
        module = layers.get(prefix)
        rank = None
        if ParamWrapper and isinstance(module, ParamWrapper):
            experts = int(module.num_experts)
            parameter = module.parameter_name
            owner = prefix
            while owner.endswith(".base_layer"):
                owner = owner[: -len(".base_layer")]
            prefix = f"{owner}.{parameter}"
            rank = tensor.shape[0 if kind == "lora_A" else 1] // experts
            if kind == "lora_A":
                tensor = tensor.reshape(experts, rank, tensor.shape[-1])
            elif _expert_lora_b_layout(lora_config, parameter) == "grouped_by_expert":
                tensor = tensor.reshape(tensor.shape[0], experts, rank).permute(1, 0, 2)
            else:
                tensor = tensor.reshape(tensor.shape[0], rank, experts).permute(2, 0, 1)
            tensor = tensor.contiguous()
            # PEFT >= 0.19 swaps a (experts, out, in) parameter's features so lora_A reads the last
            # dim; PEFT 0.18 reads dim 1, giving A^T @ B^T (b = A^T, a = B^T in llama.cpp).
            if kind == "lora_A" and not getattr(module, "_did_swap_in_out_features", False):
                swapped_roles.append(prefix)
        else:
            for leaf, checkpoint_leaf in _GGUF_LORA_LEAVES.items():
                if prefix.endswith("." + leaf):
                    prefix = prefix[: -len(leaf)] + checkpoint_leaf
                    break
            if module is not None:
                rank = tensor.shape[0 if kind == "lora_A" else -1]
        if kind == "lora_B" and module is not None and rank:
            if live_name in converted:
                from ._utils import _configured_scaling
                scaling = float(_configured_scaling(model, module, adapter))
            else:
                scaling = float(module.scaling[adapter])
            factor = scaling * rank / alpha if alpha else scaling
            if abs(factor - 1.0) > 1e-6:
                tensor = (tensor.float() * factor).to(tensor.dtype)
        out[f"{prefix}.{kind}.weight"] = tensor
    if any(_LORA_KEY.match(k) and ("vision_tower." in k or ".visual." in k) for k in out):
        print(
            "Unsloth: llama.cpp converts Inkling 2's text backbone only, so the vision tower's LoRA weights are "
            "not in the GGUF LoRA; export the merged model to keep them."
        )
    for prefix in swapped_roles:
        A, B = out[f"{prefix}.lora_A.weight"], out[f"{prefix}.lora_B.weight"]
        out[f"{prefix}.lora_A.weight"] = B.transpose(1, 2).contiguous()
        out[f"{prefix}.lora_B.weight"] = A.transpose(1, 2).contiguous()

    directory = tempfile.mkdtemp(prefix = "unsloth-inkling2-lora-gguf-")
    try:
        save_file(
            out, os.path.join(directory, "adapter_model.safetensors"), metadata = {"format": "pt"}
        )
        for filename in ("adapter_config.json", "README.md"):
            if os.path.isfile(os.path.join(adapter_dir, filename)):
                shutil.copyfile(
                    os.path.join(adapter_dir, filename), os.path.join(directory, filename)
                )
    except BaseException:
        shutil.rmtree(directory, ignore_errors = True)
        raise
    return directory


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
    """True when accelerate offloads the weights (on meta) or its hook runs on another device."""
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


# ------------------------------------------------------------------------------------------- pipeline parallel


class Inkling2PipelineAdapter:
    """How pipeline parallelism (models/pp_native.py) splits an Inkling 2 text model.

    Every decoder layer reads ``block_states`` (block attention residuals: the inputs of the block boundary layers,
    ``layer_idx % n_layers_per_attn_res_block == 0``) beside ``hidden_states``, the final norm reads them through
    ``final_input``, and Inkling-2-Mini runs layers ``[loop_layer_index_start, loop_layer_index_end)`` several times,
    each later pass restarting from the pre-loop states with messages (``message_input``) from the previous pass.
    So a stage boundary carries two tensors, ``hidden_states`` [batch, seq, hidden] and ``block_states``
    [batch, seq, blocks, hidden] (the block count is fixed by the cut), the looped layers stay on one stage (a
    stage starts at or before the loop start or at or after the loop end), ``final_input`` lives with the last
    stage and ``message_input`` with the stage holding the loop. The text model's forward runs only this stage's
    layers (``stage_forward``, the transformers 5.24 forward restricted to ``[start, end)``)."""

    def __init__(self, decoder):
        self.config = decoder.config
        config = self.config
        self.num_layers = int(config.num_hidden_layers)
        self.block = int(getattr(config, "n_layers_per_attn_res_block", 0) or 0)
        start = getattr(config, "loop_layer_index_start", None)
        end = getattr(config, "loop_layer_index_end", None)
        passes = getattr(config, "loop_num_passes", None)
        self.loop = None
        if hasattr(decoder, "message_input") and None not in (start, end, passes):
            self.loop = (int(start), int(end), int(passes))
        # Decoder children beside embed / layers / norm, with the layer whose stage owns them (None = the last).
        self.extra_modules = {"final_input": None}
        if self.loop is not None:
            self.extra_modules["message_input"] = self.loop[0]

    def cut_points(self):
        """Layer indices where a stage may start: anywhere but strictly inside the looped layers."""
        points = range(1, self.num_layers)
        if self.loop is None:
            return list(points)
        start, end, _ = self.loop
        return [i for i in points if not start < i < end]

    def blocks_before(self, cut):
        """Entries of block_states when the stage starting at ``cut`` begins (the boundary layers before it)."""
        return sum(1 for i in range(min(cut, self.num_layers)) if i % self.block == 0)

    def carry(self, cut):
        """Tensors crossing the boundary before layer ``cut``: [(trailing shape, dtype)], dtype None = the activation
        dtype (both stay in the model's hidden dtype in transformers' forward)."""
        hidden = int(self.config.hidden_size)
        return [((hidden,), None), ((self.blocks_before(cut), hidden), None)]

    def owner_layer(self, name):
        return self.extra_modules.get(name)

    def install(self, decoder, ctx, stage_output):
        install_inkling2_stage_forward(decoder, ctx, stage_output)
        pin_inkling2_dequant_plan(decoder.layers[ctx.start : ctx.end])


def inkling2_pipeline_adapter(decoder):
    """An Inkling2PipelineAdapter for an Inkling 2 text model whose forward is the one stage_forward mirrors, else None
    (not Inkling 2). Raises ValueError for an Inkling 2 text model whose layout it does not know."""
    if decoder is None or not type(decoder).__name__.startswith("Inkling2"):
        return None
    if not is_inkling2_config(getattr(decoder, "config", None)):
        return None
    config = decoder.config
    layers = getattr(decoder, "layers", None)
    missing = [
        name
        for name in ("embed_tokens", "layers", "norm", "final_input")
        if getattr(decoder, name, None) is None
    ]
    block = getattr(config, "n_layers_per_attn_res_block", None)
    if (
        missing
        or not isinstance(block, int)
        or block < 1
        or len(layers) != config.num_hidden_layers
    ):
        raise ValueError(
            f"Unsloth: pipeline parallelism does not know this Inkling 2 text model layout ({type(decoder).__name__}"
            f"{', missing ' + str(missing) if missing else ''}); it mirrors transformers 5.24's."
        )
    adapter = Inkling2PipelineAdapter(decoder)
    if adapter.loop is None and getattr(config, "loop_layer_index_start", None) is not None:
        raise ValueError(
            f"Unsloth: pipeline parallelism found Inkling 2 loop settings without a message_input module in "
            f"{type(decoder).__name__}; this layout is not supported."
        )
    if adapter.loop is not None:
        start, end, passes = adapter.loop
        if not (0 < start < end <= adapter.num_layers) or passes < 1:
            raise ValueError(
                f"Unsloth: pipeline parallelism does not support the Inkling 2 loop [{start}, {end}) x {passes}."
            )
    return adapter


def _modeling(decoder):
    import sys
    return sys.modules[type(decoder).__module__]


def inkling2_stage_layers(decoder, start, end, hidden_states, block_states, layer_kwargs):
    """Decoder layers ``[start, end)`` of an Inkling 2 text model on (hidden_states, block_states), as transformers
    5.24's text forward runs them (Mini: the looped layers' passes and messages when the loop starts here)."""
    import torch

    config = decoder.config
    layers = decoder.layers
    loop = None
    if (
        hasattr(decoder, "message_input")
        and getattr(config, "loop_layer_index_start", None) is not None
    ):
        loop = (config.loop_layer_index_start, config.loop_layer_index_end, config.loop_num_passes)

    def run(layer, h, b, *extra, **more):
        new = layer(h, b, *extra, **layer_kwargs[layer.attention_type], **more)
        if layer.is_block_boundary:
            b = torch.cat([b.to(h.device), h.unsqueeze(-2)], dim = -2)
        return new, b

    h, b = hidden_states, block_states
    i = start
    while i < end:
        if loop is not None and loop[0] < i < loop[1]:
            raise RuntimeError(
                f"Unsloth: an Inkling 2 pipeline stage starts at layer {i}, inside the looped layers "
                f"[{loop[0]}, {loop[1]})."
            )
        if loop is None or i != loop[0]:
            h, b = run(layers[i], h, b)
            i += 1
            continue
        loop_start, loop_end, passes = loop
        if end < loop_end:
            raise RuntimeError(
                f"Unsloth: an Inkling 2 pipeline stage ends at layer {end}, inside the looped layers "
                f"[{loop_start}, {loop_end})."
            )
        num_loop_layers = loop_end - loop_start
        pre_h, pre_b = h, b
        num_pre_loop_blocks = pre_b.shape[-2]
        for layer in layers[loop_start:loop_end]:
            h, b = run(layer, h, b)
        for loop_idx in range(1, passes):
            previous_b = b[..., num_pre_loop_blocks + 1 :, :]
            previous_h = h
            h, b = pre_h, pre_b
            cache_offset = config.num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start
            for loop_layer_idx, layer in enumerate(layers[loop_start:loop_end]):
                messages = decoder.message_input(previous_b, previous_h, loop_layer_idx)
                h, b = run(layer, h, b, messages, cache_offset = cache_offset)
        i = loop_end
    return h, b


# Model-level flags the text forward's output recorder consumes; never handed to the layers.
_STAGE_DROPPED_KWARGS = (
    "output_attentions",
    "output_hidden_states",
    "output_router_logits",
    "return_dict",
)


def _stage_text_forward(
    decoder, ctx, stage_output, input_ids, attention_mask, position_ids, inputs_embeds, kwargs
):
    import torch

    modeling = _modeling(decoder)
    if inputs_embeds is None:
        inputs_embeds = decoder.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device = inputs_embeds.device).unsqueeze(
            0
        )
    for key in _STAGE_DROPPED_KWARGS:
        kwargs.pop(key, None)
    if ctx.is_first:
        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(
            *inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1]
        )
    else:
        hidden_states, block_states = ctx.stage_input
    mapping = attention_mask
    if not isinstance(mapping, dict):
        mask_kwargs = {
            "config": decoder.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache": None,
            "position_ids": position_ids,
        }
        mapping = {
            "full_attention": modeling.create_causal_mask(**mask_kwargs),
            "sliding_attention": modeling.create_sliding_window_causal_mask(**mask_kwargs),
            "linear_attention": modeling.create_recurrent_attention_mask(**mask_kwargs),
        }
    distance = {
        "full_attention": modeling.get_relative_distance(inputs_embeds, None, is_sliding = False),
        "sliding_attention": modeling.get_relative_distance(inputs_embeds, None, is_sliding = True),
    }
    layer_kwargs = {
        kind: dict(
            attention_mask = mapping[kind],
            relative_distance = distance[kind],
            conv_mask = mapping["linear_attention"],
            cache = None,
            position_ids = position_ids,
            **kwargs,
        )
        for kind in ("full_attention", "sliding_attention")
    }
    hidden_states, block_states = inkling2_stage_layers(
        decoder, ctx.start, ctx.end, hidden_states, block_states, layer_kwargs
    )
    if not ctx.is_last:
        # The layers' short convolutions hand out time-major (transposed) outputs; the wire needs dense rows.
        hidden_states, block_states = hidden_states.contiguous(), block_states.contiguous()
        if torch.is_grad_enabled() and not block_states.requires_grad:
            # Frozen-only: detach to a leaf so the next stage's gradient has somewhere to go.
            block_states = block_states.detach().requires_grad_()
        raise stage_output((hidden_states, block_states))
    hidden_states = decoder.final_input(block_states, hidden_states)[0]
    hidden_states = decoder.norm(hidden_states).to(inputs_embeds.dtype)
    return modeling.MoeModelOutputWithPast(last_hidden_state = hidden_states, cache = None)


def install_inkling2_stage_forward(decoder, ctx, stage_output):
    """Run only this pipeline stage's layers in the text model's forward while ``ctx`` is active (a pipeline step);
    the stock forward otherwise (a replica put back for generate / exports). ``stage_output`` is the exception a
    stage before the last raises with (hidden_states, block_states)."""
    import functools

    original = decoder.__dict__.get("forward")
    stock = type(decoder).forward

    def stage(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        cache = None,
        inputs_embeds = None,
        use_cache = None,
        **kwargs,
    ):
        return _stage_text_forward(
            self, ctx, stage_output, input_ids, attention_mask, position_ids, inputs_embeds, kwargs
        )

    try:  # FlexAttention keeps a per-forward block mask cache (inkling2_flex)
        from .inkling2_flex import _wrap_text_forward
        stage = _wrap_text_forward(stage)
    except Exception:
        pass

    @functools.wraps(stock)
    def forward(*args, **kwargs):
        if not ctx.active:
            if original is not None:
                return original(*args, **kwargs)
            return stock(decoder, *args, **kwargs)
        if args:
            raise RuntimeError(
                "Unsloth: the Inkling 2 pipeline stage forward takes keyword arguments only "
                f"(got {len(args)} positional)."
            )
        if (
            kwargs.get("cache") is not None
            or kwargs.get("past_key_values") is not None
            or kwargs.get("use_cache")
        ):
            raise RuntimeError(
                "Unsloth: the Inkling 2 pipeline stage forward trains without a KV cache."
            )
        kwargs.pop("past_key_values", None)
        return stage(decoder, **kwargs)

    forward._unsloth_inkling2_pipeline_stage = True
    decoder.forward = forward
    return forward


_DEQUANT_CHUNK_ENV = "UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS"


def _expert_stack_shape(param):
    while hasattr(param, "base_layer"):
        param = param.base_layer
    shape = getattr(param, "_original_shape", None) or getattr(
        getattr(param, "quant_state", None), "shape", None
    )
    if shape is None and getattr(param, "dim", lambda: 0)() == 3:
        shape = param.shape
    return tuple(int(x) for x in shape) if shape is not None and len(shape) == 3 else None


def pin_inkling2_dequant_plan(layers):
    """Under pipeline parallel, fix unsloth_zoo's expert dequant chunking for this stage's run.

    zoo reads an Inkling 2 expert stack in chunks when the stack exceeds twice ``_unsloth_dequant_scratch_fraction``
    of the card and does not fit the free memory at that moment. Its checkpoint replay reuses the first pass's
    decision only inside zoo's own checkpoint; under torch's non-reentrant checkpoint the recompute decides again,
    and a pipeline stage holding several micro-batches sees other free memory then (real Inkling-2-Mini on a T4,
    stage 0 = layers 0..17: ``CheckpointError: ... saved during forward: 202, during recomputation: 200``). On a card
    where the decision depends on free memory, always chunk with zoo's own chunk size (the chunks give the whole
    stack's values bit for bit); elsewhere zoo never chunks and nothing is set. An explicit
    UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS or UNSLOTH_MOE_DEQUANT_MAX_MB wins. Returns the chunk set, else None."""
    import os

    import torch

    if _DEQUANT_CHUNK_ENV in os.environ or os.environ.get(
        "UNSLOTH_MOE_DEQUANT_MAX_MB", "auto"
    ).strip().lower() not in (
        "",
        "auto",
    ):
        return None
    for layer in layers:
        for module in layer.modules():
            fraction = getattr(type(module), "_unsloth_dequant_scratch_fraction", None)
            if not fraction:
                continue
            shapes, device = [], None
            for name in ("gate_up_proj", "down_proj"):
                param = getattr(module, name, None)
                shape = _expert_stack_shape(param) if param is not None else None
                if shape is None:
                    continue
                shapes.append(shape)
                base = param
                while hasattr(base, "base_layer"):
                    base = base.base_layer
                device = getattr(base, "device", device)
            if (
                not shapes
                or device is None
                or device.type != "cuda"
                or not torch.cuda.is_available()
            ):
                return None
            element = 2  # zoo dequantizes to the 16-bit compute dtype
            per_expert = max(a * b for _, a, b in shapes) * element
            scratch = shapes[0][0] * per_expert
            cap = float(fraction) * torch.cuda.get_device_properties(device).total_memory
            if cap <= 0 or scratch <= 2 * cap:
                return None  # zoo keeps the whole stack whatever the free memory: already one plan
            chunk = max(1, int(cap // per_expert))
            os.environ[_DEQUANT_CHUNK_ENV] = str(chunk)
            print(
                f"Unsloth: Inkling 2 pipeline stage: expert stacks dequantized {chunk} experts at a time "
                f"({_DEQUANT_CHUNK_ENV}={chunk}), so checkpoint recomputes repeat the forward's plan."
            )
            return chunk
    return None
