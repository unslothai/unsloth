# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""REAP expert pruning (Router-weighted Expert Activation Pruning, arXiv:2510.13999).

Same saliency, expert selection and schedule as llm-compressor's ``REAPPruningModifier``: each
decoder layer is calibrated on the outputs of the already pruned layers before it, its lowest
saliency experts are dropped, and the pruned layer produces the next layer's inputs. Expert
outputs come from the model's own fused experts kernel (one call with every routed (token, slot)
pair as its own row) instead of per-expert Linear modules, so no extra dependency is needed.
"""

import inspect
import json
from collections.abc import MutableMapping
import os
import time

import torch
import torch.nn as nn

__all__ = ["reap_prune", "raise_if_unsaved_reap"]

_NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
_NUM_EXPERTS_MODULE_KEYS = ("num_experts", "n_experts", "n_routed_experts", "num_local_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "top_k", "moe_top_k", "top_k_experts")
_ROUTER_ATTRS = ("gate", "router")
_DEFAULT_DATASET = "mlabonne/open-perfectblend"
_TAP_CHUNK_BYTES = 256 << 20
_STATEFUL_LAYERS = (
    "Unsloth: REAP calibrates one decoder layer at a time, so it does not support decoder layers "
    "that hand extra state to the next layer (GLM-5 / HY-V4 sparse-attention top-k indices, ZAYA "
    "router states)."
)
# llm-compressor's fallback for tokenizers without a chat template (PerfectBlendDataset).
_DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}\n"
    "{% if message['role'] == 'user' %}\n"
    "{{ '<|user|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'system' %}\n"
    "{{ '<|system|>\n' + message['content'] + eos_token }}\n"
    "{% elif message['role'] == 'assistant' %}\n"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n"
    "{% endif %}\n"
    "{% if loop.last and add_generation_prompt %}\n"
    "{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
)


class _StopForward(Exception):
    pass


def _configs(model):
    config = model.config
    text_config = getattr(config, "text_config", None)
    return [c for c in (config, text_config) if c is not None]


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _first_attr(
    configs,
    keys,
    per_layer = False,
):
    for config in reversed(configs):  # text_config first
        for key in keys:
            value = getattr(config, key, None)
            if _is_int(value):
                return value
            # HunYuan stores `moe_topk` per layer (`[8] * 32`); the largest bounds every layer.
            if (
                per_layer
                and isinstance(value, (list, tuple))
                and value
                and all(map(_is_int, value))
            ):
                return max(value)
    return None


def _moe_parts(module, num_experts):
    """(experts, router) when ``module`` is a fused-experts MoE block, else None."""
    experts = getattr(module, "experts", None)
    if not isinstance(experts, nn.Module) or isinstance(experts, nn.ModuleList):
        return None
    router = next(
        (
            getattr(module, a)
            for a in _ROUTER_ATTRS
            if isinstance(getattr(module, a, None), nn.Module)
        ),
        None,
    )
    if router is None:
        return None
    try:
        params = list(inspect.signature(experts.forward).parameters.values())
    except (TypeError, ValueError):
        params = []
    # The tap needs forward(hidden_states, top_k_index, top_k_weights); Llama4 passes one tensor.
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if params and len(positional) < 3 and not any(p.kind == p.VAR_POSITIONAL for p in params):
        return None
    if not any(
        p.dim() == 3 and p.shape[0] == num_experts for p in experts.parameters(recurse = False)
    ):
        return None
    return experts, router


def _find_layers(model, num_experts):
    """(owner, layers): the ModuleList of decoder layers holding the MoE blocks and its parent."""
    best, stacks = None, set()
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
            continue
        n_moe = sum(
            any(_moe_parts(m, num_experts) is not None for m in layer.modules()) for layer in layers
        )
        if n_moe and (best is None or n_moe > best[0]):
            best = (n_moe, module, layers)
        if n_moe:
            stacks.add(id(layers))
    if best is None:
        return None, None
    if len(stacks) > 1:
        # Pruning one stack would still rewrite the expert counts the other stacks read.
        raise ValueError(
            "Unsloth: REAP found MoE layers in several decoder stacks (e.g. separate multimodal "
            "decoders); it prunes models with one MoE decoder stack."
        )
    return best[1], best[2]


def _layer_moes(layer, num_experts):
    found, seen = [], set()
    for name, module in layer.named_modules():
        parts = _moe_parts(module, num_experts)
        if parts is not None and id(parts[0]) not in seen:
            seen.add(id(parts[0]))
            found.append((name, module, parts[0], parts[1]))
    return found


def _layer_moe(layer, num_experts):
    found = _layer_moes(layer, num_experts)
    return found[0] if found else None


def _check_weights(experts, router, layer_name):
    # The router recursively: HunYuan / AfMoE keep its Linear in a child module (bnb would quantize it).
    for owner, kind, recurse in ((experts, "experts", False), (router, "router", True)):
        for name, p in owner.named_parameters(recurse = recurse):
            if not p.is_floating_point() or p.element_size() < 2 or p.is_meta:
                raise RuntimeError(
                    f"Unsloth: REAP needs 16-bit or 32-bit MoE weights, but {layer_name} {kind}.{name} "
                    f"is {p.dtype}{' on the meta device' if p.is_meta else ''}. Load the model with "
                    "`load_in_4bit = False` (and without fp8 / offloaded weights), then run REAP."
                )


def _row_text(row, tokenizer, chat):
    if "text" in row and isinstance(row["text"], str):
        return row["text"]
    messages = row.get("messages", row.get("conversations"))
    if not isinstance(messages, list):
        raise ValueError(
            "Unsloth: each REAP calibration row needs `input_ids`, `text`, `messages` or "
            f"`conversations`; got columns {list(row)}."
        )
    messages = [
        {
            "role": {"human": "user", "gpt": "assistant"}.get(
                m.get("role", m.get("from")), m.get("role", m.get("from"))
            ),
            "content": m.get("content", m.get("value")),
        }
        for m in messages
    ]
    kwargs = (
        {} if getattr(chat, "chat_template", None) else {"chat_template": _DEFAULT_CHAT_TEMPLATE}
    )
    return chat.apply_chat_template(messages, tokenize = False, add_generation_prompt = False, **kwargs)


def _text_tokenizer(tokenizer):
    """The text tokenizer of a processor. Only a processor is unwrapped: MistralCommonBackend is
    itself an HF tokenizer but keeps its raw mistral_common backend in ``.tokenizer``."""
    try:
        from transformers import ProcessorMixin
    except ImportError:
        return tokenizer
    return tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer


def _calibration_ids(dataset, tokenizer, num_samples, max_seq_length):
    """Token id lists, truncated to ``max_seq_length`` like llm-compressor's tokenize step."""
    text_tokenizer = _text_tokenizer(tokenizer)
    chat = tokenizer if getattr(tokenizer, "chat_template", None) else text_tokenizer
    if dataset is None:
        dataset = _DEFAULT_DATASET
    if isinstance(dataset, str):
        from datasets import load_dataset
        dataset = load_dataset(dataset, split = "train", streaming = True)
    ids = []
    for row in dataset:
        if len(ids) >= num_samples:
            break
        if isinstance(row, str):
            row = {"text": row}
        elif isinstance(row, (list, tuple)):
            row = {"input_ids": row}
        if "input_ids" in row:
            sample = list(row["input_ids"])
            mask = row.get("attention_mask")
            if mask is not None:  # padded pretokenized rows: keep the real tokens only
                sample = [t for t, m in zip(sample, list(mask)) if m]
            sample = sample[:max_seq_length]
        else:
            if tokenizer is None:
                raise ValueError(
                    "Unsloth: REAP needs a tokenizer unless the dataset has `input_ids`."
                )
            sample = text_tokenizer(
                _row_text(row, tokenizer, chat), truncation = True, max_length = max_seq_length
            )["input_ids"]
        if len(sample):
            ids.append(sample)
    if not ids:
        raise ValueError("Unsloth: REAP got no calibration samples from the dataset.")
    return ids


def _batches(ids, max_batch_tokens, pad_id):
    """Length-sorted right-padded batches. Right padding cannot change real tokens in a causal
    model, so no attention mask is passed and padded positions are only masked out of saliency."""
    order = sorted(range(len(ids)), key = lambda i: -len(ids[i]))
    batches, current = [], []
    for i in order:
        width = len(ids[current[0]]) if current else len(ids[i])
        if current and (len(current) + 1) * width > max_batch_tokens:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    out = []
    for batch in batches:
        width = len(ids[batch[0]])
        input_ids = torch.full((len(batch), width), pad_id, dtype = torch.long)
        valid = torch.zeros((len(batch), width), dtype = torch.bool)
        for row, i in enumerate(batch):
            input_ids[row, : len(ids[i])] = torch.tensor(ids[i], dtype = torch.long)
            valid[row, : len(ids[i])] = True
        out.append((input_ids, valid))
    return out


def _to(
    obj,
    device,
    memo = None,
):
    """Moves nested tensors; ``memo`` keeps tensors shared across layers (rotary tables) shared."""
    if torch.is_tensor(obj):
        if memo is None:
            return obj.to(device, non_blocking = True)
        if id(obj) not in memo:
            memo[id(obj)] = obj.to(device)
        return memo[id(obj)]
    if isinstance(obj, tuple):
        return tuple(_to(o, device, memo) for o in obj)
    if isinstance(obj, list):
        return [_to(o, device, memo) for o in obj]
    if isinstance(obj, dict):
        return {k: _to(v, device, memo) for k, v in obj.items()}
    return obj


def _tensor_bytes(obj, seen = None):
    """Bytes of the distinct tensors nested in ``obj`` (masks shared by every layer count once)."""
    seen = set() if seen is None else seen
    if torch.is_tensor(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        return obj.numel() * obj.element_size()
    if isinstance(obj, (tuple, list)):
        return sum(_tensor_bytes(o, seen) for o in obj)
    if isinstance(obj, dict):
        return sum(_tensor_bytes(o, seen) for o in obj.values())
    return 0


def _capture_inputs(owner, layers, input_ids):
    """Embedding output plus the per-layer kwargs (masks, rotary embeddings, ...) the model
    passes to each decoder layer, by running the model with every layer stubbed out."""
    captured = [None] * len(layers)
    first = {}
    originals = []
    for i, layer in enumerate(layers):
        originals.append(layer.__dict__.get("forward"))

        def stub(
            *args,
            _i = i,
            **kwargs,
        ):
            hidden = args[0] if args else kwargs.pop("hidden_states")
            if _i == 0:
                first["hidden"] = hidden
            captured[_i] = (tuple(args[1:]), kwargs)
            if _i == len(layers) - 1:
                raise _StopForward
            return hidden

        layer.forward = stub
    try:
        owner(input_ids = input_ids, use_cache = False)
    except _StopForward:
        pass
    except (TypeError, ValueError) as error:
        if "unpack" not in str(error):
            raise
        raise ValueError(_STATEFUL_LAYERS) from error
    finally:
        for layer, original in zip(layers, originals):
            if original is None:
                del layer.forward
            else:
                layer.forward = original
    return first["hidden"], captured


def _layer_device(layer, default):
    for p in layer.parameters():
        return p.device
    return default


def _hidden(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _collective_device():
    """(device, accumulation dtype) for a collective on the current process group: CUDA for NCCL,
    CPU for gloo, else the current accelerator (HCCL on Ascend, XCCL on Intel XPU). MPS, XPU and
    NPU have no (or slow) float64, so they accumulate in float32 like ``_Saliency.add``."""
    backend = str(torch.distributed.get_backend())
    if "nccl" in backend:
        device = torch.device("cuda", torch.cuda.current_device())
    elif backend == "gloo":
        device = torch.device("cpu")
    else:
        # torch 2.6 names the index getter current_device_idx.
        accel = torch.accelerator
        index = getattr(accel, "current_device_index", None) or accel.current_device_idx
        device = torch.device(accel.current_accelerator().type, index())
    return device, torch.float64 if device.type in ("cuda", "cpu") else torch.float32


# Written next to a pruned checkpoint saved by reap_prune(..., save_directory).
REAP_PROVENANCE_FILE = "unsloth_reap.json"


def _stream_in(layer, device):
    """``(device the layer runs on, whether it was moved)``: a CPU-resident layer (not under
    accelerate hooks) is moved to ``device`` for its turn."""
    layer_device = _layer_device(layer, device)
    if layer_device.type == "cpu" and device.type != "cpu" and not hasattr(layer, "_hf_hook"):
        layer.to(device)
        return device, True
    return layer_device, False


def _ranks_agree(error, what):
    """Every rank learns whether any rank failed ``what``, so all stop together instead of some
    waiting in a later collective. Re-raises this rank's own error. No-op without a process group."""
    dist = torch.distributed
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() < 2:
        if error is not None:
            raise error
        return
    device, acc = _collective_device()
    failed = torch.tensor([1.0 if error is not None else 0.0], dtype = acc, device = device)
    dist.all_reduce(failed)
    if error is not None:
        raise error
    if failed.item():
        raise RuntimeError(f"Unsloth: REAP stopped because another rank failed to {what}.")


class _Saliency:
    """Running sum of ``gate * ||expert output||`` and routed-token count per expert (float64)."""

    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.total = None
        self.count = None

    def add(
        self,
        expert_ids,
        weights,
        norms,
        mask = None,
    ):
        # MPS, consumer Intel XPUs and Ascend NPUs have no (or slow, partial) float64.
        acc = torch.float64 if norms.device.type in ("cuda", "cpu") else torch.float32
        if self.total is None:
            self.total = torch.zeros(self.num_experts, dtype = acc, device = norms.device)
            self.count = torch.zeros(self.num_experts, dtype = acc, device = norms.device)
        ones = torch.ones_like(norms, dtype = acc) if mask is None else mask.to(acc)
        self.total.index_add_(0, expert_ids, weights.to(acc) * norms.to(acc) * ones)
        self.count.index_add_(0, expert_ids, ones)

    def all_reduce(self):
        """Sum over the ranks of a distributed job, so every rank keeps the same experts (a DDP
        job with sharded calibration data would otherwise pair unrelated experts across ranks)."""
        dist = torch.distributed
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() < 2:
            return
        device, acc = _collective_device()
        both = torch.zeros(2, self.num_experts, dtype = acc, device = device)
        if self.total is not None:
            both[0] = self.total.to(device, acc)
            both[1] = self.count.to(device, acc)
        dist.all_reduce(both)
        self.total, self.count = both[0], both[1]

    def mean(self):
        if self.total is None:
            return torch.zeros(self.num_experts, dtype = torch.float64)
        return self.total.cpu().double() / self.count.cpu().double().clamp(min = 1.0)


def _retained(saliency, n_drop, n_drop_per_group, n_group):
    """Experts kept after dropping the lowest saliency ones (per group for group-limited routers)."""
    num_experts = saliency.numel()
    if n_drop_per_group is None:
        drop = set(torch.topk(saliency, n_drop, largest = False).indices.tolist())
        return [i for i in range(num_experts) if i not in drop]
    group_size = num_experts // n_group
    kept = []
    for g in range(n_group):
        lo = g * group_size
        drop = {
            lo + i
            for i in torch.topk(
                saliency[lo : lo + group_size], n_drop_per_group, largest = False
            ).indices.tolist()
        }
        kept.extend(i for i in range(lo, lo + group_size) if i not in drop)
    return kept


@torch.no_grad()
def _prune_block(block, experts, router, retained, num_experts):
    index = torch.tensor(retained, dtype = torch.long)
    # Experts on dim 0; anything per expert inside the router (HunYuan `gate.wg`, AfMoE `router.gate`,
    # Ernie's [1, E] `moe_statics` bias); the block's own expert biases.
    targets = [(experts, None)] + [(m, None) for m in router.modules()]
    targets.append((block, ("e_score_correction_bias", "expert_bias")))
    for module, only in targets:
        tensors = list(module.named_parameters(recurse = False))
        tensors += list(module.named_buffers(recurse = False))
        # Inkling's router weight carries the shared experts' rows after the routed ones.
        shared = getattr(module, "n_shared_experts", 0) if module is not experts else 0
        shared = shared if isinstance(shared, int) else 0
        for name, tensor in tensors:
            if only is not None and name not in only or not tensor.dim():
                continue
            if shared and tensor.shape[0] == num_experts + shared:
                kept = torch.cat([index, torch.arange(num_experts, num_experts + shared)])
                new = tensor.data.index_select(0, kept.to(tensor.device)).contiguous()
                setattr(module, name, nn.Parameter(new, requires_grad = tensor.requires_grad))
                continue
            if tensor.shape[0] == num_experts:
                dim = 0
            elif module is not experts and tensor.shape[-1] == num_experts:
                dim = -1
            else:
                continue
            new = tensor.data.index_select(dim, index.to(tensor.device)).contiguous()
            if isinstance(tensor, nn.Parameter):
                setattr(module, name, nn.Parameter(new, requires_grad = tensor.requires_grad))
            else:
                module._buffers[name] = new
        for key in _NUM_EXPERTS_MODULE_KEYS:
            value = getattr(module, key, None)
            if isinstance(value, int) and value == num_experts:
                setattr(module, key, len(retained))
        if shared and getattr(module, "n_total_experts", None) == num_experts + shared:
            module.n_total_experts = len(retained) + shared
        if module is not experts and getattr(module, "out_features", None) == num_experts:
            module.out_features = len(retained)
    # Kernel configs Unsloth's MoE backend cached for the old expert count.
    if getattr(experts, "_unsloth_moe_configs", None) is not None:
        experts._unsloth_moe_configs = None


class _ExpertsTap:
    """Wraps an experts module's forward to record per (token, slot) expert output norms."""

    def __init__(
        self,
        experts,
        saliency,
        post = None,
    ):
        self.experts = experts
        self.saliency = saliency
        # Nemotron-H latent MoE maps the summed expert output back to the hidden size after the
        # experts, so ||f_j|| is measured after that shared Linear (without its bias, added once).
        self.post = post
        self.valid = None
        self.stop = False
        self.original = experts.__dict__.get("forward")
        self.forward = experts.forward
        experts.forward = self

    def __call__(self, hidden_states, top_k_index, top_k_weights, *args, **kwargs):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        top_k_index = top_k_index.reshape(hidden_states.shape[0], -1)
        top_k_weights = top_k_weights.reshape(hidden_states.shape[0], -1)
        n_tokens, top_k = top_k_index.shape
        valid = self.valid if self.valid is not None and self.valid.numel() == n_tokens else None
        step = max(
            1, _TAP_CHUNK_BYTES // (top_k * hidden_states.shape[-1] * hidden_states.element_size())
        )
        outs = []
        for start in range(0, n_tokens, step):
            x = hidden_states[start : start + step]
            index = top_k_index[start : start + step]
            gate = top_k_weights[start : start + step]
            n = x.shape[0]
            # One row per routed (token, slot) pair with gate 1 gives each expert's raw output f_j(t).
            rows = self.forward(
                x.repeat_interleave(top_k, dim = 0),
                index.reshape(-1, 1),
                torch.ones((n * top_k, 1), dtype = gate.dtype, device = gate.device),
                *args,
                **kwargs,
            )
            scored = rows if self.post is None else nn.functional.linear(rows, self.post.weight)
            norms = torch.linalg.vector_norm(
                scored,
                dim = -1,
                dtype = torch.float64 if rows.dtype == torch.float64 else torch.float32,
            )
            mask = None
            if valid is not None:
                # Masking by multiplication avoids the host sync boolean indexing needs.
                mask = (
                    valid[start : start + step]
                    .to(norms.device, non_blocking = True)
                    .repeat_interleave(top_k)
                )
            self.saliency.add(index.reshape(-1).long(), gate.reshape(-1), norms, mask)
            if not self.stop:
                out = rows.view(n, top_k, -1) * gate.unsqueeze(-1).to(rows.dtype)
                outs.append(out.sum(1).to(rows.dtype))
        if self.stop:
            raise _StopForward
        return (outs[0] if len(outs) == 1 else torch.cat(outs)).reshape(shape)

    def remove(self):
        if self.original is None:
            del self.experts.forward
        else:
            self.experts.forward = self.original


def _residual_split(layer, block):
    """(norm, block) when the layer ends ``residual + block(norm(residual))`` so a pruned layer's
    output can be rebuilt from the residual alone; verified bitwise per layer before use."""
    norm = getattr(layer, "post_attention_layernorm", None)
    if (
        isinstance(norm, nn.Module)
        and not isinstance(norm, nn.ModuleList)
        and getattr(layer, "mlp", None) is block
    ):
        return norm
    return None


def _resolve_dataset(dataset, tokenizer):
    """The calibration rows to read: a DatasetDict's train (or only) split. Raises when text rows
    come without a tokenizer. Data dependent, so it runs where every rank agrees on failures."""
    if isinstance(dataset, dict):
        # load_dataset(...) without split= gives a DatasetDict, which iterates its split names.
        if "train" in dataset or len(dataset) == 1:
            dataset = dataset["train"] if "train" in dataset else next(iter(dataset.values()))
        else:
            raise ValueError(
                f"Unsloth: REAP got a dataset with the splits {sorted(dataset)}; pass one of them, "
                "e.g. `dataset['validation']`."
            )
    # A dataset name (or the default) is text; other rows are checked as they are read, since a
    # streaming dataset built from a generator declares no columns.
    if (dataset is None or isinstance(dataset, str)) and tokenizer is None:
        raise ValueError("Unsloth: REAP needs a tokenizer unless the dataset has `input_ids`.")
    return dataset


def _model_parallel_layout(model):
    """Name of a model-parallel layout (expert / tensor / pipeline / FSDP / sequence), else None."""
    distributed = getattr(getattr(model, "config", None), "distributed_config", None)

    def field(name):
        if isinstance(distributed, dict):
            return distributed.get(name)
        return getattr(distributed, name, None)

    if field("enable_expert_parallel"):
        return "expert parallel"
    for name, label in (
        ("tp_size", "tensor parallel"),
        ("pp_size", "pipeline parallel"),
        ("fsdp_size", "FSDP sharded"),
    ):
        size = field(name)
        if isinstance(size, int) and size > 1:
            return label
    if field("enable_sequence_parallel"):
        return "sequence parallel"
    tp_size = getattr(model, "_tp_size", None)
    if isinstance(tp_size, int) and tp_size > 1:
        return "tensor parallel"
    return None


def _has_unmerged_adapters(model):
    # PEFT < 0.19 leaves `peft_config` on the model after `merge_and_unload`, so look for modules whose
    # class (or a base, as for custom LoRA layers) comes from PEFT: PeftModel, tuners, adapter wrappers.
    return any(
        (cls.__module__ or "").split(".", 1)[0] == "peft"
        for module in model.modules()
        for cls in type(module).__mro__
    )


@torch.no_grad()
def reap_prune(
    model,
    tokenizer = None,
    sparsity = 0.25,
    dataset = None,
    num_calibration_samples = 512,
    max_seq_length = 2048,
    max_batch_tokens = 16384,
    report_path = None,
    device = None,
    offload_hidden_states = None,
    save_directory = None,
):
    """Prune ``sparsity`` of the routed experts in every MoE layer with REAP, in place.

    Mirrors llm-compressor's ``REAPPruningModifier`` (same saliency ``S_j = mean(g_j * ||f_j||)``,
    selection and layer-by-layer calibration), on Unsloth's fused MoE kernels.

    Args:
        model: a 16-bit MoE model (e.g. from ``FastLanguageModel.from_pretrained(..., load_in_4bit = False)``),
            before ``get_peft_model`` or after merging LoRA.
        tokenizer: tokenizer or processor, used to format and tokenize calibration text.
        sparsity: fraction of experts removed per MoE layer, in (0, 1).
        dataset: Hugging Face dataset name, ``datasets`` dataset, or list of strings / token id lists.
            Rows may carry ``input_ids``, ``text``, ``messages`` or ``conversations``.
            Defaults to llm-compressor's REAP example dataset ``mlabonne/open-perfectblend``.
        num_calibration_samples: number of calibration rows.
        max_seq_length: truncation length per row.
        max_batch_tokens: padded tokens per calibration batch.
        report_path: optional ``.json`` file for ``{layer name: [kept expert ids]}``.
        device: device layers run on; layers stored on the CPU are moved there one at a time.
        offload_hidden_states: keep calibration activations on the CPU (default: only when they
            would take more than a quarter of the free GPU memory).
        save_directory: also save the pruned model (and tokenizer) here and load later merges from
            it. Needed before ``get_peft_model`` when the LoRA model will be saved merged or to GGUF,
            since those rebuild the base weights from the checkpoint on disk.

    Returns:
        The pruned model. ``model.config`` carries the new expert count, so ``save_pretrained``,
        ``save_pretrained_merged`` and GGUF export write the smaller model.
    """
    if not 0.0 < float(sparsity) < 1.0:
        raise ValueError(f"Unsloth: REAP sparsity must be in (0, 1), got {sparsity}.")
    if _has_unmerged_adapters(model):
        raise RuntimeError(
            "Unsloth: run REAP on the base model before `get_peft_model`, or merge the LoRA "
            "adapters first (`model = model.merge_and_unload()`)."
        )
    if getattr(model, "vllm_engine", None) is not None:
        raise RuntimeError(
            "Unsloth: REAP cannot update the attached vLLM engine, so `fast_generate` would keep the "
            "unpruned experts. Load the model without `fast_inference = True` to prune it."
        )
    if model.__dict__.get("_unsloth_mtp_head") is not None:
        raise RuntimeError(
            "Unsloth: REAP does not prune the attached MTP head, so a save would write it with the "
            "old expert count. Run `reap_prune` before `get_mtp_head`."
        )
    layout = _model_parallel_layout(model)
    if layout is not None:
        # Each rank would hold a slice of the experts (or, pipelined, other layers); plain DP is fine.
        raise RuntimeError(
            f"Unsloth: REAP does not support {layout} models. Load the model on one device (or with "
            "`device_map`, or plain data parallel) to prune it."
        )
    configs = _configs(model)
    num_experts = _first_attr(configs, _NUM_EXPERTS_KEYS)
    top_k = _first_attr(configs, _TOP_K_KEYS, per_layer = True)
    if num_experts is None or top_k is None:
        raise ValueError(
            "Unsloth: REAP needs an MoE model with an expert count and top_k in its config."
        )
    n_group = _first_attr(configs, ("n_group", "num_expert_group"))
    topk_group = _first_attr(configs, ("topk_group", "top_k_group"))
    if n_group is not None and n_group > 1 and topk_group is None:
        # The router still splits experts into groups; pruning them globally would shift the
        # group boundaries. A lone topk_group (or n_group 1) routes ungrouped.
        raise ValueError(
            f"Unsloth: REAP found n_group = {n_group} but no topk_group in the config, so it cannot "
            "prune per router group."
        )
    if n_group is None or n_group <= 1 or topk_group is None:
        n_group = topk_group = None

    # Calibration runs each decoder stack alone on right padded text, which needs causal attention.
    if getattr(model, "_unsloth_slow_diffusion", False) or any(
        getattr(c, "use_bidirectional_attention", None) == "all"
        or str(getattr(c, "model_type", "")).startswith("diffusion")
        for c in configs
    ):
        raise ValueError(
            "Unsloth: REAP supports causal language models, not bidirectional or diffusion ones."
        )
    for config in configs:
        if getattr(config, "zero_expert_num", None):
            raise ValueError(
                "Unsloth: REAP does not support MoE routers with zero (identity) experts."
            )
    owner, layers = _find_layers(model, num_experts)
    if layers is None:
        raise RuntimeError(
            "Unsloth: REAP found no MoE layers with fused experts. It needs transformers 5.x MoE "
            "modules (an `experts` module with 3D expert weights next to a `gate` / `router`)."
        )
    if any(p.startswith("prev_") for p in inspect.signature(layers[0].forward).parameters):
        raise ValueError(_STATEFUL_LAYERS)
    # Non-causal text attention (openai/privacy-filter) carries no config flag.
    if any(getattr(m, "is_causal", None) is False for layer in layers for m in layer.modules()):
        raise ValueError(
            "Unsloth: REAP supports causal language models, not bidirectional or diffusion ones."
        )

    layer_names = {id(m): n for n, m in model.named_modules()}
    for layer in layers:
        moes = _layer_moes(layer, num_experts)
        if len(moes) > 1:
            raise ValueError(
                "Unsloth: REAP supports one MoE block per decoder layer; "
                f"{layer_names[id(layer)]} has {len(moes)} (e.g. separate text and vision experts)."
            )
        moe = moes[0] if moes else None
        if moe is not None:
            _check_weights(moe[2], moe[3], layer_names[id(moe[1])])
        if moe is not None and any(hasattr(m, "tid2eid") for m in moe[3].modules()):
            raise ValueError(
                "Unsloth: REAP does not support hash-routed MoE layers (fixed token to expert "
                "tables, e.g. DeepSeek-V4), since dropped experts cannot be remapped."
            )

    n_drop = int(num_experts * float(sparsity))
    if n_drop == 0:
        raise ValueError(
            f"Unsloth: REAP sparsity={sparsity} drops 0 of {num_experts} experts; raise the sparsity."
        )
    n_drop_per_group = None
    if n_group is not None:
        if num_experts % n_group:
            raise ValueError(
                f"Unsloth: {num_experts} experts are not divisible by n_group={n_group}."
            )
        n_drop_per_group = round(n_drop / n_group)
        if n_drop_per_group == 0:
            raise ValueError(
                f"Unsloth: REAP sparsity={sparsity} drops 0 experts per router group "
                f"({num_experts // n_group} experts in each of {n_group} groups); raise the sparsity."
            )
        if n_drop_per_group * n_group != n_drop:
            print(
                f"Unsloth: REAP group-limited routing drops {n_drop_per_group} experts per group, "
                f"{n_drop_per_group * n_group} in total instead of {n_drop}."
            )
        n_drop = n_drop_per_group * n_group
        if num_experts // n_group - n_drop_per_group < 2:
            # DeepSeek style routers score each group by the sum of its top 2 experts.
            raise ValueError(
                f"Unsloth: REAP sparsity={sparsity} leaves fewer than 2 experts in each of the "
                f"{n_group} router groups; lower the sparsity."
            )
        available = topk_group * (num_experts // n_group - n_drop_per_group)
    else:
        available = num_experts - n_drop
    if top_k > available:
        raise ValueError(
            f"Unsloth: REAP sparsity={sparsity} is too aggressive: the router picks top_k={top_k} "
            f"experts per token but only {available} would stay reachable. Lower the sparsity."
        )
    kept_count = num_experts - n_drop

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(device)

    pad_id = 0
    text_tokenizer = _text_tokenizer(tokenizer)
    if text_tokenizer is not None:
        pad_id = text_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = text_tokenizer.eos_token_id or 0
    error = None
    try:
        dataset = _resolve_dataset(dataset, tokenizer)
        ids = _calibration_ids(dataset, tokenizer, num_calibration_samples, max_seq_length)
        batches = _batches(ids, max(max_batch_tokens, max(len(x) for x in ids)), pad_id)
    except Exception as e:
        error = e
    # One rank's empty or malformed shard must stop every rank before the per-layer collectives.
    _ranks_agree(error, "prepare its calibration data")
    n_tokens = sum(len(x) for x in ids)

    report, saliencies = {}, {}
    pruning = None
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    start = time.time()
    try:
        hidden, kwargs_per_batch, valid_per_batch = [], [], []
        embed_device = _layer_device(owner, device)
        store = None
        capture_error = None
        try:
            for input_ids, valid in batches:
                h, kwargs = _capture_inputs(owner, layers, input_ids.to(embed_device))
                if not hidden:
                    # Decided from the first batch, so activations never pile up on the GPU first.
                    if offload_hidden_states is None and device.type == "cuda":
                        padded = sum(x.numel() for x, _ in batches) * h.shape[-1] * h.element_size()
                        # Eager attention keeps a [batch, 1, seq, seq] mask per batch in the kwargs.
                        padded += _tensor_bytes(kwargs) * len(batches)
                        offload_hidden_states = padded > torch.cuda.mem_get_info(device)[0] // 4
                    store = torch.device("cpu") if offload_hidden_states else None
                if store is not None:
                    h = h.to(store).pin_memory() if torch.cuda.is_available() else h.to(store)
                    # Rotary tables and any materialized masks go too, or they pile up per batch.
                    kwargs = _to(kwargs, store, {})
                hidden.append(h)
                kwargs_per_batch.append(kwargs)
                valid_per_batch.append(valid.reshape(-1))
        except Exception as e:
            capture_error = e
        _ranks_agree(capture_error, "capture its calibration activations")
        width = hidden[0].shape[-1]

        print(
            f"Unsloth: REAP pruning {n_drop} of {num_experts} experts per MoE layer "
            f"({len(ids)} samples, {n_tokens} tokens, {len(batches)} batches, hidden {width})."
        )
        rebuilt_layers = 0
        for index, layer in enumerate(layers):
            move_error = None
            try:
                layer_device, moved = _stream_in(layer, device)
            except Exception as e:
                layer_device, moved, move_error = device, True, e
            post_error = move_error

            try:

                def run(b, h):
                    args, kwargs = kwargs_per_batch[b][index]
                    # Gemma 4's shared_kv_states (a UserDict shared across layers) follows offloaded activations.
                    shared = [
                        v
                        for v in kwargs.values()
                        if isinstance(v, MutableMapping) and not isinstance(v, dict)
                    ]
                    # Moved every call: under a device_map the previous layer may sit on another GPU.
                    for m in shared:
                        m.update(_to(dict(m), layer_device))
                    try:
                        return _hidden(
                            layer(h, *_to(args, layer_device), **_to(kwargs, layer_device))
                        )
                    finally:
                        if store is not None:
                            for m in shared:
                                m.update(_to(dict(m), store))

                moe = _layer_moe(layer, num_experts)
                norm = None
                if moe is not None:
                    _, block, experts, router = moe
                    block_name = layer_names[id(block)]
                    stage_error = None
                    try:
                        if move_error is not None:
                            raise move_error
                        norm = _residual_split(layer, block)
                        residuals = []
                        if norm is not None:
                            hook = norm.register_forward_pre_hook(
                                lambda module, args: residuals.append(args[0])
                            )
                            try:
                                h0 = hidden[0].to(layer_device, non_blocking = True)
                                reference = run(0, h0)
                                try:
                                    same = len(residuals) == 1 and torch.equal(
                                        reference,
                                        residuals[0]
                                        + _hidden(block(norm(residuals[0]))).reshape(
                                            residuals[0].shape
                                        ),
                                    )
                                except (
                                    Exception
                                ):  # block needs more than hidden states (DeepSeek-V4)
                                    same = False
                                if not same:
                                    norm = None
                            finally:
                                residuals.clear()
                                hook.remove()

                        saliency = _Saliency(num_experts)
                        post = getattr(block, "fc2_latent_proj", None)
                        tap = _ExpertsTap(
                            experts, saliency, post if isinstance(post, nn.Linear) else None
                        )
                        # Pass 2 recomputes the layer output, so pass 1 stops right after the experts
                        # (not under accelerate hooks, whose post-forward offload must run).
                        tap.stop = not any(hasattr(m, "_hf_hook") for m in layer.modules())
                        hook = None
                        if norm is not None:
                            hook = norm.register_forward_pre_hook(
                                lambda module, args: residuals.append(args[0])
                            )
                        try:
                            for b in range(len(batches)):
                                tap.valid = valid_per_batch[b]
                                try:
                                    run(b, hidden[b].to(layer_device, non_blocking = True))
                                except _StopForward:
                                    pass
                                if norm is not None:
                                    # The residual is all a verified layer needs later, so it replaces the input.
                                    r = residuals.pop()
                                    hidden[b] = (
                                        r if store is None else r.to(store, non_blocking = True)
                                    )
                        finally:
                            tap.remove()
                            if hook is not None:
                                hook.remove()
                    except Exception as e:
                        stage_error = e
                    _ranks_agree(stage_error, f"score {block_name}")

                    saliency.all_reduce()
                    if saliency.total is None or not saliency.count.sum().item():
                        raise RuntimeError(
                            f"Unsloth: REAP saw no tokens reach {block_name}'s experts module; its MoE "
                            "forward computes the experts itself, so REAP cannot score them."
                        )
                    scores = saliency.mean()
                    saliencies[block_name] = scores.tolist()
                    retained = _retained(scores, n_drop, n_drop_per_group, n_group)
                    rebuilt_layers += norm is not None
                    assert len(retained) == kept_count
                    # Tensors change from here on: no longer the checkpoint any earlier save wrote.
                    model._unsloth_reap_saved = None
                    pruning = block_name
                    try:
                        _prune_block(block, experts, router, retained, num_experts)
                        report[block_name] = retained
                        pruning = None
                    except Exception as e:
                        post_error = e

                try:
                    for b in range(len(batches) if post_error is None else 0):
                        h = hidden[b].to(layer_device, non_blocking = True)
                        if norm is not None:
                            out = h + _hidden(block(norm(h))).reshape(h.shape)
                        else:
                            out = run(b, h)
                        hidden[b] = out if store is None else out.to(store, non_blocking = True)
                except Exception as e:
                    post_error = e
                _ranks_agree(post_error, f"run decoder layer {index + 1}")
            finally:
                if moved:
                    layer.to("cpu")
            print(
                f"Unsloth: REAP layer {index + 1}/{len(layers)} done ({time.time() - start:.1f}s)."
            )
    except BaseException:
        if report or pruning is not None:
            # Partly pruned: mark it REAP modified and unsaved so the PEFT / save guards refuse it.
            partial = dict(report)
            if pruning is not None:
                partial.setdefault(pruning, [])
            model._unsloth_reap_report = partial
            model._unsloth_reap_saved = None
            print(
                f"Unsloth: REAP stopped after pruning {len(report)} MoE layers, so the model now mixes "
                "pruned and unpruned layers. Reload it before running REAP again."
            )
        raise
    finally:
        for module, training in modes:
            module.training = training

    if not report:
        raise RuntimeError("Unsloth: REAP found no MoE layers to prune.")
    for config in configs:
        for key in _NUM_EXPERTS_KEYS:
            if getattr(config, key, None) == num_experts:
                setattr(config, key, kept_count)
        # MTP layers live outside `layers` and are not pruned; they would be rebuilt at the new count.
        for key in (
            "num_nextn_predict_layers",
            "num_mtp_layers",
            "mtp_num_layers",
            "mtp_num_hidden_layers",
        ):
            value = getattr(config, key, None)
            if isinstance(value, int) and value > 0:
                setattr(config, key, 0)
                print(
                    f"Unsloth: REAP set {key} = 0, since the MTP layers keep the old expert count."
                )
    # Wrappers cache the count too (Mixtral / Jamba pass model.num_experts to the router aux loss).
    for module in model.modules():
        for key in _NUM_EXPERTS_MODULE_KEYS:
            value = getattr(module, key, None)
            if isinstance(value, int) and value == num_experts:
                setattr(module, key, kept_count)
    model._unsloth_reap_report = report
    model._unsloth_reap_saliency = saliencies
    dist = torch.distributed
    multi_rank = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if save_directory is not None:
        save_directory = os.path.abspath(save_directory)
        source_model = getattr(model.config, "_name_or_path", None) or None
        if source_model and os.path.exists(source_model):
            # This folder may be pushed to the Hub: name a local source without its path.
            source_model = os.path.basename(os.path.normpath(source_model))
        error = None
        if not multi_rank or dist.get_rank() == 0:
            try:
                model.save_pretrained(save_directory)
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_directory)
                # Survives a reload, unlike the in-process marker: the adapter push guard reads it.
                with open(
                    os.path.join(save_directory, REAP_PROVENANCE_FILE), "w", encoding = "utf-8"
                ) as f:
                    json.dump({"source_model": source_model, "kept_experts": report}, f, indent = 1)
            except Exception as e:
                error = e
        _ranks_agree(error, f"save the pruned model to {save_directory}")
        # Merged / GGUF saves read config._name_or_path; PEFT's adapter config reads model.name_or_path.
        for config in configs:
            config._name_or_path = save_directory
        for module in model.modules():
            if "name_or_path" in module.__dict__:
                module.name_or_path = save_directory
        if getattr(model, "_unsloth_modelscope_snapshot", None):
            model._unsloth_modelscope_snapshot = save_directory
    model._unsloth_reap_saved = save_directory
    if report_path is not None:
        error = None
        if not multi_rank or dist.get_rank() == 0:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok = True)
                with open(report_path, "w", encoding = "utf-8") as f:
                    json.dump(report, f)
            except Exception as e:
                error = e
        _ranks_agree(error, f"write the REAP report to {report_path}")
    print(
        f"Unsloth: REAP kept {kept_count} of {num_experts} experts in {len(report)} MoE layers "
        f"in {time.time() - start:.1f}s ({rebuilt_layers} reused their attention output)."
    )
    if save_directory is None:
        print(
            "Unsloth: to LoRA finetune the pruned model, pass `save_directory` (or save the pruned "
            "model and load it again) before `get_peft_model`."
        )
    return model


def raise_if_unsaved_reap(model):
    """LoRA adapters and merged saves record the base checkpoint on disk, which still holds every
    expert after an in-memory prune."""
    if getattr(model, "_unsloth_reap_report", None) and not getattr(
        model, "_unsloth_reap_saved", None
    ):
        raise RuntimeError(
            "Unsloth: this model was REAP pruned in memory only, but LoRA adapters and merged saves "
            f"point at the base checkpoint `{model.config._name_or_path}`, which still has every "
            'expert. Call `reap_prune(..., save_directory = "pruned_model")` (or save the pruned '
            "model and load it again) before `get_peft_model`."
        )


def _is_token_rows(dataset):
    if isinstance(dataset, (list, tuple)) and dataset:
        first = dataset[0]
        return isinstance(first, (list, tuple)) or (
            isinstance(first, dict) and "input_ids" in first
        )
    column_names = getattr(dataset, "column_names", None)
    return bool(column_names) and "input_ids" in column_names
