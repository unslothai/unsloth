# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""REAP expert pruning (Router-weighted Expert Activation Pruning, arXiv:2510.13999).

Same saliency, expert selection and schedule as llm-compressor's ``REAPPruningModifier``: each
decoder layer is calibrated on the outputs of the already pruned layers before it, its lowest
saliency experts are dropped, and the pruned layer produces the next layer's inputs. Expert
outputs come from the model's own fused experts kernel (one call with every routed (token, slot)
pair as its own row) instead of per-expert Linear modules, so no extra dependency is needed.
"""

import json
import os
import re
import time

import torch
import torch.nn as nn

__all__ = ["reap_prune"]

_NUM_EXPERTS_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
_NUM_EXPERTS_MODULE_KEYS = ("num_experts", "n_experts", "n_routed_experts", "num_local_experts")
_TOP_K_KEYS = ("num_experts_per_tok", "top_k", "moe_top_k")
_ROUTER_ATTRS = ("gate", "router")
_DEFAULT_DATASET = "mlabonne/open-perfectblend"
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


def _first_attr(configs, keys):
    for config in reversed(configs):  # text_config first
        for key in keys:
            value = getattr(config, key, None)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
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
    if not any(
        p.dim() == 3 and p.shape[0] == num_experts for p in experts.parameters(recurse = False)
    ):
        return None
    return experts, router


def _find_layers(model, num_experts):
    """(owner, layers): the ModuleList of decoder layers holding the MoE blocks and its parent."""
    best = None
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) == 0:
            continue
        n_moe = sum(
            any(_moe_parts(m, num_experts) is not None for m in layer.modules()) for layer in layers
        )
        if n_moe and (best is None or n_moe > best[0]):
            best = (n_moe, module, layers)
    if best is None:
        return None, None
    return best[1], best[2]


def _layer_moe(layer, num_experts):
    for name, module in layer.named_modules():
        parts = _moe_parts(module, num_experts)
        if parts is not None:
            return name, module, parts[0], parts[1]
    return None


def _check_weights(experts, router, layer_name):
    for owner, kind in ((experts, "experts"), (router, "router")):
        for name, p in owner.named_parameters(recurse = False):
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


def _calibration_ids(dataset, tokenizer, num_samples, max_seq_length):
    """Token id lists, truncated to ``max_seq_length`` like llm-compressor's tokenize step."""
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    chat = tokenizer if getattr(tokenizer, "chat_template", None) else text_tokenizer
    if dataset is None:
        dataset = _DEFAULT_DATASET
    if isinstance(dataset, str):
        from datasets import load_dataset

        # Streaming reads only the rows used instead of the whole dataset.
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
            sample = list(row["input_ids"])[:max_seq_length]
        else:
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


def _to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking = True)
    if isinstance(obj, tuple):
        return tuple(_to(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


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


class _Saliency:
    """Running sum of ``gate * ||expert output||`` and routed-token count per expert (float64)."""

    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.total = None
        self.count = None

    def add(self, expert_ids, weights, norms):
        if self.total is None:
            self.total = torch.zeros(self.num_experts, dtype = torch.float64, device = norms.device)
            self.count = torch.zeros(self.num_experts, dtype = torch.float64, device = norms.device)
        self.total.index_add_(0, expert_ids, weights.to(torch.float64) * norms.to(torch.float64))
        self.count.index_add_(0, expert_ids, torch.ones_like(norms, dtype = torch.float64))

    def mean(self):
        if self.total is None:
            return torch.zeros(self.num_experts, dtype = torch.float64)
        return self.total.cpu() / self.count.cpu().clamp(min = 1.0)


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
    for module, sliced in ((experts, None), (router, None), (block, "e_score_correction_bias")):
        for name, param in list(module.named_parameters(recurse = False)):
            if param.dim() and param.shape[0] == num_experts and sliced in (None, name):
                new = param.data.index_select(0, index.to(param.device)).contiguous()
                setattr(module, name, nn.Parameter(new, requires_grad = param.requires_grad))
        for name, buffer in list(module.named_buffers(recurse = False)):
            if buffer.dim() and buffer.shape[0] == num_experts and sliced in (None, name):
                module._buffers[name] = buffer.index_select(0, index.to(buffer.device)).contiguous()
        for key in _NUM_EXPERTS_MODULE_KEYS:
            if getattr(module, key, None) == num_experts:
                setattr(module, key, len(retained))
        if isinstance(getattr(module, "out_features", None), int) and module is router:
            module.out_features = len(retained)
    # Kernel configs Unsloth's MoE backend cached for the old expert count.
    if getattr(experts, "_unsloth_moe_configs", None) is not None:
        experts._unsloth_moe_configs = None


class _ExpertsTap:
    """Wraps an experts module's forward to record per (token, slot) expert output norms."""

    def __init__(self, experts, saliency):
        self.experts = experts
        self.saliency = saliency
        self.valid = None
        self.original = experts.__dict__.get("forward")
        self.forward = experts.forward
        experts.forward = self

    def __call__(self, hidden_states, top_k_index, top_k_weights, *args, **kwargs):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, shape[-1])
        top_k_index = top_k_index.reshape(hidden_states.shape[0], -1)
        top_k_weights = top_k_weights.reshape(hidden_states.shape[0], -1)
        n_tokens, top_k = top_k_index.shape
        # One row per routed (token, slot) pair with gate 1 gives each expert's raw output f_j(t).
        rows = self.forward(
            hidden_states.repeat_interleave(top_k, dim = 0),
            top_k_index.reshape(-1, 1),
            torch.ones(
                (n_tokens * top_k, 1), dtype = top_k_weights.dtype, device = top_k_weights.device
            ),
            *args,
            **kwargs,
        )
        norms = torch.linalg.vector_norm(
            rows if rows.dtype == torch.float64 else rows.float(), dim = -1
        )
        expert_ids = top_k_index.reshape(-1).long()
        weights = top_k_weights.reshape(-1)
        valid = self.valid
        if valid is not None and valid.numel() == n_tokens:
            keep = valid.to(norms.device).repeat_interleave(top_k)
            expert_ids, weights, norms = expert_ids[keep], weights[keep], norms[keep]
        self.saliency.add(expert_ids, weights, norms)
        out = (rows.view(n_tokens, top_k, -1) * top_k_weights.unsqueeze(-1).to(rows.dtype)).sum(1)
        return out.to(rows.dtype).reshape(shape)

    def remove(self):
        if self.original is None:
            del self.experts.forward
        else:
            self.experts.forward = self.original


def _residual_split(layer, block):
    """(norm, block) when the layer ends ``residual + block(norm(residual))`` so a pruned layer's
    output can be rebuilt from the residual alone; verified bitwise per layer before use."""
    norm = getattr(layer, "post_attention_layernorm", None)
    if isinstance(norm, nn.Module) and getattr(layer, "mlp", None) is block:
        return norm
    return None


def reap_prune(
    model,
    tokenizer = None,
    sparsity = 0.25,
    dataset = None,
    num_calibration_samples = 512,
    max_seq_length = 2048,
    max_batch_tokens = 16384,
    report_path = None,
    ignore = None,
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
        ignore: regex patterns (``re.search``) of MoE block names left untouched.
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
    if hasattr(model, "peft_config"):
        raise RuntimeError(
            "Unsloth: run REAP on the base model before `get_peft_model`, or merge the LoRA "
            "adapters first (`model = model.merge_and_unload()`)."
        )
    if dataset is None or isinstance(dataset, str) or not _is_token_rows(dataset):
        if tokenizer is None:
            raise ValueError("Unsloth: REAP needs a tokenizer unless the dataset has `input_ids`.")
    configs = _configs(model)
    num_experts = _first_attr(configs, _NUM_EXPERTS_KEYS)
    top_k = _first_attr(configs, _TOP_K_KEYS)
    if num_experts is None or top_k is None:
        raise ValueError(
            "Unsloth: REAP needs an MoE model with an expert count and top_k in its config."
        )
    n_group = _first_attr(configs, ("n_group",))
    topk_group = _first_attr(configs, ("topk_group", "top_k_group"))
    if (n_group is None) != (topk_group is None):
        n_group = topk_group = None

    owner, layers = _find_layers(model, num_experts)
    if layers is None:
        raise RuntimeError(
            "Unsloth: REAP found no MoE layers with fused experts. It needs transformers 5.x MoE "
            "modules (an `experts` module with 3D expert weights next to a `gate` / `router`)."
        )
    ignore = list(ignore or [])

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
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if text_tokenizer is not None:
        pad_id = text_tokenizer.pad_token_id
        if pad_id is None:
            pad_id = text_tokenizer.eos_token_id or 0
    ids = _calibration_ids(dataset, tokenizer, num_calibration_samples, max_seq_length)
    batches = _batches(ids, max(max_batch_tokens, max(len(x) for x in ids)), pad_id)
    n_tokens = sum(len(x) for x in ids)

    report, saliencies = {}, {}
    was_training = model.training
    model.eval()
    start = time.time()
    try:
        hidden, kwargs_per_batch, valid_per_batch = [], [], []
        embed_device = _layer_device(owner, device)
        for input_ids, valid in batches:
            h, kwargs = _capture_inputs(owner, layers, input_ids.to(embed_device))
            hidden.append(h)
            kwargs_per_batch.append(kwargs)
            valid_per_batch.append(valid.reshape(-1))
        width = hidden[0].shape[-1]
        if offload_hidden_states is None and device.type == "cuda":
            padded = sum(h.numel() for h in hidden) * hidden[0].element_size()
            free = torch.cuda.mem_get_info(device)[0]
            offload_hidden_states = padded > free // 4
        store = torch.device("cpu") if offload_hidden_states else None
        if store is not None:
            hidden = [
                h.to(store).pin_memory() if torch.cuda.is_available() else h.to(store)
                for h in hidden
            ]

        print(
            f"Unsloth: REAP pruning {n_drop} of {num_experts} experts per MoE layer "
            f"({len(ids)} samples, {n_tokens} tokens, {len(batches)} batches, hidden {width})."
        )
        rebuilt_layers = 0
        layer_names = {id(m): n for n, m in model.named_modules()}
        for index, layer in enumerate(layers):
            moved = False
            layer_device = _layer_device(layer, device)
            if (
                layer_device.type == "cpu"
                and device.type != "cpu"
                and not hasattr(layer, "_hf_hook")
            ):
                layer.to(device)
                moved = True
                layer_device = device

            def run(b, h):
                args, kwargs = kwargs_per_batch[b][index]
                return _hidden(layer(h, *_to(args, layer_device), **_to(kwargs, layer_device)))

            moe = _layer_moe(layer, num_experts)
            if moe is not None:
                block_name = layer_names[id(moe[1])]
                if any(re.search(pattern, block_name) for pattern in ignore):
                    moe = None
            norm = None
            if moe is not None:
                _, block, experts, router = moe
                _check_weights(experts, router, block_name)
                norm = _residual_split(layer, block)
                residuals = []
                if norm is not None:
                    hook = norm.register_forward_pre_hook(
                        lambda module, args: residuals.append(args[0])
                    )
                    try:
                        h0 = hidden[0].to(layer_device, non_blocking = True)
                        reference = run(0, h0)
                        rebuilt = residuals[0] + _hidden(block(norm(residuals[0]))).reshape(
                            residuals[0].shape
                        )
                        if not torch.equal(reference, rebuilt):
                            norm = None
                    finally:
                        residuals.clear()
                        hook.remove()

                saliency = _Saliency(num_experts)
                tap = _ExpertsTap(experts, saliency)
                hook = None
                if norm is not None:
                    hook = norm.register_forward_pre_hook(
                        lambda module, args: residuals.append(args[0])
                    )
                try:
                    for b in range(len(batches)):
                        tap.valid = valid_per_batch[b]
                        run(b, hidden[b].to(layer_device, non_blocking = True))
                        if norm is not None:
                            # The residual is all a verified layer needs later, so it replaces the input.
                            r = residuals.pop()
                            hidden[b] = r if store is None else r.to(store, non_blocking = True)
                finally:
                    tap.remove()
                    if hook is not None:
                        hook.remove()

                scores = saliency.mean()
                saliencies[block_name] = scores.tolist()
                retained = _retained(scores, n_drop, n_drop_per_group, n_group)
                rebuilt_layers += norm is not None
                assert len(retained) == kept_count
                _prune_block(block, experts, router, retained, num_experts)
                report[block_name] = retained

            for b in range(len(batches)):
                h = hidden[b].to(layer_device, non_blocking = True)
                if norm is not None:
                    out = h + _hidden(block(norm(h))).reshape(h.shape)
                else:
                    out = run(b, h)
                hidden[b] = out if store is None else out.to(store, non_blocking = True)
            if moved:
                layer.to("cpu")
            print(
                f"Unsloth: REAP layer {index + 1}/{len(layers)} done ({time.time() - start:.1f}s)."
            )
    except BaseException:
        if report:
            print(
                f"Unsloth: REAP stopped after pruning {len(report)} MoE layers, so the model now mixes "
                "pruned and unpruned layers. Reload it before running REAP again."
            )
        raise
    finally:
        if was_training:
            model.train()

    if not report:
        raise RuntimeError("Unsloth: REAP pruned no layers (all MoE layers ignored?).")
    for config in configs:
        for key in _NUM_EXPERTS_KEYS:
            if getattr(config, key, None) == num_experts:
                setattr(config, key, kept_count)
    model._unsloth_reap_report = report
    model._unsloth_reap_saliency = saliencies
    if save_directory is not None:
        save_directory = os.path.abspath(save_directory)
        model.save_pretrained(save_directory)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_directory)
        # Merged / GGUF saves of a LoRA on top read the base checkpoint from here, not the original.
        for config in configs:
            config._name_or_path = save_directory
    model._unsloth_reap_saved = save_directory
    if report_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok = True)
        with open(report_path, "w", encoding = "utf-8") as f:
            json.dump(report, f)
    print(
        f"Unsloth: REAP kept {kept_count} of {num_experts} experts in {len(report)} MoE layers "
        f"in {time.time() - start:.1f}s ({rebuilt_layers} reused their attention output)."
    )
    if save_directory is None:
        print(
            "Unsloth: to LoRA finetune the pruned model and save it merged or to GGUF, pass "
            "`save_directory` (or save the pruned model and load it again) before `get_peft_model`."
        )
    return model


def _is_token_rows(dataset):
    if isinstance(dataset, (list, tuple)) and dataset:
        first = dataset[0]
        return isinstance(first, (list, tuple)) or (
            isinstance(first, dict) and "input_ids" in first
        )
    column_names = getattr(dataset, "column_names", None)
    return bool(column_names) and "input_ids" in column_names
