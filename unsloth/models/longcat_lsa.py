# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""LongCat-Flash-Lite-Sparse (``LongcatCausalLM``) on transformers' own ``longcat_flash``.

The checkpoint ships no ``model_type``, no ``auto_map`` and no modeling code; it is served
only by SGLang. Its network is LongCat-Flash-Lite (transformers' native ``longcat_flash``
decoder plus the n-gram embedding of the Lite remote code) with LongCat Sparse Attention
(LSA) on top: one DSA-style lightning indexer per decoder layer picks ``index_topk`` keys
for each query, and the second attention of the layer reuses that choice.

When a sequence is no longer than ``index_topk`` (2048) the top-k selects every causal
position (SGLang's indexer skips its logits for exactly that reason), so the model is the
dense network and the indexer weights take no part in the forward. That is what this module
builds: the dense decoder, the n-gram embedding, and the indexer weights kept as frozen
parameters so they load and save with the checkpoint. Longer sequences train with dense
attention, which differs from the sparse attention used at inference, and a one-time
warning says so.

Deltas against the Lite checkpoint, all translated here:
- n-gram weights are ``model.oe_embed_tokens{i}`` / ``model.oe_embed_proj{i}`` instead of
  ``model.ngram_embeddings.embedders.{i}`` / ``post_projs.{i}`` (the same rename SGLang does),
- the n-gram fields are ``oe_vocab_size_ratio`` / ``oe_neighbor_num`` / ``oe_split_num``,
- ``rope_scaling.rope_type`` is ``deepseek_yarn``: SGLang rewrites every MLA rope to
  ``deepseek_yarn`` (Lite's ``yarn`` included), and it computes the same inverse
  frequencies and cos/sin scale as transformers' ``yarn`` with ``mscale`` and
  ``mscale_all_dim`` set; the attention softmax ``mscale**2`` factor is applied by
  ``longcat_flash`` for any non-default rope type,
- SGLang builds the MLA ``q_a_layernorm`` / ``kv_a_layernorm`` with ``rms_norm_eps``
  where ``longcat_flash`` uses 1e-6, and runs the F32 router in F32.

The multi-token-prediction head (``model.mtp.*``) is for speculative decoding only and is
not loaded. Identity (zero) experts follow transformers, vLLM and Meituan's HF code, which
scale them by ``routed_scaling_factor``; SGLang's kernel leaves them unscaled.
"""

import sys
import warnings

import torch
from torch import nn

__all__ = [
    "LONGCAT_LSA_MODEL_TYPE",
    "is_longcat_lsa_config_dict",
    "register_longcat_lsa",
]

LONGCAT_LSA_MODEL_TYPE = "longcat_flash_lsa"
_LONGCAT_LSA_ARCHITECTURES = ("LongcatCausalLM",)

# Checkpoint key prefix -> module key prefix (``oe_embed_tokens3`` -> ``embedders.3``). Plain
# prefixes with no groups or escapes, so the pair also runs backwards when saving.
_OE_RENAMES = (
    ("model.oe_embed_tokens", "model.ngram_embeddings.embedders."),
    ("model.oe_embed_proj", "model.ngram_embeddings.post_projs."),
)


def is_longcat_lsa_config_dict(config_dict) -> bool:
    """A LongCat-Flash-Lite-Sparse style config: saved by this module, or the published
    checkpoint's config.json, which names ``LongcatCausalLM`` and carries no model_type."""
    if not isinstance(config_dict, dict):
        return False
    model_type = config_dict.get("model_type")
    if model_type == LONGCAT_LSA_MODEL_TYPE:
        return True
    if model_type:
        return False
    architectures = config_dict.get("architectures") or []
    if not any(arch in _LONGCAT_LSA_ARCHITECTURES for arch in architectures):
        return False
    # The n-gram embedding is part of the network; a config without it is not this model.
    return config_dict.get("oe_vocab_size_ratio") is not None or (
        config_dict.get("ngram_vocab_size_ratio") is not None
    )


def _translate_rope_scaling(rope_scaling):
    if not isinstance(rope_scaling, dict):
        return rope_scaling
    rope_scaling = dict(rope_scaling)
    rope_type = rope_scaling.pop("type", None) or rope_scaling.get("rope_type")
    if rope_type == "deepseek_yarn":
        rope_type = "yarn"
    if rope_type is not None:
        rope_scaling["rope_type"] = rope_type
    return rope_scaling


_CLASSES = None


_NGRAM_TRACKING_CACHES = {}


def _ngram_tracking_cache_class(cls):
    """``cls`` with its beam reorder, batch select / repeat and crop also applied to the token
    history the n-gram embedding reads, so the history follows the key/value layers."""
    tracked = _NGRAM_TRACKING_CACHES.get(cls)
    if tracked is not None:
        return tracked

    class NgramTrackingCache(cls):
        def reorder_cache(self, beam_idx, *args, **kwargs):
            out = super().reorder_cache(beam_idx, *args, **kwargs)
            history = self._unsloth_ngram_history
            self._unsloth_ngram_history = history.index_select(0, beam_idx.to(history.device))
            return out

        def batch_select_indices(self, indices, *args, **kwargs):
            out = super().batch_select_indices(indices, *args, **kwargs)
            history = self._unsloth_ngram_history
            if torch.is_tensor(indices):
                indices = indices.to(history.device)
            self._unsloth_ngram_history = history[indices]
            return out

        def batch_repeat_interleave(self, repeats, *args, **kwargs):
            out = super().batch_repeat_interleave(repeats, *args, **kwargs)
            self._unsloth_ngram_history = self._unsloth_ngram_history.repeat_interleave(
                repeats, dim = 0
            )
            return out

        def crop(self, *args, **kwargs):
            out = super().crop(*args, **kwargs)
            self._unsloth_ngram_history = self._unsloth_ngram_history[..., : self.get_seq_length()]
            return out

    NgramTrackingCache.__name__ = cls.__name__
    NgramTrackingCache.__qualname__ = cls.__qualname__
    _NGRAM_TRACKING_CACHES[cls] = NgramTrackingCache
    return NgramTrackingCache


def _classes():
    """Build the config and model classes on first use, so importing Unsloth does not import
    transformers' longcat_flash modules."""
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES

    from transformers.models.longcat_flash.configuration_longcat_flash import (
        LongcatFlashConfig,
    )
    from transformers.models.longcat_flash.modeling_longcat_flash import (
        LongcatFlashForCausalLM,
        LongcatFlashModel,
        LongcatFlashPreTrainedModel,
    )

    class LongcatLsaConfig(LongcatFlashConfig):
        model_type = LONGCAT_LSA_MODEL_TYPE

        def __init__(
            self,
            oe_vocab_size_ratio = None,
            oe_neighbor_num = None,
            oe_split_num = None,
            index_n_heads = None,
            index_head_dim = None,
            index_topk = None,
            **kwargs,
        ):
            # The Lite remote code spells the n-gram fields differently; accept both.
            if oe_vocab_size_ratio is None:
                oe_vocab_size_ratio = kwargs.pop("ngram_vocab_size_ratio", None)
            if oe_neighbor_num is None:
                oe_neighbor_num = kwargs.pop("emb_neighbor_num", None)
            if oe_split_num is None:
                oe_split_num = kwargs.pop("emb_split_num", None)
            self.oe_vocab_size_ratio = oe_vocab_size_ratio
            self.oe_neighbor_num = oe_neighbor_num
            self.oe_split_num = oe_split_num
            self.index_n_heads = index_n_heads
            self.index_head_dim = index_head_dim
            self.index_topk = index_topk
            if "rope_scaling" in kwargs:
                kwargs["rope_scaling"] = _translate_rope_scaling(kwargs["rope_scaling"])
            if isinstance(kwargs.get("rope_parameters"), dict):
                kwargs["rope_parameters"] = _translate_rope_scaling(kwargs["rope_parameters"])
            # The rotary table is sized by head_dim, which LongCat configs omit (the class
            # default 64 only matches because qk_rope_head_dim is 64 there).
            if "head_dim" not in kwargs and "qk_rope_head_dim" in kwargs:
                kwargs["head_dim"] = kwargs["qk_rope_head_dim"]
            super().__init__(**kwargs)

    class _FrozenWeight(nn.Module):
        """A bare ``weight`` so the checkpoint key loads and saves, but no Linear for
        LoRA's ``all-linear`` or bitsandbytes to pick up."""

        def __init__(self, *shape):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(*shape), requires_grad = False)

    class LongcatLsaIndexer(nn.Module):
        """The lightning indexer's weights. At ``seq_len <= index_topk`` its top-k is every
        causal position, so the forward never reads them."""

        def __init__(self, config):
            super().__init__()
            heads, head_dim = config.index_n_heads, config.index_head_dim
            self.wq_b = _FrozenWeight(heads * head_dim, config.q_lora_rank)
            self.wk = _FrozenWeight(head_dim, config.hidden_size)
            self.k_norm = _FrozenWeight(head_dim)
            self.weights_proj = _FrozenWeight(heads, config.hidden_size)

    class LongcatNgramEmbedding(nn.Module):
        """The n-gram embedding of LongCat-Flash-Lite (``NgramEmbedding`` in the Lite remote
        code), with the per-row EOS reset vectorized. The model's ``embed_tokens`` output is
        passed in at call time; passing the module itself would let an accelerate hook on a
        split model move the whole token table to this module's card."""

        def __init__(self, config):
            super().__init__()
            self.vocab_size = config.vocab_size
            self.m = int(config.oe_vocab_size_ratio * config.vocab_size)
            self.k = int(config.oe_split_num)
            self.n = int(config.oe_neighbor_num)
            eos = config.eos_token_id
            self.eos_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos
            num_embedders = self.k * (self.n - 1)
            emb_dim = config.hidden_size // num_embedders
            self.embedders = nn.ModuleList(
                nn.Embedding(int(self.m + i * 2 + 1), emb_dim) for i in range(num_embedders)
            )
            self.post_projs = nn.ModuleList(
                nn.Linear(emb_dim, config.hidden_size, bias = False) for _ in range(num_embedders)
            )

        def _shifted(self, context, shift):
            """``context`` shifted right by ``shift`` inside each EOS-terminated segment
            (the EOS closes its own segment), zero where the shift leaves the segment."""
            length = context.shape[-1]
            positions = torch.arange(length, device = context.device)
            ends = torch.where(
                context == self.eos_token_id, positions + 1, torch.zeros_like(positions)
            )
            starts = torch.cummax(ends, dim = -1).values
            starts = torch.cat([torch.zeros_like(starts[..., :1]), starts[..., :-1]], dim = -1)
            source = positions - shift
            valid = source >= starts
            gathered = torch.gather(
                context, -1, source.clamp(min = 0).expand_as(context).contiguous()
            )
            return torch.where(valid, gathered, torch.zeros_like(gathered))

        def forward(
            self,
            word_embeds,
            input_ids,
            ngram_context = None,
        ):
            seq_len = input_ids.shape[-1]
            context = input_ids
            if ngram_context is not None:
                context = torch.cat([ngram_context[..., -(self.n - 1) :], input_ids], dim = -1)
            x = word_embeds
            context = context.long()
            shifted = {i: self._shifted(context, i - 1) for i in range(2, self.n + 1)}
            for i in range(2, self.n + 1):
                for j in range(self.k):
                    index = (i - 2) * self.k + j
                    mod = int(self.m + index * 2 + 1)
                    ngram_ids = context.clone()
                    power = 1
                    for order in range(2, i + 1):
                        power = (power * self.vocab_size) % mod
                        ngram_ids = ngram_ids + shifted[order] * power
                    ngram_ids = (ngram_ids % mod)[..., -seq_len:]
                    embedder = self.embedders[index]
                    projected = self.post_projs[index](
                        embedder(ngram_ids.to(embedder.weight.device))
                    )
                    x = x + projected.to(x.device)
            return x / (1 + self.k * (self.n - 1))

    class LongcatLsaModel(LongcatFlashModel):
        config_class = LongcatLsaConfig

        def __init__(self, config):
            super().__init__(config)
            self.ngram_embeddings = LongcatNgramEmbedding(config)
            # transformers 5's fused LongcatFlashExperts sizes gate_up_proj for routed plus zero
            # experts, but only routed experts have weights (the checkpoint stacks
            # n_routed_experts, down_proj is sized the same) and the identity experts never read
            # a row, so size it to the routed experts or the checkpoint cannot load.
            routed = config.n_routed_experts
            for layer in self.layers:
                # SGLang, the checkpoint's only serving stack, builds the MLA latent norms with
                # rms_norm_eps; longcat_flash leaves them at the RMSNorm default of 1e-6.
                for attn in getattr(layer, "self_attn", ()):
                    for norm_name in ("q_a_layernorm", "kv_a_layernorm"):
                        norm = getattr(attn, norm_name, None)
                        if norm is not None and hasattr(norm, "variance_epsilon"):
                            norm.variance_epsilon = config.rms_norm_eps
                experts = getattr(getattr(layer, "mlp", None), "experts", None)
                gate_up = getattr(experts, "gate_up_proj", None)
                if isinstance(gate_up, nn.Parameter) and gate_up.shape[0] > routed:
                    experts.gate_up_proj = nn.Parameter(
                        torch.empty(
                            routed, *gate_up.shape[1:], dtype = gate_up.dtype, device = gate_up.device
                        )
                    )
            if getattr(config, "index_topk", None) and getattr(config, "index_n_heads", None):
                for layer in self.layers:
                    layer.self_attn[0].indexer = LongcatLsaIndexer(config)
            self.post_init()

        def forward(
            self,
            input_ids = None,
            attention_mask = None,
            position_ids = None,
            past_key_values = None,
            inputs_embeds = None,
            use_cache = None,
            **kwargs,
        ):
            if inputs_embeds is None and input_ids is not None:
                topk = getattr(self.config, "index_topk", None)
                cached = 0
                if past_key_values is not None:
                    try:
                        cached = int(past_key_values.get_seq_length())
                    except Exception:
                        cached = 0
                total = cached + input_ids.shape[-1]
                if topk and total > topk and not getattr(self, "_unsloth_lsa_warned", False):
                    self._unsloth_lsa_warned = True
                    warnings.warn(
                        f"Unsloth: LongCat sparse attention selects every key only up to {topk} "
                        f"tokens. This {total} token sequence runs dense attention, "
                        "which differs from the sparse attention used at inference.",
                        stacklevel = 2,
                    )
                history = getattr(past_key_values, "_unsloth_ngram_history", None)
                keep = self.ngram_embeddings.n - 1
                context = None if history is None else history[..., -keep:]
                inputs_embeds = self.ngram_embeddings(
                    self.embed_tokens(input_ids), input_ids, context
                )
                if use_cache is None:
                    use_cache = getattr(self.config, "use_cache", False)
                if use_cache and past_key_values is None:
                    from transformers.cache_utils import DynamicCache
                    try:
                        past_key_values = DynamicCache(config = self.config)
                    except TypeError:
                        past_key_values = DynamicCache()
                if past_key_values is not None:
                    # The whole id history rides on the cache and follows its beam reorders and
                    # crops; the key/value layers alone cannot give the n-gram ids back.
                    seen = input_ids if history is None else torch.cat([history, input_ids], dim = -1)
                    past_key_values._unsloth_ngram_history = seen
                    cache_class = type(past_key_values)
                    if cache_class not in _NGRAM_TRACKING_CACHES.values():
                        past_key_values.__class__ = _ngram_tracking_cache_class(cache_class)
            return super().forward(
                input_ids = None,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache = use_cache,
                **kwargs,
            )

    class LongcatCausalLM(LongcatFlashForCausalLM):
        # The published architecture name, so a saved config keeps it.
        config_class = LongcatLsaConfig
        _keys_to_ignore_on_load_unexpected = [r"model\.mtp\..*"]
        # The router and the indexer's weights_proj ship in F32 and SGLang runs them in F32;
        # keep them there under bf16 loads (and in saved checkpoints).
        _keep_in_fp32_modules_strict = ["router", "weights_proj"]

        def __init__(self, config):
            LongcatFlashPreTrainedModel.__init__(self, config)
            self.model = LongcatLsaModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)
            self.post_init()

        # transformers 5 renames the n-gram keys both ways through the conversion registry.
        # transformers 4 only applies key_mapping to VLM class names, so pass it on load and
        # undo it on save, keeping saved checkpoints in the published key layout.
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            if not _has_conversion_registry() and kwargs.get("key_mapping") is None:
                kwargs["key_mapping"] = {"^" + src: dst for src, dst in _OE_RENAMES}
            loaded = super().from_pretrained(*args, **kwargs)
            # Loading swaps the parameters in, dropping requires_grad = False; the indexer
            # takes no part in the dense forward, so it stays frozen.
            model = loaded[0] if isinstance(loaded, tuple) else loaded
            for module in model.modules():
                if isinstance(module, LongcatLsaIndexer):
                    module.requires_grad_(False)
            return loaded

        def save_pretrained(self, *args, **kwargs):
            if not _has_conversion_registry():
                state_dict = kwargs.get("state_dict")
                if state_dict is None:
                    state_dict = self.state_dict()
                kwargs["state_dict"] = {
                    _to_checkpoint_key(key): value for key, value in state_dict.items()
                }
            return super().save_pretrained(*args, **kwargs)

    _CLASSES = (LongcatLsaConfig, LongcatCausalLM)
    return _CLASSES


def _has_conversion_registry():
    try:
        from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    except Exception:
        return False
    return True


def _to_checkpoint_key(key):
    for source, target in _OE_RENAMES:
        if key.startswith(target):
            return source + key[len(target) :]
    return key


_CONVERSIONS_REGISTERED = False


def _register_conversions():
    """transformers 5 merges the per-expert checkpoint weights into 3-D stacks through the
    conversion registry, keyed by model_type; give this model_type longcat_flash's own
    conversions plus the n-gram renames."""
    global _CONVERSIONS_REGISTERED
    if _CONVERSIONS_REGISTERED:
        return
    try:
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )
        from transformers.core_model_loading import WeightRenaming
    except Exception:
        return
    renames = [
        WeightRenaming(source_patterns = source, target_patterns = target)
        for source, target in _OE_RENAMES
    ]
    base = get_checkpoint_conversion_mapping("longcat_flash") or []
    register_checkpoint_conversion_mapping(
        LONGCAT_LSA_MODEL_TYPE, renames + list(base), overwrite = True
    )
    _CONVERSIONS_REGISTERED = True
    # peft converts per-expert LoRA adapters (trained on transformers 4) onto the stacked
    # experts by model_type through this table, and copies it when it is imported. Added
    # after the registration above, which builds the conversion cache from the table.
    for module_name in (
        "transformers.conversion_mapping",
        "peft.utils.transformers_weight_conversion",
    ):
        table = getattr(sys.modules.get(module_name), "_MODEL_TO_CONVERSION_PATTERN", None)
        if isinstance(table, dict) and "longcat_flash" in table:
            table.setdefault(LONGCAT_LSA_MODEL_TYPE, table["longcat_flash"])


def register_longcat_lsa():
    """Register the config and model classes with the Auto factories (idempotent)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config_class, model_class = _classes()
    try:
        AutoConfig.register(LONGCAT_LSA_MODEL_TYPE, config_class)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(config_class, model_class)
    except ValueError:
        pass
    _register_conversions()
    return config_class, model_class


def load_longcat_lsa_config(pretrained_model_name_or_path, *args, **kwargs):
    config_class, _ = register_longcat_lsa()
    kwargs.pop("trust_remote_code", None)
    kwargs.pop("code_revision", None)
    return config_class.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
