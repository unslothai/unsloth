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

"""SenseNova-Loop (``sensenova/SenseNova-Loop-9B-Preview``, remote code over ``qwen3_5``).

The checkpoint ships its own 395-line modeling file (auto_map, class
``SenseNovaLoopForConditionalGeneration``), so the compiler self-gates on remote code and the
model keeps its own forward: layers 4-27 execute twice (56 executions from 32 stored layers)
with one cache slot per visit, and the fp32 MHC routers (``mhc_attn``, ``mhc_ffn``) ride
``_keep_in_fp32_modules`` on the remote classes. Nothing is registered with the Auto
factories; detection is config based, and the loader chains call
:func:`register_sensenova_loop` after the plain remote-code load.

Three load-time gaps are fixed here:
(a) the loop forward only accepts a DynamicCache (it rebuilds empty foreign caches itself),
    so generation must never be handed a static one;
(b) the tokenizer ships Mistral's broken pre-tokenizer regex, needing transformers'
    documented ``fix_mistral_regex = True``;
(c) bitsandbytes cannot quantize ``nn.Embedding``, so the 1.89 GiB bf16 table
    (248,320 x 4096) does not fit an 8 GB card beside 4-bit weights: it is re-stored as
    int8 plus a bf16 per-row scale and dequantized on gather.
"""

import os

import torch
from torch import nn

__all__ = [
    "SENSENOVA_LOOP_MODEL_TYPE",
    "is_sensenova_loop_config_dict",
    "register_sensenova_loop",
    "SenseNovaLoopInt8Embedding",
]

SENSENOVA_LOOP_MODEL_TYPE = "sensenova_loop"
_SENSENOVA_LOOP_ARCHITECTURES = ("SenseNovaLoopForConditionalGeneration",)

# 8192 rows x 4096 in fp32 is a 128 MiB transient; the whole table is never fp32 at once.
_INT8_ROWS_PER_CHUNK = 8192


def is_sensenova_loop_config_dict(config_dict) -> bool:
    """Remote-code checkpoint, so detection is config based: ``model_type``, else architectures."""
    if not isinstance(config_dict, dict):
        return False
    model_type = config_dict.get("model_type")
    if model_type == SENSENOVA_LOOP_MODEL_TYPE:
        return True
    if model_type:
        return False
    architectures = config_dict.get("architectures") or []
    return any(arch in _SENSENOVA_LOOP_ARCHITECTURES for arch in architectures)


class SenseNovaLoopInt8Embedding(nn.Module):
    """``nn.Embedding`` replacement: int8 rows plus a bf16 per-row scale, dequantized on gather.

    (c): the checkpoint's ``model.language_model.embed_tokens.weight`` stays bf16 under
    bitsandbytes (1.89 GiB for 248,320 x 4096), which does not fit an 8 GB card beside the
    4-bit weights. int8 storage drops it to about 0.95 GiB plus a 0.5 MiB scale vector; a
    gather dequantizes only the rows read (one 4096-wide vector per token), negligible
    compute. Saved checkpoints write the original bf16 ``weight`` key back, so reloads work
    anywhere.
    """

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        dtype = None,
        padding_idx = None,
        device = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        dtype = dtype or torch.bfloat16
        self.int8_weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype = torch.int8, device = device),
            requires_grad = False,
        )
        self.scale = nn.Parameter(
            torch.empty(num_embeddings, 1, dtype = dtype, device = device),
            requires_grad = False,
        )

    def forward(self, input_ids):
        ids = input_ids.reshape(-1)
        rows = self.int8_weight.index_select(0, ids).to(self.scale.dtype)
        out = rows * self.scale.index_select(0, ids)
        return out.reshape(*input_ids.shape, self.embedding_dim)


def _quantize_rows(weight, dtype):
    """Per-row absmax int8 quantization; chunked so fp32 is never the size of the table."""
    rows, hidden = weight.shape
    quant = torch.empty(rows, hidden, dtype = torch.int8)
    scale = torch.empty(rows, 1, dtype = dtype)
    for start in range(0, rows, _INT8_ROWS_PER_CHUNK):
        stop = min(start + _INT8_ROWS_PER_CHUNK, rows)
        chunk = weight[start:stop].to(torch.float32)
        row_scale = (chunk.abs().amax(dim = 1, keepdim = True) / 127.0).clamp(min = 1e-9)
        quant[start:stop] = (chunk / row_scale).round_().clamp_(-128, 127).to(torch.int8)
        scale[start:stop] = row_scale.to(dtype)
    return quant, scale


def _dequantized_state_dict(state_dict):
    """Swap ``embed_tokens.int8_weight`` + ``.scale`` back to one float ``weight`` (checkpoint key)."""
    int8_keys = [key for key in state_dict if key.endswith(".int8_weight")]
    if not int8_keys:
        return state_dict
    out = dict(state_dict)
    for key in int8_keys:
        prefix = key[: -len("int8_weight")]
        scale = out.pop(prefix + "scale", None)
        weight = out.pop(key)
        if scale is not None:
            weight = weight.to(scale.dtype) * scale
        out[prefix + "weight"] = weight
    return out


def _convert_embedding_to_int8(text_model):
    """(c): re-store ``embed_tokens`` as int8. The bf16 table is staged through host RAM and
    the old module is dropped before the int8 copy lands on the GPU, so the card never holds
    both at once; the end state is all-VRAM with no host-RAM offload."""
    embedding = getattr(text_model, "embed_tokens", None)
    if isinstance(embedding, SenseNovaLoopInt8Embedding):
        return embedding
    # Plain tables only: a PEFT-wrapped embedding (modules_to_save) keeps its adapter
    # semantics, and exotic Embedding subclasses keep their own forward.
    if type(embedding) is not nn.Embedding:
        return None
    weight = embedding.weight
    if not weight.is_floating_point() or weight.dim() != 2:
        return None
    dtype, device = weight.dtype, weight.device
    host_weight = weight.detach().to("cpu", copy = True)
    rows, hidden = host_weight.shape
    quant, scale = _quantize_rows(host_weight, dtype)
    del host_weight
    module = SenseNovaLoopInt8Embedding(
        rows,
        hidden,
        dtype = dtype,
        padding_idx = embedding.padding_idx,
    )
    module.int8_weight.data = quant
    module.scale.data = scale
    text_model.embed_tokens = module
    module.to(device)
    return module


def _wrap_save_pretrained(model):
    """Round-trip the int8 table: saved checkpoints keep the original bf16 ``weight`` key."""
    if getattr(model, "_unsloth_sensenova_loop_save", False):
        return
    original = model.save_pretrained

    def save_pretrained(*args, **kwargs):
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            state_dict = model.state_dict()
        kwargs["state_dict"] = _dequantized_state_dict(state_dict)
        return original(*args, **kwargs)

    model.save_pretrained = save_pretrained
    model._unsloth_sensenova_loop_save = True


def _fix_mistral_regex(tokenizer):
    """(b): this checkpoint's tokenizer ships Mistral's broken pre-tokenizer regex, so
    transformers documents ``fix_mistral_regex = True`` for it. Unsloth's tokenizer repair
    reinstalls the repo backend, which would overwrite a fix applied during the load, so
    this runs post-load through transformers' own classmethod."""
    backend = getattr(tokenizer, "_tokenizer", None)
    patch = getattr(type(tokenizer), "_patch_mistral_regex", None)
    name = getattr(tokenizer, "name_or_path", None)
    if backend is not None and patch is not None and name is not None:
        try:
            patch(backend, name, fix_mistral_regex = True, is_local = os.path.isdir(str(name)))
        except Exception:
            pass
    # Visible intent even when detection declines (offline) or the classmethod is absent.
    setattr(tokenizer, "fix_mistral_regex", True)


def register_sensenova_loop(model, tokenizer = None):
    """Install the SenseNova-Loop load-time fixes on a loaded model (idempotent).

    Called by the FastModel chain after the weights and tokenizer are in; the remote-code
    load itself is untouched, since loader.py keeps its ``trust_remote_code`` handling and
    the compiler self-gates on remote code.
    """
    config = getattr(model, "config", None)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else None
    if not is_sensenova_loop_config_dict(config_dict):
        return model
    if getattr(model, "_unsloth_sensenova_loop", False):
        return model
    # Walk down to the text model: PeftModel/LoraModel wrappers and the Qwen3_5 composite
    # (model.language_model) both sit between the top model and the decoder that owns
    # embed_tokens. A fresh load resolves in two hops; adapter loads resolve in four.
    text_model = model
    for _ in range(5):
        if getattr(text_model, "embed_tokens", None) is not None:
            break
        candidate = getattr(text_model, "language_model", None)
        if candidate is None:
            candidate = getattr(text_model, "model", None)
        if candidate is None:
            break
        text_model = candidate
    _convert_embedding_to_int8(text_model)
    # Wrap the save path on the transformer model itself; a PeftModel saves adapters only
    # and does not accept a state_dict kwarg. LoraModel forwards attributes, so the wrapped
    # bound method still saves the full model with the dequantized table.
    save_owner = model
    if type(save_owner).__module__.startswith("peft."):
        save_owner = save_owner.get_base_model()
    _wrap_save_pretrained(save_owner)
    # (a): the loop forward rejects any cache but DynamicCache and rebuilds only EMPTY
    # foreign caches. unsloth_base_fast_generate forces "static" on transformers >= 4.56
    # (UNSLOTH_DISABLE_STATIC_GENERATION is ignored there), so _supports_static_cache =
    # False is the switch that keeps generation dynamic; the env var covers older versions.
    model._supports_static_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and getattr(
        generation_config, "cache_implementation", None
    ) in (None, "static", "hybrid", "sliding_window"):
        generation_config.cache_implementation = "dynamic"
    if tokenizer is not None:
        _fix_mistral_regex(tokenizer)
    model._unsloth_sensenova_loop = True
    return model
