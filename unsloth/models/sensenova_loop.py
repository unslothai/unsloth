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
    (248,320 x 4096) makes the stock checkpoint's 4-bit load (about 7.36 GiB of weights)
    fail the load-time fit check on an 8 GB card before any post-load step can run.
    :func:`convert_sensenova_loop_checkpoint` rewrites the table's shard to int8 plus a bf16
    per-row scale (about 6.42 GiB of weights after conversion), and the patched build loads
    it as int8 end to end; a post-load conversion remains as the fallback for stock
    checkpoints.
"""

import functools
import json
import os
import shutil
import sys

import torch
from torch import nn

__all__ = [
    "SENSENOVA_LOOP_MODEL_TYPE",
    "is_sensenova_loop_config_dict",
    "register_sensenova_loop",
    "convert_sensenova_loop_checkpoint",
    "prepare_sensenova_loop_int8_load",
    "verify_sensenova_loop_int8_build",
    "SenseNovaLoopInt8Embedding",
]

SENSENOVA_LOOP_MODEL_TYPE = "sensenova_loop"
_SENSENOVA_LOOP_ARCHITECTURES = ("SenseNovaLoopForConditionalGeneration",)
_EMBED_KEY = "model.language_model.embed_tokens.weight"
_SCALE_KEY = "model.language_model.embed_tokens.weight_scale"

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
    """``nn.Embedding`` replacement storing the table as int8 plus a bf16 per-row scale.

    (c): the parameters are named ``weight`` (int8) and ``weight_scale`` (bf16), matching the
    converted checkpoint keys, so the int8 shard loads straight into the module with no key
    renames and the table is never materialized as bf16. A gather dequantizes only the rows
    read (one 4096-wide vector per token), negligible compute. Saved checkpoints write one
    bf16 ``weight`` back via ``register_sensenova_loop``'s save wrapper, so vanilla
    transformers reloads them.
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
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype = torch.int8, device = device),
            requires_grad = False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(num_embeddings, 1, dtype = dtype, device = device),
            requires_grad = False,
        )

    def forward(self, input_ids):
        ids = input_ids.reshape(-1)
        rows = self.weight.index_select(0, ids).to(self.weight_scale.dtype)
        out = rows * self.weight_scale.index_select(0, ids)
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
    """Swap int8 ``embed_tokens.weight`` + ``weight_scale`` back to one bf16 ``weight``."""
    int8_keys = [
        key
        for key in state_dict
        if key.endswith(".embed_tokens.weight")
        and getattr(state_dict[key], "dtype", None) == torch.int8
    ]
    if not int8_keys:
        return state_dict
    out = dict(state_dict)
    for key in int8_keys:
        scale = out.pop(key[: -len("weight")] + "weight_scale", None)
        weight = out.pop(key)
        if scale is not None:
            out[key] = weight.to(scale.dtype) * scale
        # Without its scale the int8 table is meaningless; leave it out.
    return out


def convert_sensenova_loop_checkpoint(checkpoint, output_dir = None, token = None, revision = None):
    """(c), checkpoint side: store the token table as int8 plus a per-row bf16 scale.

    Rewrites ONLY the shard holding ``model.language_model.embed_tokens.weight`` (the
    248,320 x 4096 bf16 table, about 2.03 GB): the same key is stored as int8 plus a new
    bf16 ``model.language_model.embed_tokens.weight_scale``, and the safetensors index gains
    the scale entry with the shard total adjusted. Every other file is untouched, so the disk
    cost is one shard write (briefly both versions on disk). In place by default;
    ``output_dir`` writes a full copy with only that shard and the index changed. Idempotent:
    a shard already carrying ``weight_scale`` is left alone.

    After conversion a plain 4-bit load carries about 6.42 GiB of weights (the int8 table in
    place of the bf16 one) against about 6.76 GiB usable on an 8 GB card, so it passes the
    load-time fit check with no offload_layers and no host-RAM dependence: the check never
    sees the bf16 table, which no post-load step could have removed anyway.
    """
    if not os.path.isdir(checkpoint):
        from huggingface_hub import snapshot_download

        checkpoint = snapshot_download(
            checkpoint,
            revision = revision,
            token = token,
            allow_patterns = ["*.safetensors", "*.json"],
        )
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    sharded = os.path.isfile(index_path)
    if sharded:
        with open(index_path, encoding = "utf-8") as file:
            index = json.load(file)
        weight_map = index["weight_map"]
    else:
        if not os.path.isfile(os.path.join(checkpoint, "model.safetensors")):
            raise RuntimeError(
                f"Unsloth: {checkpoint} has no safetensors weights; only safetensors "
                "checkpoints can be converted."
            )
        weight_map = {_EMBED_KEY: "model.safetensors"}
    if _SCALE_KEY in weight_map:
        return output_dir or checkpoint  # already converted
    shard_name = weight_map.get(_EMBED_KEY)
    if shard_name is None:
        raise RuntimeError(
            f"Unsloth: {_EMBED_KEY} not found in {checkpoint}; not a SenseNova-Loop checkpoint?"
        )
    shard_path = os.path.join(checkpoint, shard_name)
    target_dir = output_dir or checkpoint
    os.makedirs(target_dir, exist_ok = True)

    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(shard_path, framework = "pt", device = "cpu") as file:
        metadata = file.metadata() or {}
        tensors = {key: file.get_tensor(key) for key in file.keys()}
    weight = tensors.pop(_EMBED_KEY)
    if not weight.is_floating_point() or weight.dim() != 2:
        raise RuntimeError(f"Unsloth: {_EMBED_KEY} is not a float [vocab, hidden] table.")
    old_bytes = weight.numel() * weight.element_size()
    quant, scale = _quantize_rows(weight, weight.dtype)
    del weight
    tensors[_EMBED_KEY] = quant
    tensors[_SCALE_KEY] = scale

    new_shard = os.path.join(target_dir, shard_name)
    save_file(tensors, new_shard + ".tmp", metadata = metadata)
    os.replace(new_shard + ".tmp", new_shard)
    if output_dir is not None:
        for name in os.listdir(checkpoint):
            source = os.path.join(checkpoint, name)
            if name != shard_name and os.path.isfile(source):
                shutil.copy2(source, os.path.join(target_dir, name))
    if sharded:
        index["weight_map"][_SCALE_KEY] = shard_name
        index["total_size"] = (
            int(index.get("total_size", 0))
            - old_bytes
            + quant.numel() * quant.element_size()
            + scale.numel() * scale.element_size()
        )
        with open(
            os.path.join(target_dir, "model.safetensors.index.json"), "w", encoding = "utf-8"
        ) as file:
            json.dump(index, file, indent = 2)
    return target_dir


def _config_scale_dtype(config):
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }.get(str(dtype or "").lower(), torch.bfloat16)


def _patch_remote_build_for_int8(config):
    """Wrap the remote ``_configure_text_model`` (once) so the build swaps the table in.

    Both remote classes build their text model through the module-level
    ``_configure_text_model``; the wrapper replaces ``embed_tokens`` with
    SenseNovaLoopInt8Embedding right after it runs, before any weight is loaded. The gate is
    the config flag set by :func:`prepare_sensenova_loop_int8_load`, so a stock checkpoint
    in the same process still builds the plain ``nn.Embedding``.
    """
    module = sys.modules.get(getattr(type(config), "__module__", ""))
    original = getattr(module, "_configure_text_model", None)
    if original is None or getattr(original, "_unsloth_sensenova_int8", False):
        return

    @functools.wraps(original)
    def _configure_text_model(text, build_config):
        original(text, build_config)
        if not getattr(build_config, "_unsloth_sensenova_int8_embedding", False):
            return
        text_config = getattr(text, "config", None)
        vocab = getattr(text_config, "vocab_size", None)
        hidden = getattr(text_config, "hidden_size", None)
        if vocab and hidden:
            text.embed_tokens = SenseNovaLoopInt8Embedding(
                vocab,
                hidden,
                dtype = getattr(build_config, "_unsloth_sensenova_int8_scale_dtype", None)
                or torch.bfloat16,
                padding_idx = getattr(text_config, "pad_token_id", None),
            )

    _configure_text_model._unsloth_sensenova_int8 = True
    module._configure_text_model = _configure_text_model


def _checkpoint_has_int8_embedding(model_name, token = None, revision = None):
    """Header-only check for the converted token table (the ``weight_scale`` key)."""
    try:
        if os.path.isdir(model_name):
            index_path = os.path.join(model_name, "model.safetensors.index.json")
            if os.path.isfile(index_path):
                with open(index_path, encoding = "utf-8") as file:
                    return _SCALE_KEY in json.load(file).get("weight_map", {})
            single = os.path.join(model_name, "model.safetensors")
            if os.path.isfile(single):
                from huggingface_hub import parse_safetensors_file_metadata

                return _SCALE_KEY in parse_safetensors_file_metadata(single).tensors
            return False
        from huggingface_hub import get_safetensors_metadata

        metadata = get_safetensors_metadata(model_name, revision = revision, token = token)
        return _SCALE_KEY in metadata.weight_map
    except Exception:
        return False


def prepare_sensenova_loop_int8_load(model_name, config, token = None, revision = None):
    """Set up the in-load int8 table when the checkpoint was converted (loader.py calls this).

    The remote build is patched to construct SenseNovaLoopInt8Embedding, whose parameter
    names match the converted checkpoint keys, so the table loads as int8 end to end and no
    bf16 copy is ever materialized. Returns True when the checkpoint is converted, False for
    a stock checkpoint, which keeps the post-load conversion fallback.
    """
    if not _checkpoint_has_int8_embedding(model_name, token = token, revision = revision):
        return False
    config._unsloth_sensenova_int8_embedding = True
    config._unsloth_sensenova_int8_scale_dtype = _config_scale_dtype(config)
    _patch_remote_build_for_int8(config)
    return True


def verify_sensenova_loop_int8_build(checkpoint, token = None, revision = None):
    """CPU-only wiring check (no weights read, no GPU touched): run the converted-checkpoint
    prepare, build the model on the meta device with transformers' own init contexts, and
    return the ``embed_tokens`` class name.

    Raises when the checkpoint is not converted, the remote class cannot be resolved, or the
    build patch failed to engage (a plain nn.Embedding would load the int8 table as garbage).
    """
    import transformers.initialization as _init
    from contextlib import ExitStack

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        checkpoint, token = token, revision = revision, trust_remote_code = True
    )
    if not prepare_sensenova_loop_int8_load(
        checkpoint, config, token = token, revision = revision
    ):
        raise RuntimeError(
            f"Unsloth: {checkpoint} does not carry the int8 token table ({_SCALE_KEY} "
            "missing). Run convert_sensenova_loop_checkpoint first."
        )
    module = sys.modules.get(getattr(type(config), "__module__", ""))
    auto_map = getattr(config, "auto_map", {}) or {}
    remote_class = (auto_map.get("AutoModelForCausalLM") or "").split(".")[-1]
    model_class = getattr(module, remote_class, None)
    if model_class is None:
        raise RuntimeError(f"Unsloth: could not resolve the remote {remote_class} class.")
    # transformers 5 builds under these contexts in from_pretrained (get_init_context).
    with ExitStack() as stack:
        stack.enter_context(_init.no_tie_weights())
        stack.enter_context(_init.meta_device_safe_creation_ops())
        stack.enter_context(torch.device("meta"))
        model = model_class(config)
    text_model = getattr(getattr(model, "model", model), "language_model", None)
    if text_model is None:
        text_model = getattr(model, "language_model", None)
    embedding = getattr(text_model, "embed_tokens", None)
    if not isinstance(embedding, SenseNovaLoopInt8Embedding):
        raise RuntimeError(
            f"Unsloth: the build produced {type(embedding).__name__} instead of "
            f"{SenseNovaLoopInt8Embedding.__name__}; the int8 build patch did not engage."
        )
    return type(embedding).__name__


def _convert_embedding_to_int8(text_model):
    """(c) fallback: re-store a loaded bf16 ``embed_tokens`` as int8. The bf16 table is
    staged through host RAM and the old module is dropped before the int8 copy lands on the
    GPU, so the card never holds both at once; the end state is all-VRAM with no host-RAM
    offload. Used only when the checkpoint itself was not converted."""
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
    module.weight.data = quant
    module.weight_scale.data = scale
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


def register_sensenova_loop(model, tokenizer = None, require_int8 = False):
    """Install the SenseNova-Loop load-time fixes on a loaded model (idempotent).

    Called by the FastModel chain after the weights and tokenizer are in; the remote-code
    load itself is untouched, since loader.py keeps its ``trust_remote_code`` handling and
    the compiler self-gates on remote code. ``require_int8`` is True when the loader
    detected a converted checkpoint: the build patch must have reached the model, since a
    plain ``nn.Embedding`` would hold the int8 table cast to bf16 (garbage), and that must
    fail loudly instead of training or generating silently.
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
    embedding = getattr(text_model, "embed_tokens", None)
    if require_int8 and not isinstance(embedding, SenseNovaLoopInt8Embedding):
        raise RuntimeError(
            "Unsloth: this SenseNova-Loop checkpoint stores the token table as int8, but "
            "the model was built with a plain nn.Embedding, which would hold the int8 "
            "values cast to bf16 (garbage). The int8 build patch did not reach the model; "
            "please report this with your transformers version."
        )
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
