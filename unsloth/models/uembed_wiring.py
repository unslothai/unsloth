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

"""Single-forward dense + sparse output wiring for UEmbed (Qwen3.5) embedders.

UEmbed derives BOTH of its vectors from one causal forward pass: the dense vector is the
hidden state preceding the trailing EOS block, and the sparse (SPLADE) vector is produced
by `SpladeHead` from the very same hidden states. Running the transformer twice - once per
output - would double the cost of every training step and every encode.

`UEmbedSparseOutput` is therefore a sentence-transformers module, appended after the dense
pooling: it reads `token_embeddings` + `attention_mask` (already in the features dict, so
the transformer is not touched again) and writes `sparse_embedding` beside
`sentence_embedding`. That single dict is what `SentenceTransformerTrainer` hands to
`UEmbedUnifiedLoss`, which needs both keys.

`encode()` is wrapped so callers can ask for `output_mode = "dense" | "sparse" | "both"`.
The default, `"dense"`, delegates straight to the untouched sentence-transformers
implementation, so existing embedders keep their exact return object. The wrapper is only
installed on models that actually carry a `SpladeHead`, i.e. UEmbed checkpoints shipping
`sparse_weights.pt`.

Torch-only, so it imports without an accelerator and without importing `unsloth`.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from typing import Any

import torch
from torch import nn


# Features-dict keys of the sentence-transformers pipeline. `sparse_embedding` is the one
# this module adds; the rest are produced upstream by the Transformer / Pooling modules.
TOKEN_EMBEDDINGS_KEY = "token_embeddings"
ATTENTION_MASK_KEY = "attention_mask"
SENTENCE_EMBEDDING_KEY = "sentence_embedding"
SPARSE_EMBEDDING_KEY = "sparse_embedding"

# `encode(output_mode = ...)` values. "dense" is the default and is a pass-through.
OUTPUT_MODE_DENSE = "dense"
OUTPUT_MODE_SPARSE = "sparse"
OUTPUT_MODE_BOTH = "both"
OUTPUT_MODES = (OUTPUT_MODE_DENSE, OUTPUT_MODE_SPARSE, OUTPUT_MODE_BOTH)

_CONFIG_FILENAME = "config.json"


def _load_sibling(module_name: str, filename: str):
    """Import a sibling by package or directly, without importing all of `unsloth`."""
    try:
        return __import__(f"{__package__}.{module_name}", fromlist = [module_name])
    except (ImportError, TypeError):
        pass

    name = f"unsloth_{module_name}_direct"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _uembed_splade():
    return _load_sibling("uembed_splade", "uembed_splade.py")


def _uembed_pooling():
    return _load_sibling("uembed_pooling", "uembed_pooling.py")


class UEmbedSparseOutput(nn.Module):
    """sentence-transformers module that adds UEmbed's sparse vector to the features dict.

    It consumes the hidden states the transformer already emitted, so it costs one head
    projection, never a second forward. The wrapped `SpladeHead` is a submodule, which is
    what puts its parameters in `model.parameters()` (and therefore in the optimizer) and
    keeps them trainable next to the LoRA adapter.
    """

    config_keys = ["mode", "num_eos_tokens"]

    def __init__(
        self,
        head: nn.Module,
        mode: str | None = None,
    ) -> None:
        super().__init__()
        splade = _uembed_splade()
        # Duck-typed on purpose: `SpladeHead` can legitimately be reached through more
        # than one module object (package import vs the standalone file-path load), and an
        # identity check would then reject a perfectly valid head.
        if not isinstance(head, nn.Module) or not hasattr(head, "num_eos_tokens"):
            raise ValueError(
                f"Unsloth: UEmbedSparseOutput needs a SpladeHead, got " f"{type(head).__name__}."
            )
        if mode is None:
            # `splade.last` reads the trailing EOS block; without one, only `splade.max`
            # is defined for the checkpoint.
            mode = splade.SPLADE_LAST if head.num_eos_tokens > 0 else splade.SPLADE_MAX
        if not splade.is_splade_pooling_mode(mode):
            raise ValueError(
                f"Unsloth: unknown SPLADE pooling mode {mode!r}; expected one of "
                f"{sorted(splade.SPLADE_POOLING_MODES)}."
            )
        self.head = head
        self.mode = mode

    @property
    def num_eos_tokens(self) -> int:
        return self.head.num_eos_tokens

    def set_mode(self, mode: str) -> None:
        """Switch between `splade.last` and `splade.max` without rebuilding the head."""
        splade = _uembed_splade()
        if not splade.is_splade_pooling_mode(mode):
            raise ValueError(
                f"Unsloth: unknown SPLADE pooling mode {mode!r}; expected one of "
                f"{sorted(splade.SPLADE_POOLING_MODES)}."
            )
        self.mode = mode

    def forward(self, features: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if TOKEN_EMBEDDINGS_KEY not in features:
            raise KeyError(
                f"Unsloth: the features dict has no `{TOKEN_EMBEDDINGS_KEY}`, so the "
                f"UEmbed sparse head has no hidden states to pool. It must run after the "
                f"Transformer module of the SentenceTransformer."
            )
        token_embeddings = features[TOKEN_EMBEDDINGS_KEY]
        attention_mask = features.get(ATTENTION_MASK_KEY)
        if attention_mask is None or attention_mask.size(-1) != token_embeddings.size(1):
            # Same fallback as sentence-transformers' Pooling: no mask means no padding.
            attention_mask = torch.ones(
                token_embeddings.shape[:-1],
                device = token_embeddings.device,
                dtype = torch.long,
            )
        features[SPARSE_EMBEDDING_KEY] = self.head(token_embeddings, attention_mask, self.mode)
        return features

    def get_config_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "num_eos_tokens": self.head.num_eos_tokens}

    def save(self, output_path: str, *args: Any, **kwargs: Any) -> None:
        """Write the module config plus the sparse heads in UEmbed's sidecar layout."""
        splade = _uembed_splade()
        os.makedirs(output_path, exist_ok = True)
        with open(os.path.join(output_path, _CONFIG_FILENAME), "w", encoding = "utf-8") as file:
            json.dump(self.get_config_dict(), file, indent = 2)
        torch.save(
            {
                splade.SPARSE_LM_HEADS_KEY: [
                    weight.detach().cpu() for weight in self.head.sparse_lm_heads
                ],
                splade.SPARSE_BIAS_KEY: [bias.detach().cpu() for bias in self.head.sparse_bias],
            },
            os.path.join(output_path, splade.SPARSE_WEIGHTS_FILENAME),
        )

    @classmethod
    def load(cls, input_path: str, *args: Any, **kwargs: Any) -> UEmbedSparseOutput:
        subfolder = kwargs.get("subfolder", "")
        if subfolder:
            if os.path.isdir(input_path):
                input_path = os.path.join(input_path, subfolder)
            else:
                try:
                    from sentence_transformers.util import load_dir_path
                except ImportError as exception:
                    raise RuntimeError(
                        "Unsloth: sentence-transformers is required to resolve a remote "
                        "serialized UEmbedSparseOutput module."
                    ) from exception
                input_path = load_dir_path(
                    model_name_or_path = input_path,
                    subfolder = subfolder,
                    token = kwargs.get("token"),
                    cache_folder = kwargs.get("cache_folder"),
                    revision = kwargs.get("revision"),
                    local_files_only = kwargs.get("local_files_only", False),
                )

        splade = _uembed_splade()
        config: dict[str, Any] = {}
        config_path = os.path.join(input_path, _CONFIG_FILENAME)
        if os.path.isfile(config_path):
            with open(config_path, encoding = "utf-8") as file:
                config = json.load(file)
        head = splade.SpladeHead.from_checkpoint(
            input_path, num_eos_tokens = config.get("num_eos_tokens")
        )
        return cls(head, config.get("mode"))

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


# -- pipeline wiring --------------------------------------------------------------------
def find_uembed_sparse_output(model: Any) -> UEmbedSparseOutput | None:
    """The model's sparse output module, or None for a plain dense embedder."""
    children = getattr(model, "children", None)
    if children is None:
        return None
    for module in children():
        if isinstance(module, UEmbedSparseOutput):
            return module
    return None


def require_uembed_sparse_output(model: Any) -> UEmbedSparseOutput:
    """Same as `find_uembed_sparse_output`, but says what is wrong instead of None."""
    module = find_uembed_sparse_output(model)
    if module is None:
        raise ValueError(
            "Unsloth: this model has no UEmbed sparse head, so it cannot produce a "
            "`sparse_embedding`. Sparse output needs a checkpoint that ships "
            "`sparse_weights.pt` (a UEmbed model); load one with "
            "`FastSentenceTransformer.from_pretrained(...)`, or ask for the dense "
            "output only."
        )
    return module


def _match_dtype_and_device(module: nn.Module, model: Any) -> None:
    """Move the head onto the backbone's device (and float dtype, when it has one)."""
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return
    reference = next(iter(parameters()), None)
    if reference is None:
        return
    if reference.is_floating_point():
        # Quantized backbones expose integer parameters; casting to those would destroy
        # the head, so only a real float dtype is copied.
        module.to(device = reference.device, dtype = reference.dtype)
    else:
        module.to(device = reference.device)


def _append_module(model: Any, module: nn.Module) -> None:
    names = {name for name, _ in model.named_children()}
    index = len(names)
    while str(index) in names:
        index += 1
    model.add_module(str(index), module)


def _patch_encode(model: Any) -> None:
    """Give `encode` an `output_mode`; the default delegates to the original untouched."""
    if getattr(model, "_unsloth_uembed_original_encode", None) is not None:
        return
    original_encode = getattr(model, "encode", None)
    if original_encode is None:
        # A bare module chain (no `encode`) is still perfectly trainable: the loss reads
        # the features dict, not `encode`. Nothing to wrap, so leave it alone.
        return

    def encode(
        sentences,
        *args: Any,
        output_mode: str = OUTPUT_MODE_DENSE,
        **kwargs: Any,
    ):
        if output_mode == OUTPUT_MODE_DENSE:
            return original_encode(sentences, *args, **kwargs)
        return encode_uembed(
            model,
            sentences,
            *args,
            output_mode = output_mode,
            encode_fn = original_encode,
            **kwargs,
        )

    model.encode = encode
    model._unsloth_uembed_original_encode = original_encode


def patch_uembed_sparse_encode(model: Any) -> bool:
    """Restore UEmbed's process-local encode wrapper for a loaded sparse module chain.

    Sentence-transformers serializes modules, not instance method wrappers. This is a
    no-op for dense-only chains and is idempotent when attachment or sidecar reload runs
    more than once.
    """
    if find_uembed_sparse_output(model) is None:
        return False
    _patch_encode(model)
    return True


def restore_uembed_inference_input_format(model: Any) -> bool:
    """Restore process-local UEmbed formatting from a serialized sparse module chain.

    ``UEmbedSparseOutput`` and its positive ``num_eos_tokens`` are serialized in
    ``modules.json`` and the module config, unlike the replaced Transformer preprocess
    method. Their presence is therefore the trusted opt-in signal on the native
    sentence-transformers inference path; dense-only models are never modified.
    """
    sparse_output = find_uembed_sparse_output(model)
    if sparse_output is None or sparse_output.num_eos_tokens <= 0:
        return False

    try:
        transformer_module = model[0]
    except (KeyError, IndexError, TypeError) as exception:
        raise RuntimeError(
            "Unsloth: serialized UEmbed sparse metadata was found, but the "
            "SentenceTransformer has no first Transformer module to restore input "
            "formatting on."
        ) from exception

    pooling = _uembed_pooling()
    processor = pooling._module_processor(transformer_module)
    # Assignment replaces any serialized TemplateProcessing rather than stacking EOS
    # blocks, while the input formatter itself has an explicit idempotence marker.
    pooling.build_eos_post_processor(processor, sparse_output.num_eos_tokens)
    attached = pooling.attach_uembed_input_format(transformer_module)
    model._unsloth_uembed_instruction = True
    return attached


def attach_uembed_sparse_output(
    model: Any,
    head: nn.Module,
    mode: str | None = None,
) -> bool:
    """Append the sparse head to `model`'s module chain and teach `encode` about it.

    Args:
        model: the SentenceTransformer (its module chain must already end in the dense
            pooling, so `token_embeddings` is still in the features dict).
        head: a `SpladeHead`, or an already-built `UEmbedSparseOutput`.
        mode: `splade.last` (default when the checkpoint has an EOS block) or `splade.max`.

    Returns:
        True when the sparse output was attached, False when the model already had one
        (idempotent, so a reload cannot stack two heads).
    """
    if find_uembed_sparse_output(model) is not None:
        patch_uembed_sparse_encode(model)
        return False

    module = head if isinstance(head, UEmbedSparseOutput) else UEmbedSparseOutput(head, mode)
    _match_dtype_and_device(module, model)
    _append_module(model, module)
    patch_uembed_sparse_encode(model)
    return True


def _resolve_sparse_weights_dir(
    model_name_or_path: str,
    token: str | bool | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
) -> str | None:
    """Directory holding `sparse_weights.pt`, or None when the checkpoint has none.

    A missing file is the documented "not a sparse checkpoint" case and stays silent.
    Anything else (network / auth failure) is reported rather than silently dropping the
    sparse half of a UEmbed model.
    """
    splade = _uembed_splade()
    filename = splade.SPARSE_WEIGHTS_FILENAME

    if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path):
        local = os.path.join(model_name_or_path, filename)
        return model_name_or_path if os.path.isfile(local) else None

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
        try:
            from huggingface_hub.errors import RemoteEntryNotFoundError
        except ImportError:
            # huggingface_hub < 1 uses EntryNotFoundError for remote 404s. Its
            # LocalEntryNotFoundError subclass is excluded explicitly below.
            RemoteEntryNotFoundError = EntryNotFoundError
    except ImportError as exception:
        raise RuntimeError(
            f"Unsloth: cannot resolve optional `{filename}` for `{model_name_or_path}` "
            f"because `huggingface_hub` is unavailable. Install `huggingface_hub` or "
            f"load from a complete local checkpoint."
        ) from exception

    try:
        path = hf_hub_download(
            model_name_or_path,
            filename,
            token = token,
            cache_dir = cache_dir,
            revision = revision,
        )
    except Exception as exception:
        if isinstance(exception, RemoteEntryNotFoundError) and not isinstance(
            exception, LocalEntryNotFoundError
        ):
            return None
        raise RuntimeError(
            f"Unsloth: failed to resolve `{filename}` for `{model_name_or_path}`. Only a "
            f"confirmed missing file is treated as an optional dense-only checkpoint; "
            f"this failure may be caused by the network, an invalid repository/revision, "
            f"authentication, or gated-repository access. Check connectivity and pass a "
            f"token with access. Original error: {exception}"
        ) from exception
    return os.path.dirname(path)


def attach_uembed_sparse_checkpoint(
    model: Any,
    model_name_or_path: str,
    num_eos_tokens: int | None = None,
    token: str | bool | None = None,
    cache_dir: str | None = None,
    revision: str | None = None,
    mode: str | None = None,
) -> bool:
    """Attach the checkpoint's `sparse_weights.pt` heads, if it ships any.

    Returns False (leaving the model dense-only) when the checkpoint carries no sparse
    sidecar, so non-UEmbed embedders are unaffected.
    """
    directory = _resolve_sparse_weights_dir(
        model_name_or_path, token = token, cache_dir = cache_dir, revision = revision
    )
    if directory is None:
        if num_eos_tokens is not None and num_eos_tokens > 0:
            raise RuntimeError(
                f"Unsloth: `{model_name_or_path}` declares num_eos_tokens = "
                f"{num_eos_tokens}, so it is a UEmbed checkpoint, but its required "
                f"`sparse_weights.pt` is missing. Restore the complete checkpoint or "
                f"select a revision that contains the sparse heads; refusing to silently "
                f"return a dense-only model."
            )
        return False

    splade = _uembed_splade()
    head = splade.SpladeHead.from_checkpoint(
        directory,
        num_eos_tokens = num_eos_tokens,
        token = token,
        cache_dir = cache_dir,
        revision = revision,
    )
    return attach_uembed_sparse_output(model, head, mode)


# -- encode -----------------------------------------------------------------------------
def _stack(rows: list[Any], key: str) -> torch.Tensor:
    values = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or key not in row:
            raise KeyError(
                f"Unsloth: row {index} of the encode output has no `{key}`. The UEmbed "
                f"sparse module must stay in the model's module chain for `encode` to "
                f"return sparse vectors."
            )
        values.append(row[key])
    return torch.stack(values) if values else torch.empty(0)


def _finalize(
    embeddings: torch.Tensor, single: bool, convert_to_tensor: bool, convert_to_numpy: bool
) -> Any:
    if single and embeddings.shape[0] > 0:
        embeddings = embeddings[0]
    if convert_to_tensor:
        return embeddings
    if convert_to_numpy:
        embeddings = embeddings.detach().cpu()
        if embeddings.dtype in (torch.bfloat16, torch.float16):
            embeddings = embeddings.float()
        return embeddings.numpy()
    return embeddings


def encode_uembed(
    model: Any,
    sentences: Any,
    *args: Any,
    output_mode: str = OUTPUT_MODE_BOTH,
    encode_fn: Any = None,
    **kwargs: Any,
):
    """Encode once and return the dense vector, the sparse vector, or both.

    `"both"` and `"sparse"` run the model's own `encode` a SINGLE time asking for every
    feature it produced, which is where `sparse_embedding` rides along - there is no
    second forward pass. `"dense"` is a plain pass-through.
    """
    if output_mode not in OUTPUT_MODES:
        raise ValueError(
            f"Unsloth: unknown `output_mode` {output_mode!r}; expected one of "
            f"{list(OUTPUT_MODES)}."
        )
    if encode_fn is None:
        encode_fn = getattr(model, "_unsloth_uembed_original_encode", None) or model.encode
    if output_mode == OUTPUT_MODE_DENSE:
        return encode_fn(sentences, *args, **kwargs)

    require_uembed_sparse_output(model)
    if kwargs.get("output_value", None) is not None:
        raise ValueError(
            f"Unsloth: `output_value = {kwargs['output_value']!r}` cannot be combined with "
            f"`output_mode = {output_mode!r}`; the sparse path already reads every output "
            f"of the single forward pass."
        )
    kwargs.pop("output_value", None)
    convert_to_tensor = bool(kwargs.pop("convert_to_tensor", False))
    convert_to_numpy = bool(kwargs.pop("convert_to_numpy", True))

    rows = encode_fn(
        sentences,
        *args,
        output_value = None,
        convert_to_numpy = False,
        convert_to_tensor = False,
        **kwargs,
    )
    single = isinstance(rows, dict)
    row_list = [rows] if single else list(rows)

    sparse = _finalize(
        _stack(row_list, SPARSE_EMBEDDING_KEY), single, convert_to_tensor, convert_to_numpy
    )
    if output_mode == OUTPUT_MODE_SPARSE:
        return sparse
    dense = _finalize(
        _stack(row_list, SENTENCE_EMBEDDING_KEY), single, convert_to_tensor, convert_to_numpy
    )
    return {SENTENCE_EMBEDDING_KEY: dense, SPARSE_EMBEDDING_KEY: sparse}
