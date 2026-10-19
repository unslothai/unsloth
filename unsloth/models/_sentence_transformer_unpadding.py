# SPDX-License-Identifier: AGPL-3.0-only

"""Training-only encoder unpadding for mean-pooled BERT / RoBERTa SentenceTransformers.

Embeddings and pooling stay stock; only valid tokens enter the encoder, whose output is
scattered back to the padded layout before pooling. Checkpoints are unchanged.
"""

from types import MethodType

import torch

_ATTENTION = "unsloth_sentence_varlen"
_MASK = "_unsloth_sentence_padding_mask"
_SEQUENCES = "_unsloth_sentence_sequences"
# Below this many padded slots MiniLM training was CPU-launch-bound (RTX 5090).
_MIN_AUTO_TOKENS = 8192


def _unpadding_backend():
    from ..utils import attention_dispatch as ad
    from ..utils.packing import _XFormersBidirectionalMask

    backend = ad.select_attention_backend(use_varlen = True)
    if backend == ad.FLASH_VARLEN:
        return backend
    if backend == ad.XFORMERS and _XFormersBidirectionalMask is not None:
        return backend
    return None


def _sentence_attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from ..utils.attention_dispatch import AttentionConfig, AttentionContext, run_attention

    packed = kwargs.pop(_SEQUENCES, None)
    if packed is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    seq_info, backend = packed
    heads, tokens, head_dim = query.shape[1:]
    dropout, scale = kwargs.get("dropout", 0.0), kwargs.get("scaling")
    config = AttentionConfig(
        backend = backend,
        n_kv_heads = key.shape[1],
        n_groups = heads // key.shape[1],
        flash_varlen_kwargs = {"dropout_p": dropout, "softmax_scale": scale},
        xformers_kwargs = {"p": dropout, "scale": scale},
        sdpa_kwargs = {"dropout_p": dropout, "scale": scale},
    )
    context = AttentionContext(
        bsz = 1,
        q_len = tokens,
        kv_seq_len = tokens,
        n_heads = heads,
        head_dim = head_dim,
        requires_grad = module.training,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
        is_causal = False,
    )
    return run_attention(config = config, context = context, Q = query, K = key, V = value), None


def _mean_pooling_pipeline(model):
    from sentence_transformers.models import Dense, Normalize, Pooling, Transformer

    modules = list(model.children())
    if len(modules) < 2 or type(modules[0]) is not Transformer or type(modules[1]) is not Pooling:
        return False
    pooling = modules[1]
    modes = getattr(pooling, "pooling_mode", None)
    if modes is not None:
        if modes not in ("mean", ["mean"]):
            return False
    elif pooling.get_pooling_mode_str() != "mean":
        return False
    return all(type(module) in (Dense, Normalize) for module in modules[2:])


def _encoder_forward(
    encoder,
    hidden_states,
    attention_mask = None,
    *args,
    **kwargs,
):
    mask = kwargs.pop(_MASK, None)
    original_forward = encoder._unsloth_original_forward
    if args or mask is None:
        return original_forward(hidden_states, attention_mask, *args, **kwargs)
    # FP32 hidden states are only packed under autocast, where Q/K/V come out half precision.
    if (
        torch.compiler.is_compiling()
        or not encoder.training
        or not hidden_states.is_cuda
        or hidden_states.shape[:2] != mask.shape
        or mask.numel() < encoder._unsloth_min_tokens
        or (hidden_states.dtype == torch.float32 and not torch.is_autocast_enabled())
        or kwargs.get("encoder_hidden_states") is not None
        or kwargs.get("past_key_values") is not None
        or (backend := _unpadding_backend()) is None
    ):
        return original_forward(hidden_states, attention_mask = attention_mask, **kwargs)

    keep = mask == 1
    lengths = keep.sum(dim = 1, dtype = torch.int32)
    summary = torch.cat((lengths, ((mask == 0) | keep).all().to(torch.int32).view(1))).tolist()
    row_lengths, binary = summary[:-1], summary[-1]
    batch, width, channels = hidden_states.shape
    if not binary or min(row_lengths) == 0 or sum(row_lengths) == batch * width:
        return original_forward(hidden_states, attention_mask = attention_mask, **kwargs)

    indices = keep.view(-1).nonzero().flatten()
    cu_seqlens = torch.nn.functional.pad(lengths.cumsum(0, dtype = torch.int32), (1, 0))
    seq_info = (lengths, cu_seqlens, max(row_lengths))
    packed = hidden_states.reshape(-1, channels).index_select(0, indices).unsqueeze(0)
    output = original_forward(
        packed, attention_mask = None, **kwargs, **{_SEQUENCES: (seq_info, backend)}
    )
    restored = output.last_hidden_state.new_zeros((batch * width, channels))
    output.last_hidden_state = restored.index_copy(
        0, indices, output.last_hidden_state.squeeze(0)
    ).view(batch, width, channels)
    return output


def _sentence_forward(model, input, **kwargs):
    original_forward = model._unsloth_original_forward
    if (
        torch.compiler.is_compiling()
        or not model.training
        # A routed keyword mask overrides the one in features.
        or "attention_mask" in kwargs
        or not _mean_pooling_pipeline(model)
    ):
        return original_forward(input, **kwargs)
    mask = input.get("attention_mask")
    base = model[0].auto_model
    config = base.config
    flags = ("output_hidden_states", "output_attentions")
    if (
        hasattr(base, "_orig_mod")
        or config._attn_implementation != _ATTENTION
        or not isinstance(mask, torch.Tensor)
        or mask.ndim != 2
        or mask.numel() == 0
        or input.get("modality", "text") != "text"
        or any(getattr(config, f) or kwargs.get(f) or input.get(f) for f in flags)
    ):
        return original_forward(input, **kwargs)
    result = original_forward({**input, _MASK: mask}, **kwargs)
    result.pop(_MASK, None)
    input.update(result)
    return result


def disable_sentence_transformer_unpadding(model):
    """Restore stock forwards, e.g. before torch.compile; caller-replaced forwards are kept."""
    if not getattr(model, "_unsloth_unpadding_installed", False):
        return False
    transformer = model[0]
    base = transformer.auto_model
    base = getattr(base, "_orig_mod", base)
    if hasattr(base, "get_base_model"):
        base = base.get_base_model()
    encoder = base.encoder
    if (
        getattr(model.forward, "__func__", None) is not _sentence_forward
        or getattr(encoder.forward, "__func__", None) is not _encoder_forward
    ):
        return False
    model.forward = model._unsloth_original_forward
    encoder.forward = encoder._unsloth_original_forward
    if base.config._attn_implementation == _ATTENTION:
        base.config._attn_implementation = "sdpa"
    if transformer.model_forward_params is not None:
        transformer.model_forward_params = set(transformer.model_forward_params) - {_MASK}
    del model._unsloth_original_forward, model._unsloth_unpadding_installed
    del encoder._unsloth_original_forward, encoder._unsloth_min_tokens
    return True


def enable_sentence_transformer_unpadding(model, auto = False):
    """Pack valid tokens through FlashAttention varlen / xFormers during training.

    Returns False (model untouched) unless: Transformers 5, stock BertModel / RobertaModel
    with absolute positions and SDPA, a mean-only pooling pipeline, and a bidirectional
    varlen backend. ``auto`` skips batches under _MIN_AUTO_TOKENS padded slots.
    """
    import transformers

    if getattr(model, "_unsloth_unpadding_installed", False):
        return True
    if int(transformers.__version__.split(".")[0]) < 5 or not _mean_pooling_pipeline(model):
        return False
    backend = _unpadding_backend()
    if backend is None:
        return False

    from transformers.models.bert.modeling_bert import BertModel
    from transformers.models.roberta.modeling_roberta import RobertaModel

    transformer = model[0]
    base = transformer.auto_model
    # Exact types (seen through unsloth_zoo's bf16 autocast subclass): PEFT-wrapped,
    # remote-code and compiled models keep the stock path.
    base_type = getattr(type(base), "_unsloth_autocast_base", type(base))
    if base_type not in (BertModel, RobertaModel):
        return False
    config = base.config
    head_dim = config.hidden_size // config.num_attention_heads
    if (
        config.is_decoder
        or config.add_cross_attention
        or head_dim > 256
        or (backend == "xformers" and head_dim % 8 != 0)
        or getattr(config, "position_embedding_type", "absolute") != "absolute"
        or getattr(config, "chunk_size_feed_forward", 0)
        or config._attn_implementation != "sdpa"
        or not hasattr(transformer, "model_forward_params")
    ):
        return False
    modalities = getattr(transformer, "modality_config", None)
    if modalities is not None and modalities.get("text") != {
        "method": "forward",
        "method_output_name": "last_hidden_state",
    }:
        return False

    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask

    AttentionInterface.register(_ATTENTION, _sentence_attention)
    # Without a registered mask builder the padded fallback would silently lose its padding mask.
    AttentionMaskInterface.register(_ATTENTION, sdpa_mask)
    config._attn_implementation = _ATTENTION
    # None means ST forwards every feature key unfiltered.
    if transformer.model_forward_params is not None:
        transformer.model_forward_params = set(transformer.model_forward_params) | {_MASK}
    encoder = base.encoder
    # Bound methods (not closures) so deepcopy rebinds to the copy.
    encoder._unsloth_original_forward = encoder.forward
    encoder._unsloth_min_tokens = _MIN_AUTO_TOKENS if auto else 0
    encoder.forward = MethodType(_encoder_forward, encoder)
    model._unsloth_original_forward = model.forward
    model.forward = MethodType(_sentence_forward, model)
    model._unsloth_unpadding_installed = True
    return True
