# SPDX-License-Identifier: Apache-2.0

"""Training-only encoder compaction for ordinary mean-pooled sentence models.

Embeddings (including positions) and pooling remain the model's own implementations.
Only valid tokens enter the encoder; its output is restored before pooling. Per-call
metadata travels through Transformers' attention kwargs, including checkpoint replay.
"""

from types import MethodType

import torch


_ATTENTION = "unsloth_sentence_varlen"
_MASK = "_unsloth_sentence_padding_mask"
_SEQUENCES = "_unsloth_sentence_sequences"
# Smaller MiniLM batches were CPU-launch-bound in the matched training benchmark.
_MIN_UNPADDING_TOKENS = 8192


def _sentence_attention(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from ..utils.attention_dispatch import AttentionConfig, AttentionContext, run_attention

    sequences = kwargs.pop(_SEQUENCES, None)
    if sequences is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    heads, tokens, head_dim = query.shape[1:]
    config = AttentionConfig(
        backend = "flash_varlen",
        n_kv_heads = key.shape[1],
        n_groups = heads // key.shape[1],
        flash_varlen_kwargs = {
            "causal": False,
            "dropout_p": kwargs.get("dropout", 0.0),
            "softmax_scale": kwargs.get("scaling"),
        },
        sdpa_kwargs = {
            "is_causal": False,
            "dropout_p": kwargs.get("dropout", 0.0),
            "scale": kwargs.get("scaling"),
        },
    )
    context = AttentionContext(
        bsz = 1,
        q_len = tokens,
        kv_seq_len = tokens,
        n_heads = heads,
        head_dim = head_dim,
        requires_grad = module.training,
        seq_info = sequences,
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
        if modes != "mean" and modes != ["mean"]:
            return False
    elif not getattr(pooling, "pooling_mode_mean_tokens", False) or any(
        getattr(pooling, name, False)
        for name in (
            "pooling_mode_cls_token",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken",
        )
    ):
        return False
    return all(type(module) in (Dense, Normalize) for module in modules[2:])


def _encoder_forward(
    encoder,
    hidden_states,
    attention_mask = None,
    *args,
    **kwargs,
):
    from ..utils import attention_dispatch

    original_forward = encoder._unsloth_original_forward
    mask = kwargs.pop(_MASK, None)
    if args:
        return original_forward(hidden_states, attention_mask, *args, **kwargs)
    # Keep upstream torch.compile policy: its padded graph is still useful for
    # long runs. Dynamic compaction is used by the eager training path.
    if (
        torch.compiler.is_compiling()
        or mask is None
        or not encoder.training
        # Autocast keeps embedding LayerNorm output in FP32 while Q/K/V linear
        # projections use its FP16/BF16 dtype. Keep those FP32 residuals intact.
        or (
            hidden_states.dtype not in (torch.float16, torch.bfloat16)
            and not (hidden_states.dtype == torch.float32 and torch.is_autocast_enabled())
        )
        or not hidden_states.is_cuda
        or hidden_states.shape[:2] != mask.shape
        or hidden_states.device != mask.device
        or mask.numel() < encoder._unsloth_min_tokens
        or attention_dispatch.select_attention_backend(use_varlen = True)
        != attention_dispatch.FLASH_VARLEN
        or kwargs.get("encoder_hidden_states") is not None
        or kwargs.get("past_key_values") is not None
        or kwargs.get("use_cache", False)
    ):
        return original_forward(hidden_states, attention_mask = attention_mask, **kwargs)

    # Read validity and lengths once; nonzero also synchronizes CUDA to size its
    # output. Never cache by mask identity: data loaders can mutate the tensor.
    keep = mask == 1
    lengths = keep.sum(dim = 1, dtype = torch.int32)
    summary = torch.cat((lengths, ((mask == 0) | keep).all().to(torch.int32).reshape(1))).tolist()
    batch, width, channels = hidden_states.shape
    total = sum(summary[:-1])
    if (
        not summary[-1]
        or min(summary[:-1], default = 0) == 0
        or total >= batch * width * (1 - encoder._unsloth_padding_threshold)
    ):
        return original_forward(hidden_states, attention_mask = attention_mask, **kwargs)

    indices = keep.reshape(-1).nonzero().flatten()
    cumulative = torch.nn.functional.pad(lengths.cumsum(0, dtype = torch.int32), (1, 0))
    packed = hidden_states.reshape(-1, channels).index_select(0, indices).unsqueeze(0)
    output = original_forward(
        packed,
        attention_mask = None,
        **{**kwargs, _SEQUENCES: (lengths, cumulative, max(summary[:-1]))},
    )
    restored = output.last_hidden_state.new_zeros((batch * width, channels))
    output.last_hidden_state = restored.index_copy(
        0, indices, output.last_hidden_state.squeeze(0)
    ).view(batch, width, channels)
    return output


def _sentence_forward(model, input, **kwargs):
    features = input
    if (
        torch.compiler.is_compiling()
        or not model.training
        # ST can route a keyword mask that overrides the mask in features.
        or "attention_mask" in kwargs
        or not getattr(model, "_unsloth_use_unpadding", True)
        or not _mean_pooling_pipeline(model)
    ):
        return model._unsloth_original_forward(features, **kwargs)
    mask = features.get("attention_mask")
    base = model[0].auto_model
    config = base.config
    can_pack = (
        # Guard before passing metadata: a compiled inner model can graph-break
        # back to eager execution before the encoder's own compile check.
        not hasattr(base, "_orig_mod")
        and config._attn_implementation == _ATTENTION
        and not any(layer.chunk_size_feed_forward for layer in base.encoder.layer)
        and isinstance(mask, torch.Tensor)
        and mask.ndim == 2
        and mask.numel() > 0
        and features.get("modality", "text") == "text"
        and not config.output_hidden_states
        and not config.output_attentions
        and not kwargs.get("output_hidden_states", False)
        and not kwargs.get("output_attentions", False)
        and not features.get("output_hidden_states", False)
        and not features.get("output_attentions", False)
    )
    if not can_pack:
        return model._unsloth_original_forward(features, **kwargs)
    result = model._unsloth_original_forward({**features, _MASK: mask}, **kwargs)
    result.pop(_MASK, None)
    features.update(result)
    return result


def disable_sentence_transformer_unpadding(model):
    """Restore installer-owned execution before upstream compiles the padded model."""
    if not getattr(model, "_unsloth_unpadding_installed", False):
        return False
    transformer = model[0]
    base = transformer.auto_model
    base = getattr(base, "_orig_mod", base)
    if hasattr(base, "get_base_model"):
        base = base.get_base_model()
    encoder = getattr(base, "encoder", None)
    # Do not overwrite forwards replaced by callers after installation.
    if (
        getattr(model.forward, "__func__", None) is not _sentence_forward
        or getattr(getattr(encoder, "forward", None), "__func__", None) is not _encoder_forward
    ):
        return False
    model.forward = model._unsloth_original_forward
    encoder.forward = encoder._unsloth_original_forward
    if base.config._attn_implementation == _ATTENTION:
        base.config._attn_implementation = "sdpa"
    if transformer.model_forward_params is not None:
        transformer.model_forward_params = set(transformer.model_forward_params) - {_MASK}
    del model._unsloth_original_forward
    del model._unsloth_unpadding_installed
    del model._unsloth_use_unpadding
    del encoder._unsloth_original_forward
    del encoder._unsloth_padding_threshold
    del encoder._unsloth_min_tokens
    return True


def enable_sentence_transformer_unpadding(
    model,
    padding_threshold = 0.0,
    auto = False,
):
    """Install an execution optimization without changing checkpoint topology.

    Unsupported versions, architectures, attention backends and module layouts retain
    their ordinary forward. The supported path requires Transformers 5's attention
    kwargs and FlashAttention; SDPA/xFormers-only installations keep padded execution.
    Automatic mode avoids packing small, launch-bound batches. Explicitly enabled
    packing can still save activation memory there at the expense of step time.
    """
    import transformers
    from ..utils import attention_dispatch

    if getattr(model, "_unsloth_unpadding_installed", False):
        return True
    if not 0.0 <= padding_threshold < 1.0:
        raise ValueError("padding_threshold must be in [0, 1)")
    if int(transformers.__version__.split(".")[0]) < 5 or not _mean_pooling_pipeline(model):
        return False
    if (
        attention_dispatch.select_attention_backend(use_varlen = True)
        != attention_dispatch.FLASH_VARLEN
    ):
        return False

    transformer = model[0]
    base = transformer.auto_model
    # Install before PEFT wraps the base. Prompt tuning, remote model classes and
    # already-compiled modules do not have the verified encoder contract.
    from transformers.models.bert.modeling_bert import BertModel
    from transformers.models.roberta.modeling_roberta import RobertaModel

    if type(base) not in (BertModel, RobertaModel):
        return False
    config = base.config
    if (
        config.is_decoder
        or config.add_cross_attention
        or config.hidden_size // config.num_attention_heads > 256
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
    # Padded fallback needs the SAME mask as ordinary SDPA. An unregistered mask
    # backend would silently remove padding masks before the encoder sees them.
    AttentionMaskInterface.register(_ATTENTION, sdpa_mask)
    config._attn_implementation = _ATTENTION
    # Newer ST uses None for a model accepting **kwargs (unfiltered pass-through).
    if transformer.model_forward_params is not None:
        transformer.model_forward_params = set(transformer.model_forward_params) | {_MASK}
    encoder = base.encoder
    # Bound methods remain bound to the copied module under deepcopy, unlike
    # closures capturing the original model. These attributes add no parameters.
    encoder._unsloth_original_forward = encoder.forward
    encoder._unsloth_padding_threshold = padding_threshold
    encoder._unsloth_min_tokens = _MIN_UNPADDING_TOKENS if auto else 0
    encoder.forward = MethodType(_encoder_forward, encoder)
    model._unsloth_original_forward = model.forward
    model.forward = MethodType(_sentence_forward, model)
    model._unsloth_unpadding_installed = True
    model._unsloth_use_unpadding = True
    return True
