# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load a whole-pipeline GGUF (stable-diffusion.cpp ``convert`` of an SDXL checkpoint) through diffusers.

Pipeline-level ``from_single_file`` treats every component as packed GGUF bytes (text encoders fail 816 vs 768), so
only the denoiser loads quantized; the rest is dequantized once and fed to the pipeline loader in place of the read.
"""

import threading
from typing import Any, Optional

_DENOISER_PREFIX = "model.diffusion_model."
_PATCH_LOCK = threading.Lock()


def whole_pipeline_gguf_resident_mib(path: Optional[str], *, dense_bytes: int = 2) -> Optional[int]:
    """Resident MiB as ``load_whole_pipeline_gguf`` lays it out: quantized denoiser linears packed, every other tensor
    dense at ``dense_bytes`` per element. None when the header cannot be read."""
    try:
        from gguf import GGMLQuantizationType, GGUFReader
        tensors = GGUFReader(str(path)).tensors
    except Exception:  # noqa: BLE001 - caller falls back to the packed-file estimate
        return None
    unquantized = {
        GGMLQuantizationType.F32,
        GGMLQuantizationType.F16,
        GGMLQuantizationType.BF16,
    }
    total = 0
    for t in tensors:
        packed = (
            t.name.startswith(_DENOISER_PREFIX)
            and len(t.shape) == 2
            and t.tensor_type not in unquantized
        )
        total += int(t.n_bytes) if packed else int(t.n_elements) * dense_bytes
    return int(total * 1.05 / (1024 * 1024)) if total else None


def split_whole_pipeline_checkpoint(checkpoint: dict, dtype: Any) -> tuple[dict, dict]:
    """(denoiser state dict still GGUF-packed, every other tensor dequantized at ``dtype``)."""
    from diffusers.quantizers.gguf.utils import GGUFParameter, dequantize_gguf_tensor

    denoiser: dict = {}
    rest: dict = {}
    for key, value in checkpoint.items():
        if key.startswith(_DENOISER_PREFIX):
            denoiser[key] = value
        elif isinstance(value, GGUFParameter):
            rest[key] = dequantize_gguf_tensor(value).to(dtype)
        else:
            rest[key] = value
    return denoiser, rest


def load_whole_pipeline_gguf(
    pipeline_cls: Any,
    denoiser_cls: Any,
    path: str,
    pipe_kwargs: dict,
    *,
    dtype: Any,
    denoiser_attr: str = "unet",
    logger: Optional[Any] = None,
) -> Any:
    """Denoiser stays GGUF-quantized, text encoders and VAE load dense at ``dtype``; ``pipe_kwargs`` as for the
    safetensors ``from_single_file``."""
    import diffusers
    import diffusers.loaders.single_file as single_file_mod
    from diffusers.loaders.single_file_utils import load_single_file_checkpoint

    checkpoint = load_single_file_checkpoint(
        path,
        local_files_only = pipe_kwargs.get("local_files_only"),
        cache_dir = pipe_kwargs.get("cache_dir"),
        token = pipe_kwargs.get("token"),
    )
    denoiser_sd, rest = split_whole_pipeline_checkpoint(checkpoint, dtype)
    del checkpoint
    if not denoiser_sd:
        raise ValueError(f"'{path}' carries no {denoiser_attr} tensors ({_DENOISER_PREFIX}*).")
    model_kwargs = {
        key: pipe_kwargs[key]
        for key in ("local_files_only", "cache_dir", "token")
        if pipe_kwargs.get(key) is not None
    }
    denoiser = denoiser_cls.from_single_file(
        denoiser_sd,
        config = pipe_kwargs.get("config"),
        subfolder = denoiser_attr,
        quantization_config = diffusers.GGUFQuantizationConfig(compute_dtype = dtype),
        torch_dtype = dtype,
        **model_kwargs,
    )
    del denoiser_sd
    original = single_file_mod.load_single_file_checkpoint

    def _read(link, *args, **kwargs):
        if str(link) == str(path):
            return rest
        return original(link, *args, **kwargs)

    # Module-global swap: serialized so concurrent single-file loads never see another load's tensors
    with _PATCH_LOCK:
        single_file_mod.load_single_file_checkpoint = _read
        try:
            pipe = pipeline_cls.from_single_file(path, **{denoiser_attr: denoiser}, **pipe_kwargs)
        finally:
            single_file_mod.load_single_file_checkpoint = original
    if logger is not None:
        logger.info(
            "diffusion.gguf_pipeline: %s loaded from a whole-pipeline GGUF (%s GGUF-quantized, %d other tensors "
            "dequantized at %s)",
            pipeline_cls.__name__,
            denoiser_attr,
            len(rest),
            dtype,
        )
    return pipe
