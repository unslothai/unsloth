# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build a pre-quantized transformer checkpoint for the Studio diffusion fast path.

Quantise a model's dense bf16 DiT transformer ONCE and save the quantized state dict, so
the backend can load the already-quantized weights at runtime (meta-init +
load_state_dict(assign=True)) instead of materialising the dense bf16 on the GPU. That
drops the transformer GPU load peak ~2x and the download ~2x for fp8 (measured on Z-Image:
12.9 -> 6.3 GB peak, 12 -> 6.28 GB on disk), with bit-identical output -- it is the exact
same torchao config + min_features filter the runtime path uses, applied ahead of time.

Run on one CUDA (Blackwell / Ada / Hopper) GPU. fp8 works on torch 2.9+; the FP4/MX schemes
need the newer kernels (see scripts/nvfp4_t211_probe.py).

  python scripts/build_prequant_checkpoint.py \
      --base Tongyi-MAI/Z-Image-Turbo --family z-image --scheme fp8 \
      --out outputs/quant_research/prequant_fp8/transformer_fp8.pt [--upload-repo ORG/REPO]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base", required = True, help = "diffusers base repo (carries the transformer subfolder)"
    )
    p.add_argument("--family", required = True, help = "diffusion family name/alias (e.g. z-image)")
    p.add_argument("--scheme", required = True, help = "quant scheme: int8 | fp8 | nvfp4 | mxfp8")
    p.add_argument("--out", required = True, help = "output .pt path for the checkpoint")
    p.add_argument(
        "--subfolder",
        default = "transformer",
        help = "transformer config/weights subfolder in the base repo; a dual-expert video "
        "pipeline's second DiT builds from transformer_2",
    )
    p.add_argument("--min-features", type = int, default = 512)
    p.add_argument("--dtype", default = "bfloat16", choices = ["bfloat16"])
    p.add_argument("--hf-token", default = None)
    p.add_argument(
        "--ltx23-single-file",
        default = None,
        help = "repo_id:filename of an LTX-2.3 single-file checkpoint; the transformer is "
        "assembled via the runtime's load_ltx23_transformer (2.3 key renames + config "
        "overrides on the --base LTX-2 config) instead of from_pretrained, and "
        "base_model_id records the single-file repo (the 2.3 weights identity).",
    )
    p.add_argument(
        "--upload-repo", default = None, help = "optional HF repo id to upload the checkpoint to"
    )
    p.add_argument("--upload-revision", default = None)
    args = p.parse_args(argv)

    sys.path.insert(0, str(BACKEND))
    import torch
    import torchao
    import diffusers

    from core.inference.diffusion_families import detect_family
    from core.inference.diffusion_prequant import PREQUANT_FORMAT, prequant_filename

    # Reuse the runtime quant factory + filter so offline == runtime (the LPIPS-0 invariant).
    from core.inference.diffusion_transformer_quant import (
        FP8_GRANULARITY,
        TQ_FP8,
        TQ_SCHEMES,
        _REQUIRE_BF16_SCHEMES,
        _make_quant_config,
        _resolve_fast_accum,
        exclude_tokens_for_scheme,
        make_filter_fn,
    )
    from torchao.quantization import quantize_

    scheme = args.scheme.strip().lower()
    if scheme not in TQ_SCHEMES:
        print(f"error: --scheme must be one of {TQ_SCHEMES} (not 'auto')", flush = True)
        return 2
    fam = detect_family(args.base, override = args.family)
    if fam is None:
        # Video families (Wan / HunyuanVideo) register in their own module; same duck-typed
        # surface (name / transformer_class), so the rest of the build is family-agnostic.
        from core.inference.video_families import detect_video_family

        fam = detect_video_family(args.base, override = args.family)
    if fam is None:
        print(f"error: unknown family '{args.family}'", flush = True)
        return 2
    transformer_cls = getattr(diffusers, fam.transformer_class)

    print(f"== build prequant ({fam.name}/{scheme}, min_feat={args.min_features}) ==", flush = True)
    t0 = time.time()
    base_model_id = args.base
    if args.ltx23_single_file:
        # LTX-2.3 ships one .safetensors carrying DiT + connectors + VAEs; the runtime
        # assembles the transformer via load_ltx23_transformer (2.3-only key renames +
        # config overrides merged into the LTX-2 base config). Reuse that EXACT path so
        # offline == runtime, and stamp the single-file repo as the weights identity.
        from huggingface_hub import hf_hub_download

        from core.inference.video_ltx2 import _split_checkpoint, load_ltx23_transformer

        sf_repo, sf_name = args.ltx23_single_file.split(":", 1)
        print(f"  loading 2.3 single file {sf_repo}:{sf_name} ...", flush = True)
        local = hf_hub_download(sf_repo, sf_name, token = args.hf_token)
        from diffusers.loaders.single_file_utils import load_single_file_checkpoint

        state = load_single_file_checkpoint(str(local))
        groups = _split_checkpoint(state)
        del state
        transformer = load_ltx23_transformer(
            groups["dit"],
            base_repo = args.base,
            torch_dtype = torch.bfloat16,
            is_gguf = False,
            hf_token = args.hf_token,
        ).to("cuda")
        del groups
        base_model_id = sf_repo
    else:
        print(
            f"  loading dense transformer from {args.base} (subfolder={args.subfolder}) ...",
            flush = True,
        )
        transformer = transformer_cls.from_pretrained(
            args.base, subfolder = args.subfolder, torch_dtype = torch.bfloat16, token = args.hf_token
        ).to("cuda")
    print(f"  quantising in place ({scheme}) ...", flush = True)
    # Mirror the runtime path EXACTLY (offline == runtime, LPIPS-0 invariant): int8 skips the
    # M=1 modulation / conditioning-embedder projections (else the checkpoint bakes them int8 and
    # crashes at the first denoise step); the scaled_mm schemes exclude only a family's
    # padded-conditioning embedder. Pass the family so the offline set matches runtime.
    exclude_name_tokens = exclude_tokens_for_scheme(scheme, fam.name)
    # fp8 / mxfp8 assert a bf16 weight, so skip any non-bf16 Linear the transformer keeps in fp32
    # (a mixed-precision DiT's _keep_in_fp32_modules), else quantize_ raises. nvfp4 handles fp32.
    # Mirrors the runtime scheme gate so the offline layer set matches.
    require_bf16 = scheme in _REQUIRE_BF16_SCHEMES
    # fp8 bakes the accumulate mode into the kernels; record the resolved choice so the loader
    # can refuse a checkpoint whose baked value contradicts an explicit runtime request.
    fast_accum = _resolve_fast_accum(None) if scheme == TQ_FP8 else None
    quantize_(
        transformer,
        _make_quant_config(scheme),
        filter_fn = make_filter_fn(
            args.min_features,
            exclude_name_tokens = exclude_name_tokens,
            require_bf16 = require_bf16,
        ),
    )

    # Move the state dict to CPU for a portable, GPU-free artifact.
    state_dict = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in transformer.state_dict().items()
    }
    metadata = {
        "base_model_id": base_model_id,
        "family": fam.name,
        "scheme": scheme,
        "min_features": args.min_features,
        # Skipped layers, the non-bf16 gate, and (fp8) the baked accumulate mode -- all let the
        # loader reject a checkpoint that would not match the runtime path.
        "exclude_name_tokens": list(exclude_name_tokens),
        "require_bf16": require_bf16,
        "fast_accum": fast_accum,
        "torch_dtype": args.dtype,
        "quant_backend": "torchao",
        "transformer_class": fam.transformer_class,
        "torch_version": torch.__version__,
        "torchao_version": getattr(torchao, "__version__", "?"),
        "diffusers_version": diffusers.__version__,
    }
    # Record the fp8 granularity so the loader can reject a stale per-tensor checkpoint
    # (the runtime now requires per-row; see FP8_GRANULARITY).
    if scheme == TQ_FP8:
        metadata["fp8_granularity"] = FP8_GRANULARITY
    if args.ltx23_single_file:
        # The 2.3 transformer config is the LTX-2 base config plus the 2.3 overrides; no
        # diffusers repo carries it as a subfolder, so bake the merged dict for a future
        # meta-init (the current loader path receives the module via transformer_override).
        metadata["transformer_config"] = dict(transformer.config)
    ckpt = {
        "format": PREQUANT_FORMAT,
        "metadata": metadata,
        "state_dict": state_dict,
    }

    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    torch.save(ckpt, out)
    size_gb = out.stat().st_size / 1e9
    print(f"  saved {out}  ({size_gb:.2f} GB) in {time.time() - t0:.0f}s", flush = True)
    print(f"  metadata: {ckpt['metadata']}", flush = True)

    if args.upload_repo:
        from huggingface_hub import HfApi

        dest = prequant_filename(scheme)
        print(f"  uploading -> {args.upload_repo}:{dest} ...", flush = True)
        api = HfApi(token = args.hf_token)
        api.create_repo(args.upload_repo, exist_ok = True)
        api.upload_file(
            path_or_fileobj = str(out),
            path_in_repo = dest,
            repo_id = args.upload_repo,
            revision = args.upload_revision,
        )
        print(f"  uploaded {dest} to {args.upload_repo}", flush = True)

    print("BUILD-PREQUANT-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
