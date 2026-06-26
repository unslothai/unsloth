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
    p.add_argument("--min-features", type = int, default = 512)
    p.add_argument("--dtype", default = "bfloat16", choices = ["bfloat16"])
    p.add_argument("--hf-token", default = None)
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
        TQ_SCHEMES,
        _make_quant_config,
        make_filter_fn,
    )
    from torchao.quantization import quantize_

    scheme = args.scheme.strip().lower()
    if scheme not in TQ_SCHEMES:
        print(f"error: --scheme must be one of {TQ_SCHEMES} (not 'auto')", flush = True)
        return 2
    fam = detect_family(args.base, override = args.family)
    if fam is None:
        print(f"error: unknown family '{args.family}'", flush = True)
        return 2
    transformer_cls = getattr(diffusers, fam.transformer_class)

    print(f"== build prequant ({fam.name}/{scheme}, min_feat={args.min_features}) ==", flush = True)
    print(f"  loading dense transformer from {args.base} (subfolder=transformer) ...", flush = True)
    t0 = time.time()
    transformer = transformer_cls.from_pretrained(
        args.base, subfolder = "transformer", torch_dtype = torch.bfloat16, token = args.hf_token
    ).to("cuda")
    print(f"  quantising in place ({scheme}) ...", flush = True)
    quantize_(transformer, _make_quant_config(scheme), filter_fn = make_filter_fn(args.min_features))

    # Move the state dict to CPU for a portable, GPU-free artifact.
    state_dict = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in transformer.state_dict().items()
    }
    ckpt = {
        "format": PREQUANT_FORMAT,
        "metadata": {
            "base_model_id": args.base,
            "family": fam.name,
            "scheme": scheme,
            "min_features": args.min_features,
            "torch_dtype": args.dtype,
            "quant_backend": "torchao",
            "transformer_class": fam.transformer_class,
            "torch_version": torch.__version__,
            "torchao_version": getattr(torchao, "__version__", "?"),
            "diffusers_version": diffusers.__version__,
        },
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
