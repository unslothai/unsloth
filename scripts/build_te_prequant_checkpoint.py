# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build a pre-cast text-encoder checkpoint for the Studio TE prequant path.

Apply the runtime layerwise-fp8 STORAGE cast (``diffusion_precision._cast_fp8``) to a
model's dense text encoder ONCE and save the cast state dict, so the backend can load the
~half-size artifact (meta-init + ``load_state_dict(assign=True)``, see
``core/inference/diffusion_te_prequant.py``) instead of downloading the full bf16 encoder
and casting on every load. The cast is a deterministic storage transform, so the loaded
encoder is bit-identical to dense-load-then-cast by construction. CPU-runnable: the cast
touches storage dtypes only, no kernels.

  python scripts/build_te_prequant_checkpoint.py \
      --base Lightricks/LTX-2 --family ltx-2 --component text_encoder \
      --out outputs/te_prequant/ltx2/text_encoder_fp8.pt
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
        "--base", required = True, help = "diffusers base repo (carries the component subfolder)"
    )
    p.add_argument("--family", required = True, help = "diffusion/video family name or alias")
    p.add_argument(
        "--component",
        default = "text_encoder",
        help = "pipeline component attribute (also the repo subfolder)",
    )
    p.add_argument(
        "--config-subfolder",
        default = None,
        help = "where the encoder lives inside --base (default: the component name; "
        "pass '' for a standalone encoder repo whose config sits at the root, "
        "e.g. HiDream's Llama text_encoder_4)",
    )
    p.add_argument("--scheme", default = "fp8", choices = ["fp8"])
    p.add_argument("--out", required = True, help = "output .pt path for the checkpoint")
    p.add_argument("--dtype", default = "bfloat16", choices = ["bfloat16"])
    p.add_argument("--hf-token", default = None)
    args = p.parse_args(argv)

    sys.path.insert(0, str(BACKEND))
    import torch
    import transformers

    from core.inference.diffusion_precision import _cast_fp8
    from core.inference.diffusion_te_prequant import TE_PREQUANT_FORMAT

    # The family is metadata for forensics; detection lives in different modules per
    # branch (diffusion_families vs video_families), so resolve best-effort by name.
    family = args.family.strip().lower()

    subfolder = args.component if args.config_subfolder is None else args.config_subfolder
    from_pretrained_kwargs = {"token": args.hf_token}
    if subfolder:
        from_pretrained_kwargs["subfolder"] = subfolder

    print(f"== build TE prequant ({family}/{args.component}/{args.scheme}) ==", flush = True)
    print(f"  loading dense encoder from {args.base} (subfolder={subfolder!r}) ...", flush = True)
    t0 = time.time()
    config = transformers.AutoConfig.from_pretrained(args.base, **from_pretrained_kwargs)
    # Prefer the checkpoint's own architecture (what the diffusers pipeline instantiates,
    # e.g. Gemma3ForConditionalGeneration); AutoModel.from_config would give the bare base
    # class and record a te_class whose state dict the pipeline cannot use.
    arch = (getattr(config, "architectures", None) or [None])[0]
    if arch and hasattr(transformers, arch):
        encoder_cls_name = arch
    else:
        encoder = transformers.AutoModel.from_config(config)
        encoder_cls_name = type(encoder).__name__
        del encoder
    encoder = getattr(transformers, encoder_cls_name).from_pretrained(
        args.base,
        torch_dtype = torch.bfloat16,
        **from_pretrained_kwargs,
    )
    print(f"  casting in place (layerwise {args.scheme}) ...", flush = True)

    class _Target:
        dtype = torch.bfloat16

    _cast_fp8(encoder, _Target())

    state_dict = {
        k: (v.detach().to("cpu") if hasattr(v, "detach") else v)
        for k, v in encoder.state_dict().items()
    }
    metadata = {
        "base_model_id": args.base,
        "family": family,
        "scheme": args.scheme,
        "component": args.component,
        "te_class": encoder_cls_name,
        "torch_dtype": args.dtype,
        "cast_backend": "diffusers_layerwise",
        # str(): torch.__version__ is a TorchVersion object; pickling it into the
        # checkpoint makes torch.load(weights_only=True) reject the whole artifact.
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
    }
    ckpt = {
        "format": TE_PREQUANT_FORMAT,
        "metadata": metadata,
        "state_dict": state_dict,
    }
    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    torch.save(ckpt, out)
    size_gb = out.stat().st_size / 1e9
    print(f"  saved {out}  ({size_gb:.2f} GB) in {time.time() - t0:.0f}s", flush = True)
    print(f"  metadata: {metadata}", flush = True)
    print("BUILD-TE-PREQUANT-DONE", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
