# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SAM 3 / 3.1 segmenter builder; reuses diffusion_segment_quant so Studio's loader checks pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required = True, help = "transformers-format Sam3Model directory (bf16)")
    p.add_argument(
        "--base-model-id", required = True, help = "the repo the artifact belongs to (recorded)"
    )
    p.add_argument(
        "--source-revision", default = None, help = "the repo commit the source was taken from"
    )
    p.add_argument("--scheme", required = True, choices = ("fp8", "int8"))
    p.add_argument("--convrot-groupsize", type = int, default = 0)
    p.add_argument("--out", required = True, help = "output .safetensors path")
    p.add_argument("--device", default = "cuda")
    args = p.parse_args(argv)
    if not args.out.endswith(".safetensors"):
        print("error: --out must end in .safetensors", flush = True)
        return 2

    sys.path.insert(0, str(BACKEND))
    import torch
    from transformers import Sam3Model

    from core.inference.diffusion_segment_quant import quantize_segmenter_, write_prequant_file

    started = time.time()
    model = Sam3Model.from_pretrained(args.src, dtype = torch.bfloat16, device_map = args.device).eval()
    quant = quantize_segmenter_(model, args.scheme, convrot_group = args.convrot_groupsize)
    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    tmp = out.with_name(out.name + ".tmp")
    metadata = write_prequant_file(
        str(tmp),
        model,
        quant,
        base_model_id = args.base_model_id,
        source_revision = args.source_revision,
    )
    tmp.replace(out)
    fmt = metadata["format"]
    digest = hashlib.sha256()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    print(
        json.dumps(
            {
                "out": str(out),
                "bytes": out.stat().st_size,
                "sha256": digest.hexdigest(),
                "format": fmt,
                "scheme": args.scheme,
                "quantized_linears": quant["quantized_linears"],
                "rotated_linears": len(quant.get("activation_rotation_fqns") or ()),
                "exclude_name_tokens": quant["exclude_name_tokens"],
                "fingerprinted": metadata["fingerprint"]["count"],
                "seconds": round(time.time() - started, 1),
            }
        ),
        flush = True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
