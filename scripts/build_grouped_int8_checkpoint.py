# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Packed W6A8 / W4A8 single file from a floating one (round to nearest, ConvRot group 256), in the per-layer
``comfy_quant`` layout Studio and other single-file runtimes read.

    python scripts/build_grouped_int8_checkpoint.py SRC.safetensors OUT.safetensors --format w6a8_int8 \\
        --like REFERENCE_QUANTIZED.safetensors [--group-size 16] [--device cuda] [--base-model ORG/REPO]

``--like`` copies the layer set of an existing quantized file of the same model (its declared layers); without it
``--layers-regex`` picks layers by name."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"


def main() -> int:
    parser = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("src")
    parser.add_argument("out")
    parser.add_argument("--format", required = True, choices = ("w6a8_int8", "asym_w4a8_int8"))
    parser.add_argument("--like", help = "quantized file whose declared layers are converted")
    parser.add_argument(
        "--layers-regex", help = "layer names (no .weight) to convert, when there is no --like file"
    )
    parser.add_argument("--group-size", type = int, default = 16)
    parser.add_argument("--rotation", type = int, default = 256)
    parser.add_argument("--device", default = "cpu")
    parser.add_argument("--base-model", help = "recorded in the header (unsloth_comfy_conversion)")
    args = parser.parse_args()
    if not args.like and not args.layers_regex:
        parser.error("pass --like or --layers-regex")
    sys.path.insert(0, str(BACKEND))
    from core.inference.diffusion_grouped_int8 import export_grouped_file

    pattern = re.compile(args.layers_regex) if args.layers_regex else None
    record = {"scheme": args.format, "group_size": args.group_size, "rounding": "nearest"}
    if args.base_model:
        record["base_model_id"] = args.base_model
    result = export_grouped_file(
        args.src,
        args.out,
        args.format,
        like = args.like,
        layers = (lambda name: bool(pattern.search(name))) if pattern else None,
        group_size = args.group_size,
        rotation = args.rotation,
        device = args.device,
        record = record,
    )
    print(json.dumps({k: (v if k != "kept" else len(v)) for k, v in result.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
