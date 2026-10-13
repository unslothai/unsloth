# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build an int8 ConvRot weight-only text-encoder checkpoint for the Unsloth TE prequant path.

Every decoder projection of the dense encoder (q/k/v/o and gate/up/down in each
``model.language_model.layers.N``) is rotated by the group-256 Hadamard and stored int8 with a
float32 per-output-channel scale (``diffusion_te_prequant.quantize_int8_convrot_weight``); the
vision tower, embedding table and norms are copied bf16, bit for bit; ``lm_head`` is left out
(the pipelines that load this file read hidden states only). Each quantized Linear carries a
``<name>.comfy_quant`` JSON blob naming the format, and the header is the plain-tensor container
the TE prequant loader reads (``tensor_names`` + ``unsloth_format`` + ``unsloth_metadata``).

Reads one source tensor at a time, so host RAM peaks at about the output size. Deterministic: the
same source and group give a byte-identical file.

  python scripts/build_te_int8_convrot_checkpoint.py \\
      --base Qwen/Qwen-Image-2.1 --family qwen-image-2.1 \\
      --out Qwen-Image-2.1-text_encoder-INT8-ConvRot.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent / "studio" / "backend"
DECODER_LINEAR = re.compile(
    r"^model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


_ST_DTYPES = {"float32": "F32", "bfloat16": "BF16", "float16": "F16", "int8": "I8", "uint8": "U8"}


def _write_safetensors(path: Path, tensors: dict, metadata: dict) -> None:
    """safetensors with a canonical header (sorted keys) so a rebuild is byte-identical; ``save_file`` orders the
    metadata map differently per run. Widest dtype first, so every tensor starts at its own alignment."""
    import torch

    order = sorted(tensors, key = lambda n: (-tensors[n].element_size(), n))
    header: dict = {"__metadata__": dict(sorted(metadata.items()))}
    offset = 0
    for name in order:
        t = tensors[name]
        size = t.numel() * t.element_size()
        header[name] = {
            "dtype": _ST_DTYPES[str(t.dtype).replace("torch.", "")],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    blob = json.dumps(header, sort_keys = True, separators = (",", ":")).encode("utf-8")
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as fh:
        fh.write(len(blob).to_bytes(8, "little"))
        fh.write(blob)
        for name in order:
            fh.write(tensors[name].contiguous().view(torch.uint8).numpy().tobytes())


def _download_component(base: str, component: str, token) -> Path:
    """The component folder of ``base``, data files only: the build reads JSON and safetensors (safe_open), never
    anything executable, so nothing else is fetched."""
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        base,
        allow_patterns = [f"{component}/*.json", f"{component}/*.safetensors"],
        token = token,
    )
    return Path(root) / component


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required = True, help = "diffusers base repo carrying text_encoder/")
    p.add_argument("--family", required = True)
    p.add_argument("--component", default = "text_encoder")
    p.add_argument("--src", default = None, help = "local dense encoder folder (skips the download)")
    p.add_argument("--out", required = True, help = "output .safetensors path")
    p.add_argument("--group", type = int, default = 256)
    p.add_argument(
        "--device", default = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"
    )
    p.add_argument("--hf-token", default = None)
    args = p.parse_args(argv)

    sys.path.insert(0, str(BACKEND))
    import safetensors
    import torch
    import transformers
    from safetensors import safe_open

    from core.inference.diffusion_te_prequant import (
        TE_PREQUANT_FORMAT_INT8_CONVROT,
        quantize_int8_convrot_weight,
    )
    from core.inference.prequant_safetensors import UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY

    src = (
        Path(args.src)
        if args.src
        else _download_component(args.base, args.component, args.hf_token)
    )
    config = json.loads((src / "config.json").read_text(encoding = "utf-8"))
    index_path = src / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding = "utf-8"))["weight_map"]
    else:
        with safe_open(str(src / "model.safetensors"), framework = "pt") as handle:
            weight_map = {name: "model.safetensors" for name in handle.keys()}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    quant = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": args.group}
    blob = torch.tensor(list(json.dumps(quant).encode("utf-8")), dtype = torch.uint8)
    tensors: dict = {}
    quantized = 0
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(src / shard), framework = "pt", device = "cpu") as handle:
            for name in sorted(n for n, s in weight_map.items() if s == shard):
                if name == "lm_head.weight":
                    continue
                tensor = handle.get_tensor(name)
                if DECODER_LINEAR.match(name):
                    q, scale = quantize_int8_convrot_weight(tensor.to(device), args.group)
                    prefix = name[: -len(".weight")]
                    tensors[name] = q.cpu().contiguous()
                    tensors[prefix + ".weight_scale"] = scale.cpu().contiguous()
                    tensors[prefix + ".comfy_quant"] = blob.clone()
                    quantized += 1
                else:
                    tensors[name] = tensor.to(torch.bfloat16).contiguous()
    if not quantized:
        raise SystemExit("no decoder projections matched; is this a Qwen3-VL text encoder?")
    metadata = {
        "base_model_id": args.base,
        "family": args.family,
        "scheme": "int8",
        "component": args.component,
        "te_class": (config.get("architectures") or [None])[0],
        "torch_dtype": "bfloat16",
        "quant": dict(
            quant, weight_only = True, scale = "per_output_channel_absmax_over_127", linears = quantized
        ),
        "lm_head": "dropped",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "safetensors_version": safetensors.__version__,
    }
    names = sorted(tensors)
    header = {
        UNSLOTH_FORMAT_KEY: TE_PREQUANT_FORMAT_INT8_CONVROT,
        UNSLOTH_METADATA_KEY: json.dumps(metadata, sort_keys = True),
        "tensor_names": json.dumps(names),
    }
    header.update({name: json.dumps({"_type": "Tensor"}) for name in names})
    out = Path(args.out)
    out.parent.mkdir(parents = True, exist_ok = True)
    tmp = out.with_name(out.name + ".tmp")
    _write_safetensors(tmp, tensors, header)
    os.replace(tmp, out)
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
                "quantized_linears": quantized,
                "tensors": len(names),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
