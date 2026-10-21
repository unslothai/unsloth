# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build the hosted int8 ConvRot Gemma 4 encoder LTX-2.5's text-encoder sidecar loads instead of quantizing at load.

The file IS the sidecar's own at-load build, written once: this runs the worker's ``_load`` (transformers 5.10
sidecar, ``from_pretrained(torch_dtype = bfloat16)``, the dead-tail trim, then ``quantize_int8_convrot_`` on the card)
and saves the resulting state dict. So every key is the loaded model's own name (transformers 5 renames some
checkpoint keys at load), every decoder projection is the ``Int8ConvRotLinear`` buffers (``<name>.weight`` int8,
``<name>.weight_scale`` float32 [out, 1]) and every other tensor is the bf16 tensor the at-load path holds. The tied
``lm_head`` has no key (it is re-tied to the embedding at load). The header is the plain-tensor container the prequant
readers share (``unsloth_format`` + ``unsloth_metadata`` + ``tensor_names``), written canonically, so a rebuild on the
same device kind is byte-identical.

  python scripts/build_ltx25_te_int8_convrot.py \\
      --out outputs/te_default/unsloth/LTX-2.5-FP8/LTX-2.5-text_encoder-INT8-ConvRot.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "studio" / "backend"


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default = "Lightricks/LTX-2.5-Diffusers")
    p.add_argument(
        "--repo", default = None, help = "local pack folder or Hub id to read (default: --base)"
    )
    p.add_argument("--out", required = True)
    p.add_argument(
        "--device", default = "cuda:0", help = "the at-load path quantizes on the card when it has one"
    )
    p.add_argument(
        "--venv-dir", default = None, help = "the transformers 5.10 sidecar (default: Unsloth's)"
    )
    p.add_argument("--hf-token", default = None)
    args = p.parse_args(argv)
    sys.path.insert(0, str(BACKEND))
    venv_dir = args.venv_dir
    if venv_dir is None:
        from utils.transformers_version import _VENV_T5_510_DIR
        venv_dir = _VENV_T5_510_DIR
    result = build(
        args.repo or args.base,
        Path(args.out),
        base = args.base,
        device = args.device,
        venv_dir = venv_dir,
        hf_token = args.hf_token,
    )
    print(json.dumps(result))
    return 0


def build(
    repo: str,
    out: Path,
    *,
    base: str,
    device: str,
    venv_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict:
    """Load ``repo``'s encoder the worker's way, quantized, and write it to ``out``; its size, sha256 and counts."""
    sys.path.insert(0, str(BACKEND))
    sys.path.insert(0, str(ROOT / "scripts"))
    from core.inference import video_ltx25_te_worker as worker

    if venv_dir:
        worker._activate_sidecar(venv_dir)
    import safetensors
    import torch

    from build_te_int8_convrot_checkpoint import _write_safetensors
    from core.inference.prequant_safetensors import UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY
    from core.inference.video_ltx25 import LTX25_TE_HOSTED_FORMAT
    from core.inference.video_minimax_h3_te import _int8_convrot_linear_class

    device = _available_device(torch, device)
    on_card = device.startswith(("cuda", "xpu"))
    config = {
        "repo": repo,
        "hf_token": hf_token,
        "device": device,
        "dtype": "bfloat16",
        "resident": on_card,
        "te_quant": worker.TE_QUANT_INT8_CONVROT,
    }
    _tokenizer, model = worker._load(config)
    if config.get("_te_source") != "quantized_at_load":
        raise SystemExit(f"expected the at-load build, got {config.get('_te_source')!r}")
    quantized = sum(1 for m in model.modules() if isinstance(m, _int8_convrot_linear_class()))
    state = model.state_dict()
    # The tied head is the embedding under a second name; the loader re-ties it, so it never gets a key of its own.
    # The dead-tail trim normally drops it already, but a kept head (or a trim that did not engage) must not ship a
    # second 262k-vocab copy.
    if _head_is_tied(state):
        state.pop("lm_head.weight")
    tensors = {k: v.detach().to("cpu").contiguous() for k, v in state.items()}
    del model
    names = sorted(tensors)
    try:
        import transformers

        cfg = transformers.AutoConfig.from_pretrained(
            repo, subfolder = "text_encoder", token = hf_token
        )
        te_class = (getattr(cfg, "architectures", None) or [None])[0]
        transformers_version = transformers.__version__
    except Exception:  # noqa: BLE001 -- forensic metadata only
        te_class, transformers_version = None, None
    metadata = {
        "base_model_id": base,
        "family": "ltx-2.5",
        "scheme": "int8",
        "component": "text_encoder",
        "te_class": te_class or "Gemma4UnifiedForConditionalGeneration",
        "torch_dtype": "bfloat16",
        "quant": {
            "format": "int8_convrot",
            "group_size": worker.CONVROT_GROUP,
            "weight_only": True,
            "scale": "per_output_channel_absmax_over_127",
            "linears": quantized,
            "quantized_on": torch.device(device).type if on_card else "cpu",
        },
        "lm_head": "tied to the input embedding, no key",
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "safetensors_version": safetensors.__version__,
    }
    header = {
        UNSLOTH_FORMAT_KEY: LTX25_TE_HOSTED_FORMAT,
        UNSLOTH_METADATA_KEY: json.dumps(metadata, sort_keys = True),
        "tensor_names": json.dumps(names),
    }
    # Plain tensors only, so the shared reader (prequant_safetensors.load_plain_prequant_safetensors) accepts it.
    header.update({name: json.dumps({"_type": "Tensor"}) for name in names})
    out.parent.mkdir(parents = True, exist_ok = True)
    tmp = out.with_name(out.name + ".tmp")
    _write_safetensors(tmp, tensors, header)
    os.replace(tmp, out)
    digest = hashlib.sha256()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            digest.update(chunk)
    return {
        "out": str(out),
        "bytes": out.stat().st_size,
        "sha256": digest.hexdigest(),
        "quantized_linears": quantized,
        "tensors": len(names),
    }


def _available_device(torch, device: str) -> str:
    """``device`` when its backend is usable here, else ``cpu``: the worker moves a resident build onto the device."""
    kind = torch.device(device).type
    backend = getattr(torch, kind, None) if kind in ("cuda", "xpu") else None
    if backend is None:
        return device
    is_available = getattr(backend, "is_available", None)
    if callable(is_available) and is_available():
        return device
    print(f"{device} is not available on this host; building on the CPU", file = sys.stderr)
    return "cpu"


def _head_is_tied(state: dict) -> bool:
    """Whether ``lm_head.weight`` is the same storage as another tensor in ``state`` (the input embedding)."""
    head = state.get("lm_head.weight")
    if head is None:
        return False
    return any(
        name != "lm_head.weight" and t.shape == head.shape and t.data_ptr() == head.data_ptr()
        for name, t in state.items()
    )


if __name__ == "__main__":
    raise SystemExit(main())
