# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Examine which Linear layers the int8 dense-quant filter would select, to find the large M=1
modulation/embedder projections that crash torch._int_mm (M>16). Loads each transformer on the
META device (no weights, no GPU) from its base-repo config, lists nn.Linear fqn/in/out, and marks
those that pass min_features=512. CPU-only, fast."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))

# (label, transformer_class, base_repo)
MODELS = [
    ("flux.1-dev", "FluxTransformer2DModel", "black-forest-labs/FLUX.1-dev"),
    ("qwen-image", "QwenImageTransformer2DModel", "Qwen/Qwen-Image"),
    ("z-image", "ZImageTransformer2DModel", "Tongyi-MAI/Z-Image-Turbo"),
    ("flux.2-klein-4b", "Flux2Transformer2DModel", "black-forest-labs/FLUX.2-klein-4B"),
]
MIN = 512


def main() -> int:
    import diffusers
    import torch
    from accelerate import init_empty_weights

    tok = os.environ.get("HF_TOKEN")
    for label, cls_name, base in MODELS:
        cls = getattr(diffusers, cls_name, None)
        if cls is None:
            print(f"\n### {label}: {cls_name} NOT in diffusers"); continue
        try:
            cfg = cls.load_config(base, subfolder="transformer", token=tok)
            with init_empty_weights():
                model = cls.from_config(cfg)
        except Exception as e:  # noqa: BLE001
            print(f"\n### {label}: load failed {type(e).__name__}: {e}"); continue
        lins = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        selected = [(n, m) for n, m in lins if m.in_features >= MIN and m.out_features >= MIN]
        print(f"\n### {label}: {len(lins)} Linear, {len(selected)} pass min_features={MIN}")
        # Heuristic: a modulation/embedder Linear is one OUTSIDE the repeated transformer blocks,
        # i.e. its fqn does not contain a numeric block index, OR out==k*in (k>=3) AdaLN shape.
        sus = []
        for n, m in selected:
            depth_idx = any(p.isdigit() for p in n.split("."))
            ratio = m.out_features / m.in_features if m.in_features else 0
            tag = []
            if not depth_idx:
                tag.append("NO-BLOCK-IDX")
            if ratio >= 3:
                tag.append(f"out={ratio:.0f}xin")
            if any(t in n.lower() for t in ("norm", "embed", "time", "guidance", "modulation", "adaln", "cond")):
                tag.append("NAME")
            if tag:
                sus.append((n, m.in_features, m.out_features, ",".join(tag)))
        # Print the distinct fqn shapes (collapse block indices to {i})
        import re
        seen = {}
        for n, i, o, tag in sus:
            key = re.sub(r"\.\d+\.", ".{i}.", n)
            seen.setdefault((key, i, o, tag), 0)
            seen[(key, i, o, tag)] += 1
        print(f"  SUSPECT (M=1 risk) distinct patterns:")
        for (key, i, o, tag), cnt in sorted(seen.items()):
            print(f"    [{cnt:>3}x] {key:55s} {i:>6}->{o:<6} [{tag}]")
        # Also show a few non-suspect selected names for contrast (the real FLOP linears)
        good = [n for n, m in selected if (n, m.in_features, m.out_features) not in
                {(s[0], s[1], s[2]) for s in sus}][:6]
        print(f"  kept-for-int8 examples: {good}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
