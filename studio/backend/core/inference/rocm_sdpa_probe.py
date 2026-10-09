# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One ROCm SDPA backend per process, so a deferred launch error cannot poison the next probe."""

from __future__ import annotations

import json
import sys

RESULT_PREFIX = "UNSLOTH_SDPA_PROBE "


def probe(device: str, dtype_name: str, backend_name: str) -> str:
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    dtype = getattr(torch, dtype_name)
    backend = getattr(SDPBackend, backend_name)
    generator = torch.Generator(device = device).manual_seed(0)
    ran = False
    with torch.inference_mode():
        for length, causal in ((256, True), (1024, False)):
            q, k, v = (
                torch.randn(1, 4, length, 128, device = device, dtype = dtype, generator = generator)
                for _ in range(3)
            )
            try:
                with sdpa_kernel([SDPBackend.MATH]):
                    reference = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, is_causal = causal
                    )
                reference = reference.float()
                if not bool(torch.isfinite(reference).all().item()):
                    return "unknown"
            except Exception:
                return "unknown"
            if backend_name == "MATH":
                ran = True
                continue
            try:
                with sdpa_kernel([backend]):
                    actual = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, is_causal = causal
                    )
                # A failed HIP launch can surface only at this checked consumer, not at synchronize().
                actual = actual.float()
                actual.sum().item()
                if not torch.allclose(actual, reference, atol = 2e-2, rtol = 2e-2):
                    return "failed"
                ran = True
            except torch.OutOfMemoryError:
                return "unknown"
            except Exception as exc:
                if "No available kernel" not in str(exc):
                    return "failed"
    return "available" if ran else "unavailable"


if __name__ == "__main__":
    try:
        result = probe(*sys.argv[1:])
    except Exception:
        result = "unknown"
    print(RESULT_PREFIX + json.dumps(result), flush = True)
