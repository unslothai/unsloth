// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Loaded build row for the backend's `lora_fuse` status: whether the attached LoRA set is folded into the weights. */
const LORA_FUSE_LABELS: Record<string, { label: string; detail: string }> = {
  fused: {
    label: "Fused",
    detail: "The LoRA set is folded into the denoiser weights, so renders skip the adapter math.",
  },
  pending: {
    label: "Not fused yet",
    detail: "A new LoRA set runs unfused once; the next render with the same set can fold it.",
  },
  disabled: {
    label: "Not fused (fusion off)",
    detail: "LoRA fusion is off on this install (UNSLOTH_DIFFUSION_LORA_FUSE=1 turns it on).",
  },
  bit_exact: {
    label: "Not fused (bit-exact)",
    detail: "Speed mode Off keeps the unfused forward bit for bit, so the LoRA set is not folded.",
  },
  offload: {
    label: "Not fused (offload)",
    detail: "The denoiser weights move between CPU and GPU every render, so the LoRA set cannot be folded.",
  },
  quantized: {
    label: "Not fused (quantized)",
    detail: "A quantized transformer keeps the adapters it baked in at load.",
  },
  unsupported: {
    label: "Not fused",
    detail: "These adapters or weights cannot be folded exactly (for example DoRA, GGUF or manual-cast layers).",
  },
};

export function loraFuseLabel(
  status: string | null | undefined,
): { label: string; detail: string } | null {
  if (!status) return null;
  return LORA_FUSE_LABELS[status] ?? null;
}
