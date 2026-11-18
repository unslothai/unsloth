// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolvedSelectValue, type ResolvedControl } from "./resolved-precision.ts";

export const INT8_ATTENTION_CHOICES = ["auto", "on", "off"] as const;
export type Int8AttentionChoice = (typeof INT8_ATTENTION_CHOICES)[number];

export const INT8_ATTENTION_LABEL = "INT8 attention";

export const INT8_ATTENTION_OPTIONS: [Int8AttentionChoice, string][] = [
  ["auto", "Auto"],
  ["on", "On"],
  ["off", "Off"],
];

export const INT8_ATTENTION_HINT =
  "Studio's INT8 attention kernel. Auto turns it on where it is measured accurate and faster for this model and GPU (for example MiniMax-H3 video, FLUX, Wan, Qwen-Image-2.1). On engages it wherever it passes the accuracy check. Off keeps the attention selected above. SageAttention (INT8) stays a separate choice in the Attention select.";

export const ATTENTION_SELECT_HINT =
  "Attention kernel. Auto upgrades to cuDNN fused attention on NVIDIA when a speed profile is active. sage is SageAttention 2, one INT8 attention option (without a local install Unsloth fetches the Hugging Face kernels-hub build, which runs on Ampere, Ada and Hopper GPUs, and any other GPU keeps the default): fast (10-40%) but can black-frame some families (Qwen, Wan), so it never engages automatically. Studio's own INT8 attention kernel is the separate INT8 attention control below.";

/** Null leaves the select alone (older backend, sd.cpp, MLX); a declined "on" snaps to "off". */
export function int8AttentionSelection(
  resolved: ResolvedControl | undefined | null,
): Int8AttentionChoice | null {
  return resolvedSelectValue(resolved, (v) => {
    const text = v.trim().toLowerCase();
    if (text === "none") return "off";
    return INT8_ATTENTION_CHOICES.find((o) => o === text) ?? null;
  });
}
