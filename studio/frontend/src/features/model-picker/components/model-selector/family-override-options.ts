// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const LABELS: Readonly<Record<string, string>> = {
  "flux.1": "FLUX.1",
  "flux.2-klein": "FLUX.2 Klein",
  "flux.2-dev": "FLUX.2 Dev",
  "qwen-image": "Qwen Image",
  "qwen-image-edit": "Qwen Image Edit",
  "z-image": "Z-Image",
  "krea-2": "Krea 2",
  "minimax-h3": "MiniMax-H3",
  "ltx-2": "LTX-2",
  "wan2.2-ti2v-5b": "Wan2.2 TI2V 5B",
  "wan2.2-t2v-a14b": "Wan2.2 T2V A14B",
  "hunyuanvideo-1.5": "HunyuanVideo 1.5 (480p)",
  "hunyuanvideo-1.5-720p": "HunyuanVideo 1.5 (720p)",
};

/** Build the selector from the backend registry, so adding a family cannot leave the UI stale. */
export function familyOverrideOptions(
  supportedFamilies: readonly string[] | null | undefined,
): [string, string][] {
  const names = [...new Set(supportedFamilies ?? [])].filter(Boolean);
  return [
    ["auto", "Auto (detect)"],
    ...names.map((name): [string, string] => [name, LABELS[name] ?? name]),
  ];
}
