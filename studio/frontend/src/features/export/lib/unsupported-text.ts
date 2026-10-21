// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of config/hardware-verdict.ts (startup bundle): only the lazy Export route needs it.

/** A Desktop Chat only user never typed the --no-torch the backend names: point at Repair instead. */
export function exportUnsupportedText(
  reason: string | null,
  message: string | null,
  canRepair: boolean,
): string {
  if (reason === "no_torch" && canRepair)
    return "Export was left out of this install (Chat only). Use Repair installation in Settings to add it.";
  return (
    message ??
    "Export requires a supported accelerator (NVIDIA, AMD, or Intel GPU, or Apple Silicon) with PyTorch or MLX installed."
  );
}
