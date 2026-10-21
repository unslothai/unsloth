// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Not in config/hardware-verdict.ts: that module is in the startup bundle and this text is only
// needed by the lazy Export route.

/** Why the Export page is disabled. A Desktop Chat only user never typed the --no-torch that the
 * backend's no_torch text names, so they get the Train hint's Repair installation advice instead. */
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
