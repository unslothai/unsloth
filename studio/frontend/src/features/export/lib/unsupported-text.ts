// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Kept out of config/hardware-verdict.ts (startup bundle): only the lazy Export route needs it.

/** A Desktop Chat only user never typed the --no-torch the backend names. */
export function exportUnsupportedText(
  reason: string | null,
  message: string | null,
  canUpgrade: boolean,
): string {
  if (reason === "no_torch" && canUpgrade)
    return "Export was left out of this install (Chat only). Use Upgrade to full install in Settings to add it.";
  return (
    message ??
    "Export requires a supported accelerator (NVIDIA, AMD, or Intel GPU, or Apple Silicon) with PyTorch or MLX installed."
  );
}
