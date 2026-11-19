// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";

export type TauriRepairController = {
  /** Reruns the bundled installer over the managed environment, then restarts the backend. */
  repairInstall: () => Promise<void>;
  /** Attached to a terminal-started backend: startRepair would switch to the repair screen before
   * start_managed_repair refuses, so the consumer hides itself. */
  isExternalServer: boolean;
};

// Null outside Tauri and on the startup screen; the consumer then renders nothing.
export const TauriRepairContext = createContext<TauriRepairController | null>(
  null,
);

export function useTauriRepairController(): TauriRepairController | null {
  return useContext(TauriRepairContext);
}

/** Off Mac the full install trains on the GPU, or on the CPU when there is none (`cpu_training`); `chat_only_hint` marks a Mac the MLX wheels skip. */
export function canUpgradeToFullInstall(
  chatOnlyReason: string | null,
  chatOnlyHint: string | null,
  verdictMeasured: boolean,
  repair: TauriRepairController | null,
): boolean {
  return (
    verdictMeasured &&
    chatOnlyReason === "no_torch" &&
    chatOnlyHint === null &&
    repair !== null &&
    !repair.isExternalServer
  );
}

/** The full install brings video wherever it brings a video device: not on a GPU-less host (it trains on the CPU there).
 * On a Mac the only hint is the MLX wheel floor, which blocks training alone: the full install skips MLX and still
 * brings PyTorch with MPS, which video runs on. Every other hint (a closed AMD node) blocks video too. */
export function canUpgradeForVideo(
  chatOnlyReason: string | null,
  chatOnlyHint: string | null,
  verdictMeasured: boolean,
  repair: TauriRepairController | null,
  host: { deviceType: string | null; cpuTraining: boolean },
): boolean {
  if (host.cpuTraining) return false;
  const hint = host.deviceType === "mac" ? null : chatOnlyHint;
  return canUpgradeToFullInstall(chatOnlyReason, hint, verdictMeasured, repair);
}
