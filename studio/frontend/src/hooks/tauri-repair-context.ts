// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";

/** What Settings needs from the backend controller to offer a manual repair. */
export type TauriRepairController = {
  /** Reruns the bundled installer over the managed environment, then restarts the backend. */
  repairInstall: () => Promise<void>;
  /**
   * True when the desktop is attached to a backend someone started in a terminal.
   *
   * start_managed_repair refuses to touch that environment, but startRepair switches the
   * whole shell to the repairing screen before the refusal comes back, so an offered
   * action would drop a connected user on the repair-error screen. The consumer hides
   * itself instead. The settings update row already suppresses its action this way.
   */
  isExternalServer: boolean;
};

// Null outside Tauri and on the startup screen, where the app shell has not mounted. The
// consumer renders nothing rather than offering an action that cannot run.
export const TauriRepairContext = createContext<TauriRepairController | null>(
  null,
);

export function useTauriRepairController(): TauriRepairController | null {
  return useContext(TauriRepairContext);
}

/**
 * Whether a click can turn this Chat only install into a full one. Only a measured `no_torch`
 * verdict qualifies: that install left PyTorch out by request, and the manual repair reruns the
 * installer with it. Off Mac the backend reports `no_torch` only when its PyTorch-free GPU
 * inventory finds a card a GPU build could use; a CPU-only host reports `no_gpu`. On a Mac the
 * MLX wheels skip, the installer adds PyTorch but not MLX, so the backend sends a
 * `chat_only_hint` saying why and Train stays off. Every other chat-only reason keeps its own hint.
 */
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
