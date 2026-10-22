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

/** Whether the manual repair would turn this Chat only install into one that trains. The backend
 * reports `no_torch` off Mac only with a GPU a PyTorch build could use, and sets `chat_only_hint` on
 * a Mac the MLX wheels skip. */
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
