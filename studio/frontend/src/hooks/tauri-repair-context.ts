// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createContext, useContext } from "react";

/** What Settings needs from the backend controller to offer a manual repair. */
export type TauriRepairController = {
  /** Reruns the bundled installer over the managed environment, then restarts the backend. */
  repairInstall: () => Promise<void>;
};

// Null outside Tauri and on the startup screen, where the app shell has not mounted. The
// consumer renders nothing rather than offering an action that cannot run. Mirrors
// TauriUpdateContext, which the settings update row already uses this way.
export const TauriRepairContext = createContext<TauriRepairController | null>(
  null,
);

export function useTauriRepairController(): TauriRepairController | null {
  return useContext(TauriRepairContext);
}
