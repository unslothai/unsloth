// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** Lives outside the composer pill so the keyboard shortcut works before MCP is enabled. */
interface McpServersDialogState {
  open: boolean;
  setOpen: (open: boolean) => void;
}

export const useMcpServersDialogStore = create<McpServersDialogState>(
  (set) => ({
    open: false,
    setOpen: (open) => set({ open }),
  }),
);
