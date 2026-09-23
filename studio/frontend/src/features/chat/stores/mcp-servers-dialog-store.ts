// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** One MCP servers dialog for the chat, opened from either side: the composer pill's menu and the
 *  keyboard shortcut. The pill only exists once MCP is on for the chat, and it ships off, so the
 *  dialog cannot live there or the shortcut would do nothing until the user found the pill. */
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
