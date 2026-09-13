// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

/** Composer-only choice. Persist the purpose on the attachment when sending. */
export const useMcpImageSelection = create<{
  attachmentId: string | null;
  select: (id: string | null) => void;
}>((set) => ({
  attachmentId: null,
  select: (attachmentId) => set({ attachmentId }),
}));

export function isSelectedMcpImage(id: string): boolean {
  return useMcpImageSelection.getState().attachmentId === id;
}

export function clearSelectedMcpImage(id?: string): void {
  const selection = useMcpImageSelection.getState();
  if (id === undefined || selection.attachmentId === id) {
    selection.select(null);
  }
}
