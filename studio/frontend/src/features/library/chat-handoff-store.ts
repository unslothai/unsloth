// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

// "Chat about this" hands files to a composer that may not be mounted yet (the chat page mounts
// on navigation) or already is (it stays mounted off-route). Keyed by the composer's attachment
// target, so only the fresh chat that was opened for the files picks them up. Kept free of other
// imports: the chat thread reads it, and must not pull the Library in.
export interface LibraryChatHandoff {
  files: File[];
}

interface LibraryChatHandoffState {
  pending: { targetKey: string; handoff: LibraryChatHandoff } | null;
  offer: (targetKey: string, handoff: LibraryChatHandoff) => void;
  take: (targetKey: string) => LibraryChatHandoff | null;
}

export const useLibraryChatHandoffStore = create<LibraryChatHandoffState>(
  (set, get) => ({
    pending: null,
    offer: (targetKey, handoff) => set({ pending: { targetKey, handoff } }),
    take: (targetKey) => {
      const pending = get().pending;
      if (!pending || pending.targetKey !== targetKey) return null;
      set({ pending: null });
      return pending.handoff;
    },
  }),
);
