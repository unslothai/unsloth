// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT } from "../auth/session-events.ts";

// "Chat about this" hands files to a composer that may not be mounted yet (the chat page mounts
// on navigation) or already is (it stays mounted off-route). Keyed by the composer's attachment
// target, so only the fresh chat that was opened for the files picks them up. Kept free of other
// imports but an event name: the chat thread reads it, and must not pull the Library in.
export interface LibraryChatHandoff {
  files: File[];
}

interface LibraryChatHandoffState {
  pending: { targetKey: string; handoff: LibraryChatHandoff } | null;
  held: { targetKey: string; files: File[] } | null;
  offer: (targetKey: string, handoff: LibraryChatHandoff) => void;
  take: (targetKey: string) => LibraryChatHandoff | null;
}

let session = 0;

export const useLibraryChatHandoffStore = create<LibraryChatHandoffState>(
  (set, get) => ({
    pending: null,
    held: null,
    offer: (targetKey, handoff) => set({ pending: { targetKey, handoff }, held: null }),
    take: (targetKey) => {
      const pending = get().pending;
      if (!pending || pending.targetKey !== targetKey) return null;
      set({ pending: null });
      return pending.handoff;
    },
  }),
);

export async function attachLibraryChatFiles(
  targetKey: string,
  add: (file: File) => Promise<unknown>,
  retryHeld = false,
): Promise<number> {
  const store = useLibraryChatHandoffStore;
  const started = session;
  const held = store.getState().held;
  let files: File[];
  if (retryHeld) {
    if (held?.targetKey !== targetKey) return 0;
    store.setState({ held: null });
    files = held.files;
  } else {
    files = store.getState().take(targetKey)?.files ?? [];
  }
  const refused: File[] = [];
  for (const file of files) {
    if (session !== started) return 0;
    try {
      await add(file);
    } catch {
      refused.push(file);
    }
  }
  if (refused.length === 0 || session !== started) return 0;
  store.setState(({ held: now }) => ({
    held: {
      targetKey,
      files: [...(now?.targetKey === targetKey ? now.files : []), ...refused],
    },
  }));
  return refused.length;
}

// Module state outlives a sign-out; the next account must not be handed these files.
if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    session += 1;
    useLibraryChatHandoffStore.setState({ pending: null, held: null });
  });
}
