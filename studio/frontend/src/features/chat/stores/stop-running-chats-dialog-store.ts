// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

type Resolver = (confirmed: boolean) => void;

// One at a time: a new request declines any pending one so no promise leaks.
let pendingResolver: Resolver | null = null;

interface StopRunningChatsDialogStore {
  open: boolean;
  /** How many conversations the pending action would stop. */
  count: number;
  /** Titles of those conversations, when known, for the dialog body. */
  titles: string[];
  /** What the user is about to do, e.g. "Loading a different model". */
  action: string;
  /** The set includes an embeddings/completions/audio request, which is not a chat. */
  hasNonChat: boolean;
  requestConfirm: (args: {
    count: number;
    titles?: string[];
    action?: string;
    hasNonChat?: boolean;
  }) => Promise<boolean>;
  resolve: (confirmed: boolean) => void;
}

export const useStopRunningChatsDialogStore =
  create<StopRunningChatsDialogStore>()((set) => ({
    open: false,
    count: 0,
    titles: [],
    action: "",
    hasNonChat: false,
    requestConfirm: ({ count, titles = [], action = "", hasNonChat = false }) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        set({ open: true, count, titles, action, hasNonChat });
      }),
    resolve: (confirmed) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      set({ open: false, count: 0, titles: [], action: "", hasNonChat: false });
      resolver?.(confirmed);
    },
  }));
