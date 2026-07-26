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
  requestConfirm: (args: {
    count: number;
    titles?: string[];
    action?: string;
  }) => Promise<boolean>;
  resolve: (confirmed: boolean) => void;
}

export const useStopRunningChatsDialogStore =
  create<StopRunningChatsDialogStore>()((set) => ({
    open: false,
    count: 0,
    titles: [],
    action: "",
    requestConfirm: ({ count, titles = [], action = "" }) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        set({ open: true, count, titles, action });
      }),
    resolve: (confirmed) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      set({ open: false, count: 0, titles: [], action: "" });
      resolver?.(confirmed);
    },
  }));
