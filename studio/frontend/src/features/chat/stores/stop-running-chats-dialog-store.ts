// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

type Resolver = (confirmed: boolean) => void;

export type StopRunningChatsEffect = "reload" | "unload";

// One at a time: a new request declines any pending one so no promise leaks.
let pendingResolver: Resolver | null = null;

interface StopRunningChatsDialogStore {
  open: boolean;
  count: number;
  titles: string[];
  action: string;
  hasNonChat: boolean;
  effect: StopRunningChatsEffect;
  requestConfirm: (args: {
    count: number;
    titles?: string[];
    action?: string;
    hasNonChat?: boolean;
    effect?: StopRunningChatsEffect;
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
    effect: "reload",
    requestConfirm: ({
      count,
      titles = [],
      action = "",
      hasNonChat = false,
      effect = "reload",
    }) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        set({ open: true, count, titles, action, hasNonChat, effect });
      }),
    resolve: (confirmed) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      set({
        open: false,
        count: 0,
        titles: [],
        action: "",
        hasNonChat: false,
        effect: "reload",
      });
      resolver?.(confirmed);
    },
  }));
