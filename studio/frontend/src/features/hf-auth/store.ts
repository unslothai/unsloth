// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export type HfTokenWarningDecision = "anonymous" | "replace" | "cancel";
type Resolver = (decision: HfTokenWarningDecision) => void;

let pendingResolver: Resolver | null = null;

interface HfTokenWarningStore {
  open: boolean;
  requestDecision: () => Promise<HfTokenWarningDecision>;
  resolve: (decision: HfTokenWarningDecision) => void;
}

export const useHfTokenWarningStore = create<HfTokenWarningStore>((set) => ({
  open: false,
  requestDecision: () =>
    new Promise<HfTokenWarningDecision>((resolve) => {
      pendingResolver?.("cancel");
      pendingResolver = resolve;
      set({ open: true });
    }),
  resolve: (decision) => {
    const resolver = pendingResolver;
    pendingResolver = null;
    set({ open: false });
    resolver?.(decision);
  },
}));
