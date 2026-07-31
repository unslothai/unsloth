// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export type HfTokenWarningDecision = "anonymous" | "replace" | "cancel";
type Resolver = (decision: HfTokenWarningDecision) => void;

let pendingResolver: Resolver | null = null;
let pendingOwner: unknown;

interface HfTokenWarningStore {
  open: boolean;
  allowAnonymous: boolean;
  requestDecision: (
    allowAnonymous: boolean,
    owner?: unknown,
  ) => Promise<HfTokenWarningDecision>;
  resolve: (decision: HfTokenWarningDecision, owner?: unknown) => void;
}

export const useHfTokenWarningStore = create<HfTokenWarningStore>((set) => ({
  open: false,
  allowAnonymous: true,
  requestDecision: (allowAnonymous, owner) =>
    new Promise<HfTokenWarningDecision>((resolve) => {
      pendingResolver?.("cancel");
      pendingResolver = resolve;
      pendingOwner = owner;
      set({ open: true, allowAnonymous });
    }),
  resolve: (decision, owner) => {
    if (owner !== undefined && pendingOwner !== owner) return;
    const resolver = pendingResolver;
    pendingResolver = null;
    pendingOwner = undefined;
    set({ open: false, allowAnonymous: true });
    resolver?.(decision);
  },
}));
