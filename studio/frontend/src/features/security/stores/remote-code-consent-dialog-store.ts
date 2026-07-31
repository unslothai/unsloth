// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { RemoteCodeScan } from "../types";

type Resolver = (confirmed: boolean) => void;

// One in-flight consent at a time; a new request resolves any prior pending one as
// declined so its promise never leaks.
let pendingResolver: Resolver | null = null;
let pendingOwner: unknown;

interface RemoteCodeConsentDialogStore {
  open: boolean;
  scan: RemoteCodeScan | null;
  requestConsent: (scan: RemoteCodeScan, owner?: unknown) => Promise<boolean>;
  resolve: (confirmed: boolean, owner?: unknown) => void;
}

export const useRemoteCodeConsentDialogStore = create<RemoteCodeConsentDialogStore>()(
  (set) => ({
    open: false,
    scan: null,
    requestConsent: (scan, owner) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        pendingOwner = owner;
        set({ open: true, scan });
      }),
    resolve: (confirmed, owner) => {
      if (owner !== undefined && pendingOwner !== owner) return;
      const resolver = pendingResolver;
      pendingResolver = null;
      pendingOwner = undefined;
      set({ open: false, scan: null });
      resolver?.(confirmed);
    },
  }),
);
