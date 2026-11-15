// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { ConnectionCheck } from "../connection-status";

interface ConnectionChecksState {
  /** Last check per connection id. In memory only: a result is stale by the next launch. */
  checks: Record<string, ConnectionCheck>;
  checking: Record<string, true>;
  setChecking: (providerId: string, checking: boolean) => void;
  setCheck: (providerId: string, check: ConnectionCheck) => void;
  clearCheck: (providerId: string) => void;
}

export const useConnectionChecksStore = create<ConnectionChecksState>(
  (set) => ({
    checks: {},
    checking: {},
    setChecking: (providerId, checking) =>
      set((state) => {
        const next = { ...state.checking };
        if (checking) next[providerId] = true;
        else delete next[providerId];
        return { checking: next };
      }),
    setCheck: (providerId, check) =>
      set((state) => ({ checks: { ...state.checks, [providerId]: check } })),
    clearCheck: (providerId) =>
      set((state) => {
        if (!(providerId in state.checks)) return state;
        const next = { ...state.checks };
        delete next[providerId];
        return { checks: next };
      }),
  }),
);
