// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { LlmCompressorExportProbe } from "../api/llm-compressor-consent-api";

type Resolver = (consented: boolean) => void;

let pendingResolver: Resolver | null = null;

interface LlmCompressorConsentStore {
  open: boolean;
  probe: LlmCompressorExportProbe | null;
  requestConsent: (probe: LlmCompressorExportProbe) => Promise<boolean>;
  resolve: (consented: boolean) => void;
}

export const useLlmCompressorConsentStore = create<LlmCompressorConsentStore>()(
  (set) => ({
    open: false,
    probe: null,
    requestConsent: (probe) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        set({ open: true, probe });
      }),
    resolve: (consented) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      set({ open: false, probe: null });
      resolver?.(consented);
    },
  }),
);
