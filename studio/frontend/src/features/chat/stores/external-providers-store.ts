// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  loadExternalProviders,
  saveExternalProviders,
  type ExternalProviderConfig,
} from "../external-providers";

interface ExternalProvidersState {
  providers: ExternalProviderConfig[];
  setProviders: (providers: ExternalProviderConfig[]) => void;
}

export const useExternalProvidersStore = create<ExternalProvidersState>(
  (set) => ({
    providers: loadExternalProviders(),
    setProviders: (providers) => {
      set({ providers });
      saveExternalProviders(providers);
    },
  }),
);
