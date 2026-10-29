// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  loadConnectionsEnabled,
  loadExternalProviders,
  saveConnectionsEnabled,
  saveExternalProviders,
  type ExternalProviderConfig,
} from "../external-providers";

interface ExternalProvidersState {
  providers: ExternalProviderConfig[];
  connectionsEnabled: boolean;
  setProviders: (providers: ExternalProviderConfig[]) => void;
  setConnectionsEnabled: (enabled: boolean) => void;
}

/** Connection ids whose settings save is awaiting the backend; auto reload holds its writes. */
export const providerSavesInFlight = new Set<string>();

export const useExternalProvidersStore = create<ExternalProvidersState>(
  (set) => ({
    providers: loadExternalProviders(),
    connectionsEnabled: loadConnectionsEnabled(),
    setProviders: (providers) => {
      set({ providers });
      saveExternalProviders(providers);
    },
    setConnectionsEnabled: (enabled) => {
      set({ connectionsEnabled: enabled });
      saveConnectionsEnabled(enabled);
    },
  }),
);
