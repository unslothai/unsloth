// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  CONNECTIONS_ENABLED_KEY,
  DISABLED_EXTERNAL_PROVIDERS_KEY,
  EXTERNAL_PROVIDERS_KEY,
  loadConnectionsEnabled,
  loadExternalProviders,
  saveConnectionsEnabled,
  saveExternalProviders,
  type ExternalProviderConfig,
} from "../external-providers";

interface ExternalProvidersState {
  providers: ExternalProviderConfig[];
  /** Switched off in Settings > Connections: listed there, offered nowhere else. Every sync refills it; persisted so the rows' browser-only options outlive a reload. */
  disabledProviders: ExternalProviderConfig[];
  connectionsEnabled: boolean;
  setProviders: (providers: ExternalProviderConfig[]) => void;
  setDisabledProviders: (providers: ExternalProviderConfig[]) => void;
  setConnectionsEnabled: (enabled: boolean) => void;
}

/** Connection ids whose settings save is awaiting the backend; auto reload holds its writes. */
export const providerSavesInFlight = new Set<string>();

export const useExternalProvidersStore = create<ExternalProvidersState>(
  (set) => ({
    providers: loadExternalProviders(),
    disabledProviders: loadExternalProviders(DISABLED_EXTERNAL_PROVIDERS_KEY),
    connectionsEnabled: loadConnectionsEnabled(),
    setProviders: (providers) => {
      set({ providers });
      saveExternalProviders(providers);
    },
    setDisabledProviders: (disabledProviders) => {
      set({ disabledProviders });
      saveExternalProviders(disabledProviders, DISABLED_EXTERNAL_PROVIDERS_KEY);
    },
    setConnectionsEnabled: (enabled) => {
      set({ connectionsEnabled: enabled });
      saveConnectionsEnabled(enabled);
    },
  }),
);

// Voice settings sync across tabs and drop a provider id this tab does not list.
if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === EXTERNAL_PROVIDERS_KEY || event.key === null) {
      useExternalProvidersStore.setState({ providers: loadExternalProviders() });
    }
    if (event.key === DISABLED_EXTERNAL_PROVIDERS_KEY || event.key === null) {
      useExternalProvidersStore.setState({
        disabledProviders: loadExternalProviders(DISABLED_EXTERNAL_PROVIDERS_KEY),
      });
    }
    if (event.key === CONNECTIONS_ENABLED_KEY || event.key === null) {
      useExternalProvidersStore.setState({
        connectionsEnabled: loadConnectionsEnabled(),
      });
    }
  });
}
