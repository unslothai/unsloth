// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  CONNECTIONS_ENABLED_KEY,
  EXTERNAL_PROVIDERS_KEY,
  loadConnectionsEnabled,
  loadExternalProviders,
  saveConnectionsEnabled,
  saveExternalProviders,
  type ExternalProviderConfig,
} from "../external-providers";
import { forgetServedContextWindows } from "../model-catalog";

interface ExternalProvidersState {
  providers: ExternalProviderConfig[];
  connectionsEnabled: boolean;
  setProviders: (providers: ExternalProviderConfig[]) => void;
  setConnectionsEnabled: (enabled: boolean) => void;
}

// A saved Base URL change makes the old server's windows wrong for a persistent clamp.
function forgetMovedConnections(
  before: readonly ExternalProviderConfig[],
  after: readonly ExternalProviderConfig[],
): void {
  const was = new Map(before.map((p) => [p.id, p.baseUrl]));
  for (const p of after) {
    const old = was.get(p.id);
    if (old !== undefined && old !== p.baseUrl) forgetServedContextWindows(p.id);
  }
}

/** Connection ids whose settings save is awaiting the backend; auto reload holds its writes. */
export const providerSavesInFlight = new Set<string>();

export const useExternalProvidersStore = create<ExternalProvidersState>(
  (set, get) => ({
    providers: loadExternalProviders(),
    connectionsEnabled: loadConnectionsEnabled(),
    setProviders: (providers) => {
      forgetMovedConnections(get().providers, providers);
      set({ providers });
      saveExternalProviders(providers);
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
      const providers = loadExternalProviders();
      forgetMovedConnections(useExternalProvidersStore.getState().providers, providers);
      useExternalProvidersStore.setState({ providers });
    }
    if (event.key === CONNECTIONS_ENABLED_KEY || event.key === null) {
      useExternalProvidersStore.setState({
        connectionsEnabled: loadConnectionsEnabled(),
      });
    }
  });
}
