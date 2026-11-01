// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSessionEpoch, hasAuthToken } from "@/features/auth";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { syncExternalProvidersFromBackend } from "@/features/chat/sync-external-providers";
import { hydrateHfTokenFromBackend } from "@/features/hub/stores/hf-token-store";
import { runCredentialBootstrap } from "./reconciliation";

export function bootstrapPersistedCredentials(): Promise<void> {
  const sessionEpoch = getAuthSessionEpoch();
  const isCurrent = () =>
    hasAuthToken() && getAuthSessionEpoch() === sessionEpoch;
  return runCredentialBootstrap({
    hydrateHfToken: hydrateHfTokenFromBackend,
    getProviders: () => useExternalProvidersStore.getState().providers,
    syncProviders: (providers) =>
      syncExternalProvidersFromBackend(providers, isCurrent),
    setProviders: (providers) =>
      useExternalProvidersStore.getState().setProviders(providers),
    isCurrent,
  });
}

let startupBootstrap: { sessionEpoch: number; run: Promise<void> } | null =
  null;
let startupBootstrapRequested = false;

/** Starts the gate's first run with the entry, so it overlaps the route checks. Once per page. */
export function startStartupCredentialBootstrap(): void {
  if (startupBootstrapRequested) return;
  startupBootstrapRequested = true;
  if (!hasAuthToken()) return;
  startupBootstrap = {
    sessionEpoch: getAuthSessionEpoch(),
    run: bootstrapPersistedCredentials(),
  };
}

/** The run the entry started, once, while it still belongs to this session. */
export function takeStartupCredentialBootstrap(): Promise<void> | null {
  const startup = startupBootstrap;
  startupBootstrap = null;
  if (
    startup &&
    hasAuthToken() &&
    getAuthSessionEpoch() === startup.sessionEpoch
  ) {
    return startup.run;
  }
  return null;
}
