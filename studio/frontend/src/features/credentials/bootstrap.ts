// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken } from "@/features/auth";

import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { syncExternalProvidersFromBackend } from "@/features/chat/sync-external-providers";
import { hydrateHfTokenFromBackend } from "@/features/hub/stores/hf-token-store";
import { runCredentialBootstrap } from "./reconciliation";


export function bootstrapPersistedCredentials(): Promise<void> {
  const store = useExternalProvidersStore.getState();

  const sessionToken = getAuthToken();
  const isCurrent = () => sessionToken !== null && getAuthToken() === sessionToken;
  return runCredentialBootstrap({
    hydrateHfToken: hydrateHfTokenFromBackend,
    getProviders: () => store.providers,
    syncProviders: (providers) =>
      syncExternalProvidersFromBackend(providers, isCurrent),
    setProviders: (providers) => store.setProviders(providers),

    isCurrent,
  });
}
