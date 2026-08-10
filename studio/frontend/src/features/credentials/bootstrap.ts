// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken } from "@/features/auth";

import { setLegacyProviderCredentialAccess } from "@/features/chat/external-providers";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { syncExternalProvidersFromBackend } from "@/features/chat/sync-external-providers";
import {
  hydrateHfTokenFromBackend,
  setLegacyHfCredentialAccess,
} from "@/features/hub/stores/hf-token-store";
import {
  authSubjectFromJwt,
  legacyCredentialOwnerAction,
} from "./migration-owner";
import { runCredentialBootstrap } from "./reconciliation";


const LEGACY_CREDENTIAL_OWNER_KEY = "unsloth_legacy_credential_owner";

function prepareLegacyCredentialMigration(sessionToken: string): boolean {
  if (typeof window === "undefined") return false;
  const currentOwner = authSubjectFromJwt(sessionToken);
  if (!currentOwner) return false;
  try {
    const storedOwner = window.localStorage.getItem(
      LEGACY_CREDENTIAL_OWNER_KEY,
    );
    const action = legacyCredentialOwnerAction(storedOwner, currentOwner);
    if (action === "claim") {
      window.localStorage.setItem(LEGACY_CREDENTIAL_OWNER_KEY, currentOwner);
    }
    return action !== "ignore";
  } catch {
    // Preserve browser-wide migration input but never expose it without ownership.
    return false;
  }
}


export function bootstrapPersistedCredentials(): Promise<void> {
  const sessionToken = getAuthToken();
  const allowLegacyMigration = sessionToken
    ? prepareLegacyCredentialMigration(sessionToken)
    : false;
  setLegacyProviderCredentialAccess(allowLegacyMigration);
  setLegacyHfCredentialAccess(allowLegacyMigration);
  const store = useExternalProvidersStore.getState();
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
