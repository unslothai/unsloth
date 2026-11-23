// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSessionEpoch, hasAuthToken } from "@/features/auth";
import type { ExternalProviderConfig } from "@/features/chat";
import { useExternalProvidersStore } from "@/features/chat/stores/external-providers-store";
import { syncExternalProvidersFromBackend } from "@/features/chat/sync-external-providers";
import { hydrateHfTokenFromBackend } from "@/features/hub/stores/hf-token-store";
import { runCredentialBootstrap } from "./reconciliation";

let credentialBootstrapRevision = 0;

function credentialBootstrapDependencies() {
  const bootstrapRevision = ++credentialBootstrapRevision;
  const sessionEpoch = getAuthSessionEpoch();
  const isCurrent = () =>
    hasAuthToken() && getAuthSessionEpoch() === sessionEpoch &&
    bootstrapRevision === credentialBootstrapRevision;
  return {
    hydrateHfToken: hydrateHfTokenFromBackend,
    getProviders: () => useExternalProvidersStore.getState().providers,
    syncProviders: (providers: ExternalProviderConfig[]) =>
      syncExternalProvidersFromBackend(providers, isCurrent),
    setProviders: (providers: ExternalProviderConfig[]) =>
      useExternalProvidersStore.getState().setProviders(providers),
    isCurrent,
  };
}

export function bootstrapPersistedCredentials(): Promise<void> {
  return runCredentialBootstrap(credentialBootstrapDependencies());
}

let startupBootstrap: {
  sessionEpoch: number;
  run: Promise<void>;
  retry: () => Promise<void>;
  handoff?: Promise<void>;
} | null = null;
let startupBootstrapRequested = false;

export function startStartupCredentialBootstrap(): void {
  if (startupBootstrapRequested) return;
  startupBootstrapRequested = true;
  if (!hasAuthToken()) return;
  const sessionEpoch = getAuthSessionEpoch();
  const dependencies = credentialBootstrapDependencies();
  const failedTasks = new Set<"hfToken" | "providers">();
  const run = runCredentialBootstrap({
    ...dependencies,
    onFailure: (task) => failedTasks.add(task),
  });
  startupBootstrap = {
    sessionEpoch,
    run,
    retry: async () => {
      if (!dependencies.isCurrent() || !failedTasks.size) return;
      await runCredentialBootstrap({
        ...dependencies,
        hydrateHfToken: () =>
          failedTasks.has("hfToken")
            ? dependencies.hydrateHfToken()
            : Promise.resolve(),
        syncProviders: (providers) =>
          failedTasks.has("providers")
            ? dependencies.syncProviders(providers)
            : Promise.resolve(providers),
        setProviders: (providers) => {
          if (failedTasks.has("providers")) dependencies.setProviders(providers);
        },
      });
    },
  };
}

export function takeStartupCredentialBootstrap(): Promise<void> | null {
  const startup = startupBootstrap;
  if (
    startup &&
    hasAuthToken() &&
    getAuthSessionEpoch() === startup.sessionEpoch
  ) {
    startup.handoff ??= startup.run.then(startup.retry).finally(() => {
      if (startupBootstrap === startup) startupBootstrap = null;
    });
    return startup.handoff;
  }
  startupBootstrap = null;
  return null;
}
