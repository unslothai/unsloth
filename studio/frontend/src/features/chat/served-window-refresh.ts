// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listProviderModels } from "./api/providers-api";
import {
  type ExternalProviderConfig,
  getExternalProviderApiKey,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  toExternalBackendProviderType,
} from "./external-providers";
import { learnServedContextWindows, servedWindowsListedFrom } from "./model-catalog";
import { SERVED_WINDOW_PROVIDERS } from "./provider-capabilities";
import { useExternalProvidersStore } from "./stores/external-providers-store";

const asked = new Set<string>();
const inFlight = new Set<string>();

/** Lists a self-hosted connection once per endpoint per page load, so a restarted server's window is re-read. */
export function refreshServedContextWindow(
  provider: ExternalProviderConfig | null | undefined,
  modelId: string | null | undefined,
): void {
  if (!provider || !modelId || provider.decisionsOnly) return;
  if (!SERVED_WINDOW_PROVIDERS.has(provider.providerType)) return;
  const key = `${provider.id}|${provider.baseUrl}`;
  // Re-list when the stored windows came from another endpoint, e.g. after editing A -> B -> A.
  if (asked.has(key) && servedWindowsListedFrom(provider.id) === provider.baseUrl) return;
  if (inFlight.has(key)) return;
  asked.add(key);
  inFlight.add(key);
  listProviderModels({
    providerType: toExternalBackendProviderType(provider.providerType),
    providerId: provider.id,
    apiKey: provider.hasApiKey ? "" : getExternalProviderApiKey(provider.id),
    baseUrl: provider.baseUrl || null,
    apiType:
      provider.providerType === LEGACY_CUSTOM_PROVIDER_TYPE ? provider.apiType : undefined,
    extraHeaders: provider.extraHeaders,
  })
    .then((models) => {
      // A Base URL edited mid-request makes this answer the old server's.
      const live = useExternalProvidersStore.getState().providers.find((p) => p.id === provider.id);
      if (live?.baseUrl === provider.baseUrl) {
        learnServedContextWindows(provider.id, models, provider.baseUrl);
      }
    })
    // An offline server keeps the fallback; the next endpoint or session asks again.
    .catch(() => asked.delete(key))
    .finally(() => inFlight.delete(key));
}
