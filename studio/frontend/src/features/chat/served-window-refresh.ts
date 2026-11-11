// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listProviderModels } from "./api/providers-api";
import {
  type ExternalProviderConfig,
  getExternalProviderApiKey,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  toExternalBackendProviderType,
} from "./external-providers";
import { learnServedContextWindows, servedContextWindow } from "./model-catalog";
import { SERVED_WINDOW_PROVIDERS } from "./provider-capabilities";

const asked = new Set<string>();

/** Lists a self-hosted connection's models once per endpoint so a saved connection learns its window too. */
export function refreshServedContextWindow(
  provider: ExternalProviderConfig | null | undefined,
  modelId: string | null | undefined,
): void {
  if (!provider || !modelId || provider.decisionsOnly) return;
  if (!SERVED_WINDOW_PROVIDERS.has(provider.providerType)) return;
  if (servedContextWindow(provider.id, modelId) != null) return;
  const key = `${provider.id}|${provider.baseUrl}`;
  if (asked.has(key)) return;
  asked.add(key);
  listProviderModels({
    providerType: toExternalBackendProviderType(provider.providerType),
    providerId: provider.id,
    apiKey: provider.hasApiKey ? "" : getExternalProviderApiKey(provider.id),
    baseUrl: provider.baseUrl || null,
    apiType:
      provider.providerType === LEGACY_CUSTOM_PROVIDER_TYPE ? provider.apiType : undefined,
    extraHeaders: provider.extraHeaders,
  })
    .then((models) => learnServedContextWindows(provider.id, models))
    // An offline server keeps the fallback; the next endpoint or session asks again.
    .catch(() => asked.delete(key));
}
