// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  listProviderConfigs,
  listProviderModels,
  updateProviderConfig,
} from "./api/providers-api";
import {
  getExternalProviderApiKey,
  type ExternalProviderConfig,
} from "./external-providers";
import { useExternalProvidersStore } from "./stores/external-providers-store";

/** Keeps manual IDs and the user's picks, drops IDs the server no longer lists, enables new ones. */
export function mergeReloadedModels(
  models: readonly string[],
  previousCatalog: readonly string[],
  catalog: readonly string[],
): string[] {
  const known = new Set(previousCatalog);
  const live = new Set(catalog);
  const selected = new Set(models);
  const next = new Set([
    ...models.filter((id) => !known.has(id) && !live.has(id)),
    ...catalog.filter((id) => selected.has(id) || !known.has(id)),
  ]);
  return next.size > 0 ? [...next] : [...catalog];
}

const sameList = (a: readonly string[], b: readonly string[]) =>
  a.length === b.length && a.every((id, i) => id === b[i]);

function autoReloadConnections(): ExternalProviderConfig[] {
  const { providers, connectionsEnabled } = useExternalProvidersStore.getState();
  if (!connectionsEnabled) return [];
  return providers.filter(
    (p) => p.providerType === "llama_cpp" && p.autoReloadModels === true,
  );
}

export function startLlamaCppAutoReload(intervalMs = 10_000): () => void {
  // Connection id -> endpoint it was last reloaded at. Absent: offline or not reloaded yet.
  const online = new Map<string, string>();
  const inFlight = new Set<string>();
  let stopped = false;
  const endpoint = (p: ExternalProviderConfig) => `${p.baseUrl}|${p.hasApiKey === true}`;

  async function probe(provider: ExternalProviderConfig) {
    const key = endpoint(provider);
    inFlight.add(provider.id);
    try {
      const listed = await listProviderModels({
        providerType: "llama_cpp",
        providerId: provider.id,
        baseUrl: provider.baseUrl || null,
        apiKey: provider.hasApiKey ? "" : getExternalProviderApiKey(provider.id),
      });
      if (stopped || online.get(provider.id) === key) return;
      const catalog = [...new Set(listed.map((m) => m.id.trim()).filter(Boolean))];
      // A server still starting can list nothing: keep the last good selection.
      if (catalog.length === 0) return;
      // Merge against the saved row, not this tab's copy, so tabs agree on what the user picked.
      const saved = (await listProviderConfigs()).find((c) => c.id === provider.id);
      const latest = autoReloadConnections().find((p) => p.id === provider.id);
      if (stopped || !saved || !latest || endpoint(latest) !== key) return;
      const hasSaved = (saved.available_models?.length ?? 0) > 0;
      const previousModels = hasSaved ? (saved.models ?? []) : latest.models;
      const previousCatalog = hasSaved
        ? (saved.available_models ?? [])
        : (latest.availableModels ?? []);
      const models = mergeReloadedModels(previousModels, previousCatalog, catalog);
      if (!sameList(models, previousModels) || !sameList(catalog, previousCatalog)) {
        await updateProviderConfig(provider.id, { models, availableModels: catalog });
      }
      if (stopped) return;
      const { providers, setProviders } = useExternalProvidersStore.getState();
      const current = providers.find((p) => p.id === provider.id);
      // A manual save landed meanwhile: leave it, and re-merge on the next probe.
      if (current?.models !== latest.models) return;
      if (
        !sameList(models, current.models) ||
        !sameList(catalog, current.availableModels ?? [])
      ) {
        setProviders(
          providers.map((p) =>
            p.id === provider.id ? { ...p, models, availableModels: catalog } : p,
          ),
        );
      }
      online.set(provider.id, key);
    } catch {
      online.delete(provider.id);
    } finally {
      inFlight.delete(provider.id);
    }
  }

  function probeAll(onlyNew: boolean) {
    const connections = autoReloadConnections();
    for (const id of online.keys()) {
      if (!connections.some((p) => p.id === id)) online.delete(id);
    }
    for (const p of connections) {
      if (inFlight.has(p.id) || (onlyNew && online.has(p.id))) continue;
      void probe(p);
    }
  }

  const unsubscribe = useExternalProvidersStore.subscribe(() => probeAll(true));
  const timer = setInterval(() => probeAll(false), intervalMs);
  probeAll(false);
  return () => {
    stopped = true;
    clearInterval(timer);
    unsubscribe();
  };
}
