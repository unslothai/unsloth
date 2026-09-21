// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSessionEpoch, hasAuthToken } from "@/features/auth";
import {
  listProviderConfigs,
  listProviderModels,
  testProviderConnection,
  updateProviderConfig,
} from "./api/providers-api";
import {
  EXTERNAL_PROVIDERS_KEY,
  getExternalProviderApiKey,
  loadExternalProviders,
  type ExternalProviderConfig,
} from "./external-providers";
import {
  useExternalProvidersStore,
  withProviderModelUpdate,
} from "./stores/external-providers-store";

interface ConnectionState {
  baseUrl: string;
  hasApiKey: boolean;
  connected: boolean;
  busy: boolean;
}

export function startLlamaCppAutoReload(intervalMs = 10_000): () => void {
  const connections = new Map<string, ConnectionState>();
  const sessionEpoch = getAuthSessionEpoch();
  let stopped = false;
  const authenticated = () =>
    !stopped && hasAuthToken() && getAuthSessionEpoch() === sessionEpoch;
  const eligible = (provider: ExternalProviderConfig) =>
    provider.providerType === "llama_cpp" && provider.autoReloadModels === true;

  function current(id: string, connection: ConnectionState) {
    const state = useExternalProvidersStore.getState();
    if (
      !authenticated() ||
      !state.connectionsEnabled ||
      connections.get(id) !== connection
    )
      return;
    return state.providers.find(
      (provider) =>
        provider.id === id &&
        eligible(provider) &&
        provider.baseUrl === connection.baseUrl &&
        (provider.hasApiKey === true) === connection.hasApiKey,
    );
  }

  async function poll(
    provider: ExternalProviderConfig,
    connection: ConnectionState,
  ) {
    if (connection.busy) return;
    connection.busy = true;
    try {
      const payload = {
        providerType: "llama_cpp",
        providerId: provider.id,
        baseUrl: provider.baseUrl || null,
        apiKey: provider.hasApiKey
          ? ""
          : getExternalProviderApiKey(provider.id),
      };
      const result = await testProviderConnection(payload);
      if (!current(provider.id, connection)) return;
      if (!result.success) {
        connection.connected = false;
        return;
      }
      if (connection.connected) return;
      const listed = await listProviderModels(payload);
      await withProviderModelUpdate(provider.id, async () => {
        if (!current(provider.id, connection)) return;
        const saved = (await listProviderConfigs()).find(
          (item) => item.id === provider.id,
        );
        const latest = current(provider.id, connection);
        if (!latest || !saved || saved.base_url !== latest.baseUrl) return;
        const availableModels = [
          ...new Set(listed.map((model) => model.id.trim()).filter(Boolean)),
        ];
        // an empty catalog during server startup must not erase the last working selection.
        if (availableModels.length === 0) return;
        const savedModels = saved.models ?? latest.models;
        const savedCatalog =
          saved.available_models ?? latest.availableModels ?? savedModels;
        const previousCatalog = new Set(savedCatalog);
        const selected = new Set(savedModels);
        const manualModels = savedModels.filter(
          (id) => !previousCatalog.has(id),
        );
        const selectedModels = [
          ...new Set([
            ...manualModels,
            ...availableModels.filter(
              (id) => selected.has(id) || !previousCatalog.has(id),
            ),
          ]),
        ];
        const models =
          selectedModels.length > 0 ? selectedModels : availableModels;
        const previousModels = JSON.stringify([
          latest.models,
          latest.availableModels,
        ]);
        const nextModels = JSON.stringify([models, availableModels]);
        if (JSON.stringify([savedModels, savedCatalog]) !== nextModels) {
          await updateProviderConfig(provider.id, { models, availableModels });
        }
        if (previousModels !== nextModels) {
          const afterSave = current(provider.id, connection);
          if (
            !afterSave ||
            JSON.stringify([afterSave.models, afterSave.availableModels]) !==
              previousModels
          )
            return;
          const state = useExternalProvidersStore.getState();
          state.setProviders(
            state.providers.map((item) =>
              item.id === provider.id
                ? { ...item, models, availableModels }
                : item,
            ),
          );
        }
        connection.connected = true;
      });
    } catch {
      // retry on the next probe, retaining the last successful catalog through outages.
      connection.connected = false;
    } finally {
      connection.busy = false;
    }
  }

  function reconcile() {
    const state = useExternalProvidersStore.getState();
    const providers =
      authenticated() && state.connectionsEnabled
        ? state.providers.filter(eligible)
        : [];
    for (const [id, connection] of connections) {
      if (
        !providers.some(
          (provider) =>
            provider.id === id &&
            provider.baseUrl === connection.baseUrl &&
            (provider.hasApiKey === true) === connection.hasApiKey,
        )
      ) {
        connections.delete(id);
      }
    }
    for (const provider of providers) {
      if (connections.has(provider.id)) continue;
      const connection = {
        baseUrl: provider.baseUrl,
        hasApiKey: provider.hasApiKey === true,
        connected: false,
        busy: false,
      };
      connections.set(provider.id, connection);
      void poll(provider, connection);
    }
  }

  function syncPreference(event: StorageEvent) {
    if (
      !authenticated() ||
      event.storageArea !== localStorage ||
      (event.key !== null && event.key !== EXTERNAL_PROVIDERS_KEY)
    ) return;
    const saved = new Map(
      loadExternalProviders().map((provider) => [provider.id, provider]),
    );
    const state = useExternalProvidersStore.getState();
    const providers = state.providers.map((provider) => {
      if (provider.providerType !== "llama_cpp") return provider;
      const savedProvider = saved.get(provider.id);
      const autoReloadModels = savedProvider?.autoReloadModels === true;
      const baseUrl = savedProvider?.baseUrl ?? provider.baseUrl;
      const hasApiKey = savedProvider?.hasApiKey === true;
      return provider.autoReloadModels === autoReloadModels &&
        provider.baseUrl === baseUrl &&
        provider.hasApiKey === hasApiKey
        ? provider
        : { ...provider, autoReloadModels, baseUrl, hasApiKey };
    });
    if (providers.some((provider, index) => provider !== state.providers[index])) {
      useExternalProvidersStore.setState({ providers });
    }
  }

  window.addEventListener("storage", syncPreference);
  const unsubscribe = useExternalProvidersStore.subscribe(reconcile);
  reconcile();
  const timer = setInterval(() => {
    reconcile();
    for (const [id, connection] of connections) {
      const provider = current(id, connection);
      if (provider) void poll(provider, connection);
    }
  }, intervalMs);
  return () => {
    stopped = true;
    clearInterval(timer);
    unsubscribe();
    window.removeEventListener("storage", syncPreference);
    connections.clear();
  };
}
