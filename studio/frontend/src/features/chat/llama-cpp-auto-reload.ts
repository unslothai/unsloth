// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthSessionEpoch, hasAuthToken } from "@/features/auth";
import {
  listProviderModels,
  testProviderConnection,
  updateProviderConfig,
} from "./api/providers-api";
import {
  getExternalProviderApiKey,
  type ExternalProviderConfig,
} from "./external-providers";
import {
  useExternalProvidersStore,
  withProviderModelUpdate,
} from "./stores/external-providers-store";

interface ConnectionState {
  baseUrl: string;
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
        provider.baseUrl === connection.baseUrl,
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
        const latest = current(provider.id, connection);
        if (!latest) return;
        const availableModels = [
          ...new Set(listed.map((model) => model.id.trim()).filter(Boolean)),
        ];
        // an empty catalog during server startup must not erase the last working selection.
        if (availableModels.length === 0) return;
        const previousCatalog = new Set(
          latest.availableModels ?? latest.models,
        );
        const selected = new Set(latest.models);
        const manualModels = latest.models.filter(
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
        const unchanged =
          previousModels === JSON.stringify([models, availableModels]);
        if (!unchanged) {
          await updateProviderConfig(provider.id, { models, availableModels });
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
            provider.id === id && provider.baseUrl === connection.baseUrl,
        )
      ) {
        connections.delete(id);
      }
    }
    for (const provider of providers) {
      if (connections.has(provider.id)) continue;
      const connection = {
        baseUrl: provider.baseUrl,
        connected: false,
        busy: false,
      };
      connections.set(provider.id, connection);
      void poll(provider, connection);
    }
  }

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
    connections.clear();
  };
}
