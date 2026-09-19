// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  loadConnectionsEnabled,
  loadExternalProviders,
  saveConnectionsEnabled,
  saveExternalProviders,
  type ExternalProviderConfig,
} from "../external-providers";

interface ExternalProvidersState {
  providers: ExternalProviderConfig[];
  connectionsEnabled: boolean;
  setProviders: (providers: ExternalProviderConfig[]) => void;
  setConnectionsEnabled: (enabled: boolean) => void;
}

export const useExternalProvidersStore = create<ExternalProvidersState>(
  (set) => ({
    providers: loadExternalProviders(),
    connectionsEnabled: loadConnectionsEnabled(),
    setProviders: (providers) => {
      set({ providers });
      saveExternalProviders(providers);
    },
    setConnectionsEnabled: (enabled) => {
      set({ connectionsEnabled: enabled });
      saveConnectionsEnabled(enabled);
    },
  }),
);

const pendingModelUpdates = new Map<string, Promise<unknown>>();

export async function withProviderModelUpdate<T>(
  providerId: string,
  update: () => Promise<T>,
): Promise<T> {
  const previous = pendingModelUpdates.get(providerId) ?? Promise.resolve();
  const next = previous.catch(() => {}).then(update);
  pendingModelUpdates.set(providerId, next);
  try {
    return await next;
  } finally {
    if (pendingModelUpdates.get(providerId) === next) {
      pendingModelUpdates.delete(providerId);
    }
  }
}
