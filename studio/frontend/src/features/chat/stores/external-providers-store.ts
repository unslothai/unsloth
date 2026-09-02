// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import { ACCOUNT_CHANGED_EVENT } from "../../../lib/account-transition.ts";
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

// Hydrated at module load from storage that a different account's sign-in has
// just cleared, so re-read it rather than keep showing the previous account's
// provider names, base URLs and model lists until a reload.
if (typeof window !== "undefined") {
  window.addEventListener(ACCOUNT_CHANGED_EVENT, () => {
    useExternalProvidersStore.setState({
      providers: loadExternalProviders(),
      connectionsEnabled: loadConnectionsEnabled(),
    });
  });
}
