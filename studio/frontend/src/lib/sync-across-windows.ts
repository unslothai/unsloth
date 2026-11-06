// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type PersistedStore = { persist: { getOptions: () => { name?: string }; rehydrate: () => unknown } };

export function syncAcrossWindows(store: PersistedStore): void {
  if (typeof window === "undefined") return;
  window.addEventListener("storage", (event) => {
    if (event.key === store.persist.getOptions().name) void store.persist.rehydrate();
  });
}
