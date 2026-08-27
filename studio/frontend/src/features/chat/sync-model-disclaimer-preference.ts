// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  loadChatPreferences,
  migrateChatPreferences,
  updateChatPreferences,
} from "./api/chat-preferences";
import { useChatPreferencesStore } from "./stores/chat-preferences-store";

const STORAGE_KEY = "unsloth_chat_preferences";

let mutationRevision = 0;
// Preserve request order across hydration, refreshes, and writes.
let operationQueue: Promise<void> = Promise.resolve();

function enqueue(operation: () => Promise<void>): Promise<void> {
  const next = operationQueue.then(operation, operation);
  operationQueue = next.catch(() => undefined);
  return next;
}

export function readLegacyModelDisclaimer(): boolean | undefined {
  if (typeof window === "undefined") {
    return undefined;
  }
  try {
    const raw = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY) ?? "null",
    ) as {
      state?: { showModelDisclaimer?: unknown };
    } | null;
    const value = raw?.state?.showModelDisclaimer;
    return value === true ? true : undefined;
  } catch {
    return undefined;
  }
}

async function hydrate(
  migrateLegacy: boolean,
  expectedMutation: number,
): Promise<void> {
  const settings = migrateLegacy
    ? await migrateChatPreferences(readLegacyModelDisclaimer())
    : await loadChatPreferences();
  if (expectedMutation !== mutationRevision) {
    return;
  }
  useChatPreferencesStore
    .getState()
    .setShowModelDisclaimer(settings.showModelDisclaimer);
}

function enqueueHydration(migrateLegacy: boolean): Promise<void> {
  const expectedMutation = mutationRevision;
  return enqueue(() => hydrate(migrateLegacy, expectedMutation));
}

export async function hydrateModelDisclaimerPreference(): Promise<void> {
  await enqueueHydration(true);
}

export async function refreshModelDisclaimerPreference(): Promise<void> {
  await enqueueHydration(false);
}

export async function saveModelDisclaimerPreference(
  showModelDisclaimer: boolean,
): Promise<void> {
  const previous = useChatPreferencesStore.getState().showModelDisclaimer;
  const mutation = ++mutationRevision;
  useChatPreferencesStore
    .getState()
    .setShowModelDisclaimer(showModelDisclaimer);

  await enqueue(async () => {
    try {
      const saved = await updateChatPreferences(showModelDisclaimer);
      if (mutation !== mutationRevision) {
        return;
      }
      useChatPreferencesStore
        .getState()
        .setShowModelDisclaimer(saved.showModelDisclaimer);
    } catch (error) {
      if (mutation === mutationRevision) {
        try {
          await hydrate(false, mutation);
        } catch {
          useChatPreferencesStore.getState().setShowModelDisclaimer(previous);
        }
      }
      throw error;
    }
  });
}
