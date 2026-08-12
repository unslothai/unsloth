// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();
const windowListeners = new Map<string, Set<EventListener>>();
Object.assign(globalThis.window, {
  addEventListener(type: string, listener: EventListener) {
    const listeners = windowListeners.get(type) ?? new Set<EventListener>();
    listeners.add(listener);
    windowListeners.set(type, listeners);
  },
  removeEventListener(type: string, listener: EventListener) {
    windowListeners.get(type)?.delete(listener);
  },
});

function fireWindowEvent(type: string): void {
  for (const listener of windowListeners.get(type) ?? []) {
    listener(new Event(type));
  }
}

let documentLanguage = "";
Object.assign(globalThis, {
  document: {
    documentElement: {
      get lang() {
        return documentLanguage;
      },
      set lang(value: string) {
        documentLanguage = value;
      },
    },
  },
});
const navigatorState = { language: "en-US", languages: ["en-US"] };
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: navigatorState,
});

const messagesModule = await import("../src/i18n/messages.ts");
const localeStore = await import("../src/i18n/locale-store.ts");

test("only English is present before a non-English locale is requested", () => {
  assert.deepEqual(Object.keys(messagesModule.messages), ["en"]);
  assert.equal(messagesModule.translate("common.cancel"), "Cancel");
});

test("initialization waits for the saved locale catalog before committing it", async () => {
  store.set(localeStore.LOCALE_STORAGE_KEY, "de");

  const initialized = localeStore.initializeLocale();
  assert.notEqual(typeof initialized, "string");
  assert.equal(localeStore.getLocale(), "en");

  assert.equal(await initialized, "de");
  assert.equal(localeStore.getLocale(), "de");
  assert.equal(localeStore.getLocalePreference(), "de");
  assert.equal(documentLanguage, "de");
  assert.equal(messagesModule.translate("common.cancel"), "Abbrechen");
});

test("concurrent requests share one catalog load", async () => {
  const first = messagesModule.loadLocaleMessages("ko");
  const second = messagesModule.loadLocaleMessages("ko");

  assert.ok(first);
  assert.equal(second, first);
  await first;
  assert.equal(messagesModule.loadLocaleMessages("ko"), undefined);
});

test("concurrent language loads keep the latest selection", async () => {
  const first = localeStore.setLocale("fr");
  const second = localeStore.setLocale("it");
  await Promise.all([
    messagesModule.loadLocaleMessages("fr"),
    messagesModule.loadLocaleMessages("it"),
  ]);

  assert.equal(await first, "superseded");
  assert.equal(await second, "applied");

  assert.equal(localeStore.getLocale(), "it");
  assert.equal(localeStore.getLocalePreference(), "it");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "it");
  assert.equal(messagesModule.translate("common.cancel"), "Annulla");
});

test("a selection is shown as pending and persisted only after loading", async () => {
  let finishLoading!: () => void;
  const loading = new Promise<void>((resolve) => {
    finishLoading = resolve;
  });

  const selected = localeStore.setLocale("ja", {
    loadMessages: () => loading,
  });

  assert.equal(localeStore.getPendingLocalePreference(), "ja");
  assert.equal(localeStore.getLocale(), "it");
  assert.equal(localeStore.getLocalePreference(), "it");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "it");

  finishLoading();
  assert.equal(await selected, "applied");

  assert.equal(localeStore.getPendingLocalePreference(), null);
  assert.equal(localeStore.getLocale(), "ja");
  assert.equal(localeStore.getLocalePreference(), "ja");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "ja");
});

test("a failed selection keeps the active and persisted language", async () => {
  const selected = localeStore.setLocale("ko", {
    loadMessages: () => Promise.reject(new Error("catalog unavailable")),
  });

  assert.equal(localeStore.getPendingLocalePreference(), "ko");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "ja");

  assert.equal(await selected, "failed");

  assert.equal(localeStore.getPendingLocalePreference(), null);
  assert.equal(localeStore.getLocale(), "ja");
  assert.equal(localeStore.getLocalePreference(), "ja");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "ja");
});

test("cancelling a pending selection prevents a late commit", async () => {
  const controller = new AbortController();
  let finishLoading!: () => void;
  const loading = new Promise<void>((resolve) => {
    finishLoading = resolve;
  });

  const selected = localeStore.setLocale("de", {
    loadMessages: () => loading,
    signal: controller.signal,
  });
  assert.equal(localeStore.getPendingLocalePreference(), "de");

  controller.abort();
  assert.equal(localeStore.getPendingLocalePreference(), null);
  assert.equal(localeStore.getLocale(), "ja");
  assert.equal(localeStore.getLocalePreference(), "ja");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "ja");

  finishLoading();
  assert.equal(await selected, "cancelled");
  assert.equal(localeStore.getLocale(), "ja");
  assert.equal(localeStore.getLocalePreference(), "ja");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "ja");
});

test("a browser language change cannot supersede a pending explicit choice", async () => {
  assert.equal(
    localeStore.setLocale("auto", { loadMessages: () => undefined }),
    "applied",
  );
  const unsubscribe = localeStore.subscribeLocale(() => undefined);
  let finishLoading!: () => void;
  const loading = new Promise<void>((resolve) => {
    finishLoading = resolve;
  });

  const selected = localeStore.setLocale("de", {
    loadMessages: () => loading,
  });
  navigatorState.language = "fr-FR";
  navigatorState.languages = ["fr-FR"];
  fireWindowEvent("languagechange");

  assert.equal(localeStore.getPendingLocalePreference(), "de");
  finishLoading();
  assert.equal(await selected, "applied");
  unsubscribe();

  assert.equal(localeStore.getLocale(), "de");
  assert.equal(localeStore.getLocalePreference(), "de");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "de");
});

test("a browser language change refreshes a pending auto choice", async () => {
  assert.equal(
    localeStore.setLocale("it", { loadMessages: () => undefined }),
    "applied",
  );
  navigatorState.language = "de-DE";
  navigatorState.languages = ["de-DE"];
  const unsubscribe = localeStore.subscribeLocale(() => undefined);
  let finishLoading!: () => void;
  const loading = new Promise<void>((resolve) => {
    finishLoading = resolve;
  });

  const selected = localeStore.setLocale("auto", {
    loadMessages: () => loading,
  });
  navigatorState.language = "en-US";
  navigatorState.languages = ["en-US"];
  fireWindowEvent("languagechange");

  assert.equal(localeStore.getPendingLocalePreference(), null);
  assert.equal(localeStore.getLocale(), "en");
  assert.equal(localeStore.getLocalePreference(), "auto");
  finishLoading();
  assert.equal(await selected, "superseded");
  unsubscribe();

  assert.equal(localeStore.getLocale(), "en");
  assert.equal(localeStore.getLocalePreference(), "auto");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "auto");
});

test("a catalog timeout preserves the preference and finishes later", async () => {
  store.set(localeStore.LOCALE_STORAGE_KEY, "de");
  let finishLoading!: () => void;
  const loading = new Promise<void>((resolve) => {
    finishLoading = resolve;
  });

  const initialized = localeStore.initializeLocale({
    loadMessages: () => loading,
    timeoutMs: 0,
  });
  assert.notEqual(typeof initialized, "string");

  assert.equal(await initialized, "en");
  assert.equal(localeStore.getLocale(), "en");
  // Personalization sync reads this preference. The temporary English render
  // must not turn into a server-side language change.
  assert.equal(localeStore.getLocalePreference(), "de");
  assert.equal(localeStore.getPendingLocalePreference(), null);
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "de");

  finishLoading();
  await loading;
  await Promise.resolve();

  assert.equal(localeStore.getLocale(), "de");
  assert.equal(localeStore.getLocalePreference(), "de");
  assert.equal(localeStore.getPendingLocalePreference(), null);
});

test("a late initial catalog cannot replace a newer language selection", async () => {
  store.set(localeStore.LOCALE_STORAGE_KEY, "de");
  let finishLoading!: () => void;
  const loading = new Promise<void>((resolve) => {
    finishLoading = resolve;
  });

  const initialized = localeStore.initializeLocale({
    loadMessages: () => loading,
    timeoutMs: 0,
  });
  assert.notEqual(typeof initialized, "string");
  await initialized;

  await localeStore.setLocale("it", { loadMessages: () => undefined });
  finishLoading();
  await loading;
  await Promise.resolve();

  assert.equal(localeStore.getLocale(), "it");
  assert.equal(localeStore.getLocalePreference(), "it");
  assert.equal(localeStore.getPendingLocalePreference(), null);
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "it");
});

test("a failed initial catalog falls back to English", async () => {
  store.set(localeStore.LOCALE_STORAGE_KEY, "fr");

  const initialized = localeStore.initializeLocale({
    loadMessages: () => Promise.reject(new Error("catalog unavailable")),
  });
  assert.notEqual(typeof initialized, "string");

  assert.equal(await initialized, "en");
  assert.equal(localeStore.getLocale(), "en");
  assert.equal(localeStore.getLocalePreference(), "fr");
  assert.equal(localeStore.getPendingLocalePreference(), null);
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "fr");
});

test("a synchronous initial catalog failure preserves the preference", () => {
  store.set(localeStore.LOCALE_STORAGE_KEY, "ko");

  const initialized = localeStore.initializeLocale({
    loadMessages: () => {
      throw new Error("catalog unavailable");
    },
  });

  assert.equal(initialized, "en");
  assert.equal(localeStore.getLocale(), "en");
  assert.equal(localeStore.getLocalePreference(), "ko");
  assert.equal(localeStore.getPendingLocalePreference(), null);
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "ko");
});
