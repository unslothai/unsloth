// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

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

test("hydration adopts a preference whose catalog failed, rendering English", async () => {
  await localeStore.setLocale("en");
  store.delete(localeStore.LOCALE_STORAGE_KEY);

  const result = await localeStore.setLocale("de", {
    loadMessages: () => Promise.reject(new Error("chunk 404")),
    adoptOnFailure: true,
  });

  // The preference hydration applies is already the server's stored truth, so
  // refusing to adopt it would leave the local preference disagreeing with the
  // server and the next outbound save would push the stale value back over it.
  assert.equal(result, "failed");
  assert.equal(localeStore.getLocalePreference(), "de");
  assert.equal(localeStore.getLocale(), "en");
  assert.equal(localeStore.getPendingLocalePreference(), null);
  // Not persisted: storage records choices that worked, and writing this one
  // would reproduce the failure on every later load with nothing to tell it
  // apart from a deliberate pick.
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), undefined);
});

test("a user-initiated failure is not adopted", async () => {
  await localeStore.setLocale("en");

  const result = await localeStore.setLocale("ru", {
    loadMessages: () => Promise.reject(new Error("chunk 404")),
  });

  assert.equal(result, "failed");
  assert.notEqual(localeStore.getLocalePreference(), "ru");
  assert.equal(localeStore.getLocale(), "en");
});

test("adopt-on-failure still persists a change that succeeded", async () => {
  await localeStore.setLocale("en");

  const result = await localeStore.setLocale("es", {
    loadMessages: () => Promise.resolve(),
    adoptOnFailure: true,
  });

  assert.equal(result, "applied");
  assert.equal(localeStore.getLocalePreference(), "es");
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "es");
});

test("a synchronous loader throw leaves an in-flight request pending", async () => {
  await localeStore.setLocale("en");
  let finishLoading!: () => void;
  const loading = new Promise<void>((resolve) => {
    finishLoading = resolve;
  });

  const slow = localeStore.setLocale("fr", { loadMessages: () => loading });
  assert.equal(localeStore.getPendingLocalePreference(), "fr");

  // This request never becomes the pending one, so clearing the marker on its
  // way out would blank the spinner the slow request is still relying on.
  const thrown = await localeStore.setLocale("hi", {
    loadMessages: () => {
      throw new Error("sync boom");
    },
  });

  assert.equal(thrown, "failed");
  assert.equal(localeStore.getPendingLocalePreference(), "fr");

  finishLoading();
  await slow;
});

// The real component, with only its presentation imports faked, so these assert
// against the value the shipped Select is actually given.
const { LanguageSelect } = loadWithStubs<{ LanguageSelect: () => StubElement }>(
  new URL(
    "../src/features/settings/components/language-select.tsx",
    import.meta.url,
  ),
  {
    "react/jsx-runtime": stubJsxRuntime(),
    "@/components/ui/select": {
      Select: "Select",
      SelectContent: "SelectContent",
      SelectItem: "SelectItem",
      SelectTrigger: "SelectTrigger",
      SelectValue: "SelectValue",
    },
    "@/components/ui/spinner": { Spinner: "Spinner" },
    // The store getters are what the module's hooks return, read here without a
    // renderer so the test drives the same locale state the other tests do.
    "@/i18n": {
      AUTO_LOCALE: localeStore.AUTO_LOCALE,
      LOCALES: messagesModule.LOCALES,
      isLocalePreference: localeStore.isLocalePreference,
      setLocale: localeStore.setLocale,
      useLocale: localeStore.getLocale,
      useLocalePreference: localeStore.getLocalePreference,
      usePendingLocalePreference: localeStore.getPendingLocalePreference,
      useT: () => (key: string) => key,
    },
  },
);

/** The value the language menu currently shows. */
function shownLanguage(): unknown {
  return LanguageSelect().props.value;
}

/**
 * Radix's controlled Select only calls onValueChange when the picked value differs
 * from the one it holds (useControllableState: `if (value !== prop) onChange(value)`),
 * so whatever it shows is the one language the user cannot pick.
 */
function canPick(value: string): boolean {
  return shownLanguage() !== value;
}

test("the language menu shows the language in effect after a catalog failure", async () => {
  await localeStore.setLocale("en", { loadMessages: () => undefined });

  const result = await localeStore.setLocale("de", {
    loadMessages: () => Promise.reject(new Error("chunk 404")),
    adoptOnFailure: true,
  });

  assert.equal(result, "failed");
  assert.equal(localeStore.getLocalePreference(), "de");
  assert.equal(localeStore.getLocale(), "en");
  // Naming the adopted-but-failed language here would make it the value the
  // Select already holds, and picking it again would then fire nothing, so a
  // transient chunk failure would strand the user in English for good.
  assert.equal(shownLanguage(), "en");
  assert.ok(canPick("de"));
});

test("the language menu shows the preference once it is the one in effect", async () => {
  assert.equal(
    await localeStore.setLocale("de", {
      loadMessages: () => Promise.resolve(),
    }),
    "applied",
  );

  assert.equal(localeStore.getLocale(), "de");
  assert.equal(shownLanguage(), "de");
});

test("the language menu shows auto rather than the detected locale", () => {
  assert.equal(
    localeStore.setLocale("auto", { loadMessages: () => undefined }),
    "applied",
  );

  assert.equal(localeStore.getLocalePreference(), "auto");
  assert.equal(localeStore.getLocale(), "en");
  assert.equal(shownLanguage(), "auto");
});

test("the language menu shows a pending choice while its catalog loads", async () => {
  let finishLoading!: () => void;
  const loading = new Promise<void>((resolve) => {
    finishLoading = resolve;
  });

  const selected = localeStore.setLocale("it", { loadMessages: () => loading });
  assert.equal(shownLanguage(), "it");

  finishLoading();
  assert.equal(await selected, "applied");
  assert.equal(shownLanguage(), "it");
});
