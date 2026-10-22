// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { mock } from "node:test";
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

function fireStorageEvent(key: string | null, newValue: string | null): void {
  for (const listener of windowListeners.get("storage") ?? []) {
    listener({ key, newValue, storageArea: null } as unknown as Event);
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
  // The temporary English render must not become a server-side language change.
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

  // Hydration's preference is the server's truth; refusing it would push a stale value back.
  assert.equal(result, "failed");
  assert.equal(localeStore.getLocalePreference(), "de");
  assert.equal(localeStore.getLocale(), "en");
  assert.equal(localeStore.getPendingLocalePreference(), null);
  // Not persisted: storage records only choices that worked.
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

  // This request never becomes pending, so it must not clear the slow request's spinner marker.
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

test("a cross-tab language change is adopted even when its catalog fails", async () => {
  await localeStore.setLocale("en", { loadMessages: () => undefined });
  const unsubscribe = localeStore.subscribeLocale(() => undefined);
  let failLoad!: (error: Error) => void;
  const load = messagesModule.loadLocaleMessages(
    "ru",
    () =>
      new Promise((_, reject) => {
        failLoad = reject;
      }),
  );
  load?.catch(() => undefined);

  try {
    store.set(localeStore.LOCALE_STORAGE_KEY, "ru");
    fireStorageEvent(localeStore.LOCALE_STORAGE_KEY, "ru");
    failLoad(new Error("chunk 404"));
    await load?.catch(() => undefined);
    await new Promise((resolve) => setImmediate(resolve));

    // Keep the user's choice even though only English renders, or the next save uploads a stale one.
    assert.equal(localeStore.getLocalePreference(), "ru");
    assert.equal(localeStore.getLocale(), "en");
    assert.equal(localeStore.getPendingLocalePreference(), null);
    assert.equal(localeStore.getLocaleCatalogFailed(), true);
    assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "ru");
  } finally {
    unsubscribe();
    messagesModule.forgetLocaleLoad("ru");
  }
});

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
    "@/i18n": {
      AUTO_LOCALE: localeStore.AUTO_LOCALE,
      LOCALES: messagesModule.LOCALES,
      isLocalePreference: localeStore.isLocalePreference,
      setLocale: localeStore.setLocale,
      useLocale: localeStore.getLocale,
      useLocaleCatalogFailed: localeStore.getLocaleCatalogFailed,
      useLocalePreference: localeStore.getLocalePreference,
      usePendingLocalePreference: localeStore.getPendingLocalePreference,
      useT: () => (key: string) => key,
    },
  },
);

function shownLanguage(): unknown {
  return LanguageSelect().props.value;
}

/** Radix's controlled Select only fires onValueChange for a value differing from the shown one. */
function canPick(value: string): boolean {
  return shownLanguage() !== value;
}

function findStub(node: unknown, type: string): StubElement | null {
  if (Array.isArray(node)) {
    for (const child of node) {
      const found = findStub(child, type);
      if (found) return found;
    }
    return null;
  }
  if (node === null || typeof node !== "object") return null;
  const element = node as StubElement;
  if (element.type === type) return element;
  return findStub(element.props?.children, type);
}

function shownLabel(): unknown {
  return findStub(LanguageSelect(), "SelectValue")?.props.placeholder;
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
  // Naming the failed language would make re-picking it fire nothing, stranding the user in English.
  assert.equal(shownLanguage(), "");
  assert.ok(canPick("de"));
  assert.ok(canPick("en"));
  assert.equal(shownLabel(), "English");
});

test("accepting the fallback after a catalog failure is a real choice", async () => {
  await localeStore.setLocale("en", { loadMessages: () => undefined });
  store.delete(localeStore.LOCALE_STORAGE_KEY);

  assert.equal(
    await localeStore.setLocale("de", {
      loadMessages: () => Promise.reject(new Error("chunk 404")),
      adoptOnFailure: true,
    }),
    "failed",
  );
  assert.equal(localeStore.getLocalePreference(), "de");

  assert.ok(canPick("en"));
  assert.equal(
    await localeStore.setLocale("en", { loadMessages: () => undefined }),
    "applied",
  );
  assert.equal(localeStore.getLocalePreference(), "en");
  assert.equal(localeStore.getLocaleCatalogFailed(), false);
  assert.equal(store.get(localeStore.LOCALE_STORAGE_KEY), "en");
  assert.equal(shownLanguage(), "en");
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

test("auto detection stays retryable when its detected catalog fails", async () => {
  await localeStore.setLocale("auto", { loadMessages: () => undefined });

  navigatorState.language = "de-DE";
  navigatorState.languages = ["de-DE"];
  try {
    // Hydration adopts the stored preference even when its catalog never arrives.
    const result = await localeStore.setLocale("auto", {
      loadMessages: () => Promise.reject(new Error("chunk 404")),
      adoptOnFailure: true,
    });

    assert.equal(result, "failed");
    assert.equal(localeStore.getLocalePreference(), "auto");
    assert.equal(localeStore.getLocale(), "en");

    // Naming auto or English would make re-picking that value fire nothing, blocking a retry.
    assert.equal(shownLanguage(), "");
    assert.equal(shownLabel(), "English");
    assert.ok(canPick("auto"));
    assert.ok(canPick("en"));

    assert.equal(
      await localeStore.setLocale("auto", {
        loadMessages: () => Promise.resolve(),
      }),
      "applied",
    );
    assert.equal(localeStore.getLocale(), "de");
    assert.equal(shownLanguage(), "auto");
  } finally {
    navigatorState.language = "en-US";
    navigatorState.languages = ["en-US"];
  }
});

test("a refreshed auto whose new catalog fails is still retryable", async () => {
  await localeStore.setLocale("auto", { loadMessages: () => undefined });
  assert.equal(localeStore.getLocale(), "en");

  navigatorState.language = "de-DE";
  navigatorState.languages = ["de-DE"];
  try {
    const result = await localeStore.setLocale("auto", {
      loadMessages: () => Promise.reject(new Error("chunk 404")),
    });

    assert.equal(result, "failed");
    assert.equal(localeStore.getLocalePreference(), "auto");
    assert.equal(localeStore.getLocale(), "en");
    assert.equal(shownLanguage(), "");
    assert.equal(shownLabel(), "English");
    assert.ok(canPick("auto"));
    assert.ok(canPick("en"));
  } finally {
    navigatorState.language = "en-US";
    navigatorState.languages = ["en-US"];
  }
});

test("a rejected pick leaves a working auto preference named", async () => {
  assert.equal(
    localeStore.setLocale("auto", { loadMessages: () => undefined }),
    "applied",
  );

  assert.equal(
    await localeStore.setLocale("ru", {
      loadMessages: () => Promise.reject(new Error("chunk 404")),
    }),
    "failed",
  );
  assert.equal(localeStore.getLocalePreference(), "auto");
  assert.equal(localeStore.getLocale(), "en");
  assert.equal(shownLanguage(), "auto");
});

function settledWithin(request: unknown): Promise<unknown> {
  return Promise.race([
    request,
    new Promise((resolve) => setTimeout(() => resolve("pending"), 200)),
  ]);
}

test("a catalog that never settles does not hold the caller forever", async () => {
  // A stalled CDN, proxy or service worker never settles, so the store's own timeout must wake hydration.
  const outcome = await settledWithin(
    localeStore.setLocale("hi", {
      loadMessages: () => new Promise<void>(() => {}),
      adoptOnFailure: true,
      timeoutMs: 5,
    }),
  );

  assert.equal(outcome, "failed");
  assert.equal(localeStore.getLocalePreference(), "hi");
  assert.equal(localeStore.getLocale(), "en");
  assert.equal(localeStore.getPendingLocalePreference(), null);
  assert.equal(localeStore.getLocaleCatalogFailed(), true);
});

test("a catalog that arrives after the timeout still commits", async () => {
  let finishLoading!: () => void;
  const loading = new Promise<void>((resolve) => {
    finishLoading = resolve;
  });

  assert.equal(
    await settledWithin(
      localeStore.setLocale("ar", {
        loadMessages: () => loading,
        adoptOnFailure: true,
        timeoutMs: 5,
      }),
    ),
    "failed",
  );
  assert.equal(localeStore.getLocale(), "en");

  finishLoading();
  await loading;
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(localeStore.getLocale(), "ar");
  assert.equal(localeStore.getLocalePreference(), "ar");
  assert.equal(localeStore.getLocaleCatalogFailed(), false);
});

test("a pick that names no bound of its own is still bounded", async () => {
  await localeStore.setLocale("en", { loadMessages: () => undefined });

  // The menu passes no timeoutMs, and a stalled load stays in the in-flight map for re-picks.
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const selected = localeStore.setLocale("ko", {
      loadMessages: () => new Promise<void>(() => {}),
    });
    assert.equal(localeStore.getPendingLocalePreference(), "ko");

    mock.timers.tick(localeStore.LOCALE_SELECTION_TIMEOUT_MS);

    assert.equal(localeStore.getPendingLocalePreference(), null);
    assert.equal(await selected, "failed");
  } finally {
    mock.timers.reset();
  }

  assert.equal(localeStore.getLocale(), "en");
  assert.equal(localeStore.getLocalePreference(), "en");
  assert.equal(localeStore.getLocaleCatalogFailed(), false);
});

test("a timed out catalog is evicted so the next pick asks for it again", async () => {
  await localeStore.setLocale("en", { loadMessages: () => undefined });

  let requests = 0;
  const stalled = () => {
    requests += 1;
    return new Promise<unknown>(() => {});
  };
  const loadMessages = (
    locale: Parameters<typeof messagesModule.loadLocaleMessages>[0],
  ) => messagesModule.loadLocaleMessages(locale, stalled);

  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const first = localeStore.setLocale("pt-BR", { loadMessages });
    mock.timers.tick(localeStore.LOCALE_SELECTION_TIMEOUT_MS);
    assert.equal(await first, "failed");

    const retry = localeStore.setLocale("pt-BR", { loadMessages });
    mock.timers.tick(localeStore.LOCALE_SELECTION_TIMEOUT_MS);
    assert.equal(await retry, "failed");
  } finally {
    mock.timers.reset();
    messagesModule.forgetLocaleLoad("pt-BR");
  }

  assert.equal(
    requests,
    2,
    "the retry was handed the timed out load instead of requesting the catalog again",
  );
});

test("a timed out startup catalog is evicted so the first pick asks for it again", async () => {
  store.set(localeStore.LOCALE_STORAGE_KEY, "hi");

  let requests = 0;
  const stalled = () => {
    requests += 1;
    return new Promise<unknown>(() => {});
  };
  const loadMessages = (
    locale: Parameters<typeof messagesModule.loadLocaleMessages>[0],
  ) => messagesModule.loadLocaleMessages(locale, stalled);

  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const initialized = localeStore.initializeLocale({ loadMessages });
    mock.timers.tick(localeStore.LOCALE_INITIALIZATION_TIMEOUT_MS);
    assert.equal(await initialized, "en");

    const picked = localeStore.setLocale("hi", { loadMessages });
    mock.timers.tick(localeStore.LOCALE_SELECTION_TIMEOUT_MS);
    assert.equal(await picked, "failed");
  } finally {
    mock.timers.reset();
    messagesModule.forgetLocaleLoad("hi");
  }

  assert.equal(
    requests,
    2,
    "the pick was handed the timed out startup load instead of requesting the catalog again",
  );
});
