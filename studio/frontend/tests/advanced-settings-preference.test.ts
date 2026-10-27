// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  readSrcAsync,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store, storage } = installLocalStorageFake();

const storageHandlers = new Set<(event: StorageEvent) => void>();
Object.assign(globalThis.window, {
  addEventListener: (type: string, fn: (event: StorageEvent) => void) => {
    if (type === "storage") {
      storageHandlers.add(fn);
    }
  },
  removeEventListener: (type: string, fn: (event: StorageEvent) => void) => {
    if (type === "storage") {
      storageHandlers.delete(fn);
    }
  },
});
const fromAnotherTab = (key: string | null) => {
  for (const fn of [...storageHandlers]) {
    fn({ key } as StorageEvent);
  }
};

const {
  ADVANCED_SETTINGS_OPEN_KEY,
  readAdvancedSettingsOpen,
  saveAdvancedSettingsOpen,
  subscribeAdvancedSettingsOpen,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

function mounted(): { changes: () => number; unmount: () => void } {
  let seen = 0;
  const stop = subscribeAdvancedSettingsOpen(() => {
    seen += 1;
  });
  return { changes: () => seen, unmount: stop };
}

test("an untouched profile leaves the section to the model", () => {
  // null, not false: a model with non-default advanced values may still open the section.
  assert.equal(readAdvancedSettingsOpen(), null);
});

test("opening it is remembered", () => {
  saveAdvancedSettingsOpen(true);
  assert.equal(readAdvancedSettingsOpen(), true);
  assert.equal(store.get(ADVANCED_SETTINGS_OPEN_KEY), "true");
});

test("closing it is remembered as closed, not as untouched", () => {
  saveAdvancedSettingsOpen(false);
  assert.equal(readAdvancedSettingsOpen(), false);
});

test("a value it cannot parse counts as untouched", () => {
  store.set(ADVANCED_SETTINGS_OPEN_KEY, "yes");
  assert.equal(readAdvancedSettingsOpen(), null);
  store.delete(ADVANCED_SETTINGS_OPEN_KEY);
});

test("a refused write still moves the switch", () => {
  const setItem = storage.setItem;
  storage.setItem = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    saveAdvancedSettingsOpen(true);
    assert.equal(readAdvancedSettingsOpen(), true);
    saveAdvancedSettingsOpen(false);
    assert.equal(readAdvancedSettingsOpen(), false);
  } finally {
    storage.setItem = setItem;
  }
});

test("a newer choice elsewhere takes over from a refused write", () => {
  const setItem = storage.setItem;
  storage.setItem = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    saveAdvancedSettingsOpen(true);
    assert.equal(readAdvancedSettingsOpen(), true);
  } finally {
    storage.setItem = setItem;
  }

  // Storage events are not replayed, so the next read must notice on its own.
  store.set(ADVANCED_SETTINGS_OPEN_KEY, "false");
  assert.equal(readAdvancedSettingsOpen(), false);

  store.delete(ADVANCED_SETTINGS_OPEN_KEY);
  assert.equal(readAdvancedSettingsOpen(), null);
});

test("a refused write holds while storage stays put", () => {
  store.set(ADVANCED_SETTINGS_OPEN_KEY, "false");
  const setItem = storage.setItem;
  storage.setItem = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    saveAdvancedSettingsOpen(true);
    assert.equal(readAdvancedSettingsOpen(), true);
    assert.equal(readAdvancedSettingsOpen(), true);
  } finally {
    storage.setItem = setItem;
    store.delete(ADVANCED_SETTINGS_OPEN_KEY);
  }
});

test("every mounted panel hears a toggle made on another surface", () => {
  const sidebar = mounted();
  const hub = mounted();

  saveAdvancedSettingsOpen(true);
  assert.equal(sidebar.changes(), 1);
  assert.equal(hub.changes(), 1);
  assert.equal(readAdvancedSettingsOpen(), true);

  saveAdvancedSettingsOpen(false);
  assert.equal(sidebar.changes(), 2);
  assert.equal(hub.changes(), 2);
  assert.equal(readAdvancedSettingsOpen(), false);

  sidebar.unmount();
  hub.unmount();
});

test("an unmounted panel stops hearing them", () => {
  const panel = mounted();
  panel.unmount();
  saveAdvancedSettingsOpen(true);
  assert.equal(panel.changes(), 0);
});

test("a toggle in another tab repaints mounted panels", () => {
  const panel = mounted();

  store.set(ADVANCED_SETTINGS_OPEN_KEY, "false");
  fromAnotherTab(ADVANCED_SETTINGS_OPEN_KEY);
  assert.equal(panel.changes(), 1);
  assert.equal(readAdvancedSettingsOpen(), false);

  store.delete(ADVANCED_SETTINGS_OPEN_KEY);
  fromAnotherTab(null);
  assert.equal(panel.changes(), 2);

  fromAnotherTab("unsloth_model_configs");
  assert.equal(panel.changes(), 2);

  panel.unmount();
});

test("a toggle in another tab lands even with no panel mounted to hear it", () => {
  saveAdvancedSettingsOpen(true);
  store.set(ADVANCED_SETTINGS_OPEN_KEY, "false");
  assert.equal(readAdvancedSettingsOpen(), false);

  store.delete(ADVANCED_SETTINGS_OPEN_KEY);
  assert.equal(readAdvancedSettingsOpen(), null);
});

test("the reset in Settings > General clears it", async () => {
  const source = await readSrcAsync("features/settings/tabs/general-tab.tsx");
  const start = source.indexOf("const PREFS_KEYS");
  const keys = source.slice(start, source.indexOf("];", start));
  assert.ok(
    keys.includes(`"${ADVANCED_SETTINGS_OPEN_KEY}"`),
    `${ADVANCED_SETTINGS_OPEN_KEY} missing from PREFS_KEYS`,
  );
});

test("unsubscribing detaches the cross-tab listener too", () => {
  const before = storageHandlers.size;
  const panel = mounted();
  assert.equal(storageHandlers.size, before + 1);
  panel.unmount();
  assert.equal(storageHandlers.size, before);
});
