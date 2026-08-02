// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store, storage } = installLocalStorageFake();

// The preference subscribes to cross-tab writes, so the fake window needs the
// listener pair a browser has.
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

/** Subscribed so the cross-tab handler is registered, as a mounted panel is. */
function mounted(): { changes: () => number; unmount: () => void } {
  let seen = 0;
  const stop = subscribeAdvancedSettingsOpen(() => {
    seen += 1;
  });
  return { changes: () => seen, unmount: stop };
}

test("an untouched profile leaves the section to the model", () => {
  // null, not false: a model carrying non-default advanced values may still
  // open the section for itself.
  assert.equal(readAdvancedSettingsOpen(), null);
});

test("opening it is remembered", () => {
  saveAdvancedSettingsOpen(true);
  assert.equal(readAdvancedSettingsOpen(), true);
  assert.equal(store.get(ADVANCED_SETTINGS_OPEN_KEY), "true");
});

test("closing it is remembered as closed, not as untouched", () => {
  saveAdvancedSettingsOpen(false);
  // The difference that stops a non-default model reopening it.
  assert.equal(readAdvancedSettingsOpen(), false);
});

test("a value it cannot parse counts as untouched", () => {
  const panel = mounted();
  store.set(ADVANCED_SETTINGS_OPEN_KEY, "yes");
  fromAnotherTab(ADVANCED_SETTINGS_OPEN_KEY);
  assert.equal(readAdvancedSettingsOpen(), null);
  panel.unmount();
});

test("a refused write still moves the switch", () => {
  // Storage disabled, sandboxed or full. The choice is not remembered next
  // launch, but the controls have to stay reachable this session.
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

test("every mounted panel hears a toggle made on another surface", () => {
  // The sidebar copy stays mounted while collapsed, so a toggle in the picker
  // has to reach it rather than leave it on its mount-time snapshot.
  const sidebar = mounted();
  const hub = mounted();

  saveAdvancedSettingsOpen(true);
  assert.equal(sidebar.changes(), 1);
  assert.equal(hub.changes(), 1);
  assert.equal(readAdvancedSettingsOpen(), true);

  // Closing on one surface closes it on the other, including a panel that
  // opened itself for a non-default model: an explicit choice outranks that.
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

test("a toggle in another tab reaches mounted panels", () => {
  const panel = mounted();

  store.set(ADVANCED_SETTINGS_OPEN_KEY, "false");
  fromAnotherTab(ADVANCED_SETTINGS_OPEN_KEY);
  assert.equal(panel.changes(), 1);
  assert.equal(readAdvancedSettingsOpen(), false);

  // A cleared profile drops this key too, so it counts.
  store.delete(ADVANCED_SETTINGS_OPEN_KEY);
  fromAnotherTab(null);
  assert.equal(panel.changes(), 2);
  assert.equal(readAdvancedSettingsOpen(), null);

  // An unrelated key does not.
  fromAnotherTab("unsloth_model_configs");
  assert.equal(panel.changes(), 2);

  panel.unmount();
});

test("unsubscribing detaches the cross-tab listener too", () => {
  const before = storageHandlers.size;
  const panel = mounted();
  assert.equal(storageHandlers.size, before + 1);
  panel.unmount();
  assert.equal(storageHandlers.size, before);
});
