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

test("a fresh profile keeps the advanced section closed", () => {
  store.delete(ADVANCED_SETTINGS_OPEN_KEY);
  assert.equal(readAdvancedSettingsOpen(), false);
});

test("opening it once is remembered", () => {
  saveAdvancedSettingsOpen(true);
  // What the next model, quant, or reload reads.
  assert.equal(readAdvancedSettingsOpen(), true);
});

test("closing it again is remembered too", () => {
  saveAdvancedSettingsOpen(true);
  saveAdvancedSettingsOpen(false);
  assert.equal(readAdvancedSettingsOpen(), false);
});

test("an unreadable value falls back to closed", () => {
  store.set(ADVANCED_SETTINGS_OPEN_KEY, "yes");
  assert.equal(readAdvancedSettingsOpen(), false);
});

test("every mounted panel hears a toggle made on another surface", () => {
  // The sidebar copy stays mounted while collapsed, so a toggle in the picker
  // has to reach it rather than leave it on its mount-time snapshot.
  let sidebar = 0;
  let hub = 0;
  const stopSidebar = subscribeAdvancedSettingsOpen(() => {
    sidebar += 1;
  });
  const stopHub = subscribeAdvancedSettingsOpen(() => {
    hub += 1;
  });

  saveAdvancedSettingsOpen(true);
  assert.equal(sidebar, 1);
  assert.equal(hub, 1);
  assert.equal(readAdvancedSettingsOpen(), true);

  stopSidebar();
  stopHub();
});

test("an unmounted panel stops hearing them", () => {
  let calls = 0;
  const stop = subscribeAdvancedSettingsOpen(() => {
    calls += 1;
  });
  stop();
  saveAdvancedSettingsOpen(false);
  assert.equal(calls, 0);
});

test("a toggle in another tab reaches mounted panels", () => {
  let calls = 0;
  const stop = subscribeAdvancedSettingsOpen(() => {
    calls += 1;
  });

  store.set(ADVANCED_SETTINGS_OPEN_KEY, "true");
  fromAnotherTab(ADVANCED_SETTINGS_OPEN_KEY);
  assert.equal(calls, 1);
  assert.equal(readAdvancedSettingsOpen(), true);

  // A cleared profile drops this key too, so it counts.
  fromAnotherTab(null);
  assert.equal(calls, 2);

  // An unrelated key does not.
  fromAnotherTab("unsloth_model_configs");
  assert.equal(calls, 2);

  stop();
});

test("unsubscribing detaches the cross-tab listener too", () => {
  const before = storageHandlers.size;
  const noop = () => {
    // only its registration matters here
  };
  const stop = subscribeAdvancedSettingsOpen(noop);
  assert.equal(storageHandlers.size, before + 1);
  stop();
  assert.equal(storageHandlers.size, before);
});
