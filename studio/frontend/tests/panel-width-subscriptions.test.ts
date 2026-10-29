// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { PanelWidthStore } from "../src/hooks/use-panel-width.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

function setup() {
  const fake = installLocalStorageFake();
  window.innerWidth = 1000;
  let scale = 1;
  const scaleListeners = new Set<() => void>();
  let cleanups: (() => void)[] = [];
  let values: unknown[] = [];
  const { createPanelWidthStore } = loadWithStubs<{
    createPanelWidthStore: (options: {
      key: string;
      min: number;
      max: number;
      fallback: number;
    }) => PanelWidthStore;
  }>(new URL("../src/hooks/use-panel-width.ts", import.meta.url), {
    react: {
      useCallback: (callback: unknown) => callback,
      useSyncExternalStore: (subscribe: (cb: () => void) => () => void, read: () => unknown) => {
        const target = values;
        const index = target.length;
        target.push(read());
        cleanups.push(subscribe(() => { target[index] = read(); }));
        return read();
      },
    },
    "../lib/layout-scale.ts": {
      layoutScale: () => scale,
      subscribeLayoutScale: (callback: () => void) => {
        scaleListeners.add(callback);
        return () => scaleListeners.delete(callback);
      },
    },
  });
  const store = createPanelWidthStore({ key: "panel", min: 200, max: 600, fallback: 480 });
  return {
    ...fake,
    mount() {
      cleanups = [];
      values = [];
      const actions = store.useWidth();
      const unmount = cleanups;
      return { actions, values, unmount: () => unmount.forEach((stop) => stop()) };
    },
    setScale(next: number) {
      scale = next;
      for (const callback of scaleListeners) callback();
    },
  };
}

test("unmounting one panel consumer preserves resizing and scaling for the others", () => {
  const app = setup();
  const titlebar = app.mount();
  const sidebar = app.mount();
  titlebar.unmount();
  window.innerWidth = 800;
  assert.equal(app.fireWindowEvent("resize", {}), 1);
  assert.deepEqual(sidebar.values, [320, 320, 480, 1]);
  app.setScale(2);
  assert.deepEqual(sidebar.values, [200, 200, 480, 2]);
  sidebar.unmount();
  assert.equal(app.fireWindowEvent("resize", {}), 0);
  window.innerWidth = 1500;
  app.setScale(1);
  const reopened = app.mount();
  assert.equal(reopened.actions.width, 480);
  assert.equal(reopened.actions.max, 600);
  reopened.unmount();
});

test("cross-window width updates notify every consumer and preserve the uncapped preference", () => {
  const app = setup();
  const first = app.mount();
  const second = app.mount();
  app.storage.setItem("panel", "560");
  assert.equal(app.fireWindowEvent("storage", { key: "panel" }), 1);
  assert.deepEqual(first.values, [400, 400, 560, 1]);
  assert.deepEqual(second.values, first.values);
  window.innerWidth = 1500;
  app.fireWindowEvent("resize", {});
  assert.deepEqual(second.values, [560, 600, 560, 1]);
  first.unmount();
  app.storage.removeItem("panel");
  app.fireWindowEvent("storage", { key: null });
  assert.deepEqual(second.values, [480, 600, 480, 1]);
  second.unmount();
  assert.equal(app.fireWindowEvent("storage", { key: "panel" }), 0);
});
