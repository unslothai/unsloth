// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  LOW_DEVICE_SCALE_ATTRIBUTE,
  THREAD_SCROLLBAR_GUTTER_PROPERTY,
  isLowDeviceScale,
  watchLowDeviceScale,
  watchThreadScrollbarGutter,
} from "../src/app/low-device-scale.ts";

type Query = {
  dppx: number;
  listeners: Set<() => void>;
  addEventListener: (type: "change", listener: () => void) => void;
  removeEventListener: (type: "change", listener: () => void) => void;
};

function fakeWorld(initialRatio: number, initialZoom: number) {
  let ratio = initialRatio;
  let zoom = initialZoom;
  const queries: Query[] = [];
  const zoomListeners = new Set<() => void>();
  const attributes = new Set<string>();
  const source = {
    devicePixelRatio: () => ratio,
    matchResolution: (dppx: number) => {
      const query: Query = {
        dppx,
        listeners: new Set(),
        addEventListener: (_type, listener) => query.listeners.add(listener),
        removeEventListener: (_type, listener) => query.listeners.delete(listener),
      };
      queries.push(query);
      return query;
    },
  };
  const root = {
    toggleAttribute: (name: string, force: boolean) => {
      if (force) attributes.add(name);
      else attributes.delete(name);
      return force;
    },
    removeAttribute: (name: string) => {
      attributes.delete(name);
    },
  };
  return {
    source,
    root,
    flagged: () => attributes.has(LOW_DEVICE_SCALE_ATTRIBUTE),
    live: () => queries.filter((query) => query.listeners.size > 0),
    zoomListeners,
    subscribeInterfaceZoom: (listener: () => void) => {
      zoomListeners.add(listener);
      return () => zoomListeners.delete(listener);
    },
    interfaceZoom: () => zoom,
    changeRatio(next: number) {
      ratio = next;
      for (const query of queries.filter((entry) => entry.listeners.size > 0)) {
        for (const listener of [...query.listeners]) listener();
      }
    },
    changeZoom(next: number) {
      zoom = next;
      for (const listener of [...zoomListeners]) listener();
    },
  };
}

test("the device scale is the pixel ratio with WebView2's own page zoom divided out", () => {
  assert.equal(isLowDeviceScale(1, 1), true);
  assert.equal(isLowDeviceScale(1.25, 1), true);
  assert.equal(isLowDeviceScale(1.5, 1), false);
  assert.equal(isLowDeviceScale(2.25, 1.5), false);
  assert.equal(isLowDeviceScale(1.5, 1.5), true);
  assert.equal(isLowDeviceScale(1.875, 1.25), false);
  assert.equal(isLowDeviceScale(1, 0), true);
});

test("the flag follows the display and the zoom, and goes with its watcher", () => {
  const world = fakeWorld(1.5, 1);
  const stop = watchLowDeviceScale({
    source: world.source,
    interfaceZoom: world.interfaceZoom,
    subscribeInterfaceZoom: world.subscribeInterfaceZoom,
    root: world.root,
  });
  assert.equal(world.flagged(), false);

  world.changeRatio(1);
  assert.equal(world.flagged(), true, "moved to a 100% display");
  assert.equal(world.live().length, 1, "one resolution query stays armed for the ratio now in force");
  assert.equal(world.live()[0]?.dppx, 1);

  world.changeRatio(1.5);
  world.changeZoom(1.5);
  assert.equal(world.flagged(), true, "a 150% interface zoom on a 100% display still reads 1.5 dppx");

  world.changeRatio(2.25);
  assert.equal(world.flagged(), false, "back on the 150% display with the zoom kept");

  stop();
  assert.equal(world.flagged(), false);
  assert.equal(world.live().length, 0);
  assert.equal(world.zoomListeners.size, 0);
});

test("a watcher starting on a low-scale display flags it before the first change", () => {
  const world = fakeWorld(1.25, 1);
  const stop = watchLowDeviceScale({
    source: world.source,
    interfaceZoom: world.interfaceZoom,
    subscribeInterfaceZoom: world.subscribeInterfaceZoom,
    root: world.root,
  });
  assert.equal(world.flagged(), true);
  stop();
});

test("the thread gutter is the measured thin bar, measured again whenever the pixel ratio moves", () => {
  const world = fakeWorld(1, 1);
  const properties = new Map<string, string>();
  const root = {
    style: {
      setProperty: (name: string, value: string) => void properties.set(name, value),
      removeProperty: (name: string) => {
        const value = properties.get(name) ?? "";
        properties.delete(name);
        return value;
      },
    },
  };
  let thickness = 10;
  const stop = watchThreadScrollbarGutter({ source: world.source, measure: () => thickness, style: root.style });
  assert.equal(properties.get(THREAD_SCROLLBAR_GUTTER_PROPERTY), "10px");

  thickness = 8;
  world.changeRatio(1.25);
  assert.equal(properties.get(THREAD_SCROLLBAR_GUTTER_PROPERTY), "8px", "a 125% browser or desktop zoom keeps the bar's device size");
  assert.equal(world.live().length, 1);

  stop();
  assert.equal(properties.has(THREAD_SCROLLBAR_GUTTER_PROPERTY), false, "the stylesheet's default is back once the watcher goes");
  assert.equal(world.live().length, 0);
});
