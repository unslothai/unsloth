// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";

import { calculateWindowSizeBounds } from "../src/app/window-layout.ts";
import { observeDevicePixelRatio } from "../src/app/window-layout-lifecycle.ts";
import {
  getAppliedInterfaceZoom,
  setAppliedInterfaceZoom,
  subscribeAppliedInterfaceZoom,
} from "../src/features/settings/lib/interface-scale-runtime.ts";
import { readSrc } from "./helpers/kit.ts";

const provider = readSrc("app/provider.tsx");
const ratioFunction = provider.match(
  /function logicalPerCssPx\([\s\S]*?\n\}/,
)?.[0];
assert.ok(ratioFunction);
const windowStub = { devicePixelRatio: 2 };
const logicalPerCssPx = new Function(
  "window",
  "getAppliedInterfaceZoom",
  `${stripTypeScriptTypes(ratioFunction)}\nreturn logicalPerCssPx;`,
)(windowStub, getAppliedInterfaceZoom) as (monitorScale: number) => number;

Object.defineProperty(globalThis, "document", {
  configurable: true,
  value: { documentElement: { style: { setProperty: () => undefined } } },
});

const bounds = () =>
  calculateWindowSizeBounds({ width: 1920, height: 1080 }, logicalPerCssPx(2))
    .minimum;

test("startup constraints account for page zoom without counting Windows zoom twice", () => {
  try {
    setAppliedInterfaceZoom(2);
    assert.deepEqual(bounds(), { width: 920, height: 960 });
    windowStub.devicePixelRatio = 4;
    assert.deepEqual(bounds(), { width: 920, height: 960 });
    windowStub.devicePixelRatio = 6;
    assert.equal(logicalPerCssPx(2), 3);
  } finally {
    windowStub.devicePixelRatio = 2;
    setAppliedInterfaceZoom(1);
  }
});

test("applied zoom refreshes live constraints with unchanged DPR and stops on unmount", () => {
  const observerIndex = provider.indexOf(
    "observeDevicePixelRatio(ratioSource,",
  );
  assert.ok(observerIndex >= 0);
  const start = provider.lastIndexOf("  useEffect(() => {", observerIndex);
  const end = provider.indexOf("\n  }, []);", observerIndex);
  assert.ok(start >= 0 && end > start);
  const effect = provider.slice(start, end + "\n  }, []);".length);
  const minimums: Array<{ width: number; height: number }> = [];
  const mode = { current: "app" };
  let dispose = () => {};
  const scope = {
    isTauri: true,
    useEffect: (setup: () => () => void) => {
      dispose = setup();
    },
    windowPixelRatioSource: () => ({
      devicePixelRatio: () => windowStub.devicePixelRatio,
      matchResolution: () => null,
    }),
    observeDevicePixelRatio,
    subscribeAppliedInterfaceZoom,
    appliedWindowModeRef: mode,
    windowLayoutGenerationRef: { current: 1 },
    reapplyWindowSizeConstraints: async (isCurrent: () => boolean) => {
      if (isCurrent()) minimums.push(bounds());
    },
  };
  new Function(...Object.keys(scope), stripTypeScriptTypes(effect))(
    ...Object.values(scope),
  );
  try {
    setAppliedInterfaceZoom(2);
    setAppliedInterfaceZoom(1);
    assert.deepEqual(minimums, [
      { width: 920, height: 960 },
      { width: 460, height: 480 },
    ]);
    mode.current = "setup";
    setAppliedInterfaceZoom(1.5);
    assert.equal(minimums.length, 2);
    mode.current = "app";
    dispose();
    setAppliedInterfaceZoom(2);
    assert.equal(minimums.length, 2);
  } finally {
    dispose();
    setAppliedInterfaceZoom(1);
  }
});
