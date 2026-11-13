// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A page in the narrow browser overlay going full screen widens the viewport past the mobile
// breakpoint; swapping to the side panel then unmounted the full screen frame and reloaded it.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

let width = 700;
const doc: { fullscreenElement: object | null } = { fullscreenElement: null };
Object.assign(globalThis, {
  document: doc,
  window: {
    location: { protocol: "http:" },
    matchMedia: (query: string) => ({
      matches: width <= Number(/max-width: (\d+)px/.exec(query)?.[1]),
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
    }),
  },
});

const { isMobileViewport } = await import("../src/hooks/use-mobile.ts");

test("the mobile layout holds while something is full screen and follows the viewport after", () => {
  assert.equal(isMobileViewport(), true);
  doc.fullscreenElement = {};
  width = 1512;
  assert.equal(isMobileViewport(), true);
  doc.fullscreenElement = null;
  width = 700;
  assert.equal(isMobileViewport(), true);
  width = 1440;
  assert.equal(isMobileViewport(), false);
});
