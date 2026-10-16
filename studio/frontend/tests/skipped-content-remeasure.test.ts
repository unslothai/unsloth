// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/tauri-webview-resolver.mjs", import.meta.url);

const attributes = new Set<string>();
const scope = globalThis as { window?: unknown; document?: unknown };
scope.window = {
  location: { protocol: "https:" },
  localStorage: {
    getItem: () => null,
    setItem: () => undefined,
    removeItem: () => undefined,
  },
};
scope.document = {
  documentElement: {
    style: {
      setProperty: () => undefined,
      removeProperty: () => undefined,
    },
    setAttribute: (name: string) => attributes.add(name),
    removeAttribute: (name: string) => attributes.delete(name),
    toggleAttribute: (name: string, on: boolean) =>
      on ? attributes.add(name) : attributes.delete(name),
    classList: { toggle: () => undefined },
  },
};

const { applyCustomizationToDocument, DEFAULT_CUSTOMIZATION } = await import(
  "../src/features/settings/stores/appearance-custom-store.ts"
);
const { applyInterfaceScale } = await import(
  "../src/features/settings/stores/interface-scale-store.ts"
);

const HOLD = "data-interface-zooming";

test("a typography change renders skipped blocks until they are laid out at the new metrics", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  applyCustomizationToDocument(DEFAULT_CUSTOMIZATION, "light");
  assert.equal(attributes.has(HOLD), false, "the first application has nothing laid out at older metrics");
  applyCustomizationToDocument({ ...DEFAULT_CUSTOMIZATION, contrast: 70 }, "light");
  assert.equal(attributes.has(HOLD), false, "a colour change leaves every remembered size right");
  applyCustomizationToDocument({ ...DEFAULT_CUSTOMIZATION, codeFontSize: 16 }, "light");
  assert.equal(attributes.has(HOLD), true, "a skipped code block keeps the height of the old code font size");
  t.mock.timers.tick(299);
  assert.equal(attributes.has(HOLD), true);
  t.mock.timers.tick(1);
  assert.equal(attributes.has(HOLD), false);
  applyCustomizationToDocument({ ...DEFAULT_CUSTOMIZATION, codeFontSize: 16, uiFontSize: 18 }, "light");
  assert.equal(attributes.has(HOLD), true, "and a thinking chunk the height of the old UI font size");
  t.mock.timers.tick(300);
  applyCustomizationToDocument({ ...DEFAULT_CUSTOMIZATION, codeFontSize: 16, uiFontSize: 18, chatFont: "Inter" }, "dark");
  assert.equal(attributes.has(HOLD), true, "a font family changes the metrics as much as a size");
  t.mock.timers.tick(300);
  assert.equal(attributes.has(HOLD), false);
});

test("a browser scale change renders skipped blocks until they are laid out at the new scale", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  await applyInterfaceScale(110);
  await applyInterfaceScale(110);
  assert.equal(attributes.has(HOLD), false, "the scale applied before the first paint has nothing laid out at another scale");
  await applyInterfaceScale(125);
  assert.equal(attributes.has(HOLD), true, "a skipped block keeps the height it had at the old scale");
  t.mock.timers.tick(300);
  assert.equal(attributes.has(HOLD), false);
});
