// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const attributes = new Set<string>();
const vars = new Map<string, string>();
(globalThis as { document?: unknown }).document = {
  documentElement: {
    style: {
      setProperty: (name: string, value: string) => vars.set(name, value),
      removeProperty: (name: string) => vars.delete(name),
    },
    setAttribute: (name: string) => attributes.add(name),
    removeAttribute: (name: string) => attributes.delete(name),
    toggleAttribute: (name: string, on: boolean) => (on ? attributes.add(name) : attributes.delete(name)),
    classList: { toggle() {} },
  },
};

const { applyCustomizationToDocument, DEFAULT_CUSTOMIZATION, sanitizeCustomization } = await import(
  "../src/features/settings/stores/appearance-custom-store.ts"
);

test("records from before the setting keep the background image's own outline", () => {
  assert.equal(sanitizeCustomization({}).composerOutline, null);
  assert.equal(sanitizeCustomization({ composerOutline: "40" }).composerOutline, null);
  assert.equal(sanitizeCustomization({ composerOutline: 140 }).composerOutline, 100);
  assert.equal(sanitizeCustomization({ composerOutline: 12.6 }).composerOutline, 13);
});

test("a set outline reaches every composer, and 0 keeps the theme shadow", () => {
  applyCustomizationToDocument({ ...DEFAULT_CUSTOMIZATION, composerOutline: 40 }, "light");
  assert.equal(vars.get("--composer-outline"), "40%");
  assert.ok(attributes.has("data-composer-outline"));
  applyCustomizationToDocument({ ...DEFAULT_CUSTOMIZATION, composerOutline: 0 }, "light");
  assert.equal(vars.get("--composer-outline"), "0%");
  assert.ok(!attributes.has("data-composer-outline"));
  // Unset: off, except the background image's own outline over an image.
  applyCustomizationToDocument(DEFAULT_CUSTOMIZATION, "light");
  assert.ok(!vars.has("--composer-outline"));
  assert.ok(!attributes.has("data-composer-outline"));
  const css = readSrc("index.css");
  assert.match(css, /html\[data-composer-outline\] \.unsloth-composer-surface \{\s*--composer-shadow: inset 0 0 0 1px/);
  assert.match(css, /var\(--composer-outline, var\(--wallpaper-outline, 30%\)\)/);
});

test("the composer rows have their own Composer section", () => {
  const tab = readSrc("features/settings/tabs/appearance-tab.tsx");
  const section = tab.slice(tab.indexOf('t("settings.appearance.composer.title")'));
  assert.ok(section.indexOf("<ComposerPositionSliderRow />") < section.indexOf("settings.appearance.code.title"));
  assert.ok(section.indexOf("<ComposerOutlineSliderRow />") < section.indexOf("settings.appearance.code.title"));
  assert.ok(section.indexOf("<ChatWidthSelect />") < section.indexOf("settings.appearance.code.title"));
  assert.ok(section.indexOf("<SentAttachmentsSelect />") < section.indexOf("settings.appearance.code.title"));
  const preferences = tab.slice(tab.indexOf("preferencesTitle"), tab.indexOf('t("settings.appearance.composer.title")'));
  assert.doesNotMatch(preferences, /ComposerPositionSliderRow|ComposerOutlineSliderRow|ChatWidthSelect|SentAttachmentsSelect/);
});
