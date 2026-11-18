// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const attributes = new Map<string, string>();
const vars = new Map<string, string>();
(globalThis as { document?: unknown }).document = {
  documentElement: {
    style: {
      setProperty: (name: string, value: string) => vars.set(name, value),
      removeProperty: (name: string) => vars.delete(name),
    },
    setAttribute: (name: string, value: string) => attributes.set(name, value),
    removeAttribute: (name: string) => attributes.delete(name),
    toggleAttribute: (name: string, on: boolean) => (on ? attributes.set(name, "") : attributes.delete(name)),
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

test("a set outline reaches every composer, and 0 drops the glass edge too", () => {
  applyCustomizationToDocument({ ...DEFAULT_CUSTOMIZATION, composerOutline: 40 }, "light");
  assert.equal(vars.get("--composer-outline"), "40%");
  assert.equal(attributes.get("data-composer-outline"), "on");
  applyCustomizationToDocument({ ...DEFAULT_CUSTOMIZATION, composerOutline: 0 }, "light");
  assert.equal(vars.get("--composer-outline"), "0%");
  assert.equal(attributes.get("data-composer-outline"), "off");
  // Unset: off, except the background image's own outline over an image.
  applyCustomizationToDocument(DEFAULT_CUSTOMIZATION, "light");
  assert.ok(!vars.has("--composer-outline"));
  assert.ok(!attributes.has("data-composer-outline"));
  const css = readSrc("index.css");
  // !important: the glass composer rules are more specific and would otherwise keep their 14% ring.
  assert.match(css, /html\[data-composer-outline="on"\] :is\(\.unsloth-composer-surface, \.wallpaper-preview-composer\) \{\s*--composer-shadow: inset 0 0 0 1px [^;]*var\(--composer-outline\)[^;]*!important;/);
  // macOS glass has no background image, so its 14% ring needs its own selector.
  assert.match(css, /html\[data-glass="macos"\]\[data-composer-outline="off"\] \.unsloth-composer-surface,[^{]*\[data-wallpaper-ready\]\[data-wallpaper-ready\] \.unsloth-composer-surface,[^{]*\{\s*--composer-shadow: none !important;/);
  assert.match(css, /var\(--composer-outline, var\(--wallpaper-outline, 30%\)\)/);
});

test("the element outline sets every border and glass ring, and unset keeps the theme", () => {
  assert.equal(sanitizeCustomization({}).elementOutline, null);
  assert.equal(sanitizeCustomization({ elementOutline: -5 }).elementOutline, 0);
  assert.equal(sanitizeCustomization({ elementOutline: "20" }).elementOutline, null);
  applyCustomizationToDocument({ ...DEFAULT_CUSTOMIZATION, elementOutline: 0 }, "dark");
  assert.equal(vars.get("--element-outline"), "0%");
  assert.ok(attributes.has("data-element-outline"));
  applyCustomizationToDocument(DEFAULT_CUSTOMIZATION, "dark");
  assert.ok(!vars.has("--element-outline"));
  assert.ok(!attributes.has("data-element-outline"));
  const css = readSrc("index.css");
  assert.match(css, /html\[data-element-outline\],\s*html\[data-element-outline\] \.diffusion-surface \{\s*--border-base: [^;]*var\(--element-outline\)[^;]*!important;\s*--input-base: [^;]*var\(--element-outline\)[^;]*!important;/);
  assert.doesNotMatch(css, /color-mix\(in srgb, var\(--foreground\) 14%, transparent\), inset 0 1px 0 rgb\(255 255 255 \/ \.072\)/);
});

test("light glass outlines its controls by default", () => {
  const css = readSrc("index.css");
  assert.match(css, /--glass-light-edge: inset 0 0 0 1px color-mix\(in srgb, var\(--foreground\) var\(--element-outline, 12%\), transparent\), var\(--glass-light-inner\);/);
  assert.match(css, /--page-glass-edge: var\(--glass-light-edge\);/);
  assert.doesNotMatch(css, /box-shadow: var\(--glass-light-inner\);/);
  assert.doesNotMatch(css, /:where\(\[class~="border"\]\[class\*="rounded"\]\)[^{]*\{\s*border-color: transparent;/);
  const tab = readSrc("features/settings/tabs/appearance-tab.tsx");
  const preferences = tab.slice(tab.indexOf("preferencesTitle"), tab.indexOf('t("settings.appearance.code.title")'));
  assert.ok(preferences.indexOf("<ElementOutlineSliderRow />") > preferences.indexOf("<InterfaceScaleRow />"));
});

test("the composer appearance rows live only in Chat's Composer section", () => {
  const chat = readSrc("features/settings/tabs/chat-tab.tsx");
  const start = chat.indexOf('<SettingsSection title={t("composerSettings.title")}>');
  assert.ok(start !== -1);
  const section = chat.slice(start, chat.indexOf("</SettingsSection>", start));
  for (const control of ["ComposerPositionSliderRow", "ComposerOutlineSliderRow", "ChatWidthSelect", "SentAttachmentsSelect"]) {
    assert.ok(section.includes(`<${control} />`), `${control} is in Composer`);
    assert.ok(!readSrc("features/settings/tabs/appearance-tab.tsx").includes(control));
  }
});
