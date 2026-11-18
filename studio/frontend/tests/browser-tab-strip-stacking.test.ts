// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("no query container wraps the browser tab strip, so it stays above the macOS titlebar drag strip", () => {
  // Before Safari 18.5 container-type adds layout containment, a stacking context that traps the strip's z-40 (#14314).
  const panel = readSrc("features/browser/browser-panel.tsx");
  const root = panel.indexOf("export const BrowserPanel = memo(");
  const section = panel.indexOf("<section", root);
  const strip = panel.indexOf("<TabStrip ", section);
  assert.ok(root >= 0 && section > root && strip > section);
  assert.doesNotMatch(panel.slice(section, strip), /@container/);
  assert.match(panel, /className="browser-chrome relative z-40 flex /);
  assert.match(
    panel.slice(strip),
    /^<TabStrip [^\n]*\n(?:[^\n]*\n){0,3}\s*<div className="browser-panel-body @container\/browser /,
  );
});

// Not the section (a stacking context) nor a pseudo-element (macOS native pages cut holes only through elements).
test("the browser's glass frosts the tab strip and panel body, never the section", () => {
  const css = readSrc("index.css").replace(/\/\*[\s\S]*?\*\//g, "");
  const rules = [...css.matchAll(/([^{}]*\[data-slot="browser-panel"\][^{}]*)\{([^}]*)\}/g)];
  const frosted = rules.filter(([, , body]) => /backdrop-filter/.test(body));
  assert.ok(frosted.length > 0);
  for (const [, selector] of frosted) {
    for (const part of selector.split(",")) {
      if (part.trim().endsWith('[data-windows-browser="web"]::before')) continue;
      assert.match(part.trim(), /\[data-slot="browser-panel"\](?::not\(\[data-windows-browser\]\))?(:not\(\[data-browser-document\]\))? > \.browser-(chrome|panel-body)( ?\[data-native-hole\] > :not\(\[data-browser-page\]\))?$/);
    }
  }
});

test("a native page's hole drops backdrop filters and masks the wallpaper layer", () => {
  const css = readSrc("index.css");
  assert.match(css, /\[data-native-hole\] \{\n\t-webkit-backdrop-filter: none !important;\n\tbackdrop-filter: none !important;/);
  assert.match(css, /\[data-native-hole-mask\] \{[^}]*mask-image: var\(--native-hole-mask\)/);
  const view = readSrc("features/browser/native-view.ts");
  assert.match(view, /node\.hasAttribute\?\.\("data-wallpaper-workspace"\)/);
  assert.match(view, /child\.classList\.contains\("chat-wallpaper"\)/);
});

test("the docked Windows glass surface never clips the floating browser's rounded border", () => {
  const panel = readSrc("features/browser/browser-panel.tsx");
  assert.match(panel, /data-windows-browser=\{!floating && shouldUseCustomWindowTitlebar\(\)/);
});
