// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import postcss from "postcss";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const MATERIAL = readSrc("features/chat-background/material.ts");
const GATE = "html[data-glass][data-glass-cpu]";

function ruleAt(selectorStart: string): string {
  const at = CSS.indexOf(selectorStart);
  assert.ok(at > 0, `rule ${selectorStart} exists`);
  return CSS.slice(at, CSS.indexOf("}", at) + 1);
}

test("the CPU glass flag is WebKitGTK only and follows the layer flag's material and scope gate", () => {
  assert.match(MATERIAL, /import \{ isWebKitGTK \} from "@\/components\/ui\/theme-view-transition";/);
  assert.match(
    MATERIAL,
    /const WEBKIT_GTK =\s*typeof navigator !== "undefined" &&\s*isWebKitGTK\(navigator\.platform, navigator\.userAgent\);/,
    "macOS WebKit composites backdrops on the GPU and keeps real glass",
  );
  assert.match(
    MATERIAL,
    /toggleAttribute\(\s*"data-glass-cpu",\s*WEBKIT_GTK && material !== "solid" && material !== "transparent" && here,\s*\)/,
    "solid, Transparent and out-of-scope routes never carry the flag",
  );
});

test("WebKitGTK glass drops every live Glass blur, the sidebar's included", () => {
  const rule = ruleAt(`${GATE} *:not([class*="backdrop-"]),`);
  assert.doesNotMatch(rule, /sidebar-inner/, "the sidebar's blur re-ran on every composite while the thread scrolled");
  assert.ok(
    rule.includes(`${GATE} *:not([data-windows-glass-frame], [class*="backdrop-"])::before,`),
    "the framed sidebar's glass keeps its blur",
  );
  assert.ok(rule.includes(`${GATE} *:not([class*="backdrop-"])::after {`), "a component's own backdrop-* blur is not Glass's");
  assert.match(rule, /-webkit-backdrop-filter: none !important;/);
  assert.match(rule, /\n\s*backdrop-filter: none !important;/);
  assert.doesNotMatch(rule, /:has\(/, "a :has() here would re-match on every DOM change");
  const toast = ruleAt(`html[data-glass][data-glass-menus][data-glass-cpu] [data-sonner-toast][data-styled="true"] {`);
  assert.match(toast, /\n\s*backdrop-filter: none !important;/, "Sonner's own !important blur outranks the universal rule");
});

test("no later important blur can win the cascade back", () => {
  const at = CSS.indexOf(`${GATE} *:not(`);
  assert.doesNotMatch(CSS.slice(at), /backdrop-filter:\s*blur[^;]*!important/);
});

test("dialogs and menus go opaque on that path, so nothing behind them ghosts through", () => {
  assert.match(ruleAt(`html[data-glass][data-glass-menus][data-glass-cpu] [data-slot="dialog-content"] {`), /background-color: var\(--glass-page-bg\);/);
  const body = ruleAt(`${GATE} body {`);
  assert.match(body, /--glass-menu-fill: var\(--popover\);/, "resolved on body, where --popover is still opaque");
  assert.match(body, /--glass-opacity: max\(\.9, var\(--glass-opacity-set\)\);/);
  const floors = ruleAt(`${GATE} {`);
  assert.match(floors, /--glass-composer-floor: 110%;/, "over 100%, so the default strength reaches the 96% cap");
  assert.doesNotMatch(floors, /--glass-sidebar-floor/, "the page floors stay as they are");
  assert.match(ruleAt(`${GATE} [data-slot="sidebar-inner"] {`), /--glass-pane-floor: 110%;/, "the unblurred sidebar reaches the 96% fill cap");
  assert.match(
    ruleAt('html[data-glass="frosted"][data-glass-cpu] :is([data-slot="sidebar-inner"], .unsloth-composer-surface'),
    /background-image: none !important;/,
    "the Frosted grain (an SVG turbulence tile) is rendered again on every paint there",
  );
  assert.match(floors, /--glass-opacity-set: var\(--glass-opacity, 1\);/);
});

test("WebKitGTK liquid 3D draws its wide inner glows as gradients, not blurred shadows", () => {
  for (const sel of [
    'html[data-glass="liquid-3d"][data-glass-cpu] {',
    'html.dark[data-glass="liquid-3d"][data-glass-cpu] {',
    'html[data-glass="liquid-3d"][data-glass-cpu] :is(.unsloth-composer-surface',
    'html.dark[data-glass="liquid-3d"][data-glass-cpu] :is(.unsloth-composer-surface',
  ]) {
    const rule = ruleAt(sel);
    const rim = /--glass-rim: ([^;]*);/.exec(rule)?.[1] ?? "";
    assert.ok(rim, `${sel} sets the rim`);
    for (const shadow of rim.split(/,(?![^(]*\))/)) {
      const blur = /inset\s+-?[\d.]+(?:px)?\s+-?[\d.]+(?:px)?\s+([\d.]+)px/.exec(shadow)?.[1];
      assert.ok(!blur || Number(blur) <= 2, `${sel}: no wide blurred inset (${shadow.trim()})`);
    }
    assert.match(rule, /--glass-glow: linear-gradient\(/);
  }
  const glowUses = CSS.match(/background-image: var\(--glass-glow, none\), var\(--glass-sheen\);/g) ?? [];
  assert.equal(glowUses.length, 3, "composer, menus and dialogs; elsewhere --glass-glow is unset, so the layer is none");
});

test("the WebKitGTK rules are unlayered top-level rules", () => {
  const found: string[] = [];
  postcss.parse(CSS).walkRules((rule) => {
    if (!rule.selector.includes("[data-glass-cpu]")) return;
    found.push(rule.selector);
    assert.equal(rule.parent?.type, "root", `${rule.selector} must not sit inside @layer or another block`);
  });
  assert.ok(found.length >= 9, "blur, fill, dialog, toast and glow rules all present");
});

// Every surface the CPU path strips of its blur, and what keeps text behind it from reading through.
test("no surface that loses its blur is left with a see-through fill", () => {
  const plainMenus = ruleAt('html[data-glass][data-glass-menus][data-glass-plain][data-glass-cpu]:not(.dark) :is(');
  assert.match(plainMenus, /--glass-menu-fill: var\(--glass-page-popover\);/, "plain light menus set their own thin fill on the element");
  assert.match(
    ruleAt("html[data-glass][data-glass-menus][data-glass-cpu] :is(.tooltip-compact, .tooltip-rich) {"),
    /background-color: var\(--tooltip-bg\);/,
  );
  const pageSticky = ruleAt(
    'html[data-glass][data-glass-cpu][data-chat-wallpaper-workspace-ready] [data-wallpaper-workspace][data-wallpaper-route="page"] > [data-slot="sidebar-inset"] .sticky:not(',
  );
  assert.match(pageSticky, /:not\(\.library-header-row:not\(\[data-stuck\]\), \[data-page-model-header\]\)/, "headers at rest stay clear");
  assert.match(pageSticky, /background-color: var\(--glass-page-bg\);/);
  assert.match(
    ruleAt('html[data-glass][data-glass-cpu] :is([data-slot="chat-settings-panel"], [data-slot="dialog-content"], [data-slot="sheet-content"]) .sticky {'),
    /background-color: var\(--glass-page-bg\);/,
  );
  assert.match(
    ruleAt('html[data-glass][data-glass-cpu] :is([data-slot$="popover-content"], [data-slot$="menu-content"]) .sticky {'),
    /background-color: var\(--glass-page-popover\);/,
  );
  const panes = ruleAt('html[data-glass][data-glass-cpu][data-chat-wallpaper-workspace-ready] [data-slot="chat-settings-panel"],');
  assert.ok(panes.includes("> .browser-chrome,") && panes.includes("> .browser-panel-body {"));
  assert.match(panes, /background-color: var\(--glass-panel-fill\);/);
  assert.match(ruleAt("html[data-glass][data-glass-cpu]:not(.dark) {"), /--glass-control: rgb\(255 255 255 \/ \.8\);/);
  assert.match(
    ruleAt('html[data-glass][data-glass-cpu]:not(.dark)[data-chat-wallpaper-workspace-ready] [data-wallpaper-route="page"] > [data-slot="sidebar-inset"],'),
    /--page-glass-card-fill: rgb\(255 255 255 \/ \.85\);/,
  );
});

test("each CPU-path fill outranks the glass rule it replaces", () => {
  // Same selector plus [data-glass-cpu]: one more attribute always wins, whatever the order.
  const pairs: [string, string][] = [
    ['html[data-glass][data-glass-menus][data-glass-plain]:not(.dark) :is(', 'html[data-glass][data-glass-menus][data-glass-plain][data-glass-cpu]:not(.dark) :is('],
    ["html[data-glass][data-glass-menus] :is(.tooltip-compact, .tooltip-rich) {", "html[data-glass][data-glass-menus][data-glass-cpu] :is(.tooltip-compact, .tooltip-rich) {"],
    ['html[data-glass][data-chat-wallpaper-workspace-ready] [data-wallpaper-workspace][data-wallpaper-route="page"] > [data-slot="sidebar-inset"] .sticky {',
     'html[data-glass][data-glass-cpu][data-chat-wallpaper-workspace-ready] [data-wallpaper-workspace][data-wallpaper-route="page"] > [data-slot="sidebar-inset"] .sticky:not('],
    ["html[data-glass]:not(.dark) {", "html[data-glass][data-glass-cpu]:not(.dark) {"],
  ];
  for (const [base, cpu] of pairs) {
    assert.ok(CSS.includes(base), `base rule still exists: ${base}`);
    assert.ok(CSS.includes(cpu), `cpu override exists: ${cpu}`);
  }
});
