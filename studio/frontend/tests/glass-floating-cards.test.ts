// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css").replace(/\/\*[\s\S]*?\*\//g, "");

function bodiesFor(selectorPart: string): string[] {
  const out: string[] = [];
  for (const match of CSS.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    if (match[1].includes(selectorPart)) out.push(match[2]);
  }
  return out;
}

test("glass toasts and floating cards take the composer fill and no drop shadow", () => {
  const bodies = bodiesFor(".glass-card)").join("\n");
  assert.match(bodies, /background-color: color-mix\(in srgb, var\(--glass-composer-fill\)/);
  assert.match(bodies, /backdrop-filter: blur\(var\(--glass-blur\)\)/);
  assert.match(bodies, /box-shadow: var\(--composer-shadow\) !important/);
  assert.doesNotMatch(bodies, /--glass-lift/);
});

test("the composer's liquid rims cover the floating cards", () => {
  for (const glass of ['html[data-glass^="liquid"]', 'html[data-glass="liquid-3d"]']) {
    const selector = `${glass} :is(.unsloth-composer-surface, .unsloth-files-panel, .wallpaper-preview-composer, .toast-surface, .glass-card)`;
    assert.ok(CSS.includes(selector), selector);
  }
});

test("every floating card is tagged", () => {
  const tagged: [string, string][] = [
    ["components/ui/sonner.tsx", "pointer-events-auto glass-card"],
    ["features/hub/download-manager/download-manager-panel.tsx", "hub-download-panel pointer-events-auto modal-pe-live glass-card"],
    ["features/hub/download-manager/download-manager-panel.tsx", "hub-download-fab pointer-events-auto modal-pe-live glass-card"],
    ["features/loaded-models/loaded-models-indicator.tsx", "menu-soft-edgeless glass-card"],
    ["features/api-monitor/api-monitor-overlay.tsx", "menu-soft-surface glass-card"],
    ["components/floating-monitor.tsx", "settings-surface glass-card"],
  ];
  for (const [file, needle] of tagged) {
    assert.ok(readSrc(file).includes(needle), `${file}: ${needle}`);
  }
});

test("the corner rail is a scroller only while its cards overflow", () => {
  const provider = readSrc("app/provider.tsx");
  assert.equal(provider.match(/ref=\{overlayRailScope\}/g)?.length, 2);
  assert.match(CSS, /\[data-overlay-rail\]:not\(\[data-rail-scrolls\]\) \{\s*overflow: visible;/);
});
