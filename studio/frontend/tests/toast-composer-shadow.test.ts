// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");

function rules(): [string, string][] {
  const css = CSS.replace(/\/\*[\s\S]*?\*\//g, "");
  const out: [string, string][] = [];
  for (const match of css.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
    out.push([match[1].trim().replace(/\s+/g, " "), match[2]]);
  }
  return out;
}

const SURFACE = /chat-composer-surface|unsloth-composer-surface|\[data-sonner-toast\]\[data-styled='true'\]/;

test("toasts and the composer read one shadow variable", () => {
  const shadows = rules()
    .filter(([selector, body]) => SURFACE.test(selector) && /(^|[\s;])box-shadow:/.test(body))
    .filter(([selector]) => !selector.includes(":focus-within") && !selector.includes("chat-full-view-dock"))
    .map(([selector, body]) => [selector, /box-shadow:\s*([^;]+);/.exec(body)?.[1].replace(/\s*!important$/, "").trim()]);
  assert.ok(shadows.length >= 3, "the composers and the toast set a shadow");
  for (const [selector, shadow] of shadows) {
    assert.equal(shadow, "var(--composer-shadow)", `${selector} uses the shared shadow`);
  }
});

test("the shadow is set per theme and per palette", () => {
  const defs = rules().filter(([, body]) => /--composer-shadow:/.test(body)).map(([selector]) => selector);
  assert.ok(defs.includes(":root"), "light");
  assert.ok(defs.includes(".dark"), "dark");
  assert.ok(defs.includes(':root[data-palette="neon-cyberpunk"]:not(.dark)'), "Neon Cyberpunk light");
  assert.ok(defs.includes(':root[data-palette="neon-cyberpunk"].dark'), "Neon Cyberpunk dark");
});
