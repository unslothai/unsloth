// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const MAIN = readSrc("main.tsx");
const GATE = "html.tauri.render-linux.webkit-cpu-paint";
const POPUP_GATE = `${GATE}:not([data-glass], [data-glass-clear])`;
const COMPOSER_GATE = `${GATE}:not([data-glass], [data-glass-clear], [data-composer-outline="on"])`;
const SPLIT = /,(?![^(]*\))/;

/** Every rule as [selector, body], comments dropped, at-rule wrappers kept out of the selector. */
function rules(): [string, string][] {
  const css = CSS.replace(/\/\*[\s\S]*?\*\//g, "");
  const out: [string, string][] = [];
  for (const match of css.matchAll(/([^{};]+)\{([^{}]*)\}/g)) {
    out.push([match[1].trim().replace(/\s+/g, " "), match[2]]);
  }
  return out;
}

const gated = () => rules().filter(([selector]) => selector.startsWith(GATE));

test("flat edges only apply under the CPU-paint class of the Linux desktop app", () => {
  const touching = rules().filter(([, body]) => /--cpu-paint/.test(body));
  assert.ok(touching.length >= 5);
  for (const [selector] of touching) {
    for (const part of selector.split(SPLIT)) {
      assert.ok(part.trim().startsWith(GATE), `${part.trim()} is gated`);
    }
  }
});

test("every large shadowed surface takes the flat edge", () => {
  const body = (fragment: string) =>
    gated().find(([selector]) => selector.includes(fragment))?.[1] ?? "";
  for (const slot of ["dialog-content", "alert-dialog-content", "sheet-content"]) {
    assert.match(
      body(`[data-slot="${slot}"]`),
      /--popup-shadow:\s*var\(--cpu-paint-edge\);/,
    );
  }
  assert.match(
    body("[data-sonner-toast][data-styled='true']"),
    /--composer-shadow:\s*var\(--cpu-paint-edge\);/,
  );
  for (const surface of [".unsloth-composer-surface", ".unsloth-files-panel"]) {
    assert.match(
      body(surface),
      /--composer-shadow:\s*var\(--cpu-paint-composer-edge\);/,
    );
  }
});

test("the flat edges carry no blur, so nothing is re-blurred", () => {
  const values = gated().flatMap(([, b]) =>
    [...b.matchAll(/(?:--cpu-paint[\w-]*|box-shadow):\s*([^;]+);/g)].map((m) =>
      m[1].trim(),
    ),
  );
  assert.ok(values.length >= 6);
  for (const value of values) {
    if (/^var\(/.test(value)) continue;
    assert.match(value, /^0 0 0 1px /, value);
  }
});

test("surfaces whose shadow glass, Transparent, a wallpaper or the outline setting replace keep it", () => {
  for (const [selector, body] of gated()) {
    if (!/--popup-shadow|--composer-shadow|box-shadow/.test(body)) continue;
    for (const part of selector.split(SPLIT).map((p) => p.trim())) {
      if (/composer-surface|files-panel/.test(part)) {
        assert.ok(part.startsWith(COMPOSER_GATE), part);
        assert.match(part, /:not\(\[data-wallpaper-ready\] \*\)/, part);
      } else {
        assert.ok(part.startsWith(POPUP_GATE), part);
      }
    }
  }
  // Element-level overrides of these shadows only exist in the states gated out above.
  for (const [selector, body] of rules()) {
    if (selector.startsWith(GATE)) continue;
    for (const part of selector.split(SPLIT)) {
      const surface = /composer-surface|files-panel/.test(part)
        ? "composer"
        : /dialog-content"\]|sheet-content"\]|sonner-toast/.test(part) &&
            !/\s\S*(dialog-content|sheet-content|sonner-toast)\S*\s/.test(`${part} `)
          ? "popup"
          : null;
      if (!surface) continue;
      const sets =
        surface === "composer"
          ? /--composer-shadow:/.test(body)
          : /(^|[;\s])box-shadow:/.test(body);
      if (!sets) continue;
      const ok =
        surface === "composer"
          ? /data-glass|data-wallpaper-(ready|outline-off)|data-composer-outline|chat-full-view-dock/.test(part)
          : /data-glass-menus|data-glass-clear/.test(part) ||
            part === "[data-sonner-toast][data-styled='true']" ||
            part.startsWith(':is([data-slot="dialog-content"]');
      assert.ok(ok, `${part} overrides a shadow the flat edges replace`);
    }
  }
});

test("the original shadows are unchanged without the class", () => {
  const plain = rules().filter(([selector]) => !selector.startsWith(GATE));
  const body = (selector: string) =>
    plain
      .find(([s]) => s === selector)?.[1]
      .replace(/\s+/g, " ")
      .trim();
  assert.equal(
    body(
      ':is([data-slot="dialog-content"], [data-slot="alert-dialog-content"], [data-slot="sheet-content"])',
    ),
    "box-shadow: var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--popup-shadow);",
  );
  assert.match(
    CSS,
    /--popup-shadow: 0 12px 32px -12px rgb\(0 0 0 \/ 0\.16\), 0 2px 6px -2px rgb\(0 0 0 \/ 0\.06\);/,
  );
  for (const [selector, b] of plain) {
    assert.doesNotMatch(b, /--cpu-paint/, `${selector} is untouched`);
  }
});

test("main.tsx adds the class only on the shell's explicit yes, after the Linux classes", () => {
  const linux = MAIN.indexOf('classList.add("render-linux")');
  const add = MAIN.indexOf('classList.add("webkit-cpu-paint")');
  assert.ok(linux > 0 && add > linux);
  assert.match(MAIN, /if \(status\.cpuPainting\) \{\s*document\.documentElement\.classList\.add\("webkit-cpu-paint"\);/);
  assert.doesNotMatch(MAIN, /cpu-paint-shadows/);
});

test("dialogs drop their rounded clip under CPU painting, gated like the edges", () => {
  const square = gated().filter(([, b]) => /border-radius:\s*0 !important;/.test(b));
  assert.equal(square.length, 1);
  for (const part of square[0][0].split(SPLIT)) {
    assert.ok(part.trim().startsWith(POPUP_GATE), part);
    assert.match(part, /dialog-content"\]/);
  }
  assert.doesNotMatch(square[0][0], /sheet-content|composer|toast/);
});
