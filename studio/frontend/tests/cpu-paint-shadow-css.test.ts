// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");
const RUNTIME = readSrc("lib/cpu-paint-shadows.ts");
const GATE = "html.tauri.render-linux.webkit-cpu-paint";

/** Every rule as [selector, body], comments dropped, at-rule wrappers kept out of the selector. */
function rules(): [string, string][] {
  const css = CSS.replace(/\/\*[\s\S]*?\*\//g, "");
  const out: [string, string][] = [];
  for (const match of css.matchAll(/([^{};]+)\{([^{}]*)\}/g)) {
    out.push([match[1].trim().replace(/\s+/g, " "), match[2]]);
  }
  return out;
}

test("image shadows only apply under the CPU-paint class of the Linux desktop app", () => {
  const touching = rules().filter(([, body]) => /border-image|--cpu-/.test(body));
  assert.ok(touching.length >= 8, "dialogs, sheets, composer, toast and files panel are covered");
  for (const [selector] of touching) {
    for (const part of selector.split(/,(?![^(]*\))/)) {
      assert.ok(part.trim().startsWith(GATE), `${part.trim()} is gated`);
    }
  }
});

test("each large shadowed surface switches to its image", () => {
  const bodies = new Map(rules().filter(([s]) => s.startsWith(GATE)).map(([s, b]) => [s, b]));
  const find = (fragment: string) => [...bodies].find(([selector]) => selector.includes(fragment))?.[1] ?? "";
  assert.match(find('[data-slot="alert-dialog-content"]'), /--popup-shadow:\s*var\(--cpu-popup-rest\);/);
  assert.match(find('[data-slot="alert-dialog-content"]'), /border-image:\s*var\(--cpu-popup\);/);
  assert.match(bodies.get(`${GATE} [data-slot="sheet-content"]`) ?? "", /border-image:\s*var\(--cpu-sheet\);/);
  assert.match(bodies.get(`${GATE} :is(.chat-composer-surface, .unsloth-composer-surface)`) ?? "", /--composer-shadow:\s*var\(--cpu-composer-rest\);/);
  assert.match(find("[data-sonner-toast][data-styled='true']"), /--composer-shadow:\s*var\(--cpu-toast-rest\);/);
  assert.match(find(".unsloth-files-panel"), /border-image:\s*var\(--cpu-files\);/);
  // Never `fill`: the image would cover the surface's own background.
  for (const body of bodies.values()) assert.doesNotMatch(body, /border-image:[^;]*\bfill\b/);
});

test("nearest-neighbour stretching stays on the shadowed surfaces, not their content", () => {
  const all = rules();
  const pixelated = all.filter(([, b]) => /image-rendering:\s*pixelated/.test(b));
  const reset = all.filter(([, b]) => /image-rendering:\s*auto/.test(b));
  assert.equal(pixelated.length, 1);
  assert.equal(reset.length, 1);
  for (const [selector] of [...pixelated, ...reset]) {
    for (const part of selector.split(/,(?![^(]*\))/)) assert.ok(part.trim().startsWith(GATE), part);
  }
  for (const end of ["> *", "::before", "::after"]) {
    assert.ok(reset[0][0].split(/,(?![^(]*\))/).some((part) => part.trim().endsWith(end)), end);
  }
});

test("every variable the CSS reads is one the runtime publishes", () => {
  const names = [...RUNTIME.matchAll(/name: "([a-z-]+)"/g)].map((m) => m[1]);
  assert.ok(names.length >= 6);
  const read = new Set([...CSS.matchAll(/var\(--cpu-([a-z-]+?)(?:-rest)?[,)]/g)].map((m) => m[1]));
  read.delete("sheet-edge");
  for (const name of read) assert.ok(names.includes(name), `--cpu-${name} is published`);
});

test("the toast image uses Sonner's corner radius", () => {
  const radius = /TOAST_RADIUS = "([^"]+)"/.exec(RUNTIME)?.[1];
  assert.ok(radius);
  assert.ok(readSrc("components/ui/sonner.tsx").includes(`"--border-radius": "${radius}"`));
});

test("the original shadows are unchanged without the class", () => {
  const plain = rules().filter(([selector]) => !selector.startsWith(GATE));
  const body = (selector: string) => plain.find(([s]) => s === selector)?.[1].replace(/\s+/g, " ").trim();
  assert.equal(
    body(':is([data-slot="dialog-content"], [data-slot="alert-dialog-content"], [data-slot="sheet-content"])'),
    "box-shadow: var(--tw-ring-offset-shadow, 0 0 #0000), var(--tw-ring-shadow, 0 0 #0000), var(--popup-shadow);",
  );
  assert.equal(body("html.tauri.render-linux .media-stage-shadow"), "box-shadow: none;");
  assert.match(CSS, /--popup-shadow: 0 12px 32px -12px rgb\(0 0 0 \/ 0\.16\), 0 2px 6px -2px rgb\(0 0 0 \/ 0\.06\);/);
  assert.match(CSS, /--popup-shadow: 0 16px 40px -12px rgb\(0 0 0 \/ 0\.7\);/);
  for (const [selector, b] of plain) {
    assert.doesNotMatch(b, /border-image|--cpu-/, `${selector} is untouched`);
  }
});

test("Neon Cyberpunk's focus glow is a variable with the same layers, so it gets an image too", () => {
  const defs = rules().filter(([, b]) => /--composer-focus-shadow:/.test(b));
  assert.deepEqual(
    defs.map(([s]) => s),
    [':root[data-palette="neon-cyberpunk"]:not(.dark)', ':root[data-palette="neon-cyberpunk"].dark'],
  );
  const flat = (s: string) => s.replace(/\s+/g, " ").trim();
  assert.match(flat(defs[0][1]), /--composer-focus-shadow: 0 0 0 1px rgb\(230 210 0 \/ 0\.9\), 0 0 32px -4px rgb\(252 238 10 \/ 0\.9\), 0 2px 8px -2px rgb\(0 0 0 \/ 0\.1\);/);
  assert.match(flat(defs[1][1]), /--composer-focus-shadow: 0 0 0 1px rgb\(0 240 255 \/ 0\.6\), 0 0 32px -6px rgb\(0 240 255 \/ 0\.55\), 0 4px 20px 4px var\(--background\);/);
  const focus = rules().filter(([s, b]) => s.includes(":focus-within") && s.includes("neon-cyberpunk") && /box-shadow:/.test(b));
  for (const [selector, b] of focus) {
    const value = /box-shadow:\s*([^;]+);/.exec(b)?.[1].trim();
    assert.equal(value, selector.startsWith(GATE) ? "var(--cpu-composer-focus-rest)" : "var(--composer-focus-shadow)");
  }
});

test("main.tsx arms the images after the Linux classes and asks the shell first", () => {
  const main = readSrc("main.tsx");
  const linux = main.indexOf('classList.add("render-linux")');
  const arm = main.indexOf("armCpuPaintShadows(");
  assert.ok(linux > 0 && arm > linux);
  assert.match(main, /cpuPainting: \(\) => loadGpuRendering\(\)\.then\(\(status\) => status\.cpuPainting\)/);
  // Only an explicit false skips the images; an older shell or a failed call keeps them.
  assert.match(RUNTIME, /if \(cpuPainting !== false\) startCpuPaintShadows\(options\);/);
  assert.match(RUNTIME, /\(\) => startCpuPaintShadows\(options\),/);
  assert.match(RUNTIME, /classList\.contains\("tauri"\)\s*\|\|\s*!root\.classList\.contains\("render-linux"\)/);
});
