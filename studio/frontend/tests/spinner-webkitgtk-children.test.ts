// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// WebKitGTK wobbles a rotating layer, so the Linux app animates the svg children, not the svg.

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const CSS = readFileSync(path.join(ROOT, "src/index.css"), "utf8").replace(
  /\/\*[\s\S]*?\*\//g,
  "",
);

const rules = [...CSS.matchAll(/([^{}@;]+)\{([^{}]*)\}/g)].map((m) => ({
  selector: m[1].trim().replace(/\s+/g, " "),
  body: m[2],
}));
const bodyOf = (selector: string) =>
  rules.find((r) => r.selector === selector)?.body ?? "";

test("the desktop app rotates the children of a spinning svg, not the svg", () => {
  assert.match(
    bodyOf("html.tauri.render-linux svg.animate-spin"),
    /animation:\s*none/,
  );
  const child = bodyOf("html.tauri.render-linux svg.animate-spin > *");
  assert.match(child, /animation:\s*spin 1s linear infinite/);
  assert.match(child, /transform-box:\s*view-box/);
  assert.match(child, /transform-origin:\s*center/);
  for (const r of rules) {
    if (
      /svg\.animate-spin\s*\{|svg\.animate-spin > \*$/.test(r.selector) &&
      /animation:/.test(r.body)
    ) {
      assert.ok(
        r.selector.startsWith("html.tauri.render-linux "),
        `browser-wide spin override: ${r.selector}`,
      );
    }
  }
});

test("every reduced-motion spinner exception also covers the rotated children", () => {
  const exceptions = rules.filter(
    (r) =>
      r.selector.includes(".animate-spin") &&
      /animation-duration:\s*1\.5s/.test(r.body),
  );
  assert.ok(exceptions.length >= 2);
  for (const r of exceptions)
    assert.ok(r.selector.includes("svg.animate-spin > *"), r.selector);
});

test("no hand-rolled bordered spinner: those rotate a box and still wobble on the desktop", () => {
  const walk = (dir: string): string[] =>
    readdirSync(dir).flatMap((n) => {
      const p = path.join(dir, n);
      return statSync(p).isDirectory() ? walk(p) : /\.tsx$/.test(n) ? [p] : [];
    });
  const offenders = walk(path.join(ROOT, "src")).filter((f) =>
    /className="[^"]*\banimate-spin\b[^"]*\bborder-r-transparent\b|className="[^"]*\bborder-r-transparent\b[^"]*\banimate-spin\b/.test(
      readFileSync(f, "utf8"),
    ),
  );
  assert.deepEqual(
    offenders.map((f) => path.relative(ROOT, f)),
    [],
  );
});
