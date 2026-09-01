// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The parsing tests live in bundle-budget-closure.test.ts. These run the script as
 * CI runs it, because the ways a size gate goes wrong are not parsing bugs: it
 * exits 0 having measured nothing, and nobody reads a passing step.
 *
 * Every case here asserts the exit code AND that a pass came with a measurement
 * printed next to it.
 */

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import {
  copyFileSync,
  mkdirSync,
  mkdtempSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const SCRIPT = join(
  dirname(fileURLToPath(import.meta.url)),
  "..",
  "scripts",
  "check-bundle-budget.ts",
);

/**
 * Whatever this process needed to load TypeScript, the child needs too. Reusing
 * the flags rather than naming one keeps this working across the node versions
 * where `--experimental-strip-types` is required, optional, and gone.
 */
const TS_FLAGS = process.execArgv.filter((flag) =>
  /^--(experimental-)?(strip-types|transform-types)/.test(flag),
);

const MEASURED_TWO = /eager startup JS: .* raw, .* transfer, 2 chunks/;

const INDEX_HTML = `<!doctype html><html><head>
<script type="module" crossorigin src="/assets/index-aaa.js"></script>
<link rel="modulepreload" crossorigin href="/assets/react-bbb.js">
<link rel="stylesheet" crossorigin href="/assets/index-ccc.css">
</head><body><div id="root"></div></body></html>`;

/** A checkout-shaped directory: the script, and a dist/ beside it. */
function fixture(
  options: { html?: string | null; bigChunk?: boolean } = {},
): string {
  const root = mkdtempSync(join(tmpdir(), "bundle-budget-"));
  mkdirSync(join(root, "scripts"), { recursive: true });
  copyFileSync(SCRIPT, join(root, "scripts", "check-bundle-budget.ts"));
  const html = options.html === undefined ? INDEX_HTML : options.html;
  if (html !== null) {
    mkdirSync(join(root, "dist", "assets"), { recursive: true });
    writeFileSync(join(root, "dist", "index.html"), html);
    writeFileSync(
      join(root, "dist", "assets", "index-aaa.js"),
      "console.log(1)\n",
    );
    writeFileSync(
      join(root, "dist", "assets", "react-bbb.js"),
      // Incompressible, so raw and gzip both clear the budget when asked to.
      options.bigChunk ? randomBytes(6 * 1024 * 1024) : "export const a = 1\n",
    );
  }
  return root;
}

function runIn(root: string, scriptDir = root) {
  const r = spawnSync(
    process.execPath,
    [...TS_FLAGS, join(scriptDir, "scripts", "check-bundle-budget.ts")],
    { encoding: "utf8" },
  );
  return { code: r.status, out: r.stdout ?? "", err: r.stderr ?? "" };
}

test("a passing run prints the measurement it passed on", () => {
  const { code, out } = runIn(fixture());
  assert.equal(code, 0);
  assert.ok(MEASURED_TWO.test(out), out);
  assert.ok(out.includes("within budget"), out);
});

test("running through a symlinked checkout still runs the check", () => {
  // `import.meta.url` is the real path and `process.argv[1]` is the path as typed,
  // so comparing them literally turned the whole script into a silent exit 0 for
  // anyone whose checkout is reached through a symlink -- which on macOS includes
  // anything under /tmp.
  const root = fixture();
  const link = `${root}-link`;
  try {
    symlinkSync(root, link, "junction");
  } catch {
    return; // Unprivileged Windows without developer mode; nothing to assert.
  }
  const { code, out } = runIn(root, link);
  assert.equal(code, 0);
  assert.ok(
    out.includes("eager startup JS"),
    "the script ran but measured nothing",
  );
});

test("an entry with no modulepreload links is a shape change, not a small app", () => {
  // What `build.modulePreload: false` emits. Flattened into one list it reads as a
  // one-chunk app comfortably inside budget, which is the worst available answer.
  const html = INDEX_HTML.replace(/<link rel="modulepreload"[^>]*>\n?/, "");
  const { code, err } = runIn(fixture({ html }));
  assert.equal(code, 2);
  assert.ok(err.includes("modulepreload"), err);
  assert.ok(err.includes("nothing trustworthy to measure"), err);
});

test("the inlined-entry layout is measured, not rejected", () => {
  // When the entry module is nothing but imports, Vite drops the entry chunk and
  // emits one `<script type="module">` per imported chunk with no preload links.
  // That is a complete eager set, so it has to pass rather than trip the guard.
  const html = `<!doctype html><html><head>
<script type="module" crossorigin src="/assets/index-aaa.js"></script>
<script type="module" crossorigin src="/assets/react-bbb.js"></script>
</head><body></body></html>`;
  const { code, out } = runIn(fixture({ html }));
  assert.equal(code, 0);
  assert.ok(out.includes("2 chunks"), out);
});

test("preload links without a module entry are a shape change, not a measurement", () => {
  // A modulepreload link announces the entry's static import closure, so links
  // surviving while the entry does not means the entry was misread. The links are
  // numerous enough to carry a total-only guard on their own, and the chunk that
  // goes missing is the largest one on the startup path.
  const html = INDEX_HTML.replace(/<script type="module"[^>]*><\/script>\n?/, "");
  const { code, err } = runIn(fixture({ html }));
  assert.equal(code, 2);
  assert.ok(err.includes("no module entry"), err);
  assert.ok(err.includes("nothing trustworthy to measure"), err);
});

test("hrefs that are not site-root asset paths are reported, not silently skipped", () => {
  // `base: "./"` or `base: "/studio/"` emits every href in a form this does not
  // read. Measuring the empty remainder as 0 bytes would pass forever.
  const html = INDEX_HTML.replace(/"\/assets\//g, '"./assets/');
  const { code, err } = runIn(fixture({ html }));
  assert.equal(code, 2);
  assert.ok(err.includes("`base`"), err);
});

test("a referenced chunk missing from the build fails cleanly, not with a stack", () => {
  const html = INDEX_HTML.replace("react-bbb.js", "react-does-not-exist.js");
  const { code, err } = runIn(fixture({ html }));
  assert.equal(code, 2);
  assert.ok(err.includes("react-does-not-exist.js"), err);
  assert.ok(!err.includes("at Object."), "should not be an uncaught exception");
});

test("no build at all exits 2 rather than passing at zero bytes", () => {
  const { code, err } = runIn(fixture({ html: null }));
  assert.equal(code, 2);
  assert.ok(err.includes("npm run build"), err);
});

test("over the budget exits 1 and says what to do about it", () => {
  const { code, out, err } = runIn(fixture({ bigChunk: true }));
  assert.equal(code, 1);
  assert.ok(out.includes("eager startup JS"), out);
  assert.ok(err.includes("over the startup budget"), err);
  assert.ok(err.includes("raise BUDGET"), err);
});
