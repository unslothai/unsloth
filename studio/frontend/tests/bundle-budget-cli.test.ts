// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Runs the script as CI does: every pass must also print a measurement, not exit 0 silently. */

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

/** Reuse this process's TS loader flags; `--experimental-strip-types` varies across node versions. */
const TS_FLAGS = process.execArgv.filter((flag) =>
  /^--(experimental-)?(strip-types|transform-types)/.test(flag),
);

const MEASURED_TWO = /eager startup JS: .* raw, .* transfer, 2 chunks/;

const INDEX_HTML = `<!doctype html><html><head>
<script type="module" crossorigin src="/assets/index-aaa.js"></script>
<link rel="modulepreload" crossorigin href="/assets/react-bbb.js">
<link rel="stylesheet" crossorigin href="/assets/index-ccc.css">
</head><body><div id="root"></div></body></html>`;

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
  // import.meta.url is the real path and argv[1] is as typed, so symlinked checkouts (macOS /tmp)
  // must not be compared literally.
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
  const html = INDEX_HTML.replace(/<link rel="modulepreload"[^>]*>\n?/, "");
  const { code, err } = runIn(fixture({ html }));
  assert.equal(code, 2);
  assert.ok(err.includes("modulepreload"), err);
  assert.ok(err.includes("nothing trustworthy to measure"), err);
});

test("the inlined-entry layout is measured, not rejected", () => {
  // An imports-only entry makes Vite emit one module script per chunk with no preloads; that must pass.
  const html = `<!doctype html><html><head>
<script type="module" crossorigin src="/assets/index-aaa.js"></script>
<script type="module" crossorigin src="/assets/react-bbb.js"></script>
</head><body></body></html>`;
  const { code, out } = runIn(fixture({ html }));
  assert.equal(code, 0);
  assert.ok(out.includes("2 chunks"), out);
});

test("preload links without a module entry are a shape change, not a measurement", () => {
  const html = INDEX_HTML.replace(/<script type="module"[^>]*><\/script>\n?/, "");
  const { code, err } = runIn(fixture({ html }));
  assert.equal(code, 2);
  assert.ok(err.includes("no module entry"), err);
  assert.ok(err.includes("nothing trustworthy to measure"), err);
});

test("hrefs that are not site-root asset paths are reported, not silently skipped", () => {
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
