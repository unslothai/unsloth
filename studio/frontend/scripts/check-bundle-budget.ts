// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Budget for the JavaScript Studio must download, parse and execute before the
 * first screen exists.
 *
 * Two PRs (#8623, #8624) moved 1.5 MB of decoded resources off this path by hand,
 * each measuring it with a throwaway Chromium harness. Nothing then stopped the
 * next static import from putting it back, and nothing noticed when one did: a
 * single `import` of a dialog that is closed on load was carrying 2.4 MB.
 *
 * The eager set is not inferred here. Vite already computes it and writes it into
 * index.html: the entry `<script type="module">` plus one `<link rel="modulepreload">`
 * per chunk in the entry's STATIC import closure. That is exactly what the browser
 * fetches before the app boots. Chunks reachable only through `import()` carry no
 * preload link and are correctly not counted.
 *
 * Raising a budget is a normal thing to do. Doing it in the same diff as the import
 * that needed it is the point.
 */

import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = resolve(HERE, "..", "dist");

/**
 * Measured on the build these were set from, plus a little headroom. Gzip is what
 * crosses the wire; raw is what the main thread has to parse and execute, which is
 * the part that shows up as a slow launch on a weak machine.
 */
export const BUDGET = {
  // Measured 1,496.2 KB gzip / 5,207.2 KB raw over 44 chunks at 17363f8a2.
  gzipBytes: 1_600_000,
  rawBytes: 5_500_000,
  chunks: 48,
};

/** The entry script and every chunk Vite preloads for it, in document order. */
export function eagerChunksFromHtml(html: string): string[] {
  const found: string[] = [];
  const seen = new Set<string>();
  const add = (href: string) => {
    // Only the build's own emitted assets; a copied-in vendor file has no budget.
    if (!href.startsWith("/assets/") || seen.has(href)) {
      return;
    }
    seen.add(href);
    found.push(href.slice("/assets/".length));
  };

  for (const tag of html.match(/<script\b[^>]*>/g) ?? []) {
    if (!/\btype\s*=\s*["']module["']/.test(tag)) {
      continue;
    }
    const src = tag.match(/\bsrc\s*=\s*["']([^"']+)["']/);
    if (src) {
      add(src[1]);
    }
  }
  for (const tag of html.match(/<link\b[^>]*>/g) ?? []) {
    if (!/\brel\s*=\s*["']modulepreload["']/.test(tag)) {
      continue;
    }
    const href = tag.match(/\bhref\s*=\s*["']([^"']+)["']/);
    if (href) {
      add(href[1]);
    }
  }
  return found;
}

type Measured = { name: string; raw: number; gzip: number };

function measure(names: string[]): Measured[] {
  return names.map((name) => {
    const bytes = readFileSync(join(DIST, "assets", name));
    return {
      name,
      raw: bytes.byteLength,
      gzip: gzipSync(bytes, { level: 6 }).byteLength,
    };
  });
}

function kb(bytes: number): string {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function main(): number {
  let html: string;
  try {
    html = readFileSync(join(DIST, "index.html"), "utf8");
  } catch {
    console.error("no dist/index.html -- run `npm run build` first");
    return 2;
  }

  const names = eagerChunksFromHtml(html);
  if (names.length === 0) {
    // A build that emits no preload links means the shape this reads has changed,
    // which would otherwise report a comfortable 0 bytes forever.
    console.error(
      "no eager chunks found in dist/index.html; the entry script or the " +
        "modulepreload links are no longer where this expects them",
    );
    return 2;
  }

  const measured = measure(names).sort((a, b) => b.raw - a.raw);
  const raw = measured.reduce((sum, c) => sum + c.raw, 0);
  const gzip = measured.reduce((sum, c) => sum + c.gzip, 0);

  console.log(
    `eager startup JS: ${kb(raw)} raw, ${kb(gzip)} gzip, ${measured.length} chunks`,
  );
  console.log("largest:");
  for (const c of measured.slice(0, 8)) {
    console.log(
      `  ${kb(c.raw).padStart(10)} raw  ${kb(c.gzip).padStart(9)} gzip  ${c.name}`,
    );
  }

  const over: string[] = [];
  if (gzip > BUDGET.gzipBytes) {
    over.push(`gzip ${kb(gzip)} > ${kb(BUDGET.gzipBytes)}`);
  }
  if (raw > BUDGET.rawBytes) {
    over.push(`raw ${kb(raw)} > ${kb(BUDGET.rawBytes)}`);
  }
  if (measured.length > BUDGET.chunks) {
    over.push(`${measured.length} chunks > ${BUDGET.chunks}`);
  }

  if (over.length > 0) {
    console.error(`\nover the startup budget: ${over.join(", ")}`);
    console.error(
      "Something is now imported statically that the first screen does not need. " +
        "Either load it on use (React.lazy, lazyRouteComponent, or a dynamic import " +
        "at the point of use), or raise BUDGET in this file in the same PR, with the " +
        "measurement that justifies it.",
    );
    return 1;
  }
  console.log(
    `\nwithin budget (${kb(BUDGET.gzipBytes - gzip)} gzip, ${kb(BUDGET.rawBytes - raw)} raw to spare)`,
  );
  return 0;
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))
) {
  process.exit(main());
}
