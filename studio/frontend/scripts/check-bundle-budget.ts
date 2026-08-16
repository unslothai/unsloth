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
 * preload link and are correctly not counted. A parser-blocking classic
 * `<script src>` (public/theme-boot.js) is added to that: it is not Vite's, but it
 * runs before the module graph and so is part of the same wait.
 *
 * Raising a budget is a normal thing to do. Doing it in the same diff as the import
 * that needed it is the point.
 */

import { readFileSync, realpathSync } from "node:fs";
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
  // Measured 1,496.2 KB gzip / 5,207.2 KB raw at 17363f8a2.
  gzipBytes: 1_600_000,
  rawBytes: 5_500_000,
};

// The chunk count is reported but not budgeted. Splitting a page out of the entry
// raises it while lowering the bytes, which is the behaviour this is trying to
// reward; capping it would penalise the fix.

/** Reads an attribute off a single tag. HTML attribute names are case-insensitive. */
function attr(tag: string, name: string): string | undefined {
  const m = tag.match(
    new RegExp(`\\b${name}\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s"'>]+))`, "i"),
  );
  if (!m) {
    return undefined;
  }
  return m[1] ?? m[2] ?? m[3];
}

/** True for a valueless attribute like `defer`, which `attr` cannot see. */
function hasAttr(tag: string, name: string): boolean {
  return new RegExp(`\\s${name}(?=[\\s/>=])`, "i").test(tag);
}

/** `rel` is a space-separated token list, and its tokens are case-insensitive. */
function relTokens(tag: string): string[] {
  return (attr(tag, "rel") ?? "").toLowerCase().split(/\s+/).filter(Boolean);
}

/**
 * The eager set, in three buckets, as paths relative to `dist/`.
 *
 * `entry` and `preloads` are Vite's output and stay apart so the caller can tell
 * "Vite stopped emitting preload links" from "this page really does load one
 * chunk": flattened, a build with no preloads reads as a very small app.
 *
 * `blocking` is the classic `<script src>` in `<head>`, which is not Vite's and
 * carries no preload link, but is parser-blocking: it is downloaded and run before
 * the module graph starts. `public/theme-boot.js` is one, and left out it could
 * grow without limit inside a gate whose whole subject is startup JavaScript.
 */
export type EagerSet = {
  entry: string[];
  preloads: string[];
  blocking: string[];
};

/** Same-origin, build-relative, and not a traversal out of `dist/`. */
function distRelative(url: string | undefined): string | undefined {
  if (!url?.startsWith("/") || url.startsWith("//")) {
    return undefined; // Absent, external, or protocol-relative: not ours to budget.
  }
  const path = url.slice(1).split(/[?#]/)[0];
  return path && !path.split("/").includes("..") ? path : undefined;
}

/** What the browser fetches and runs before the app boots, in document order. */
export function eagerSetFromHtml(html: string): EagerSet {
  const set: EagerSet = { entry: [], preloads: [], blocking: [] };
  const seen = new Set<string>();
  const add = (into: string[], url: string | undefined, prefix = "") => {
    const path = distRelative(url);
    if (!path?.startsWith(prefix) || seen.has(path)) {
      return;
    }
    seen.add(path);
    into.push(path);
  };

  for (const tag of html.match(/<script\b[^>]*>/gi) ?? []) {
    const type = attr(tag, "type")?.toLowerCase();
    if (type === "module") {
      // Vite's entry, always one of its own hashed assets.
      add(set.entry, attr(tag, "src"), "assets/");
    } else if (!type || type === "text/javascript") {
      // A classic script blocks the parser wherever it sits, so it counts from
      // anywhere in the build, not just assets/. `defer`/`async` do not: those
      // do not hold up the first screen.
      if (!(hasAttr(tag, "defer") || hasAttr(tag, "async"))) {
        add(set.blocking, attr(tag, "src"));
      }
    }
  }
  for (const tag of html.match(/<link\b[^>]*>/gi) ?? []) {
    if (relTokens(tag).includes("modulepreload")) {
      add(set.preloads, attr(tag, "href"), "assets/");
    }
  }
  return set;
}

/** Flattened eager set, in the order the browser gets to it. */
export function eagerChunksFromHtml(html: string): string[] {
  const { entry, preloads, blocking } = eagerSetFromHtml(html);
  return [...blocking, ...entry, ...preloads];
}

type Measured = { name: string; raw: number; gzip: number };

function measure(names: string[]): Measured[] | string {
  const out: Measured[] = [];
  for (const name of names) {
    let bytes: Buffer;
    try {
      bytes = readFileSync(join(DIST, name));
    } catch {
      // index.html names a file the build did not emit. Reporting it rather than
      // a stack trace, because the alternative reading -- that the budget is fine
      // -- is the one that must never be reachable.
      return `dist/index.html references ${name}, which is not in the build`;
    }
    out.push({
      name,
      raw: bytes.byteLength,
      gzip: gzipSync(bytes, { level: 6 }).byteLength,
    });
  }
  return out;
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

  const { entry, preloads, blocking } = eagerSetFromHtml(html);
  const names = [...blocking, ...entry, ...preloads];

  // Counting Vite's own output only: a parser-blocking classic script is not
  // evidence that the module graph was read correctly, so it cannot stand in for
  // the entry when deciding whether this still understands the build.
  //
  // A code-split build of this app is dozens of chunks. One or none means the
  // shape this reads has changed and the number below would be fiction: with
  // `build.modulePreload: false` the links disappear and the entry alone measured
  // 424 KB of a 5,207 KB startup path, reporting 4.8 MB to spare. A comfortable
  // pass is the one answer this must never give by accident.
  //
  // Counting scripts and links together rather than requiring both: when the entry
  // module is nothing but imports, Vite inlines it into one `<script>` per imported
  // chunk and emits no preload links at all, which is a complete measurement.
  const fromVite = entry.length + preloads.length;
  if (fromVite < 2) {
    console.error(
      `dist/index.html yielded ${fromVite} eager chunk(s) from Vite, so there is nothing trustworthy to measure here.`,
    );
    console.error(
      'A code-split build served from the site root gives a `<script type="module" src="/assets/...">` plus one `<link rel="modulepreload" href="/assets/...">` per statically imported chunk. If the shape changed on purpose -- `build.modulePreload` turned off, a non-root or relative `base`, a different `build.assetsDir`, `renderBuiltUrl` pointing at a CDN -- teach scripts/check-bundle-budget.ts the new shape rather than leaving a gate that measures nothing.',
    );
    return 2;
  }

  const sized = measure(names);
  if (typeof sized === "string") {
    console.error(sized);
    return 2;
  }
  const measured = sized.sort((a, b) => b.raw - a.raw);
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

/**
 * True when this file was run, rather than imported by the tests.
 *
 * Compared through realpath on both sides. `import.meta.url` is already the real
 * path (node resolves modules through symlinks), while `process.argv[1]` is the
 * path as typed, so a checkout reached through a symlinked directory made the two
 * disagree and the whole check became a silent no-op that exited 0.
 */
function invokedDirectly(): boolean {
  const argv = process.argv[1];
  if (!argv) {
    return false;
  }
  const here = fileURLToPath(import.meta.url);
  try {
    return realpathSync(argv) === realpathSync(here);
  } catch {
    return resolve(argv) === resolve(here);
  }
}

if (invokedDirectly()) {
  process.exit(main());
}
