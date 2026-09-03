// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Budget for the JavaScript downloaded, parsed and executed before the first screen exists. The
 * eager set is not inferred: it is the entry `<script type="module">` and one modulepreload link
 * per chunk in its STATIC import closure, as Vite writes them into index.html, plus the
 * parser-blocking classic `<script src>`. Raise it in the same diff as the import that needed it.
 */

import { readFileSync, realpathSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = resolve(HERE, "..", "dist");

/** Transfer crosses the wire; raw is what the main thread parses and executes. */
export const BUDGET = {
  // Measured 1,496.2 KB transfer / 5,207.2 KB raw at 17363f8a2.
  // Raised for the audio placement control: same build both sides, merge base
  // 1,560.9 KB transfer against branch 1,562.6 KB, so it crossed the old 1,562.5 KB
  // ceiling by a tenth of a kilobyte.
  // Find in page is carried eagerly rather than split behind `lazy`, since the shell prevents the
  // chord's default before the chunk would exist and idle warming never runs on Safari, which
  // keeps requestIdleCallback behind a flag. It fits inside the ceiling above with 6.0 KB spare,
  // so it does not raise it.
  transferBytes: 1_620_000,
  rawBytes: 5_500_000,
};

// The chunk count is reported but not budgeted: splitting a page out of the entry raises it while
// lowering the bytes, so capping it would penalise the fix.

/** A start tag. A valueless attribute like `defer` is present with an empty-string value. */
type StartTag = { name: string; attrs: Map<string, string> };

const WHITESPACE = new Set(["\t", "\n", "\f", "\r", " "]);

function skipWhitespace(html: string, from: number): number {
  let i = from;
  while (i < html.length && WHITESPACE.has(html[i] as string)) {
    i += 1;
  }
  return i;
}

function scanUntil(html: string, from: number, stop: string): number {
  let i = from;
  while (
    i < html.length &&
    !WHITESPACE.has(html[i] as string) &&
    !stop.includes(html[i] as string)
  ) {
    i += 1;
  }
  return i;
}

function readValue(
  html: string,
  from: number,
): { value: string; next: number } {
  const i = skipWhitespace(html, from + 1);
  const quote = html[i];
  if (quote !== '"' && quote !== "'") {
    const end = scanUntil(html, i, ">");
    return { value: html.slice(i, end), next: end };
  }
  const close = html.indexOf(quote, i + 1);
  return close < 0
    ? { value: html.slice(i + 1), next: html.length }
    : { value: html.slice(i + 1, close), next: close + 1 };
}

/**
 * Reads one start tag's attributes, from just past its name. The two rules no regex over the tag
 * text can express, both live bugs when this searched that text: `>` ends the tag only OUTSIDE a
 * quoted value, and an attribute begins only after whitespace, `/` or a previous value, so
 * `data-mode="load async later"` has no `async` attribute.
 */
function readAttributes(
  html: string,
  start: number,
): { attrs: Map<string, string>; end: number; closed: boolean } {
  const attrs = new Map<string, string>();
  let i = start;
  while (i < html.length) {
    if (WHITESPACE.has(html[i] as string) || html[i] === "/") {
      i += 1;
      continue;
    }
    if (html[i] === ">") {
      return { attrs, end: i + 1, closed: true };
    }
    const nameEnd = scanUntil(html, i, "/>=");
    const name = html.slice(i, nameEnd).toLowerCase();
    i = skipWhitespace(html, nameEnd);
    let value = "";
    if (html[i] === "=") {
      ({ value, next: i } = readValue(html, i));
    }
    // The browser keeps the first of a repeated attribute, so a later `src=` cannot overwrite it.
    if (name && !attrs.has(name)) {
      attrs.set(name, value);
    }
  }
  return { attrs, end: i, closed: false };
}

/** `<!-->` and `<!--->` close there rather than running on; read as unterminated, the rest of the
 *  file disappears along with the tags in it. */
const COMMENT_END = /^-?>|--!?>/;
function endOfComment(html: string, start: number): number {
  const m = COMMENT_END.exec(html.slice(start));
  return m ? start + m.index + m[0].length : html.length;
}

const SCRIPT_END = /<\/script[\t\n\f\r >/]/i;
function endOfScriptBody(html: string, start: number): number {
  const m = SCRIPT_END.exec(html.slice(start));
  return m ? start + m.index : html.length;
}

const TAG_NAME_START = /[a-z]/i;

/**
 * Every start tag in the document. Comments and script bodies are skipped rather than scanned: a
 * tag is only a tag where the browser sees one, so a commented-out `<script src>` must not be
 * charged. A script body ends at the first `</script`, wrong only inside the `<!-- <script` escaped
 * states, where erring overcharges rather than misses.
 */
function* startTags(html: string): Generator<StartTag> {
  let i = 0;
  while (i < html.length) {
    const lt = html.indexOf("<", i);
    if (lt < 0) {
      return;
    }
    if (html.startsWith("<!--", lt)) {
      i = endOfComment(html, lt + 4);
      continue;
    }
    if (!TAG_NAME_START.test(html[lt + 1] ?? "")) {
      i = lt + 1;
      continue;
    }
    const nameEnd = scanUntil(html, lt + 1, "/>");
    const name = html.slice(lt + 1, nameEnd).toLowerCase();
    const { attrs, end, closed } = readAttributes(html, nameEnd);
    if (!closed) {
      return; // A tag the file ends in the middle of is never emitted, or fetched.
    }
    yield { name, attrs };
    i = name === "script" ? endOfScriptBody(html, end) : end;
  }
}

function attr(tag: StartTag, name: string): string | undefined {
  return tag.attrs.get(name);
}

function hasAttr(tag: StartTag, name: string): boolean {
  return tag.attrs.has(name);
}

/** `rel` is a space-separated token list, and its tokens are case-insensitive. */
function relTokens(tag: StartTag): string[] {
  return (attr(tag, "rel") ?? "").toLowerCase().split(/\s+/).filter(Boolean);
}

/** `blocking="render"` holds the first render back for any external script, `async` or not: the
 *  async/defer carve-out covers only what is IMPLICITLY render-blocking. */
function blocksRender(tag: StartTag): boolean {
  return (attr(tag, "blocking") ?? "")
    .toLowerCase()
    .split(/\s+/)
    .includes("render");
}

/**
 * Paths relative to `dist/`. `entry` and `preloads` stay apart so the caller can tell "Vite stopped
 * emitting preload links" from "this page really does load one chunk"; `blocking` is the classic
 * parser-blocking `<script src>`, which carries no preload link but runs before the module graph.
 */
export type EagerSet = {
  entry: string[];
  preloads: string[];
  blocking: string[];
};

/**
 * A `type` the browser still runs as a classic script. Exact strings, deliberately: matching is on
 * JavaScript MIME type ESSENCE, so a parameter matches nothing and `text/javascript;
 * charset=utf-8` is never even fetched. The whole frozen list, since every legacy spelling runs.
 * https://mimesniff.spec.whatwg.org/#javascript-mime-type
 */
const CLASSIC_TYPES = new Set([
  "application/ecmascript",
  "application/javascript",
  "application/x-ecmascript",
  "application/x-javascript",
  "text/ecmascript",
  "text/javascript",
  "text/javascript1.0",
  "text/javascript1.1",
  "text/javascript1.2",
  "text/javascript1.3",
  "text/javascript1.4",
  "text/javascript1.5",
  "text/jscript",
  "text/livescript",
  "text/x-ecmascript",
  "text/x-javascript",
]);

/** Same-origin, build-relative, and not a traversal out of `dist/`. */
function distRelative(url: string | undefined): string | undefined {
  if (!url?.startsWith("/") || url.startsWith("//")) {
    return undefined;
  }
  const path = url.slice(1).split(/[?#]/)[0];
  return path && !path.split("/").includes("..") ? path : undefined;
}

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

  const tags = [...startTags(html)];
  for (const tag of tags.filter((t) => t.name === "script")) {
    const type = attr(tag, "type")?.toLowerCase();
    if (type === "module") {
      add(set.entry, attr(tag, "src"), "assets/");
    } else if (!type || CLASSIC_TYPES.has(type)) {
      // `defer` counts: it runs in document order with the module entry, which is itself deferred.
      // Only `async` is out, having no ordering relationship to the first screen unless it also
      // blocks rendering, which restores one.
      if (!hasAttr(tag, "async") || blocksRender(tag)) {
        add(set.blocking, attr(tag, "src"));
      }
    }
  }
  for (const tag of tags.filter((t) => t.name === "link")) {
    if (relTokens(tag).includes("modulepreload")) {
      add(set.preloads, attr(tag, "href"), "assets/");
    }
  }
  return set;
}

export function eagerChunksFromHtml(html: string): string[] {
  const { entry, preloads, blocking } = eagerSetFromHtml(html);
  return [...blocking, ...entry, ...preloads];
}

type Measured = { name: string; raw: number; transfer: number };

/** The backend gzips the `/assets` mount only (`_AssetGZipMiddleware` in studio/backend/main.py);
 *  everything else goes out through a plain FileResponse, so its raw size IS its transfer size. */
function transferBytes(name: string, bytes: Buffer): number {
  return name.startsWith("assets/")
    ? gzipSync(bytes, { level: 6 }).byteLength
    : bytes.byteLength;
}

function measure(names: string[]): Measured[] | string {
  const out: Measured[] = [];
  for (const name of names) {
    let bytes: Buffer;
    try {
      bytes = readFileSync(join(DIST, name));
    } catch {
      // Reported rather than thrown, because the reading that must never be reachable is "fine".
      return `dist/index.html references ${name}, which is not in the build`;
    }
    out.push({
      name,
      raw: bytes.byteLength,
      transfer: transferBytes(name, bytes),
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

  // Vite's own output only: a parser-blocking classic script is no evidence the module graph was
  // read correctly. A code-split build here is dozens of chunks, so one or none means the shape
  // has changed and the number below is fiction. Counted together rather than both required, since
  // an entry of nothing but imports is inlined per chunk with no preloads; one entry is required
  // on top, because preloads announce its import closure, so links without it read it wrong.
  const fromVite = entry.length + preloads.length;
  if (entry.length === 0 || fromVite < 2) {
    console.error(
      entry.length === 0
        ? `dist/index.html yielded no module entry (and ${preloads.length} preload link(s)), so there is nothing trustworthy to measure here.`
        : `dist/index.html yielded ${fromVite} eager chunk(s) from Vite, so there is nothing trustworthy to measure here.`,
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
  const transfer = measured.reduce((sum, c) => sum + c.transfer, 0);

  console.log(
    `eager startup JS: ${kb(raw)} raw, ${kb(transfer)} transfer, ${measured.length} chunks`,
  );
  console.log("largest:");
  for (const c of measured.slice(0, 8)) {
    console.log(
      `  ${kb(c.raw).padStart(10)} raw  ${kb(c.transfer).padStart(9)} transfer  ${c.name}`,
    );
  }

  const over: string[] = [];
  if (transfer > BUDGET.transferBytes) {
    over.push(`transfer ${kb(transfer)} > ${kb(BUDGET.transferBytes)}`);
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
    `\nwithin budget (${kb(BUDGET.transferBytes - transfer)} transfer, ${kb(BUDGET.rawBytes - raw)} raw to spare)`,
  );
  return 0;
}

/** Compared through realpath on both sides: `process.argv[1]` is the path as typed, so a checkout
 *  reached through a symlink made the two disagree and the whole check became a no-op exiting 0. */
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
