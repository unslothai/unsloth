// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Budget for JavaScript fetched and run before the first screen: Vite's entry plus modulepreload
 * links (static import closure) plus parser-blocking classic scripts such as public/theme-boot.js.
 */

import { readFileSync, realpathSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const HERE = dirname(fileURLToPath(import.meta.url));
const DIST = resolve(HERE, "..", "dist");

/** Measured plus headroom. Transfer = wire bytes; raw = bytes the main thread parses. */
export const BUDGET = {
  transferBytes: 1_991_000,
  rawBytes: 6_646_000,
};

// Chunk count is not budgeted: splitting a page out raises it while lowering bytes.

/** Lowercased tag name and attributes; a valueless attribute has value "". */
type StartTag = { name: string; attrs: Map<string, string> };

/** ASCII whitespace; CR is defensive since the HTML preprocessor turns CR into LF. */
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
 * Tokenizer attribute states: `>` inside a quoted value does not end the tag, and only an
 * attribute NAME can be `async`/`type`. Regex over tag text got both wrong.
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
    // Duplicate attributes: the browser keeps the first.
    if (name && !attrs.has(name)) {
      attrs.set(name, value);
    }
  }
  return { attrs, end: i, closed: false };
}

/** Comment end; `<!-->` and `<!--->` close immediately per the comment start states. */
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
 * Every start tag in order, skipping comments and script bodies as the browser does.
 * Escaped `<!-- <script` bodies may over-count a chunk, never miss one (checked vs parse5).
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
      return;
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

function relTokens(tag: StartTag): string[] {
  return (attr(tag, "rel") ?? "").toLowerCase().split(/\s+/).filter(Boolean);
}

/**
 * blocking="render" delays first render for any external script, async included (HTML spec).
 */
function blocksRender(tag: StartTag): boolean {
  return (attr(tag, "blocking") ?? "")
    .toLowerCase()
    .split(/\s+/)
    .includes("render");
}

/**
 * Eager set relative to dist/. entry and preloads stay separate so missing preload links are
 * detectable; `blocking` holds parser-blocking classic scripts like public/theme-boot.js.
 */
export type EagerSet = {
  entry: string[];
  preloads: string[];
  blocking: string[];
};

/**
 * Types the browser runs as classic script: the full frozen JS MIME essence list, exact strings
 * (a parameter like `; charset=utf-8` makes it not run). mimesniff.spec.whatwg.org
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
      // `defer` counts (runs before DOMContentLoaded); `async` does not, unless blocking="render".
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

/** Only the `/assets` mount is gzipped by the backend; elsewhere raw size is transfer size. */
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
      // Report it: concluding the budget is fine must never be reachable here.
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

  // Shape guard: one or no Vite chunks means the build shape changed and the number is fiction.
  // Require an entry too: preloads without an entry means the entry was mis-parsed.
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
        "measurement that justifies it. If the chunk count above went up, check the " +
        "opposite first: a dynamic import of a module the startup set already carries " +
        "loads nothing later, and splits that module's graph into extra startup chunks.",
    );
    return 1;
  }
  console.log(
    `\nwithin budget (${kb(BUDGET.transferBytes - transfer)} transfer, ${kb(BUDGET.rawBytes - raw)} raw to spare)`,
  );
  return 0;
}

/** Run directly vs imported. Compare realpaths: argv[1] keeps symlinks, import.meta.url does not. */
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
