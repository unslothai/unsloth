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
 * Measured on the build these were set from, plus a little headroom.
 *
 * Transfer is what crosses the wire. Raw is what the main thread has to parse and
 * execute, which is the part that shows up as a slow launch on a weak machine.
 */
export const BUDGET = {
  // Measured 1,496.2 KB transfer / 5,207.2 KB raw at 17363f8a2.
  transferBytes: 1_600_000,
  rawBytes: 5_500_000,
};

// The chunk count is reported but not budgeted. Splitting a page out of the entry
// raises it while lowering the bytes, which is the behaviour this is trying to
// reward; capping it would penalise the fix.

/**
 * A start tag: lowercased name, and its attributes by lowercased name. A valueless
 * attribute like `defer` is present with an empty-string value, as HTML defines it.
 */
type StartTag = { name: string; attrs: Map<string, string> };

/**
 * ASCII whitespace, which is what separates one attribute from the next.
 *
 * CR is in here for safety only: the parser's input preprocessor turns every CR
 * into an LF before the tokenizer sees it, so a CRLF file cannot behave
 * differently. https://html.spec.whatwg.org/multipage/parsing.html#preprocessing-the-input-stream
 */
const WHITESPACE = new Set(["\t", "\n", "\f", "\r", " "]);

/** First index at or after `from` that is not ASCII whitespace. */
function skipWhitespace(html: string, from: number): number {
  let i = from;
  while (i < html.length && WHITESPACE.has(html[i] as string)) {
    i += 1;
  }
  return i;
}

/** First index at or after `from` that ends a name or an unquoted value. */
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

/** One attribute value, from the `=`. Quoted values run to their closing quote. */
function readValue(
  html: string,
  from: number,
): { value: string; next: number } {
  const i = skipWhitespace(html, from + 1);
  const quote = html[i];
  if (quote !== '"' && quote !== "'") {
    const end = scanUntil(html, i, ">"); // Unquoted: ends at whitespace or `>`.
    return { value: html.slice(i, end), next: end };
  }
  const close = html.indexOf(quote, i + 1);
  return close < 0
    ? { value: html.slice(i + 1), next: html.length }
    : { value: html.slice(i + 1, close), next: close + 1 };
}

/**
 * Reads the attributes of one start tag, beginning just past its name, and returns
 * where the tag ends.
 *
 * This is the tokenizer's attribute states, narrowed to what a build artefact can
 * contain. The two rules that matter, and that no regex over the tag text can
 * express:
 *
 *   - `>` ends the tag only OUTSIDE a quoted value. Inside one it is ordinary text
 *     ("anything else: append the current input character to the current
 *     attribute's value"), so `data-note="a > b"` is one attribute and the tag does
 *     not end there.
 *     https://html.spec.whatwg.org/multipage/parsing.html#attribute-value-(double-quoted)-state
 *   - An attribute begins only after whitespace, `/` or a previous value, so a NAME
 *     is the only place `async` or `type` can be read from. A value is arbitrary
 *     text: `data-mode="load async later"` contains no `async` attribute.
 *     https://html.spec.whatwg.org/multipage/parsing.html#before-attribute-name-state
 *
 * Both were live bugs while this searched the tag text instead: the first dropped
 * the entry chunk, the second dropped the parser-blocking script, and in each case
 * enough of the build survived to satisfy the shape guard below, so the gate
 * reported a comfortable pass on a startup path it had not measured.
 */
function readAttributes(
  html: string,
  start: number,
): { attrs: Map<string, string>; end: number; closed: boolean } {
  const attrs = new Map<string, string>();
  let i = start;
  while (i < html.length) {
    // Before attribute name: whitespace is ignored, and `/` (self-closing, or a
    // stray solidus) is not part of a name.
    if (WHITESPACE.has(html[i] as string) || html[i] === "/") {
      i += 1;
      continue;
    }
    if (html[i] === ">") {
      return { attrs, end: i + 1, closed: true };
    }
    const nameEnd = scanUntil(html, i, "/>=");
    const name = html.slice(i, nameEnd).toLowerCase();
    // After attribute name: whitespace may separate the name from its `=`.
    i = skipWhitespace(html, nameEnd);
    let value = "";
    if (html[i] === "=") {
      ({ value, next: i } = readValue(html, i));
    }
    // A repeated attribute is a parse error and the browser keeps the first, so a
    // later `src=` cannot overwrite the one the browser actually fetches.
    if (name && !attrs.has(name)) {
      attrs.set(name, value);
    }
  }
  return { attrs, end: i, closed: false };
}

/**
 * End of a comment, from just past its `<!--`.
 *
 * `<!-->` and `<!--->` close there rather than running on: the comment start and
 * comment start dash states both end the comment on `>`. Read as unterminated, the
 * rest of the file disappears along with the tags in it.
 * https://html.spec.whatwg.org/multipage/parsing.html#comment-start-state
 */
const COMMENT_END = /^-?>|--!?>/;
function endOfComment(html: string, start: number): number {
  const m = COMMENT_END.exec(html.slice(start));
  return m ? start + m.index + m[0].length : html.length;
}

/** End of a script element's raw text, from just past its start tag. */
const SCRIPT_END = /<\/script[\t\n\f\r >/]/i;
function endOfScriptBody(html: string, start: number): number {
  const m = SCRIPT_END.exec(html.slice(start));
  return m ? start + m.index : html.length;
}

/** Tag open state: only an ASCII letter after `<` starts a tag name. */
const TAG_NAME_START = /[a-z]/i;

/**
 * Every start tag in the document, in order.
 *
 * Comments and script bodies are skipped rather than scanned, for the same reason
 * the attribute parser exists: a tag is only a tag where the browser sees one. A
 * `<script src>` commented out during debugging is not downloaded and must not be
 * charged, and a tag written inside a string in an inline script is text.
 * https://html.spec.whatwg.org/multipage/parsing.html#script-data-state
 *
 * A script body ends at the first `</script`, which is the tokenizer's answer
 * unless the body itself contains `<!-- <script`: those escaped states let a
 * `</script>` be text. Erring there reads a bit of script body as markup, so the
 * mistake is to charge a chunk that is not there rather than to miss one. Checked
 * against parse5 over 40,000 generated documents, that is the only case left.
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
    // Tag name state: the name ends at whitespace, `/` or `>`.
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

/** Reads an attribute off a tag. Absent is undefined; valueless is `""`. */
function attr(tag: StartTag, name: string): string | undefined {
  return tag.attrs.get(name);
}

/** True for a valueless attribute like `defer`, which carries no value to read. */
function hasAttr(tag: StartTag, name: string): boolean {
  return tag.attrs.has(name);
}

/** `rel` is a space-separated token list, and its tokens are case-insensitive. */
function relTokens(tag: StartTag): string[] {
  return (attr(tag, "rel") ?? "").toLowerCase().split(/\s+/).filter(Boolean);
}

/**
 * True when the tag carries `blocking="render"`, which holds the FIRST RENDER back
 * until the resource has been fetched and evaluated.
 *
 * The spec's own reading: "Let value be the value of el's blocking attribute...
 * converted to ASCII lowercase... split on ASCII whitespace", then "An element is
 * potentially render-blocking if its blocking tokens set contains 'render'", and in
 * prepare the script element, "If el is potentially render-blocking, then block
 * rendering on el" -- which is reached for any external script, `async` or not. The
 * async/defer carve-out applies only to what is IMPLICITLY render-blocking.
 * https://html.spec.whatwg.org/multipage/urls-and-fetching.html#blocking-attributes
 * https://html.spec.whatwg.org/multipage/scripting.html#prepare-the-script-element
 *
 * Measured, not assumed: a `<script async blocking="render" src>` held back for two
 * seconds moved first contentful paint from 20 ms to 2,020 ms in Chromium 151 and
 * from 11 ms to 2,009 ms in WebKit 26.5, with Chromium reporting the request's
 * renderBlockingStatus as "blocking". Firefox has not shipped it (bugzil.la/1751383)
 * and simply treats the script as async.
 */
function blocksRender(tag: StartTag): boolean {
  return (attr(tag, "blocking") ?? "")
    .toLowerCase()
    .split(/\s+/)
    .includes("render");
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

/**
 * A `type` the browser still runs as a classic script. An absent type is the same
 * thing; anything else (`importmap`, `application/json`, a template) is not code
 * that runs, and `module` is handled on its own.
 *
 * Exact strings, deliberately. The spec matches this attribute on JavaScript MIME
 * type ESSENCE, so a parameter makes it match nothing: `text/javascript;
 * charset=utf-8` is not evaluated, and Chromium, Firefox and WebKit do not even
 * fetch such a script. Bytes the browser never requests are not startup cost.
 *
 * The whole essence list, not the four anyone would write today. The legacy
 * spellings are not historical trivia: every one of them still executes. Measured
 * in Chromium 151, a script tagged `application/x-javascript`, `text/jscript`,
 * `text/javascript1.5`, `text/livescript`, `application/x-ecmascript` or
 * `text/x-javascript` runs, while `text/javascript; charset=utf-8` and
 * `application/json` do not. The list is frozen, so this does not grow.
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

  const tags = [...startTags(html)];
  for (const tag of tags.filter((t) => t.name === "script")) {
    const type = attr(tag, "type")?.toLowerCase();
    if (type === "module") {
      // Vite's entry, always one of its own hashed assets.
      add(set.entry, attr(tag, "src"), "assets/");
    } else if (!type || CLASSIC_TYPES.has(type)) {
      // Counts from anywhere in the build, not just assets/. `defer` is included:
      // a deferred script runs after parsing but BEFORE DOMContentLoaded, in
      // document order with the module entry, which is itself deferred -- so it is
      // on exactly the timeline this budgets. Only `async` is out, having no
      // ordering relationship to the first screen at all. `async` also wins when
      // both are present, which is why it is the one tested.
      //
      // Unless it is asked to block rendering, which restores the relationship the
      // exclusion assumes is absent: `blocking="render"` is the documented way to
      // keep a boot script off the parser without letting the unthemed page paint,
      // and it delays the first screen by the whole fetch and evaluation. Not
      // counting it would leave the one thing this gate exists to bound -- bytes
      // between the navigation and the first screen -- unbounded.
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

/** Flattened eager set, in the order the browser gets to it. */
export function eagerChunksFromHtml(html: string): string[] {
  const { entry, preloads, blocking } = eagerSetFromHtml(html);
  return [...blocking, ...entry, ...preloads];
}

type Measured = { name: string; raw: number; transfer: number };

/**
 * What the browser actually downloads. The backend gzips the `/assets` mount only
 * (studio/backend/main.py mounts `_AssetGZipMiddleware` there); everything else
 * goes out through a plain FileResponse, so its raw size IS its transfer size.
 */
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
      // index.html names a file the build did not emit. Reporting it rather than
      // a stack trace, because the alternative reading -- that the budget is fine
      // -- is the one that must never be reachable.
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
  // chunk and emits no preload links at all, which is a complete measurement. So
  // the total is what decides whether this is a code-split build.
  //
  // But at least one entry is required on top of that total, because preloads
  // without one is not a build shape Vite emits: a modulepreload link exists to
  // announce the entry's static import closure, so links surviving while the entry
  // does not means the entry was read wrong, not that it is absent. Left to the
  // total alone, the app's 48 links carry the guard while the largest single chunk
  // in the startup path silently leaves the measurement -- which is exactly how
  // two mis-parses of this file's own making stayed invisible.
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
