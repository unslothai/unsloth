// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Safari only honours a clipboard write while the click that asked for it is still
// live, so both copy buttons must have their text in hand before the handler runs.
// Each test below pins one way that breaks.
//
// Read from the source: the node suite has no DOM to click in.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function read(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

const HISTORY = read("features/studio/history-card-grid.tsx");
const GGUF = read("features/hub/catalog/gguf-download-card.tsx");

/** The braced block opening after `from`, matched to its close. */
function block(source: string, from: number): string {
  const open = source.indexOf("{", from);
  assert.notEqual(open, -1, "no block at anchor");
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === "{") depth += 1;
    else if (source[i] === "}") {
      depth -= 1;
      if (depth === 0) return source.slice(open, i + 1);
    }
  }
  throw new Error("unbalanced block");
}

/** The body of the handler enclosing `anchor`. */
function handlerAt(source: string, anchor: string, opener: string): string {
  const at = source.indexOf(anchor);
  assert.notEqual(at, -1, `${anchor} not found`);
  const start = source.lastIndexOf(opener, at);
  assert.notEqual(start, -1, `${opener} not found before ${anchor}`);
  return block(source, source.indexOf("=>", start));
}

const COPY_PREVIEW = handlerAt(HISTORY, 'run.preview_ref ?? ""', "onClick={");
const COPY_PATH = block(
  GGUF,
  GGUF.indexOf("const handleCopyPath = useCallback(async () => {"),
);
const PREFETCH_PATH = block(
  GGUF,
  GGUF.indexOf("const prefetchCachedPath = useCallback(() => {"),
);

test("copy preview link writes before it awaits anything", () => {
  assert.ok(COPY_PREVIEW.includes("copyToClipboard("));
  assert.equal(
    COPY_PREVIEW.indexOf("await "),
    COPY_PREVIEW.indexOf("await copyToClipboard("),
    "something is awaited before the clipboard write",
  );
});

test("the preview link base is refreshed outside the click", () => {
  assert.ok(
    !COPY_PREVIEW.includes("fetchDeviceType"),
    "re-reading /api/health in the click costs the gesture",
  );
  assert.ok(
    HISTORY.includes("fetchDeviceType({ force: true })"),
    "without a forced re-read the link stays local",
  );
  assert.ok(HISTORY.includes('window.addEventListener("focus"'));
});

test("the prefetched model path is keyed to the model it was fetched for", () => {
  assert.match(
    GGUF,
    /const pathKey = [^\n]*repoId[^\n]*quant/,
    "the cache key must cover repoId and quant",
  );
  assert.ok(
    PREFETCH_PATH.includes("freshPath(pathKey) !== null"),
    "the guard must miss once the menu moved to another model",
  );
  assert.match(
    GGUF,
    /entry\?\.key === key && Date\.now\(\) - entry\.at < CACHED_PATH_TTL_MS/,
    "a CLI download or a delete moves the snapshot without touching inventoryVersion",
  );
  assert.ok(
    COPY_PATH.includes("freshPath(pathKey) ??"),
    "the copy must not answer from an entry the prefetch would already have replaced",
  );
  assert.match(
    GGUF,
    /cachedPathRef\.current = \{ key: pathKey, path: resolved, at: Date\.now\(\) \}/,
  );
});

test("the link base is kept fresh for as long as the grid is on screen", () => {
  assert.match(
    HISTORY,
    /setInterval\(refreshLinkBase, LINK_BASE_POLL_MS\)/,
    "one read misses a tunnel published late, or one that dies",
  );
  assert.ok(
    !/cloudflareUrl !== null/.test(HISTORY),
    "a cached URL is exactly what goes stale",
  );
  assert.ok(
    !COPY_PREVIEW.includes("setInterval") &&
      !COPY_PREVIEW.includes("fetchDeviceType"),
    "the refresh belongs outside the click",
  );
});

test("refreshes do not stack when /api/health is slow", () => {
  const effect = block(
    HISTORY,
    HISTORY.indexOf("const refreshLinkBase = () => {"),
  );
  assert.ok(
    effect.includes("pollingSince"),
    "a forced read always writes, so a slow answer would restore the old URL",
  );
  assert.ok(
    effect.includes("LINK_BASE_STALL_MS"),
    "fetchDeviceType has no timeout: a read that never settles must not hold the guard",
  );
});

test("the cached path is dropped when the inventory moves", () => {
  assert.match(
    GGUF,
    /const pathKey = [^\n]*inventoryVersion/,
    "a re-download moves the snapshot the cached path points at",
  );
  assert.match(
    GGUF,
    /if \(menuOpen\) \{\s*prefetchCachedPath\(\);/,
    "dropping it under an open menu must start the next one, not wait for an event",
  );
});

test("the pointer, the open and a cold copy share one request", () => {
  assert.ok(
    PREFETCH_PATH.includes("resolveCachedPath()"),
    "each resolve is a full scan of the Hugging Face caches",
  );
  assert.ok(
    COPY_PATH.includes("await resolveCachedPath()"),
    "a cold copy must join the prefetch already running, not start a third",
  );
  assert.match(GGUF, /if \(live\?\.key === pathKey\) \{\s*return live\.path;/);
});

test("a prefetch for the previous model cannot evict the current one", () => {
  assert.match(
    GGUF,
    /if \(pathKeyRef\.current === pathKey\) \{\s*cachedPathRef\.current/,
    "a response that outlived the switch must not commit",
  );
});

test("the model path starts resolving before the menu item is reachable", () => {
  const open = GGUF.indexOf("<DropdownMenuTrigger");
  assert.notEqual(open, -1, "no DropdownMenuTrigger");
  const trigger = GGUF.slice(
    open,
    GGUF.indexOf("</DropdownMenuTrigger>", open),
  );
  assert.ok(
    trigger.includes("onPointerEnter={armPrefetch}"),
    "on open is too late: the click would await the fetch",
  );
  assert.ok(
    trigger.includes("onPointerLeave={cancelPrefetch}"),
    "a pointer crossing a full catalog would start a cache scan per card",
  );
  assert.match(GGUF, /setTimeout\(prefetchCachedPath, HOVER_PREFETCH_MS\)/);
  assert.ok(trigger.includes("onFocus={prefetchCachedPath}"));
});

test("copy path never serves another model's cached path", () => {
  assert.ok(
    !/cachedPathRef\.current \?\?/.test(COPY_PATH),
    "an unkeyed fallback copies whatever was fetched last",
  );
  assert.ok(
    COPY_PATH.includes("freshPath(pathKey) ?? (await resolveCachedPath())"),
    "a warm cache must skip the fetch, or the await loses the gesture",
  );
});
