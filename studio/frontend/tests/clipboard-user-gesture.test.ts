// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Safari only allows a clipboard write while the click that asked for it is still
// live, so both copy buttons have to have their text ready before the handler runs.
// The two ways that breaks:
//
//   1. Anything awaited ahead of the write. The write is then rejected and the user
//      gets "Couldn't copy the link" -- no await may precede copyToClipboard.
//   2. Prefetching the text without keying it to what is on screen. The run bar keeps
//      one QuantOptionsMenu mounted and swaps repoId/quant under it, so an unkeyed
//      cache silently copies the previously selected quant's path, and the preview
//      link goes stale the moment the tunnel URL changes.
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

/** The braced block that starts at `from`, matched to its close. */
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
    "without a forced re-read the tunnel URL never lands and the link stays local",
  );
  assert.ok(HISTORY.includes('window.addEventListener("focus"'));
});

test("the prefetched model path is keyed to the model it was fetched for", () => {
  assert.match(
    GGUF,
    /const pathKey = [^\n]*repoId[^\n]*quant/,
    "the cache key must cover both repoId and quant",
  );
  assert.ok(
    PREFETCH_PATH.includes("cachedPathRef.current?.key === pathKey"),
    "the prefetch guard must miss when the menu now points at another model",
  );
  assert.ok(PREFETCH_PATH.includes("{ key: pathKey, path }"));
});

test("copy path never serves another model's cached path", () => {
  assert.ok(
    !/cachedPathRef\.current \?\?/.test(COPY_PATH),
    "an unkeyed fallback copies whatever was fetched last",
  );
  assert.ok(COPY_PATH.includes("cached?.key === pathKey"));
  assert.ok(
    COPY_PATH.includes("? cached.path"),
    "a warm cache must skip the fetch, or the await loses the gesture",
  );
});
