// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("stepping onto a match whose text a line window swapped out rebuilds the index first", () => {
  const hook = readSrc("features/find-in-page/hooks/use-find-in-page.ts");
  const step = hook.slice(
    hook.indexOf("const step = useCallback("),
    hook.indexOf("const next = useCallback("),
  );
  assert.match(
    step,
    /if \(staleRef\.current\) \{\s*const range = rangeForMatch\(indexRef\.current, target\);\s*if \(!\(range\?\.startContainer\.isConnected && range\.endContainer\.isConnected\)\) \{\s*activeStartRef\.current = target\.start;\s*search\(true, reindex\(\)\);\s*return;\s*\}\s*\}\s*apply\(true\);/,
    "a windowed fence swaps a line between one text node and its token spans as the reader moves, and the throttled rebuild came too late for a quick Enter",
  );
});

test("while the find bar is open a long fence renders every line, so the highlights keep their text", () => {
  const defer = readSrc("components/assistant-ui/code-fence-defer.tsx");
  assert.match(
    readSrc("features/find-in-page/components/find-in-page.tsx"),
    /useDocumentFlag\("data-find-bar-open", enabled && open && foreground\);/,
  );
  assert.match(defer, /const FIND_BAR_FLAG = "data-find-bar-open";/);
  assert.match(
    defer,
    /findBarObserver = new MutationObserver\(scheduleRemeasure\);\s*findBarObserver\.observe\(document\.documentElement, \{\s*attributes: true,\s*attributeFilter: \[FIND_BAR_FLAG\],\s*\}\);/,
    "find paints live ranges over a line's text, and a window move swapped that text for token spans, so the highlights vanished until the index was rebuilt",
  );
  assert.ok(defer.indexOf("findBarObserver?.disconnect();") > defer.indexOf("const unwatchWindows = "));
  assert.match(defer, /if \(printing \|\| findBarOpen\(\)\) \{\s*if \(current\.current === null && pinned\.current === null\) return;/);
  assert.ok(defer.includes("return overCap && !printing && !findBarOpen() ? EMPTY_WINDOW : NO_WINDOW;"), "a fence that mounts while the bar is open starts unwindowed");
});
