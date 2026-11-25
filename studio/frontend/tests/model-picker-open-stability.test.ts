// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

// The composer picker flickered while it opened: the list swapped sources and tabs as data landed.

const PICKER =
  "../src/components/assistant-ui/composer-model-picker/composer-model-picker.tsx";
const PAGINATED = "../src/features/hub/hooks/use-hub-paginated-search.ts";

function body(source: string, start: string, end: string): string {
  const at = source.indexOf(start);
  assert.notEqual(at, -1, `could not find ${start}`);
  const to = source.indexOf(end, at);
  assert.ok(to > at, `could not find ${end} after ${start}`);
  return source.slice(at, to);
}

test("a stale refresh of the same query keeps its rows", async () => {
  const src = await readText(PAGINATED);
  const start = body(src, "const revalidate =", "const iter = createIter");
  assert.match(start, /sameQuery &&/);
  assert.match(
    start,
    /if \(!revalidate\) \{\s*setState\(\{\s*\.\.\.\(INITIAL/,
    "only a new query may clear the rows",
  );
  const failed = body(src, ".catch((err) => {", "setState({");
  assert.match(failed, /if \(revalidate\) \{/, "a failed refresh keeps the old rows");
});

test("Discover does not show the fallback list while the Hub's first page loads", async () => {
  const src = await readText(PICKER);
  assert.match(
    src,
    /const hubFirstLoad = hubSearch\.isLoading && hubResults\.length === 0;/,
  );
  const recommended = body(src, "const recommended = useMemo", "// Hub results marked MLX");
  const gate = recommended.indexOf("if (online && hubFirstLoad)");
  const fallback = recommended.indexOf("source?.models ??");
  assert.ok(gate !== -1 && gate < fallback, "the gate must precede the fallback");
});

test("inventory readiness is latched, so a revalidation cannot flip the tab", async () => {
  const src = await readText(PICKER);
  assert.match(src, /const cachedReady = inventory\.cachedReady \|\| inventorySeenReady;/);
  assert.match(src, /const hasModels =\s*!cachedReady \|\|/);
  assert.doesNotMatch(src, /!inventory\.cachedReady/);
});

test("opening never lets Radix focus the first tab", async () => {
  const src = await readText(PICKER);
  const focus = body(src, "onOpenAutoFocus={(event) => {", "}}\n");
  const prevent = focus.indexOf("event.preventDefault();");
  const branch = focus.indexOf("if (isExpanded");
  assert.ok(prevent !== -1 && prevent < branch, "preventDefault must be unconditional");
});

test("full-bleed logo tiles render as discs, transparent marks do not", async () => {
  const src = await readText(PICKER);
  assert.match(src, /tile \? "rounded-full object-cover" : "object-contain"/);
  assert.match(src, /if \(!tile && right >= left\) \{/, "a tile is not rescaled to its ink");
});
