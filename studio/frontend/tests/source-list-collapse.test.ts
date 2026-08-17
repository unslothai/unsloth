// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

// source-list imports its sibling extensionlessly, the way vite resolves.
registerBundlerResolver();

const { SOURCES_COLLAPSED_LIMIT, hasHiddenSources, visibleSources } =
  await import("../src/features/rag/lib/source-list.ts");

const rows = (count: number) =>
  Array.from({ length: count }, (_, i) => ({ filename: `doc-${i}.pdf` }));

test("a list at or under the limit is shown whole", () => {
  for (const count of [0, 1, SOURCES_COLLAPSED_LIMIT]) {
    const shown = visibleSources(rows(count), {
      expanded: false,
      searching: false,
    });
    assert.equal(shown.length, count);
    assert.equal(
      hasHiddenSources(count, { expanded: false, searching: false }),
      false,
    );
  }
});

test("a longer list collapses to the limit", () => {
  const shown = visibleSources(rows(28), { expanded: false, searching: false });
  assert.equal(shown.length, SOURCES_COLLAPSED_LIMIT);
  assert.equal(shown[0].filename, "doc-0.pdf");
  assert.equal(
    hasHiddenSources(28, { expanded: false, searching: false }),
    true,
  );
});

test("expanding reveals everything", () => {
  const shown = visibleSources(rows(28), { expanded: true, searching: false });
  assert.equal(shown.length, 28);
  assert.equal(
    hasHiddenSources(28, { expanded: true, searching: false }),
    false,
  );
});

test("searching spans the whole list, so a match past the limit is reachable", () => {
  // The bug this guards: collapsing search results would hide matching
  // sources behind a "Show all" the user has no reason to press.
  const shown = visibleSources(rows(200), {
    expanded: false,
    searching: true,
  });
  assert.equal(shown.length, 200);
  assert.equal(
    hasHiddenSources(200, { expanded: false, searching: true }),
    false,
  );
});

test("returns a new array rather than a view of the input", () => {
  const input = rows(3);
  const shown = visibleSources(input, { expanded: true, searching: false });
  assert.notEqual(shown, input);
});
