// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the whole-pixel label spacer skips flex and grid labels", () => {
  const css = readSrc("index.css");
  const spacers = [
    ...css.matchAll(
      /\[data-slot="dropdown-menu-label"\](?::not\([^)]*\))?::after\s*\{/g,
    ),
  ];
  assert.equal(spacers.length, 1);
  assert.match(
    spacers[0]?.[0] ?? "",
    /:not\(\.flex, \.inline-flex, \.grid, \.inline-grid\)::after/,
  );
});

test("labels that pin a chip right carry a literal flex class the spacer skips", () => {
  // As a flex item the spacer pulled each chip or link off the right edge (#15232).
  const labels = [
    [
      "features/chat/permission-mode-select.tsx",
      "flex items-center justify-between gap-3",
    ],
    [
      "features/chat/permission-mode-select.tsx",
      "flex items-start justify-between gap-3",
    ],
    [
      "features/rag/components/knowledge-base-composer-button.tsx",
      "flex items-center justify-between gap-3",
    ],
  ];
  for (const [file, cls] of labels) {
    assert.ok(
      readSrc(file).includes(`<DropdownMenuLabel className="${cls}">`),
      `${file}: ${cls}`,
    );
  }
});
