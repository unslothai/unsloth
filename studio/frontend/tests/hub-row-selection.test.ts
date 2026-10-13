// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// Selecting a row must not dim it. Dark hover moved onto the --accent token
// while the selected fills stayed hand-written washes on the page, which put a
// clicked row below a merely hovered one and widened the gap as the contrast
// slider came down, since a wash falls further than the token does.

const HUB_CSS = readSrc("features/hub/hub.css");
const PICKERS = readSrc(
  "features/model-picker/components/model-selector/pickers.tsx",
);

const selectedFill = (row: string) => {
  const rule = HUB_CSS.match(
    new RegExp(
      `html\\.dark \\.hub-page \\.${row}\\[data-selected="true"\\] \\{([^}]*)\\}`,
    ),
  );
  assert.ok(rule, `${row} has no dark selected rule`);
  return rule[1] ?? "";
};

const hoverFill = (row: string) => {
  const rule = HUB_CSS.match(
    new RegExp(`html\\.dark \\.hub-page \\.${row}:hover \\{([^}]*)\\}`),
  );
  assert.ok(rule, `${row} has no dark hover rule`);
  return rule[1] ?? "";
};

for (const row of ["catalog-row", "hub-result-row"]) {
  test(`a selected ${row} is derived from the tone it must outrank`, () => {
    assert.match(hoverFill(row), /background-color: var\(--accent\)/);
    // Built ON --accent, so the pair keeps its order at every contrast
    // setting instead of relying on two curves staying in step.
    assert.match(selectedFill(row), /var\(--foreground\) 6%, var\(--accent\)/);
  });

  test(`a selected ${row} is not a wash that the slider can flatten`, () => {
    assert.doesNotMatch(selectedFill(row), /contrast-wash-gain/);
  });
}

test("the picker reads one fit per variant, companions included", () => {
  // tierOf and the rendered badge used the bare checkpoint while the
  // recommendation already judged the footprint, so a vision quant could sort
  // and badge as fitting after the star had ruled it OOM.
  assert.doesNotMatch(PICKERS, /getGgufFit\(v\.size_bytes\)/);
  const at = PICKERS.indexOf("const getVariantFit = useCallback(");
  assert.notEqual(at, -1, "getVariantFit is missing");
  assert.match(
    PICKERS.slice(at, at + 260),
    /ggufVariantFitSizeBytes\(variant\)/,
  );
  // Ordering and badge both go through it.
  assert.equal(PICKERS.match(/getVariantFit\(v\)/g)?.length, 2);
});
