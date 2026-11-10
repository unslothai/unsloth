// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Flows draws with Studio's palette and Studio's icon glyphs.

import assert from "node:assert/strict";
import test from "node:test";

import * as hugeicons from "@hugeicons/core-free-icons";
import * as local from "../src/features/flows/editor/icon-data.ts";
import { studioInk } from "../src/features/flows/editor/theme-colors.ts";
import { catalog } from "./helpers/flows.ts";

test("every catalog port and category colour maps to a Studio token", () => {
  const colours = [
    ...Object.entries(catalog.port_colors),
    ...catalog.categories.map((c) => [c.id, c.color] as const),
  ];
  for (const [name, hex] of colours) {
    assert.match(studioInk(hex), /var\(--/, `${name} (${hex}) has no Studio ink`);
  }
});

test("the port types that meet on a training flow stay distinct", () => {
  const { text, dataset, checkpoint, model, json } = catalog.port_colors;
  const inks = [text, dataset, checkpoint, model, json].map(studioInk);
  assert.equal(new Set(inks).size, inks.length);
});

test("colours the table does not know pass through", () => {
  assert.equal(studioInk("#123456"), "#123456");
});

test("the Flows icon copies match the package glyphs", () => {
  const names = Object.keys(local);
  assert.ok(names.length > 20);
  for (const name of names) {
    assert.deepEqual(
      (local as Record<string, unknown>)[name],
      (hugeicons as Record<string, unknown>)[name],
      `${name} drifted from @hugeicons/core-free-icons; regenerate icon-data.ts`,
    );
  }
});
