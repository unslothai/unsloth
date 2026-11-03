// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { studioInk } from "../src/features/flows/editor/theme-colors.ts";

const catalog = JSON.parse(
  readFileSync(new URL("./fixtures/flows-catalog.json", import.meta.url), "utf8"),
) as {
  port_colors: Record<string, string>;
  categories: { id: string; color: string }[];
};

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
