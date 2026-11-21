// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { availableAxes, MAX_COMPARE_CELLS, type CompareContext } from "../src/features/images/compare/compare-axes.ts";
import { xyGridCellSize } from "../src/features/images/image-size.ts";
import { recipeSettings, unknownRecipeKeys } from "../src/features/images/lib/image-metadata.ts";
import { readSrc } from "./helpers/kit.ts";

const backend = readFileSync(new URL("../../backend/models/inference.py", import.meta.url), "utf8");

function literal(source: RegExp, text: string): string[] {
  const body = text.match(source)?.[1];
  assert.ok(body, `no match for ${source}`);
  return [...body.matchAll(/"([a-z_]+)"/g)].map((m) => m[1]).sort();
}

test("the XYAxisSpec field union in api.ts is the backend's XYAxisField literal", () => {
  const server = literal(/XYAxisField = Literal\[([\s\S]*?)\]/, backend);
  const client = literal(/export interface XYAxisSpec \{\s*field:([\s\S]*?);/, readSrc("features/images/api.ts"));
  assert.deepEqual(client, server);
});

test("every Images page Compare axis has a server-side xy field, under the same cell cap", () => {
  const server = new Set(literal(/XYAxisField = Literal\[([\s\S]*?)\]/, backend));
  const ctx: CompareContext = {
    prompt: "a fox",
    negativePrompt: "",
    negativeCapable: true,
    negativeMinGuidance: 1,
    guidanceFixed: false,
    guidance: 6,
    samplers: [
      { id: "default", label: "Default", schedule_types: ["automatic"] },
      { id: "euler", label: "Euler", schedule_types: ["automatic", "karras"] },
    ],
    samplerChoice: { sampler: "default", scheduleType: "automatic" },
    shift: { node: "flux", base_shift: 0.5, max_shift: 1.15 },
    loraCapable: true,
    loras: [{ id: "a", weight: 1, label: "a" }],
    sdControls: { clip_skip: { min: 1, max: 12 }, cfg_rescale: { min: 0, max: 1 }, variation_seed: true },
    controlnetEngaged: false,
  } as unknown as CompareContext;
  const toServer = (id: string) => (id === "schedule" ? "schedule_type" : id.startsWith("lora:") ? "lora_weight" : id);
  for (const axis of availableAxes(ctx)) {
    if (axis.id === "nothing") continue;
    assert.ok(server.has(toServer(axis.id)), axis.id);
  }
  assert.match(backend, new RegExp(`^MAX_XY_CELLS = ${MAX_COMPARE_CELLS}$`, "m"));
});

test("reusing a server grid's recipe restores the cell size, not the grid's pixels", () => {
  assert.deepEqual(xyGridCellSize({ xy_grid: { cell_width: 768, cell_height: 512 } }), { width: 768, height: 512 });
  assert.equal(xyGridCellSize({ xy_grid: null }), null);
  assert.equal(xyGridCellSize({}), null);
  assert.match(readSrc("features/images/images-page.tsx"), /hiresBaseSize\(image\) \?\? xyGridCellSize\(image\) \?\?/);
  const dropped = { prompt: "a fox", width: 2880, height: 2441, xy_grid: { cell_width: 768, cell_height: 768, cells: [] } };
  const settings = recipeSettings(dropped, { width: 1024, height: 1024, steps: 8, guidance: 1 });
  assert.deepEqual([settings.width, settings.height], [768, 768]);
  assert.deepEqual(unknownRecipeKeys(dropped), []);
});
