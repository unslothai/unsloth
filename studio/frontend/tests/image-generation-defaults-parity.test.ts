// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { defaultsFor } from "../src/features/images/image-generation-defaults.ts";

import { readSrc } from "./helpers/kit.ts";

type Row = { key: string; steps: number; guidance: number };

// The backend table is what Studio renders when a request omits steps / guidance.
function backendRows(): Row[] {
  const source = readFileSync(
    new URL("../../backend/core/inference/diffusion_families.py", import.meta.url),
    "utf8",
  );
  const start = source.indexOf("_GENERATION_DEFAULTS: tuple");
  assert.ok(start >= 0, "backend _GENERATION_DEFAULTS not found");
  const end = source.indexOf("\n)\n", start);
  assert.ok(end > start, "end of backend _GENERATION_DEFAULTS not found");
  const rows = [...source.slice(start, end).matchAll(/\(\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)/g)].map(
    (m) => ({ key: m[1], steps: Number(m[2]), guidance: Number(m[3]) }),
  );
  assert.ok(rows.length > 40, `parsed only ${rows.length} backend rows`);
  return rows;
}

// name_key_in over the rows in order: raw substring, then separator-folded substring; first match wins.
function backendResolve(rows: Row[], identifier: string): { steps: number; guidance: number } {
  const fold = (text: string) => text.toLowerCase().replace(/[-_.\s]+/g, "-");
  const low = identifier.toLowerCase();
  const row = rows.find((r) => low.includes(r.key.toLowerCase()) || fold(low).includes(fold(r.key)));
  return row ? { steps: row.steps, guidance: row.guidance } : { steps: 9, guidance: 0 };
}

const rows = backendRows();

test("every backend generation-defaults key resolves to the backend's steps and guidance in the UI", () => {
  const mismatches = rows
    .map((row) => ({ key: row.key, ui: defaultsFor(row.key), backend: backendResolve(rows, row.key) }))
    .filter(({ ui, backend }) => ui.steps !== backend.steps || ui.guidance !== backend.guidance);
  assert.deepEqual(mismatches, []);
});

test("every catalogued image repo id resolves the same in the UI and the backend", () => {
  const catalog = readSrc("features/model-picker/components/model-selector/model-catalog.ts");
  const ids = [...new Set([...catalog.matchAll(/"([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)"/g)].map((m) => m[1]))];
  assert.ok(ids.length > 40, `found only ${ids.length} catalog ids`);
  const mismatches = ids
    .map((id) => ({ id, ui: defaultsFor(id), backend: backendResolve(rows, id) }))
    .filter(({ ui, backend }) => ui.steps !== backend.steps || ui.guidance !== backend.guidance);
  assert.deepEqual(mismatches, []);
});

test("MOSS-Image repos and GGUF files default to 25 steps at CFG 2.5, as Studio renders them", () => {
  for (const id of [
    "unsloth/MOSS-Image-SFT-GGUF",
    "unsloth/MOSS-Image-Base-GGUF",
    "unsloth/MOSS-Image-SFT-FP8",
    "OpenMOSS-Team/MOSS-Image-SFT",
    "MOSS-Image-SFT-Q4_K_M.gguf",
    "moss_image_base-Q8_0.gguf",
  ]) {
    assert.deepEqual(defaultsFor(id), { steps: 25, guidance: 2.5 }, id);
    assert.deepEqual(backendResolve(rows, id), { steps: 25, guidance: 2.5 }, id);
  }
});

test("families the UI table used to miss get the backend recipe, not the 9 / 0 fallback", () => {
  for (const [id, want] of [
    ["lodestones/Chroma1-HD", { steps: 26, guidance: 3.5 }],
    ["circlestone-labs/Anima", { steps: 30, guidance: 4 }],
    ["flux1-schnell-Q4_K_S.gguf", { steps: 4, guidance: 0 }],
    ["flux1-krea-dev-Q8_0.gguf", { steps: 20, guidance: 3.5 }],
    ["flux1-dev-Q4_K_S.gguf", { steps: 20, guidance: 3.5 }],
    ["minimax_h3_image-Q4_K_M.gguf", { steps: 6, guidance: 1 }],
    ["realrebelai/KREA-2_GGUFs", { steps: 8, guidance: 0 }],
  ] as const) {
    assert.deepEqual(defaultsFor(id), want, id);
    assert.deepEqual(backendResolve(rows, id), want, id);
  }
});
