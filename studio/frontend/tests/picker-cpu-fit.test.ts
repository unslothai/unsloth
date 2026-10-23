// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "@typescript/typescript6";
import { checkVramFit } from "../src/lib/vram.ts";

const PICKERS = readFileSync(
  new URL("../src/features/model-picker/components/model-selector/pickers.tsx", import.meta.url),
  "utf8",
);
const start = PICKERS.indexOf("function checkpointFit(");
assert.notEqual(start, -1);
const end = PICKERS.indexOf("\n}\n", start) + 2;
const checkpointFit = new Function(
  "checkVramFit",
  ts.transpileModule(PICKERS.slice(start, end), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText +
    "; return checkpointFit;",
)(checkVramFit);

// What /api/system reports on a GGUF-only Mac (no torch, so no MLX device).
const CPU = { available: false, backend: "cpu", budgetKnown: true, memoryTotalGb: 0 };

test("a checkpoint cannot run on a CPU backend, so its row dims", () => {
  assert.equal(checkpointFit(130, CPU, true), "exceeds");
  assert.equal(checkpointFit(0, CPU, true), "exceeds");
});

test("a checkpoint keeps its old verdicts everywhere else", () => {
  // Still loading the inventory: no verdict yet.
  assert.equal(checkpointFit(130, { ...CPU, budgetKnown: false }, true), null);
  // Task pickers keep their own rules.
  assert.equal(checkpointFit(130, CPU, false), null);
  const cuda = { available: true, backend: "cuda", budgetKnown: true, memoryTotalGb: 80 };
  assert.equal(checkpointFit(130, cuda, true), "exceeds");
  assert.equal(checkpointFit(20, cuda, true), "fits");
  assert.equal(checkpointFit(130, { ...cuda, available: false, backend: "mlx" }, true), null);
});

test("with no GPU a GGUF that fits in RAM keeps its warning but is not dimmed", () => {
  // The verdict still reaches the row, so the RAM fallback mark shows.
  assert.doesNotMatch(PICKERS, /fit === "ram" && budget\.memoryTotalGb <= 0/);
  assert.match(
    PICKERS,
    /const dimmed = exceeds && !\(vramStatus === "ram" && !\(gpuGb && gpuGb > 0\)\);/,
  );
});
