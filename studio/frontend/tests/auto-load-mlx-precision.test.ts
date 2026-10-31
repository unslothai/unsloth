// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import ts from "@typescript/typescript6";

const adapterPath = fileURLToPath(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
);
const source = readFileSync(adapterPath, "utf8");
const start = source.indexOf("async function resolveAutoLoadCandidate(");
assert.ok(start >= 0, "resolveAutoLoadCandidate is no longer defined in chat-adapter.ts");
const declaration = source.slice(start, source.indexOf("\n}", start) + 2);

type Candidate = { kind: string; ggufVariant: string | null };
const resolve = new Function(
  `${ts.transpileModule(declaration, {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  }).outputText}; return resolveAutoLoadCandidate;`,
)() as (
  source: object,
  remembered: string | null,
  isSkipped: (candidate: Candidate) => boolean,
) => Promise<Candidate | null>;

const mlxRepo = {
  id: "org/Model-MLX",
  loadId: "org/Model-MLX",
  kind: "model",
  maxSeqLength: 4096,
  listVariants: null,
};

test("startup restores a remembered MLX precision folder", async () => {
  const candidate = await resolve(mlxRepo, "6-bit", () => false);
  assert.equal(candidate?.ggufVariant, "6-bit");
  assert.equal(candidate?.kind, "model");
});

test("a failed precision falls back to the repo root, then stops", async () => {
  const tried = new Set(["6-bit"]);
  const skip = (candidate: Candidate) => tried.has(String(candidate.ggufVariant));
  assert.equal((await resolve(mlxRepo, "6-bit", skip))?.ggufVariant, null);
  tried.add("null");
  assert.equal(await resolve(mlxRepo, "6-bit", skip), null);
});

test("a model without a remembered precision loads its root", async () => {
  assert.equal((await resolve(mlxRepo, null, () => false))?.ggufVariant, null);
});
