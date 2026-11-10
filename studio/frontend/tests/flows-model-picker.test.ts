// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Model params keep the saved {repo, variant} shape; saved load settings notify their readers.

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";
import { loadWithStubs, stubEveryImport, stubJsxRuntime } from "./helpers/module-stubs.ts";
import { choiceFromPick, modelChoiceOf } from "../src/features/flows/editor/model-choice.ts";

registerStoreStubResolver();

const overrides = await import("../src/features/model-picker/api/model-overrides.ts");

test("values saved before the picker still load", () => {
  assert.equal(modelChoiceOf(null), null);
  assert.equal(modelChoiceOf(undefined), null);
  assert.deepEqual(modelChoiceOf({ repo: "unsloth/Qwen3-4B-GGUF", variant: "Q4_K_M" }), {
    repo: "unsloth/Qwen3-4B-GGUF",
    variant: "Q4_K_M",
  });
  assert.deepEqual(modelChoiceOf({ repo: "unsloth/Qwen3-4B" }), { repo: "unsloth/Qwen3-4B", variant: null });
  assert.deepEqual(modelChoiceOf({ repo: "", variant: null }), { repo: "", variant: null });
  assert.deepEqual(modelChoiceOf("unsloth/Qwen3-4B"), { repo: "unsloth/Qwen3-4B", variant: null });
});

test("a pick stores the repo plus the GGUF variant when the picker returns one", () => {
  assert.deepEqual(choiceFromPick("unsloth/Qwen3-4B-GGUF", { ggufVariant: "Q4_K_M" }), {
    repo: "unsloth/Qwen3-4B-GGUF",
    variant: "Q4_K_M",
  });
  assert.deepEqual(choiceFromPick("unsloth/Qwen3-4B", {}), { repo: "unsloth/Qwen3-4B", variant: null });
});

test("every saved-settings write tells its listeners once it settles", async () => {
  let heard = 0;
  const off = overrides.onModelOverrideWrite(() => heard++);
  // The test stub's authFetch rejects: a failed write still re-reads, it may have landed.
  await overrides.putModelOverride("org/model", null, null).catch(() => {});
  assert.equal(heard, 1);
  off();
  await overrides.putModelOverride("org/model", null, null).catch(() => {});
  assert.equal(heard, 1);
});

// Param fields a doc never stored show the default the run will use, not an empty box.

const { ParamField } = loadWithStubs<{
  ParamField: (p: { spec: Record<string, unknown>; value: unknown; onChange: (v: unknown) => void }) => {
    props: Record<string, unknown>;
  };
}>(
  new URL("../src/features/flows/editor/param-fields.tsx", import.meta.url),
  stubEveryImport({
    react: { __esModule: true, memo: (f: unknown) => f },
    "react/jsx-runtime": stubJsxRuntime(),
  }),
);
const field = (spec: Record<string, unknown>, value: unknown) =>
  ParamField({ spec: { id: "p", label: "P", ...spec }, value, onChange: () => undefined }).props;

const defaults: [string, Record<string, unknown>, unknown, string, unknown][] = [
  ["an unset number shows its default", { kind: "number", default: 0.7 }, undefined, "value", "0.7"],
  ["a null int shows its default", { kind: "int", default: 512 }, null, "value", "512"],
  ["a stored number wins over the default", { kind: "number", default: 0.7 }, 0.2, "value", "0.2"],
  ["a number with a null default stays empty", { kind: "number", default: null }, undefined, "value", ""],
  ["an unset toggle shows its default", { kind: "bool", default: true }, undefined, "checked", true],
  ["a stored false toggle stays off", { kind: "bool", default: true }, false, "checked", false],
  ["an unset select shows its default", { kind: "select", default: "b", options: ["a", "b"] }, undefined, "value", "b"],
  ["an empty select with no default shows the first option", { kind: "select", default: null, options: ["a", "b"] }, "", "value", "a"],
];
for (const [name, spec, value, prop, shown] of defaults) {
  test(`param field: ${name}`, () => {
    assert.equal(field(spec, value)[prop], shown);
  });
}
