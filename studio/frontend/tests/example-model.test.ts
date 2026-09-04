// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  PLACEHOLDER_EXAMPLE_MODEL,
  exampleModelOptions,
  followedExampleModel,
  pinQuant,
  resolveExampleModel,
  splitPinnedQuant,
} = await import("../src/features/settings/lib/example-model.ts");

const CATALOG = [
  { id: "unsloth/Llama-GGUF", loaded: false, quant: "Q8_0", quants: ["Q8_0"] },
  {
    id: "unsloth/Qwen3-GGUF",
    loaded: true,
    quant: "Q4_K_M",
    quants: ["Q4_K_M", "Q8_0"],
  },
  { id: "org/Mistral", loaded: false },
];

const base = {
  catalog: CATALOG,
  autoSwitch: false,
  keylessOnly: false,
  checkpoint: null,
  ggufVariant: null,
};

function resolve(overrides: Partial<Parameters<typeof resolveExampleModel>[0]>) {
  const catalog = overrides.catalog === undefined ? CATALOG : overrides.catalog;
  return resolveExampleModel({
    ...base,
    picked: null,
    ...overrides,
    catalog,
    options: exampleModelOptions(catalog),
  });
}

test("options list the resident model first, with every on-disk quant", () => {
  assert.deepEqual(exampleModelOptions(CATALOG), [
    { id: "unsloth/Qwen3-GGUF", loaded: true, quants: ["Q4_K_M", "Q8_0"] },
    { id: "unsloth/Llama-GGUF", loaded: false, quants: ["Q8_0"] },
    { id: "org/Mistral", loaded: false, quants: [] },
  ]);
  // An older server sends `quant` alone; it is still the one quant to offer.
  assert.deepEqual(exampleModelOptions([{ id: "a", quant: "Q4" }])[0].quants, [
    "Q4",
  ]);
  assert.deepEqual(exampleModelOptions(null), []);
});

test("pinning and splitting a quant round-trip", () => {
  assert.equal(pinQuant("org/a", "Q8_0"), "org/a:Q8_0");
  assert.equal(pinQuant("org/a:Q4", "Q8_0"), "org/a:Q4");
  assert.equal(pinQuant("org/a", undefined), "org/a");
  assert.deepEqual(splitPinnedQuant("org/a:Q8_0"), { repo: "org/a", quant: "Q8_0" });
  assert.deepEqual(splitPinnedQuant("org/a"), { repo: "org/a", quant: null });
  assert.deepEqual(splitPinnedQuant("hf.co/org/a"), { repo: "hf.co/org/a", quant: null });
});

test("with nothing picked the snippet follows the resident model", () => {
  const r = resolve({});
  assert.equal(r.model, "unsloth/Qwen3-GGUF:Q4_K_M");
  assert.equal(r.followed, r.model);
  assert.equal(r.option?.id, "unsloth/Qwen3-GGUF");
  assert.equal(r.servable, true);
  assert.equal(r.placeholder, false);
});

test("a picked downloaded model is named, and flagged when switching is off", () => {
  const off = resolve({ picked: "unsloth/Llama-GGUF" });
  assert.equal(off.model, "unsloth/Llama-GGUF:Q8_0");
  assert.equal(off.followed, "unsloth/Qwen3-GGUF:Q4_K_M");
  assert.equal(off.servable, false);

  const on = resolve({ picked: "unsloth/Llama-GGUF", autoSwitch: true });
  assert.equal(on.servable, true);
  // A keyless caller cannot switch, so switching being on does not help it.
  const keyless = resolve({
    picked: "unsloth/Llama-GGUF",
    autoSwitch: true,
    keylessOnly: true,
  });
  assert.equal(keyless.servable, false);
});

test("a picked quant is honoured only while the repo still holds it", () => {
  assert.equal(
    resolve({ picked: "unsloth/Qwen3-GGUF:Q8_0" }).model,
    "unsloth/Qwen3-GGUF:Q8_0",
  );
  assert.equal(
    resolve({ picked: "unsloth/Qwen3-GGUF:Q2_K" }).model,
    "unsloth/Qwen3-GGUF:Q4_K_M",
  );
  assert.equal(resolve({ picked: "org/Mistral:Q8_0" }).model, "org/Mistral");
  // Case-insensitive like Hugging Face ids.
  assert.equal(
    resolve({ picked: "UNSLOTH/qwen3-gguf" }).model,
    "unsloth/Qwen3-GGUF:Q4_K_M",
  );
});

test("a pick the catalog no longer lists falls back to following", () => {
  const r = resolve({ picked: "org/deleted-GGUF:Q4" });
  assert.equal(r.model, "unsloth/Qwen3-GGUF:Q4_K_M");
  assert.equal(r.servable, true);
  // Before /v1 answers the pick is not disproved, but nothing backs it either.
  const pending = resolve({ picked: "unsloth/Llama-GGUF", catalog: null });
  assert.equal(pending.model, null);
  assert.equal(pending.placeholder, false);
});

test("an empty catalog names the placeholder, an unanswered one names nothing", () => {
  const empty = resolve({ catalog: [] });
  assert.equal(empty.model, PLACEHOLDER_EXAMPLE_MODEL);
  assert.equal(empty.placeholder, true);
  assert.equal(empty.servable, false);
  assert.equal(empty.option, null);

  const pending = resolve({ catalog: null });
  assert.equal(pending.model, null);
  assert.equal(pending.placeholder, false);
});

test("the followed model keeps the store checkpoint only while the catalog backs it", () => {
  const backed = followedExampleModel({
    ...base,
    checkpoint: "unsloth/Qwen3-GGUF",
    ggufVariant: "Q2_K",
  });
  // The catalog's quant wins over the stored one.
  assert.equal(backed, "unsloth/Qwen3-GGUF:Q4_K_M");
  const pending = followedExampleModel({
    ...base,
    catalog: null,
    checkpoint: "unsloth/Qwen3-GGUF",
    ggufVariant: "Q2_K",
  });
  assert.equal(pending, "unsloth/Qwen3-GGUF:Q2_K");
  const unloaded = followedExampleModel({
    ...base,
    checkpoint: "unsloth/Llama-GGUF",
  });
  assert.equal(unloaded, "unsloth/Qwen3-GGUF:Q4_K_M");
  assert.equal(
    followedExampleModel({ ...base, checkpoint: "unsloth/Llama-GGUF", autoSwitch: true }),
    "unsloth/Llama-GGUF:Q8_0",
  );
  assert.equal(
    followedExampleModel({ ...base, checkpoint: "/srv/models/x.gguf" }),
    "unsloth/Qwen3-GGUF:Q4_K_M",
  );
  assert.equal(
    followedExampleModel({ ...base, catalog: [], autoSwitch: true }),
    null,
  );
});
