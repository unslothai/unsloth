// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "@typescript/typescript6";
import { classifyUnslothSupport } from "../src/features/hub/lib/unsloth-support.ts";
import { ggufRepoParams } from "../src/lib/model-size.ts";

const source = readFileSync(
  new URL("../src/features/hub/hooks/use-hub-model-search.ts", import.meta.url),
  "utf8",
);
const start = source.indexOf("function makeMapModel(");
assert.notEqual(start, -1);
const end = source.indexOf("\n}\n", start) + 2;
const makeMapModel = new Function(
  "classifyUnslothSupport", "EMBEDDING_TAGS", "isGgufLike", "estimateSizeFromDtypes", "detectBaseModel", "ggufRepoParams",
  ts.transpileModule(source.slice(start, end), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText +
    "; return makeMapModel;",
)(
  classifyUnslothSupport,
  new Set<string>(),
  (id: string) => /-gguf$/i.test(id),
  () => undefined,
  () => null,
  ggufRepoParams,
);

// unsloth/MiniMax-M3.1-Flash-GGUF as the listing reported it while only README.md was uploaded.
const PLACEHOLDER = {
  name: "unsloth/MiniMax-M3.1-Flash-GGUF",
  pipeline_tag: "image-text-to-text",
  tags: ["unsloth", "image-text-to-text", "base_model:MiniMaxAI/MiniMax-M3.1-Flash", "region:us"],
};
const UPLOADED = [
  { name: "unsloth/flint-GGUF", gguf: { total: 39785287456 }, tags: ["gguf", "unsloth"] },
  // A mirror that drops expand=gguf still reports the tag.
  { name: "unsloth/Qwen3-8B-GGUF", tags: ["gguf"] },
  // A non-GGUF repo is never judged by this rule.
  { name: "unsloth/FLUX.1-dev-FP8", tags: ["diffusers"] },
  { name: "unsloth/MiniMax-M3.1-Flash", safetensors: { total: 220490854016 }, tags: [] },
];

test("the picker drops a GGUF repo with nothing uploaded yet", () => {
  const map = makeMapModel(false, true, "", "cuda", true);
  assert.equal(map(PLACEHOLDER), null);
  for (const raw of UPLOADED) assert.equal(map(raw)?.id, raw.name);
});

test("listings that do not opt in keep the placeholder", () => {
  assert.equal(makeMapModel(false, true, "", "cuda")(PLACEHOLDER)?.id, PLACEHOLDER.name);
});

test("every Hub listing in the model picker opts in", () => {
  const picker = readFileSync(
    new URL("../src/features/model-picker/components/model-selector/pickers.tsx", import.meta.url),
    "utf8",
  );
  const calls = picker.split("useHubModelSearch(").slice(1);
  assert.ok(calls.length >= 4);
  for (const call of calls) {
    const options = call.slice(0, call.indexOf("});"));
    assert.match(options, /hideEmptyUploads: true/);
  }
});

test("a GGUF repo's size is its largest reading, not the drafter the Hub sampled", () => {
  const map = makeMapModel(false, true, "", "cuda", true);
  // The Hub's gguf.total here comes from a 1.6B DFlash drafter shipped beside the 120B quants.
  const nemotron = map({
    name: "unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-GGUF",
    gguf: { total: 1632922404, architecture: "dflash" },
    tags: ["gguf"],
  });
  assert.equal(nemotron?.totalParams, 120e9);
  // Metadata stays when it is the larger, more exact figure, or the name has no size.
  assert.equal(map({ name: "unsloth/Qwen3-30B-A3B-GGUF", gguf: { total: 30.5e9 }, tags: ["gguf"] })?.totalParams, 30.5e9);
  assert.equal(map({ name: "unsloth/flint-GGUF", gguf: { total: 39785287456 }, tags: ["gguf"] })?.totalParams, 39785287456);
  // Safetensors totals already count every weight and keep precedence.
  assert.equal(map({ name: "unsloth/MiniMax-M3.1-Flash", safetensors: { total: 220490854016 }, tags: [] })?.totalParams, 220490854016);
});
