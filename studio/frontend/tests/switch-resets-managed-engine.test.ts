// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

const runtime = readSrc("features/chat/hooks/use-chat-model-runtime.ts");
const start = runtime.indexOf("let mlxLoadProgress = false;");
const end = runtime.indexOf("const managedLoad =", start);
assert.ok(start >= 0 && end > start);
const requested = new Function(
  "useChatRuntimeStore",
  "selection",
  "keepSpeculative",
  "modelId",
  "ggufVariant",
  `${runtime.slice(start, end)}\nreturn requestedEngine;`,
) as (...args: unknown[]) => string;

type Params = {
  checkpoint: string;
  engine: string;
  enginePrecision: string;
  engineParallelism: string;
};

function store(engine: string) {
  const state: {
    params: Params;
    activeGgufVariant: null;
    setParams(params: Params): void;
  } = {
    params: {
      checkpoint: "org/A",
      engine,
      enginePrecision: "int4",
      engineParallelism: "data",
    },
    activeGgufVariant: null,
    setParams(params: Params) {
      state.params = params;
    },
  };
  return { state, useChatRuntimeStore: { getState: () => state } };
}

test("picking another model without a saved config leaves the resident's engine behind", () => {
  const { state, useChatRuntimeStore } = store("vllm");
  assert.equal(
    requested(useChatRuntimeStore, "org/B", false, "org/B", null),
    "auto",
  );
  assert.deepEqual(
    [
      state.params.engine,
      state.params.enginePrecision,
      state.params.engineParallelism,
    ],
    ["auto", "auto", "tensor"],
  );
});

test("a saved config, a staged reload or the same model keep the engine", () => {
  const withConfig = store("vllm");
  assert.equal(
    requested(
      withConfig.useChatRuntimeStore,
      { config: { engine: "sglang" } },
      false,
      "org/B",
      null,
    ),
    "sglang",
  );
  const staged = store("vllm");
  assert.equal(
    requested(staged.useChatRuntimeStore, "org/B", true, "org/B", null),
    "vllm",
  );
  const same = store("vllm");
  assert.equal(
    requested(same.useChatRuntimeStore, "org/A", false, "org/A", null),
    "vllm",
  );
  assert.equal(same.state.params.enginePrecision, "int4");
});
