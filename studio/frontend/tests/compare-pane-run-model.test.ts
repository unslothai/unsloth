// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  prepareComparePaneRun,
  registerComparePaneRunPreparer,
} from "../src/features/chat/compare-run-ownership.ts";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { checkpointMatchesPick } = await import(
  "../src/features/chat/lib/resident-model-match.ts"
);

const adapterSource = readSrc("features/chat/api/chat-adapter.ts");
const chatPageSource = readSrc("features/chat/chat-page.tsx");
const generalCompareSource = chatPageSource.slice(
  chatPageSource.indexOf("const GeneralCompareContent"),
  chatPageSource.indexOf("function formatProjectChatDate"),
);

test("a compare pane run is prepared for its own pane and pair", async () => {
  const calls: string[] = [];
  const unregister = registerComparePaneRunPreparer("pair-a", async (pane) => {
    calls.push(pane);
  });
  await prepareComparePaneRun("pair-a", "model1");
  await prepareComparePaneRun("pair-a", "model2");
  await prepareComparePaneRun("pair-b", "model1");
  await prepareComparePaneRun("pair-a", "base");
  await prepareComparePaneRun(undefined, "model1");
  assert.deepEqual(calls, ["model1", "model2"]);
  unregister();
  await prepareComparePaneRun("pair-a", "model1");
  assert.deepEqual(calls, ["model1", "model2"]);
});

test("a failed pane load stops the pane run", async () => {
  const unregister = registerComparePaneRunPreparer("pair-c", async () => {
    throw new Error("not loaded");
  });
  await assert.rejects(prepareComparePaneRun("pair-c", "model1"), /not loaded/);
  unregister();
});

test("a stale unregister keeps the newer preparer", async () => {
  const calls: string[] = [];
  const first = registerComparePaneRunPreparer("pair-d", async () => {
    calls.push("first");
  });
  const second = registerComparePaneRunPreparer("pair-d", async () => {
    calls.push("second");
  });
  first();
  await prepareComparePaneRun("pair-d", "model1");
  assert.deepEqual(calls, ["second"]);
  second();
});

test("the chat adapter prepares a compare pane before it reads the model", () => {
  const prepare = adapterSource.indexOf(
    "await prepareComparePaneRun(options.pairId, options.modelType);",
  );
  const firstRead = adapterSource.indexOf(
    "let runtime = useChatRuntimeStore.getState();",
  );
  assert.ok(prepare > 0, "the run must prepare its compare pane");
  assert.ok(prepare < firstRead, "before the checkpoint is read");
});

test("Refresh or Edit in a compare pane loads that pane's own model", () => {
  assert.match(
    generalCompareSource,
    /registerComparePaneRunPreparer\(pairId, async \(pane\) => \{\s*const sel = pane === "model1" \? model1 : model2;/,
  );
  assert.match(generalCompareSource, /await loadPaneModel\(\{\s*id: sel\.id,/);
  assert.match(generalCompareSource, /return checkpointMatchesPick\(/);
  assert.match(
    generalCompareSource,
    /\.\.\.\(sel\.ggufVariant \|\| sel\.id\.toLowerCase\(\)\.endsWith\("\.gguf"\)\s*\? \{ isGguf: true \}/,
    "a standalone GGUF pane reloads with GGUF load fields, as shared Send does",
  );
  assert.match(
    generalCompareSource,
    /if \(comparingRef\.current \|\| !sel\.id \|\| isActive\(\)\) return;/,
    "the shared send already loads each pane before its run",
  );
  assert.match(
    generalCompareSource,
    /onComparingChange=\{handleComparingChange\}/,
  );
  assert.match(chatPageSource, /loadPaneModel=\{selectModel\}/);
});

test("a standalone GGUF pane matches the quant the backend derives", () => {
  const file = "/models/Qwen3-0.6B-Q4_K_M.gguf";
  assert.ok(checkpointMatchesPick(file, "Q4_K_M", { id: file }));
  assert.ok(
    checkpointMatchesPick("unsloth/Qwen3-0.6B-GGUF", "Q4_K_M", {
      id: "unsloth/Qwen3-0.6B-GGUF",
      ggufVariant: "Q4_K_M",
    }),
  );
  assert.ok(
    !checkpointMatchesPick("unsloth/Qwen3-0.6B-GGUF", "Q8_0", {
      id: "unsloth/Qwen3-0.6B-GGUF",
      ggufVariant: "Q4_K_M",
    }),
  );
  assert.ok(
    !checkpointMatchesPick("/models/other.gguf", "Q4_K_M", { id: file }),
  );
});
