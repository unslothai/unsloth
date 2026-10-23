// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "@typescript/typescript6";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { providerModelReplaysReasoning, providerSupportsPreserveThinking } = await import("../src/features/chat/provider-capabilities.ts");
const source = readFileSync(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url), "utf8");
const tree = ts.createSourceFile("adapter.ts", source, ts.ScriptTarget.Latest, true);
let replayExpression = "";
let payloadExpression = "";
function visit(node: ts.Node): void {
  if (ts.isVariableDeclaration(node) && node.name.getText(tree) === "replayReasoning") {
    replayExpression = node.initializer!.getText(tree);
  }
  if (ts.isSpreadAssignment(node) && node.getText(tree).includes("providerSupportsPreserveThinking")) {
    payloadExpression = node.expression.getText(tree);
  }
  ts.forEachChild(node, visit);
}
visit(tree);
assert.ok(replayExpression && payloadExpression);
// Evaluates the real send-path expressions rather than a copied policy.
const evaluateWithModel = new Function("providerSupportsPreserveThinking", "externalProvider", "runtime", "isExternalRequest",
  "providerModelReplaysReasoning", "externalSelection",
  `return { replay: ${replayExpression}, fields: ${payloadExpression} };`) as
  (supports: typeof providerSupportsPreserveThinking, provider: {providerType: string}, runtime: {preserveThinking: boolean}, external: boolean,
    replays: typeof providerModelReplaysReasoning, selection: {modelId: string}) =>
    {replay: boolean; fields: {preserve_thinking?: boolean}};
const evaluate = (
  supports: typeof providerSupportsPreserveThinking,
  provider: {providerType: string},
  runtime: {preserveThinking: boolean},
  external: boolean,
  modelId = "m",
) => evaluateWithModel(supports, provider, runtime, external, providerModelReplaysReasoning, {modelId});

test("llama.cpp and vLLM send the selected value and replay reasoning only when enabled", () => {
  for (const providerType of ["llama_cpp", "vllm"]) {
    for (const enabled of [true, false]) {
      const result = evaluate(providerSupportsPreserveThinking, {providerType}, {preserveThinking: enabled}, true);
      assert.deepEqual(result, {replay: enabled, fields: {preserve_thinking: enabled}});
    }
  }
  assert.match(source, /pruneOutboundHistory\(messages, replayReasoning\)/);
  assert.match(source, /toOpenAIMessages\(\s*message,\s*replayReasoning,/);
});

test("switching providers never leaks a retained enabled preference", () => {
  for (const providerType of ["custom", "openai", "ollama", "anthropic"]) {
    assert.deepEqual(evaluate(providerSupportsPreserveThinking, {providerType}, {preserveThinking: true}, true),
      {replay: false, fields: {}});
  }
  assert.equal(evaluate(providerSupportsPreserveThinking, {providerType: ""}, {preserveThinking: false}, false).replay, true);
});

test("kimi-k3 replays reasoning whatever the preserve toggle says; k2.6 does not", () => {
  for (const enabled of [true, false]) {
    assert.deepEqual(
      evaluate(providerSupportsPreserveThinking, {providerType: "kimi"}, {preserveThinking: enabled}, true, "kimi-k3"),
      {replay: true, fields: {}},
    );
    assert.equal(
      evaluate(providerSupportsPreserveThinking, {providerType: "kimi"}, {preserveThinking: enabled}, true, "kimi-k2.6").replay,
      false,
    );
  }
});
