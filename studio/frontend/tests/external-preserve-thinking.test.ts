// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { providerSupportsPreserveThinking } = await import("../src/features/chat/provider-capabilities.ts");
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
// Evaluate the real send-path expressions, not a copied policy. The same replay flag
// must reach both pruning and serialization so reasoning-only turns survive.
const evaluate = new Function("providerSupportsPreserveThinking", "externalProvider", "runtime", "isExternalRequest",
  `return { replay: ${replayExpression}, fields: ${payloadExpression} };`) as
  (supports: typeof providerSupportsPreserveThinking, provider: {providerType: string}, runtime: {preserveThinking: boolean}, external: boolean) =>
    {replay: boolean; fields: {preserve_thinking?: boolean}};

test("llama.cpp sends the selected value and replays reasoning only when enabled", () => {
  for (const enabled of [true, false]) {
    const result = evaluate(providerSupportsPreserveThinking, {providerType: "llama_cpp"}, {preserveThinking: enabled}, true);
    assert.deepEqual(result, {replay: enabled, fields: {preserve_thinking: enabled}});
  }
  assert.match(source, /pruneOutboundHistory\(messages, replayReasoning\)/);
  assert.match(source, /toOpenAIMessages\(message, replayReasoning\)/);
});

test("switching providers never leaks a retained enabled preference", () => {
  for (const providerType of ["custom", "openai", "vllm", "ollama", "anthropic"]) {
    assert.deepEqual(evaluate(providerSupportsPreserveThinking, {providerType}, {preserveThinking: true}, true),
      {replay: false, fields: {}});
  }
  assert.equal(evaluate(providerSupportsPreserveThinking, {providerType: ""}, {preserveThinking: false}, false).replay, true);
});
