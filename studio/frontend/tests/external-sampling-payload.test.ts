// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Chat Settings panel renders Min P and Repetition Penalty off the provider's
// capability flags, and both persist per model and per thread -- but the external
// request body only ever spread temperature, top_p, max_tokens, top_k and
// presence_penalty, so on OpenRouter / vLLM / llama.cpp the two sliders moved
// and nothing changed. The local body has sent both unconditionally all along.
//
// The body is built inside a several-thousand-line function that cannot be called from
// here, so read the capability-gated spreads out of it and evaluate those: a row that
// stops existing, or that gates on the wrong flag, fails.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { getProviderCapabilities } = await import(
  "../src/features/chat/provider-capabilities.ts"
);

const PARAMS = {
  temperature: 0.6,
  topP: 0.95,
  topK: 40,
  minP: 0.07,
  repetitionPenalty: 1.15,
  presencePenalty: 0.3,
};

const source = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);
const tree = ts.createSourceFile(
  "chat-adapter.ts",
  source,
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TS,
);

function externalBodyLiteral(): ts.ObjectLiteralExpression {
  let found: ts.ObjectLiteralExpression | null = null;
  const visit = (node: ts.Node): void => {
    if (
      !found &&
      ts.isObjectLiteralExpression(node) &&
      node.properties.some(
        (property) =>
          ts.isPropertyAssignment(property) &&
          property.name.getText() === "model" &&
          property.initializer.getText() === "externalSelection.modelId",
      )
    ) {
      found = node;
      return;
    }
    ts.forEachChild(node, visit);
  };
  visit(tree);
  assert.ok(found, "external request body literal not found");
  return found;
}

const gatedSpreads = externalBodyLiteral()
  .properties.filter(ts.isSpreadAssignment)
  .map((property) => property.expression.getText())
  .filter((text) => text.includes("externalCapabilities"));

// A guard against the extraction silently matching nothing and every assertion below
// passing vacuously.
assert.ok(gatedSpreads.length >= 4, `only ${gatedSpreads.length} gated spreads`);

const buildSamplingFields = new Function(
  "externalCapabilities",
  "params",
  `return Object.assign({}, ${gatedSpreads.join(", ")});`,
) as (
  capabilities: unknown,
  params: typeof PARAMS,
) => Record<string, number>;

function bodyFor(providerType: string): Record<string, number> {
  return buildSamplingFields(getProviderCapabilities(providerType), PARAMS);
}

for (const providerType of ["vllm", "openrouter", "llama_cpp"]) {
  test(`${providerType} carries the min_p and repetition_penalty the panel offers`, () => {
    const body = bodyFor(providerType);
    assert.equal(body.min_p, PARAMS.minP);
    assert.equal(body.repetition_penalty, PARAMS.repetitionPenalty);
    // The rows that already worked must keep working.
    assert.equal(body.top_k, PARAMS.topK);
    assert.equal(body.presence_penalty, PARAMS.presencePenalty);
    assert.equal(body.temperature, PARAMS.temperature);
    assert.equal(body.top_p, PARAMS.topP);
  });
}

test("custom stays on the OpenAI-compatible baseline", () => {
  const body = bodyFor("custom");
  assert.ok(!("min_p" in body));
  assert.ok(!("repetition_penalty" in body));
  assert.ok(!("top_k" in body));
  assert.equal(body.presence_penalty, PARAMS.presencePenalty);
  assert.equal(body.temperature, PARAMS.temperature);
});

test("ollama is sent none of the three its /v1 layer drops", () => {
  // Ollama's OpenAI-compatibility layer reads only the OpenAI-documented fields; top_k,
  // min_p and repeat_penalty are native /api/chat "options", so a body carrying them is
  // answered with default sampling and no error. The panel hides all three instead.
  const body = bodyFor("ollama");
  assert.ok(!("min_p" in body));
  assert.ok(!("repetition_penalty" in body));
  assert.ok(!("top_k" in body));
  assert.equal(body.presence_penalty, PARAMS.presencePenalty);
  assert.equal(body.temperature, PARAMS.temperature);
});

test("a hosted provider's body is unchanged by the new rows", () => {
  // Anthropic, OpenAI, Gemini, Kimi and DeepSeek all declare minP and repetitionPenalty
  // false, so nothing new can appear in their bodies.
  for (const providerType of [
    "anthropic",
    "openai",
    "openai_codex",
    "gemini",
    "kimi",
    "deepseek",
    "mistral",
    "qwen",
    "huggingface",
  ]) {
    const body = bodyFor(providerType);
    assert.ok(!("min_p" in body), providerType);
    assert.ok(!("repetition_penalty" in body), providerType);
  }
});

test("an unknown provider stays on the OpenAI-compatible shape", () => {
  // A connection saved by a newer build lands on the default capability set, which must
  // not start sending extensions a strict endpoint would 400 on.
  const body = bodyFor("some-provider-this-build-never-heard-of");
  assert.ok(!("min_p" in body));
  assert.ok(!("repetition_penalty" in body));
  assert.ok(!("top_k" in body));
});

test("the panel and the request read the same capability flags", () => {
  const sheet = readFileSync(
    new URL("../src/features/chat/chat-settings-sheet.tsx", import.meta.url),
    "utf8",
  );
  // The sliders are rendered off providerCapabilities.minP / .repetitionPenalty; a body
  // gating on anything else is how the two drifted apart in the first place.
  assert.match(sheet, /Boolean\(providerCapabilities\?\.minP\)/);
  assert.match(sheet, /Boolean\(providerCapabilities\?\.repetitionPenalty\)/);
  assert.ok(
    gatedSpreads.some((text) => text.includes("externalCapabilities?.minP")),
  );
  assert.ok(
    gatedSpreads.some((text) =>
      text.includes("externalCapabilities?.repetitionPenalty"),
    ),
  );
});
