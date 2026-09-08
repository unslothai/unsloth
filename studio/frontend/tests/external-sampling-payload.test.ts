// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The external body never spread min_p / repetition_penalty, so the sliders did nothing.
// Its function is too large to call here, so the gated spreads are extracted and evaluated.

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

// Without this, an extraction that matched nothing would pass every assertion vacuously.
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
  const body = bodyFor("ollama");
  assert.ok(!("min_p" in body));
  assert.ok(!("repetition_penalty" in body));
  assert.ok(!("top_k" in body));
  assert.equal(body.presence_penalty, PARAMS.presencePenalty);
  assert.equal(body.temperature, PARAMS.temperature);
});

test("a hosted provider's body is unchanged by the new rows", () => {
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
  // A connection saved by a newer build lands here, and a strict endpoint 400s on extensions.
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
  // Gating the body on anything but these flags is how panel and request drifted apart.
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
