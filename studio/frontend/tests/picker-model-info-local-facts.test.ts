// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  LOCAL_MODEL_INFO_FIELDS,
  localModelInfoFacts,
} from "../src/features/model-picker/components/model-selector/local-model-facts.ts";

const factsByKey = (meta: Parameters<typeof localModelInfoFacts>[0]) =>
  new Map(localModelInfoFacts(meta).map((f) => [f.key, f.value]));

test("context length is reported from the header", () => {
  assert.equal(
    factsByKey({ contextLength: 131072 }).get("contextLength"),
    "128K tokens",
  );
  assert.equal(
    factsByKey({ contextLength: 262144 }).get("contextLength"),
    "256K tokens",
  );
  assert.equal(
    factsByKey({ contextLength: 40960 }).get("contextLength"),
    "40K tokens",
  );
  assert.equal(
    factsByKey({ contextLength: 1000 }).get("contextLength"),
    "1,000 tokens",
  );
});

test("unread header values drop their rows rather than showing zero", () => {
  const keys = [
    ...factsByKey({ contextLength: null, layerCount: null }).keys(),
  ];
  assert.deepEqual(keys, []);
  assert.deepEqual([...factsByKey({}).keys()], []);
});

test("a header that was not read claims nothing, not an absent template", () => {
  const notRead = factsByKey({
    contextLength: null,
    layerCount: null,
    moeLayerCount: null,
    chatTemplate: null,
  });
  assert.deepEqual([...notRead.keys()], []);

  const projector = factsByKey({
    contextLength: null,
    layerCount: null,
    moeLayerCount: 0,
    chatTemplate: null,
  });
  assert.deepEqual([...projector.keys()], []);
});

test("a dense model reports no expert layers, and is not dropped", () => {
  assert.equal(
    factsByKey({ contextLength: 4096, moeLayerCount: 0 }).get("moeLayers"),
    "None (dense)",
  );
  assert.equal(
    factsByKey({ contextLength: 4096, moeLayerCount: 24 }).get("moeLayers"),
    "24",
  );
});

test("layer count is reported when the header carries one", () => {
  assert.equal(factsByKey({ layerCount: 36 }).get("layers"), "36");
  assert.equal(
    factsByKey({ contextLength: 4096, layerCount: 0 }).has("layers"),
    false,
  );
});

test("the template row states whether one is embedded", () => {
  const embedded = factsByKey({
    contextLength: 4096,
    chatTemplate: `{%- if message.role == "system" %}{% endif %}`,
  });
  assert.equal(embedded.get("chatTemplate"), "Embedded");

  const none = factsByKey({ contextLength: 4096, chatTemplate: null });
  assert.equal(none.get("chatTemplate"), "Not available");
});

test("a withheld template is not reported as absent", () => {
  const overCap = factsByKey({
    contextLength: 8192,
    layerCount: 32,
    moeLayerCount: 0,
    chatTemplate: null,
  });
  assert.equal(overCap.get("chatTemplate"), "Not available");
  assert.notEqual(overCap.get("chatTemplate"), "None embedded");
});

test("a template read failure leaves the dims and claims nothing", () => {
  const failed = factsByKey({
    contextLength: 131072,
    layerCount: 64,
    chatTemplate: null,
  });
  assert.equal(failed.get("contextLength"), "128K tokens");
  assert.equal(failed.get("layers"), "64");
  assert.equal(failed.get("chatTemplate"), "Not available");
});

test("rows come back in the declared field order", () => {
  const facts = localModelInfoFacts({
    contextLength: 131072,
    layerCount: 24,
    moeLayerCount: 0,
    chatTemplate: `{%- if message.role == "system" %}{% endif %}`,
  });
  const order = facts.map((f) => f.key);
  const declared = LOCAL_MODEL_INFO_FIELDS.filter((f) => order.includes(f));
  assert.deepEqual(order, [...declared]);
});

test("every emitted fact has a non-empty label and value", () => {
  const facts = localModelInfoFacts({
    contextLength: 131072,
    layerCount: 24,
    moeLayerCount: 4,
    chatTemplate: `{%- if message.role == "system" %}{% endif %}`,
  });
  assert.ok(facts.length > 0);
  for (const fact of facts) {
    assert.ok(fact.label.trim().length > 0, `${fact.key} has a label`);
    assert.ok(fact.value.trim().length > 0, `${fact.key} has a value`);
  }
});

test("million-token context windows read in M, not thousands of K", () => {
  const ctx = (contextLength: number) =>
    localModelInfoFacts({ contextLength, layerCount: 32 }).find(
      (f) => f.key === "contextLength",
    )?.value;
  assert.equal(ctx(4096), "4K tokens");
  assert.equal(ctx(131_072), "128K tokens");
  assert.equal(ctx(1_048_576), "1M tokens");
  assert.equal(ctx(10_485_760), "10M tokens");
  assert.equal(ctx(1_000_000), "1,000,000 tokens");
});

test("a zeroed header count is not a reading", () => {
  assert.deepEqual(
    localModelInfoFacts({ contextLength: 0, layerCount: null }),
    [],
  );
  assert.deepEqual(
    localModelInfoFacts({ contextLength: 0, layerCount: 0 }),
    [],
  );
  const facts = localModelInfoFacts({ contextLength: 0, layerCount: 32 });
  assert.ok(facts.some((f) => f.key === "layers"));
  assert.ok(!facts.some((f) => f.key === "contextLength"));
});
