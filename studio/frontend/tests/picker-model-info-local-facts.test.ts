// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The local half of the model-info panel: what the GGUF header answers with no network.
// Template cases use the shapes real templates use, not ones invented to fit the code.

import assert from "node:assert/strict";
import test from "node:test";

import {
  LOCAL_MODEL_INFO_FIELDS,
  localModelInfoFacts,
  reasoningSupport,
} from "../src/features/model-picker/components/model-selector/local-model-facts.ts";

const factsByKey = (meta: Parameters<typeof localModelInfoFacts>[0]) =>
  new Map(localModelInfoFacts(meta).map((f) => [f.key, f.value]));

test("context length is reported from the header", () => {
  // 131072 and 262144 are what gpt-oss-20b and Qwen3.8-27B report on disk.
  assert.equal(
    factsByKey({ contextLength: 131072 }).get("contextLength"),
    "128K tokens",
  );
  assert.equal(
    factsByKey({ contextLength: 262144 }).get("contextLength"),
    "256K tokens",
  );
  // Not a clean multiple of 1024, so it stays a plain count rather than a wrong "K".
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
  const keys = [...factsByKey({ contextLength: null, layerCount: null }).keys()];
  assert.deepEqual(keys, []);
  assert.deepEqual([...factsByKey({}).keys()], []);
});

// An undownloaded model reports every field null. "None embedded" would be a finding the
// probe never made, so nothing is claimed.
test("a header that was not read claims nothing, not an absent template", () => {
  const notRead = factsByKey({
    contextLength: null,
    layerCount: null,
    moeLayerCount: null,
    chatTemplate: null,
  });
  assert.deepEqual([...notRead.keys()], []);

  // mmproj-F16.gguf: a projector, whose partial dims read reports 0 expert layers with no
  // context length. Still nothing read, so still no rows.
  const projector = factsByKey({
    contextLength: null,
    layerCount: null,
    moeLayerCount: 0,
    chatTemplate: null,
  });
  assert.deepEqual([...projector.keys()], []);
});

// 0 expert layers is a fact (dense), not a missing reading, once the header was read.
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
  assert.equal(factsByKey({ contextLength: 4096, layerCount: 0 }).has("layers"), false);
});

// Qwen3.8-27B branches on enable_thinking, so thinking can be switched off: hybrid.
test("a template with a thinking toggle is hybrid", () => {
  assert.equal(
    reasoningSupport(
      `{%- if enable_thinking is undefined or enable_thinking is true %}<think>{% endif %}`,
    ),
    "hybrid",
  );
  assert.equal(reasoningSupport("{{ '/no_think' }}"), "hybrid");
});

// gpt-oss-20b carries reasoning_effort and an analysis channel, but no toggle.
test("reasoning with no way out is always on, not hybrid", () => {
  assert.equal(
    reasoningSupport(
      `{%- set reasoning_effort = reasoning_effort|default("medium") %}<|channel|>analysis`,
    ),
    "always",
  );
  assert.equal(reasoningSupport("<think>\n</think>"), "always");
});

test("a template that renders no thinking reports none", () => {
  assert.equal(
    reasoningSupport(
      "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}",
    ),
    "none",
  );
});

test("no template means reasoning is unknown, not unsupported", () => {
  assert.equal(reasoningSupport(null), "unknown");
  assert.equal(reasoningSupport("  "), "unknown");
});

test("the reasoning row is labelled hybrid or always on", () => {
  assert.equal(
    factsByKey({
      contextLength: 4096,
      chatTemplate: "{%- if enable_thinking is true %}<think>{% endif %}",
    }).get("reasoning"),
    "Hybrid",
  );
  assert.equal(
    factsByKey({
      contextLength: 4096,
      chatTemplate: '{%- set reasoning_effort = "high" %}',
    }).get("reasoning"),
    "Always on",
  );
  // No template read, so no reasoning claim either.
  assert.equal(
    factsByKey({ contextLength: 4096, chatTemplate: null }).has("reasoning"),
    false,
  );
});

test("the template row states whether one is embedded", () => {
  const embedded = factsByKey({
    contextLength: 4096,
    chatTemplate: `{%- if message.role == "system" %}{% endif %}`,
  });
  assert.equal(embedded.get("chatTemplate"), "Embedded");

  // Header read, template genuinely absent: the panel says so rather than hiding the row.
  const none = factsByKey({ contextLength: 4096, chatTemplate: null });
  assert.equal(none.get("chatTemplate"), "None embedded");
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
