// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The local half of the model-info panel: what the GGUF header answers with no network.
// Template cases use the shapes real templates use, not ones invented to fit the code.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

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
  const keys = [
    ...factsByKey({ contextLength: null, layerCount: null }).keys(),
  ];
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
  assert.equal(
    factsByKey({ contextLength: 4096, layerCount: 0 }).has("layers"),
    false,
  );
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

test("reasoning markers do not establish an always-on capability", () => {
  assert.equal(
    reasoningSupport(
      `{%- set reasoning_effort = reasoning_effort|default("medium") %}<|channel|>analysis`,
    ),
    "detected",
  );
  assert.equal(reasoningSupport("<think>\n</think>"), "detected");
});

test("a system-prompt thinking switch is not reported as always on", () => {
  // Source: nvidia/Llama-3_3-Nemotron-Super-49B-v1 tokenizer_config.json.
  // Its system prompt controls thinking; the template only removes past thoughts.
  const template = readText("fixtures/nemotron-super-chat-template.jinja");
  assert.equal(reasoningSupport(template), "detected");
  const reasoning = localModelInfoFacts({
    contextLength: 131072,
    chatTemplate: template,
  }).find((fact) => fact.key === "reasoning");
  assert.ok(reasoning);
  assert.equal(reasoning.value, "Detected");
  assert.doesNotMatch(
    reasoning.detail ?? "",
    /always|no way to turn thinking off/i,
  );
});

// The markers recognise reasoning; they cannot prove its absence, so no match is unknown.
test("a template with no markers is unknown, not a verdict of no", () => {
  assert.equal(
    reasoningSupport(
      "{% for message in messages %}{{ message.role }}: {{ message.content }}{% endfor %}",
    ),
    "unknown",
  );
});

test("no template means reasoning is unknown, not unsupported", () => {
  assert.equal(reasoningSupport(null), "unknown");
  assert.equal(reasoningSupport("  "), "unknown");
});

test("the reasoning row distinguishes a toggle from detected markers", () => {
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
    "Detected",
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

  // Null is not absence, so the row stays neutral rather than claiming the file carries none.
  const none = factsByKey({ contextLength: 4096, chatTemplate: null });
  assert.equal(none.get("chatTemplate"), "Not available");
});

// The probe drops a template over MAX_CHAT_TEMPLATE_BYTES (65536) while still returning the
// dims it read first, so this shape belongs to a file that does carry one.
test("a withheld template is not reported as absent", () => {
  const overCap = factsByKey({
    contextLength: 8192,
    layerCount: 32,
    moeLayerCount: 0,
    chatTemplate: null,
  });
  assert.equal(overCap.get("chatTemplate"), "Not available");
  assert.notEqual(overCap.get("chatTemplate"), "None embedded");
  // Nothing to read the markers out of, so no reasoning claim either.
  assert.equal(overCap.has("reasoning"), false);
});

// Same shape when the template read throws after the dims were read: the route's probe catches
// it and returns what it already had.
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

// A real counterexample, not one built around the markers: QwQ-32B-Preview is a reasoning
// model whose published template carries no thinking markers at all. Verbatim excerpt.
const QWQ_32B_PREVIEW_TEMPLATE = `{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
`;

test("a reasoning model with no markers is not called unsupported", () => {
  assert.equal(reasoningSupport(QWQ_32B_PREVIEW_TEMPLATE), "unknown");
  assert.equal(
    factsByKey({
      contextLength: 32768,
      chatTemplate: QWQ_32B_PREVIEW_TEMPLATE,
    }).get("reasoning"),
    "Not detected",
  );
});

// A K-only unit rendered Llama 4's 10,485,760-token window as "10240K tokens".
test("million-token context windows read in M, not thousands of K", () => {
  const ctx = (contextLength: number) =>
    localModelInfoFacts({ contextLength, layerCount: 32 }).find(
      (f) => f.key === "contextLength",
    )?.value;
  assert.equal(ctx(4096), "4K tokens");
  assert.equal(ctx(131_072), "128K tokens");
  assert.equal(ctx(1_048_576), "1M tokens");
  assert.equal(ctx(10_485_760), "10M tokens");
  // Not a power of two, so neither unit divides it: a grouped count is the honest rendering.
  assert.equal(ctx(1_000_000), "1,000,000 tokens");
});

// `isReportedCount` accepts 0 so a dense model's `moeLayerCount: 0` can say "None (dense)", but
// a context length of 0 is not a reading — it cleared the header gate and then failed every row
// guard, leaving the panel asserting "Chat template: Not available" on its own, which is the one
// claim this module says must never be made from an unread header.
test("a zeroed header count is not a reading", () => {
  assert.deepEqual(
    localModelInfoFacts({ contextLength: 0, layerCount: null }),
    [],
  );
  assert.deepEqual(localModelInfoFacts({ contextLength: 0, layerCount: 0 }), []);
  // One real count is still a reading, and the other rows fall away on their own.
  const facts = localModelInfoFacts({ contextLength: 0, layerCount: 32 });
  assert.ok(facts.some((f) => f.key === "layers"));
  assert.ok(!facts.some((f) => f.key === "contextLength"));
});

// Bracket- and tag-delimited reasoning families were invisible to a `<think>`-only pattern.
// Magistral is the pointed case: this same panel offers it an Unsloth reasoning guide.
test("reasoning markers beyond the think tag are detected", () => {
  const templates = {
    magistral: "[SYSTEM_PROMPT]x[/SYSTEM_PROMPT][THINK]{{ r }}[/THINK]",
    seed: "<seed:think>{{ t }}</seed:think>",
    thinking: "Answer inside <thinking>...</thinking> first.",
    thoughtPipe: "<|start_of_thought|>{{ t }}",
  };
  for (const [name, template] of Object.entries(templates)) {
    assert.equal(reasoningSupport(template), "detected", name);
  }
});

// The giveaway idiom of a NON-thinking variant: it strips a previous turn's reasoning out of the
// history. Reading the marker it deletes as proof that it reasons inverts the answer.
test("a template that strips reasoning from history is not reasoning", () => {
  assert.equal(
    reasoningSupport(
      "{% for m in messages %}{{ m.content.split('</think>')[-1] }}{% endfor %}",
    ),
    "unknown",
  );
  // A template that strips history AND emits its own marker still reasons.
  assert.equal(
    reasoningSupport(
      "{% for m in messages %}{{ m.content.split('</think>')[-1] }}{% endfor %}{{ '<think>' }}",
    ),
    "detected",
  );
});

// "Hybrid" carries the only hard claim in this module — that reasoning can be turned off per
// request. Naming the variable is not honouring it, and a documentation note is not behaviour.
test("hybrid requires a switch the template actually branches on", () => {
  assert.equal(
    reasoningSupport("{%- set enable_thinking = true %}{{ messages[0].content }}"),
    "unknown",
  );
  assert.equal(
    reasoningSupport("{# pass /no_think to disable thinking #}{{ messages[0].content }}"),
    "unknown",
  );
  // A real branch, and the model-specific sentinel emitted for real, both still read as hybrid.
  assert.equal(
    reasoningSupport("{% if enable_thinking %}<think>{% else %}<think></think>{% endif %}"),
    "hybrid",
  );
  assert.equal(reasoningSupport("{{ '/no_think' }}"), "hybrid");
});
