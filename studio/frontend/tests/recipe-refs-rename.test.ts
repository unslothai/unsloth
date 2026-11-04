// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { replaceRef } from "../src/features/recipe-studio/utils/refs.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

test("renaming a column rewrites it wherever it is a variable root, and nowhere else", () => {
  const before =
    "{{ ticket }} / {{ ticket.issue_title }} {% if ticket.priority == 'P1' %}urgent{% endif %} " +
    "{{ ticket_id }} {{ other.ticket }} {% if x == 'the ticket' %}{% endif %} ticket";
  assert.equal(
    replaceRef(before, "ticket", "support_ticket"),
    "{{ support_ticket }} / {{ support_ticket.issue_title }} {% if support_ticket.priority == 'P1' %}urgent{% endif %} " +
      "{{ ticket_id }} {{ other.ticket }} {% if x == 'the ticket' %}{% endif %} ticket",
  );
});

test("a filter or test with the column's name is left alone", () => {
  assert.equal(
    replaceRef(
      "{{ length }} {{ response | length }} {% if x is length %}{% endif %} {{ length + 1 }}",
      "length",
      "word_count",
    ),
    "{{ word_count }} {{ response | length }} {% if x is length %}{% endif %} {{ word_count + 1 }}",
  );
});

test("a keyword argument with the column's name is left alone", () => {
  assert.equal(
    replaceRef(
      "{{ x | truncate(length=10) }} {{ f(length = 2) }} {% if length == 3 %}{% endif %}",
      "length",
      "word_count",
    ),
    "{{ x | truncate(length=10) }} {{ f(length = 2) }} {% if word_count == 3 %}{% endif %}",
  );
});

test("an escaped quote inside a string literal keeps the rest of the literal untouched", () => {
  assert.equal(
    replaceRef(`{{ "he said \\"answer\\"" ~ answer }} {{ 'it\\'s answer' }}`, "answer", "result"),
    `{{ "he said \\"answer\\"" ~ result }} {{ 'it\\'s answer' }}`,
  );
});

test("renaming a column rewrites a sampler rule that conditions on it, keeping the rule's params", async () => {
  const { applyRenameToConfigs } = await import(
    "../src/features/recipe-studio/stores/helpers/reference-sync.ts"
  );
  const configs = {
    a: {
      id: "a",
      kind: "sampler",
      name: "difficulty",
      // biome-ignore lint/style/useNamingConvention: api schema
      sampler_type: "category",
      // biome-ignore lint/style/useNamingConvention: api schema
      conditional_params: {
        "topic == 'science'": { values: ["hard"] },
        "other == 'topic'": { values: ["easy"] },
      },
    },
  } as never;
  const out = applyRenameToConfigs(configs, "topic", "subject") as Record<string, { conditional_params: object }>;
  assert.deepEqual(out.a.conditional_params, {
    "subject == 'science'": { values: ["hard"] },
    "other == 'topic'": { values: ["easy"] },
  });
});

test("renaming a column moves an AI column's image context with its prompt", async () => {
  const { applyRenameToConfigs } = await import(
    "../src/features/recipe-studio/stores/helpers/reference-sync.ts"
  );
  const configs = {
    a: { id: "a", kind: "sampler", name: "picture" },
    b: {
      id: "b",
      kind: "llm",
      name: "caption",
      prompt: "Describe {{ picture }}",
      // biome-ignore lint/style/useNamingConvention: api schema
      image_context: { enabled: true, column_name: "picture" },
    },
  };
  // biome-ignore lint/suspicious/noExplicitAny: test fixture
  const next = applyRenameToConfigs(configs as any, "picture", "photo") as any;
  assert.equal(next.b.prompt, "Describe {{ photo }}");
  assert.deepEqual(next.b.image_context, { enabled: true, column_name: "photo" });
});
