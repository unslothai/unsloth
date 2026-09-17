// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Contract for the rows the picker's model-info panel renders (issue #11017).
// The panel reports facts about a repo the user is about to load, so what these cases
// guard is a confident-looking row built from data HF never returned.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { modelInfoFacts, MODEL_INFO_FIELDS } = await import(
  "../src/features/model-picker/components/model-selector/model-info-facts.ts"
);

type Meta = Parameters<typeof modelInfoFacts>[0];

const FULL: Meta = {
  id: "unsloth/Llama-3.1-8B-Instruct",
  license: "llama3.1",
  downloads: 1_234_567,
  likes: 8900,
  totalParams: 8_030_000_000,
  sizeBytes: 16_060_000_000,
  createdAt: "2024-07-23T00:00:00.000Z",
  lastModified: "2025-01-02T00:00:00.000Z",
  library: "transformers",
  pipelineTag: "text-generation",
  tags: ["text-generation", "conversational"],
  languages: ["en", "de"],
  gated: false,
  isPrivate: false,
};

function factFor(meta: Meta, key: string) {
  return modelInfoFacts(meta).find((f) => f.key === key);
}

// `access` is excluded on purpose: it is the one field that reports a condition rather than
// a property, so an ordinary public repo like FULL should not carry the row at all. The
// gated/private case below covers it.
test("a fully populated repo yields every unconditional field", () => {
  const keys = modelInfoFacts(FULL).map((f) => f.key);
  for (const field of MODEL_INFO_FIELDS) {
    if (field === "access") continue;
    assert.ok(keys.includes(field), `missing ${field}`);
  }
});

// The panel's whole purpose is answering "what is this model?" before a load. A row
// invented from absent data is worse than no row, so missing input must drop the row
// rather than render "undefined", "NaN" or a plausible zero.
test("absent values drop their row instead of rendering a placeholder", () => {
  const facts = modelInfoFacts({ id: "acme/mystery" });
  for (const fact of facts) {
    assert.ok(
      !/undefined|NaN|null/i.test(fact.value),
      `placeholder leaked into ${fact.key}: ${fact.value}`,
    );
  }
  assert.equal(factFor({ id: "acme/mystery" }, "downloads"), undefined);
  assert.equal(factFor({ id: "acme/mystery" }, "params"), undefined);
});

// Zero downloads is a fact HF actually returned; dropping it would misreport a real
// new repo as "unknown". This is the case a plain `if (!value)` guard gets wrong.
// Popularity is not a property of the model, and the Hub page one click away shows it.
test("downloads and likes are not rendered", () => {
  const keys = modelInfoFacts(FULL).map((f) => f.key);
  assert.ok(!keys.includes("downloads" as never), "downloads row is gone");
  assert.ok(!keys.includes("likes" as never), "likes row is gone");
});

test("parameter count reads in billions", () => {
  const params = factFor(FULL, "params");
  assert.ok(params);
  assert.match(params.value, /8(\.0)?B/i);
});

// The licence row is the issue's "is it open source?" question, so it carries the
// verdict, not just the slug.
test("the licence row carries an openness verdict", () => {
  const license = factFor(FULL, "license");
  assert.ok(license);
  assert.equal(license.openness, "restricted");
  assert.match(license.value, /Llama 3\.1/);
});

test("an OSI licence reads as open", () => {
  const license = factFor({ id: "x/y", license: "apache-2.0" }, "license");
  assert.ok(license);
  assert.equal(license.openness, "open");
});

// A repo with no stated licence still gets the row: "not stated" is the answer a user
// checking openness needs, and silence would read as "fine".
test("a missing licence still renders a row", () => {
  const license = factFor({ id: "x/y" }, "license");
  assert.ok(license, "licence row dropped when unstated");
  assert.equal(license.openness, "unknown");
});

test("gated and private repos are flagged; ordinary ones are not", () => {
  assert.ok(factFor({ id: "x/y", gated: true }, "access"));
  assert.ok(factFor({ id: "x/y", isPrivate: true }, "access"));
  assert.equal(
    factFor({ id: "x/y", gated: false, isPrivate: false }, "access"),
    undefined,
  );
});

test("languages are listed, and an empty list drops the row", () => {
  const langs = factFor(FULL, "languages");
  assert.ok(langs);
  assert.match(langs.value, /en/i);
  assert.equal(factFor({ id: "x/y", languages: [] }, "languages"), undefined);
});

// An unparseable date from the API must not surface as "Invalid Date".
test("an unparseable date drops its row", () => {
  const fact = factFor({ id: "x/y", createdAt: "not-a-date" }, "created");
  assert.equal(fact, undefined);
});

test("every emitted fact has a non-empty label and value", () => {
  for (const fact of modelInfoFacts(FULL)) {
    assert.ok(fact.label.length > 0, `empty label for ${fact.key}`);
    assert.ok(fact.value.length > 0, `empty value for ${fact.key}`);
  }
});

// Row order is the reading order of the panel; a set would make it incidental.
test("facts come back in the declared field order", () => {
  const keys = modelInfoFacts(FULL).map((f) => f.key);
  const expected = MODEL_INFO_FIELDS.filter((f) => keys.includes(f));
  assert.deepEqual(keys, expected);
});
