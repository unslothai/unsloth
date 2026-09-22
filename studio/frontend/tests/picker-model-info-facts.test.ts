// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

test("a fully populated repo yields every unconditional field", () => {
  const keys = modelInfoFacts(FULL).map((f) => f.key);
  for (const field of MODEL_INFO_FIELDS) {
    if (field === "access") continue;
    assert.ok(keys.includes(field), `missing ${field}`);
  }
});

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

test("licences are displayed as reported without classification", () => {
  for (const license of ["apache-2.0", "llama3.1", "custom-license"]) {
    assert.deepEqual(factFor({ id: "x/y", license }, "license"), {
      key: "license",
      label: "License",
      value: license,
    });
  }
});

test("an absent or blank licence is explicitly unspecified", () => {
  for (const license of [undefined, null, "", "  "]) {
    assert.equal(
      factFor({ id: "x/y", license }, "license")?.value,
      "Not specified",
    );
  }
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

test("facts come back in the declared field order", () => {
  const keys = modelInfoFacts(FULL).map((f) => f.key);
  const expected = MODEL_INFO_FIELDS.filter((f) => keys.includes(f));
  assert.deepEqual(keys, expected);
});

test("small parameter counts do not round to zero", () => {
  const paramsValue = (totalParams: number) =>
    modelInfoFacts({ id: "a/b", totalParams }).find((f) => f.key === "params")
      ?.value;
  assert.equal(paramsValue(8_030_000_000), "8B");
  assert.equal(paramsValue(1_500_000), "2M");
  assert.equal(paramsValue(400_000), "0.4M");
  assert.equal(paramsValue(30_000), "30,000");
  for (const n of [1, 30_000, 400_000, 499_999]) {
    assert.notEqual(paramsValue(n), "0M", String(n));
  }
});

test("a full-precision size says so, a curated one does not", () => {
  const estimate = modelInfoFacts({
    id: "a/b",
    sizeBytes: 16_060_522_496,
    sizeIsFullPrecision: true,
  }).find((f) => f.key === "size");
  assert.ok(estimate);
  assert.match(estimate.label, /full precision/);
  assert.match(estimate.detail ?? "", /quantized download is smaller/);

  const curated = modelInfoFacts({
    id: "a/b",
    sizeBytes: 4_900_000_000,
    sizeIsFullPrecision: false,
  }).find((f) => f.key === "size");
  assert.ok(curated);
  assert.equal(curated.label, "Size");
  assert.equal(curated.detail, undefined);
});
