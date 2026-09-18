// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Contract for the picker info panel's "is this open source?" answer (issue #11017).
// The panel reports a licence's practical terms, so the risk these cases guard is a
// permissive verdict on weights that do not actually grant it.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { classifyLicense } = await import(
  "../src/features/model-picker/components/model-selector/license-openness.ts"
);

test("OSI-style licences read as open", () => {
  for (const slug of ["apache-2.0", "mit", "bsd-3-clause", "cc0-1.0"]) {
    assert.equal(classifyLicense(slug).openness, "open", slug);
  }
});

test("slugs carry a human label, not the raw slug", () => {
  assert.equal(classifyLicense("apache-2.0").label, "Apache 2.0");
  assert.equal(classifyLicense("mit").label, "MIT");
});

// Llama/Gemma ship weights publicly under an acceptable-use policy and, for Llama, a
// 700M-MAU ceiling. Calling that "open" is the single most consequential wrong answer
// this module could give, so it gets its own case.
test("community licences are restricted, never open", () => {
  for (const slug of [
    "llama2",
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama4",
    "gemma",
    "apple-ascl",
  ]) {
    assert.equal(classifyLicense(slug).openness, "restricted", slug);
  }
});

// A non-commercial grant is usable for evaluation but not for shipping; conflating it
// with Apache would mislead exactly the user who checks before deploying.
test("non-commercial and research-only grants are restricted", () => {
  for (const slug of [
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "creativeml-openrail-m",
  ]) {
    assert.equal(classifyLicense(slug).openness, "restricted", slug);
  }
});

test("proprietary terms do not grant redistribution", () => {
  assert.equal(classifyLicense("proprietary").openness, "proprietary");
});

// `unknown` and `other` name the absence of a verdict, not a refusal to share. Calling either
// "proprietary" would tell the reader redistribution is denied when the repository said no such
// thing — the same invention this module refuses to make for slugs it cannot parse.
test("tags that state no terms stay unknown, never proprietary", () => {
  for (const slug of ["unknown", "other"]) {
    const verdict = classifyLicense(slug);
    assert.equal(verdict.openness, "unknown", slug);
    assert.notEqual(verdict.openness, "proprietary", slug);
    assert.notEqual(verdict.openness, "open", slug);
    assert.match(verdict.summary, /model card/i, slug);
  }
});

// `other` can be a custom licence that grants everything Apache does, so the copy must not
// claim the terms withhold redistribution.
test("a custom-terms tag does not assert that redistribution is refused", () => {
  assert.doesNotMatch(
    classifyLicense("other").summary,
    /do not grant redistribution/i,
  );
});

// Absence of a licence is not permission. Defaulting a blank to "open" would invent a
// grant the repository never made.
test("a missing licence is unknown, not open", () => {
  for (const value of [null, undefined, "", "   "]) {
    const verdict = classifyLicense(value as string | null | undefined);
    assert.equal(verdict.openness, "unknown");
    assert.notEqual(verdict.openness, "open");
  }
});

test("unrecognised slugs fall back to unknown and keep the original text", () => {
  const verdict = classifyLicense("some-vendor-eula-v3");
  assert.equal(verdict.openness, "unknown");
  assert.equal(verdict.label, "some-vendor-eula-v3");
});

// HF lowercases its own slugs, but a hand-edited model card can carry any case.
test("classification is case- and whitespace-insensitive", () => {
  assert.equal(classifyLicense("  Apache-2.0 ").openness, "open");
  assert.equal(classifyLicense("MIT").label, "MIT");
  assert.equal(classifyLicense("Llama3.1").openness, "restricted");
});

// The panel renders this next to the verdict; an empty string would leave a bare chip.
test("every verdict carries a non-empty summary", () => {
  for (const slug of [
    null,
    "mit",
    "llama3",
    "proprietary",
    "mystery-licence",
  ]) {
    const { summary } = classifyLicense(slug as string | null);
    assert.ok(summary.length > 0, `empty summary for ${String(slug)}`);
  }
});

// A `license:` tag is arbitrary repo text, and the tables are plain objects, so a bare index
// also resolved Object.prototype: `__proto__` and `constructor` read as open, with an object
// and a function for a label. Neither is a valid React child.
test("inherited property names stay unknown, with a string label", () => {
  for (const slug of ["__proto__", "constructor", "toString", "valueOf"]) {
    const verdict = classifyLicense(slug);
    assert.equal(verdict.openness, "unknown", `${slug} must not classify`);
    assert.equal(typeof verdict.label, "string", `${slug} label must be a string`);
    assert.equal(typeof verdict.summary, "string");
  }
});

test("every verdict's label is a string, for the recognised slugs too", () => {
  for (const slug of ["apache-2.0", "llama3.1", "gemma", "proprietary", "other", ""]) {
    assert.equal(typeof classifyLicense(slug).label, "string", slug);
  }
});
