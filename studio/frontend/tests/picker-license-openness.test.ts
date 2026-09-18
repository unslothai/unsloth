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
    "apple-amlr",
  ]) {
    assert.equal(classifyLicense(slug).openness, "restricted", slug);
  }
});

// `apple-ascl` is the Apple SAMPLE CODE licence, not the research one. Its text grants use,
// reproduction, modification and redistribution in source or binary form, conditioned only on
// keeping the notice and not implying Apple's endorsement — MIT-shaped. Reading it as restricted
// warned about a licence that imposes no commercial or field-of-use limit, and the catch it
// showed named no condition at all. `apple-amlr`, one line above, is the research-only sibling
// and stays restricted.
test("the Apple sample-code licence is open, not restricted", () => {
  const ascl = classifyLicense("apple-ascl");
  assert.equal(ascl.openness, "open");
  assert.match(ascl.summary, /permits commercial use/);
  assert.equal(classifyLicense("apple-amlr").openness, "restricted");
  assert.match(classifyLicense("apple-amlr").summary, /research-only/);
});

// The EU carve-out is not a condition on the grant, it withholds it: Meta's Acceptable Use
// Policy for Llama 4 and for Llama 3.2's vision models says the Section 1(a) rights "are not
// being granted to you if you are an individual domiciled in, or a company with a principal
// place of business in, the European Union". A reader in the EU who sees only the 700M ceiling
// has been told the opposite of what the licence says.
test("the Llama releases that withhold the EU grant say so", () => {
  for (const slug of ["llama4", "llama3.2"]) {
    assert.match(classifyLicense(slug).summary, /European Union/, slug);
  }
  // The text-only releases carry no such clause and must not claim one.
  for (const slug of ["llama2", "llama3", "llama3.1", "llama3.3"]) {
    assert.doesNotMatch(classifyLicense(slug).summary, /European Union/, slug);
  }
});

// A licence slug says what the terms are, not whether the repository will hand you the files:
// meta-llama/*, google/gemma-* and most apple-amlr repos are gated, and this function is given
// the slug alone.
test("a restricted verdict does not claim the weights are public", () => {
  const verdict = classifyLicense("llama3.3");
  assert.doesNotMatch(verdict.summary, /public/);
  assert.match(verdict.summary, /downloadable/);
});

// Copyleft and attribution are part of the grant, and a single "permits commercial use,
// modification and redistribution" erased them. AGPL is the one that costs real money to get
// wrong: §13 obliges anyone offering network access to offer the source too.
test("open licences with obligations name them", () => {
  assert.match(classifyLicense("agpl-3.0").summary, /over a network is offered the source/);
  assert.match(classifyLicense("gpl-3.0").summary, /stay under the GPL/);
  assert.match(classifyLicense("lgpl-3.0").summary, /stay under the LGPL/);
  assert.match(classifyLicense("mpl-2.0").summary, /stay under the MPL/);
  assert.match(classifyLicense("cc-by-4.0").summary, /credit the author/);
  assert.match(
    classifyLicense("cc-by-sa-4.0").summary,
    /derivatives carry the same licence/,
  );
  // A licence with nothing extra to say keeps the plain sentence.
  assert.equal(
    classifyLicense("mit").summary,
    "MIT permits commercial use, modification and redistribution.",
  );
});

// Every one of these is a live Hugging Face licence tag. Falling through to "Licence unclear"
// dropped the non-commercial warning for ~966 repos while their own 4.0 siblings were
// classified, which is the failure direction that costs a user something.
test("older non-commercial spellings are restricted, like their 4.0 siblings", () => {
  for (const slug of [
    "cc-by-nc-2.0",
    "cc-by-nc-3.0",
    "cc-by-nc-sa-2.0",
    "cc-by-nc-sa-3.0",
    "cc-by-nc-nd-3.0",
  ]) {
    const verdict = classifyLicense(slug);
    assert.equal(verdict.openness, "restricted", slug);
    assert.match(verdict.summary, /non-commercial/, slug);
  }
});

test("research-only vendor licences are restricted", () => {
  for (const slug of [
    "fair-noncommercial-research-license",
    "deepfloyd-if-license",
    "intel-research",
    "h-research",
  ]) {
    const verdict = classifyLicense(slug);
    assert.equal(verdict.openness, "restricted", slug);
    assert.match(verdict.summary, /research-only/, slug);
  }
});

test("common OSI licences are not reported as unrecognised", () => {
  for (const slug of [
    "gpl-2.0",
    "lgpl-2.1",
    "afl-3.0",
    "bsl-1.0",
    "epl-2.0",
    "ecl-2.0",
    "zlib",
    "ncsa",
    "ms-pl",
    "osl-3.0",
    "eupl-1.2",
    "postgresql",
    "bsd-3-clause-clear",
    "wtfpl",
  ]) {
    assert.equal(classifyLicense(slug).openness, "open", slug);
  }
  assert.equal(classifyLicense("bigscience-openrail-m").openness, "restricted");
});

// The bare `cc` family tag spans CC0 through CC BY-NC-ND, so no verdict is possible — but it is
// a tag HF publishes, not a slug nobody recognised, and the summary should say which it is.
test("the generic Creative Commons tag explains why it is unclear", () => {
  const verdict = classifyLicense("cc");
  assert.equal(verdict.openness, "unknown");
  assert.doesNotMatch(verdict.summary, /Unrecognised/);
  assert.match(verdict.summary, /CC0/);
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
