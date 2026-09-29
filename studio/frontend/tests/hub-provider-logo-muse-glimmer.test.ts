// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  matchProviderLogo,
  resolveOwnerProviderLogo,
} from "../src/features/hub/lib/provider-logos.ts";

const META = "/hub/profile/logo/meta.svg";

test("the Unsloth re-upload of Muse Glimmer shows the Meta mark", () => {
  // The row the Model hub actually lists. Owner is unsloth, so the org rule
  // never fires and the repo name has to carry it.
  assert.equal(
    resolveOwnerProviderLogo("unsloth", "Muse-Glimmer-30B-GGUF")?.logoPath,
    META,
  );
});

test("every Muse Glimmer variant rides the same stem", () => {
  for (const repo of [
    "Muse-Glimmer-30B",
    "Muse-Glimmer-30B-GGUF",
    "Muse-Glimmer-30B-unsloth-bnb-4bit",
    "Muse-Glimmer-30B-Instruct",
    // No trailing dash in the stem, so a future minor version is picked up.
    "Muse-Glimmer2-30B-GGUF",
  ]) {
    assert.equal(matchProviderLogo(repo)?.id, "meta-llama", repo);
  }
});

test("the stem is case-sensitive and anchored at the start", () => {
  assert.equal(matchProviderLogo("muse-glimmer-30b-gguf"), null);
  assert.equal(matchProviderLogo("Retro-Muse-Glimmer-30B"), null);
  assert.equal(matchProviderLogo("Muse-30B"), null);
});

test("the org rule still covers Meta's own upload", () => {
  assert.equal(
    resolveOwnerProviderLogo("meta-models", "Muse-Glimmer-30B")?.logoPath,
    META,
  );
});

test("an ineligible owner gets nothing from the new stem", () => {
  // Only RELABELED_OWNERS resolve by repo name; everyone else keeps their avatar.
  assert.equal(
    resolveOwnerProviderLogo("someone-else", "Muse-Glimmer-30B-GGUF"),
    null,
  );
});
