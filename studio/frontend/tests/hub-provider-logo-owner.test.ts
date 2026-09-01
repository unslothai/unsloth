// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PROVIDER_LOGOS,
  matchProviderLogoByOwner,
  resolveOwnerProviderLogo,
} from "../src/features/hub/lib/provider-logos.ts";

test("every Meta org shows the Meta mark, whatever the repo is named", () => {
  for (const owner of ["meta-models", "meta-llama", "facebook"]) {
    assert.equal(matchProviderLogoByOwner(owner)?.id, "meta-llama", owner);
  }
  // Muse Glimmer matches no prefix; the org alone decides.
  assert.equal(
    resolveOwnerProviderLogo("meta-models", "Muse-Glimmer-30B-GGUF")?.logoPath,
    "/hub/profile/logo/meta.svg",
  );
  assert.equal(
    resolveOwnerProviderLogo("meta-llama", "Llama-4-Scout-17B-16E-Instruct")
      ?.logoPath,
    "/hub/profile/logo/meta.svg",
  );
});

test("owners match in full, never as a prefix", () => {
  assert.equal(matchProviderLogoByOwner("META-MODELS")?.id, "meta-llama");
  assert.equal(matchProviderLogoByOwner("  meta-models  ")?.id, "meta-llama");
  // Unrelated accounts that merely start with the same letters stay untouched.
  assert.equal(matchProviderLogoByOwner("metavoice"), null);
  assert.equal(matchProviderLogoByOwner("meta-models-community"), null);
  assert.equal(matchProviderLogoByOwner("facebookresearch"), null);
  assert.equal(matchProviderLogoByOwner(""), null);
  assert.equal(matchProviderLogoByOwner(null), null);
});

test("an unlisted owner is left to its Hub avatar", () => {
  assert.equal(resolveOwnerProviderLogo("bartowski", "Qwen3-8B-GGUF"), null);
  assert.equal(resolveOwnerProviderLogo("some-org", "Muse-Glimmer-30B"), null);
});

test("Unsloth re-uploads still resolve by repo name", () => {
  assert.equal(
    resolveOwnerProviderLogo("unsloth", "Llama-4-Scout-17B-16E-Instruct-GGUF")
      ?.id,
    "meta-llama",
  );
  assert.equal(
    resolveOwnerProviderLogo("unsloth", "DeepSeek-R1-Distill-Llama-8B-GGUF")
      ?.id,
    "deepseek-ai",
  );
  assert.equal(resolveOwnerProviderLogo("unsloth", "Qwen3-30B-A3B")?.id, "qwen");
  assert.equal(resolveOwnerProviderLogo("unsloth", undefined), null);
});

test("no org is claimed by two providers", () => {
  const seen = new Map<string, string>();
  for (const provider of PROVIDER_LOGOS) {
    for (const owner of provider.owners ?? []) {
      const key = owner.toLowerCase();
      assert.equal(
        seen.get(key),
        undefined,
        `${owner} is claimed by both ${seen.get(key)} and ${provider.id}`,
      );
      seen.set(key, provider.id);
    }
  }
});
