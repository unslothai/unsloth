// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PROVIDER_LOGOS,
  matchProviderLogoByOwner,
  providerLogoFromBaseModel,
  resolveOwnerProviderLogo,
} from "../src/features/hub/lib/provider-logos.ts";

test("an Unsloth re-upload named after no family takes the base model's mark", () => {
  // unsloth/Muse-Glimmer-30B-GGUF matches no prefix; base_model names Meta.
  assert.equal(
    resolveOwnerProviderLogo(
      "unsloth",
      "Muse-Glimmer-30B-GGUF",
      "meta-models/Muse-Glimmer-30B",
    )?.id,
    "meta-llama",
  );
  assert.equal(
    resolveOwnerProviderLogo("unsloth", "Muse-Glimmer-30B-GGUF", null),
    null,
  );
});

test("the repo name still wins over base-model provenance", () => {
  // DeepSeek's Llama distill is DeepSeek's, not Meta's, whichever base is tagged.
  assert.equal(
    resolveOwnerProviderLogo(
      "unsloth",
      "DeepSeek-R1-Distill-Llama-8B-GGUF",
      "meta-llama/Llama-3.1-8B",
    )?.id,
    "deepseek-ai",
  );
});

test("base-model owners resolve in full, never as a prefix", () => {
  assert.equal(matchProviderLogoByOwner("meta-models")?.id, "meta-llama");
  assert.equal(matchProviderLogoByOwner("META-LLAMA")?.id, "meta-llama");
  assert.equal(matchProviderLogoByOwner("metavoice"), null);
  assert.equal(matchProviderLogoByOwner(""), null);
});

test("an unknown base owner falls through to the base repo name", () => {
  assert.equal(
    providerLogoFromBaseModel("some-lab/Qwen3-30B-A3B")?.id,
    "qwen",
  );
  assert.equal(providerLogoFromBaseModel("Llama-3.2-1B")?.id, "meta-llama");
  assert.equal(providerLogoFromBaseModel("some-lab/Untitled-7B"), null);
  assert.equal(providerLogoFromBaseModel(null), null);
});

test("only relabeled owners inherit a provider mark", () => {
  assert.equal(
    resolveOwnerProviderLogo(
      "some-org",
      "Muse-Glimmer-30B-GGUF",
      "meta-models/Muse-Glimmer-30B",
    ),
    null,
  );
});

test("every declared base-model owner is unique to one provider", () => {
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
