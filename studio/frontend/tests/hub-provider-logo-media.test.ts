// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import test from "node:test";

import {
  matchProviderLogo,
  resolveOwnerProviderLogo,
} from "../src/features/hub/lib/provider-logos.ts";

const CASES: [string, string][] = [
  ["LFM2.5-24B-A2B-GGUF", "liquidai"],
  ["LFM2-8B-A1B-GGUF", "liquidai"],
  ["Inkling-2-Mini-GGUF", "thinkingmachines"],
  ["inkling-GGUF", "thinkingmachines"],
  ["flux-3-image-klein-pup-GGUF", "black-forest-labs"],
  ["FLUX.2-klein-9B-GGUF", "black-forest-labs"],
  ["LTX-2.5-GGUF", "lightricks"],
  ["MOSS-Image-SFT-GGUF", "openmoss"],
  ["Ornith-1.5-122B-GGUF", "ornith-ai"],
  ["flint-GGUF", "recursivesi"],
];

test("unsloth re-uploads of these families show their maker's logo", () => {
  for (const [repo, id] of CASES) {
    assert.equal(resolveOwnerProviderLogo("unsloth", repo)?.id, id, repo);
  }
});

test("the makers' own orgs resolve by owner", () => {
  assert.equal(resolveOwnerProviderLogo("LiquidAI", "x")?.id, "liquidai");
  assert.equal(resolveOwnerProviderLogo("Lightricks", "x")?.id, "lightricks");
  assert.equal(
    resolveOwnerProviderLogo("RecursiveSI", "flint")?.id,
    "recursivesi",
  );
});

test("each logo file ships in public", () => {
  for (const [repo] of CASES) {
    const path = matchProviderLogo(repo)?.logoPath ?? "";
    assert.ok(existsSync(new URL(`../public${path}`, import.meta.url)), path);
  }
});
