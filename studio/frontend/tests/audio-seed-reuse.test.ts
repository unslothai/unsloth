// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  audioOptionUsesSlider,
  ignoredOptionsWarning,
  parseAudioOptions,
} from "../src/features/audio/audio-options.ts";
import { readSrc } from "./helpers/kit.ts";

test("the seed row is a blank number box, not a slider", () => {
  const [seed] = parseAudioOptions([
    { name: "seed", type: "int", min: 0, max: 2147483647, default: null },
  ]);
  assert.equal(seed.default, null);
  assert.equal(audioOptionUsesSlider(seed), false);
});

test("Reuse seed shows only for a clip with a numeric seed and fills the field", () => {
  const workspace = readSrc("features/audio/pages/tts-workspace.tsx");
  assert.match(
    workspace,
    /\{onReuseSeed && typeof clip\.settings\?\.seed === "number" \? \([\s\S]{0,400}Reuse seed/,
  );
  const page = readSrc("features/audio/audio-page.tsx");
  const handler = page.slice(page.indexOf("const handleReuseSeed"));
  assert.match(handler, /specs\.some\(\(spec\) => spec\.name === "seed"\)/);
  assert.match(handler, /toast\.info\("This model doesn't take a seed\."\)/);
  assert.match(
    handler,
    /handleAudioOptionChange\("seed", seed\);\s*setAdvancedOpen\(true\);\s*toast\.success\(`Seed \$\{seed\} set for the next take\.`\)/,
  );
  assert.match(page, /onReuseSeed: handleReuseSeed/);
});

test("ignored options read as one sentence", () => {
  const warn = (...names: string[]) => ignoredOptionsWarning(names, "Kokoro");
  const them = "so they weren't sent.";
  assert.equal(warn(), null);
  assert.equal(
    warn("exaggeration"),
    "Kokoro doesn't take Exaggeration, so it wasn't sent.",
  );
  assert.equal(
    warn("exaggeration", "top_k"),
    `Kokoro doesn't take Exaggeration or Top k, ${them}`,
  );
  assert.equal(warn("a", "b", "c"), `Kokoro doesn't take A, B or C, ${them}`);
  assert.match(
    readSrc("features/audio/hooks/use-clone-generation.ts"),
    /warnIgnoredOptions\(response\.ignored_options, response\.model\)/,
  );
  assert.match(
    readSrc("features/audio/hooks/use-speech-generation.ts"),
    /warnIgnoredOptions\(generated\.ignored_options, generated\.model\)/,
  );
});
