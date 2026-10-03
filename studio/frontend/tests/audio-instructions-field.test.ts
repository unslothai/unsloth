// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { instructionsFieldKind } = await import("../src/features/audio/audio-page-policy.ts");

const fields = readSrc("features/audio/components/instructions-fields.tsx");
const host = readSrc("features/audio/audio-page.tsx");

test("a model gets the instruction field it always got, on its own page", () => {
  assert.equal(instructionsFieldKind("speak", "audiocpp_tts"), "voice");
  assert.equal(instructionsFieldKind("speak", "higgs_tts2"), "scene");
  assert.equal(instructionsFieldKind("speak", "moss_tts_local"), "style");
  assert.equal(instructionsFieldKind("music", "minimax_music3"), "music");
  assert.equal(instructionsFieldKind("speak", "snac"), null);
  assert.equal(instructionsFieldKind("speak", "csm"), null);
  assert.equal(instructionsFieldKind("speak", "audiocpp_music"), null);
});

test("the fields keep the rail's exact copy", () => {
  assert.match(fields, /\? "Scene description"/);
  assert.match(fields, /"Voice or style description"/);
  assert.match(fields, /instructionsKind === "music"\s*\? musicNeedsDescription/);
  assert.match(fields, /id="audio-instructions"/);
  assert.match(fields, /id="audio-language"/);
  assert.match(host, /instructionsKind === "style" \? \(\s*<MossLanguageField/);
});
