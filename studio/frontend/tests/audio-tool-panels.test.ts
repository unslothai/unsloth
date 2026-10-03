// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { AudioModelContext } from "../src/features/audio/tools/types.ts";
import type { AudioWorkflowId } from "../src/features/audio/workflows.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { claimedOptionNames, instructionsKindFor, panelApplies } = await import(
  "../src/features/audio/tools/select.ts"
);

const registry = readSrc("features/audio/tools/registry.tsx");
const panels = readSrc("features/audio/tools/instructions-panels.tsx");

const ctx = (overrides: Partial<AudioModelContext> = {}): AudioModelContext => ({
  audioType: null,
  audioFamily: null,
  musicGeneration: false,
  cudaMusicGeneration: false,
  musicNeedsDescription: false,
  ...overrides,
});

// The panels the rail used to inline, in order, each with the kind and workflow it serves.
const PANELS = [...registry.matchAll(/instructionsPanel\("([^"]+)", "([^"]+)", "[^"]+", \[([^\]]*)\]\)/g)].map(
  ([, id, kind, workflows]) => ({
    id,
    kind,
    families: [] as string[],
    workflows: [...workflows.matchAll(/"([^"]+)"/g)].map((m) => m[1] as AudioWorkflowId),
    appliesTo: (c: AudioModelContext) => instructionsKindFor(c) === kind,
  }),
);

test("each of today's special cases is one panel, in rail order", () => {
  assert.deepEqual(
    PANELS.map(({ id, kind, workflows }) => [id, kind, workflows.join()]),
    [
      ["voice-design", "voice", "speak"],
      ["higgs-scene", "scene", "speak"],
      ["moss-style", "style", "speak"],
      ["music-description", "music", "music"],
    ],
  );
});

test("a model gets the instruction field it always got, and nothing else", () => {
  const shown = (workflow: AudioWorkflowId, c: AudioModelContext) =>
    PANELS.filter((panel) => panelApplies(panel, workflow, c)).map((p) => p.id);
  assert.deepEqual(shown("speak", ctx({ audioType: "audiocpp_tts" })), ["voice-design"]);
  assert.deepEqual(shown("speak", ctx({ audioType: "higgs_tts2" })), ["higgs-scene"]);
  assert.deepEqual(shown("speak", ctx({ audioType: "moss_tts_local" })), ["moss-style"]);
  assert.deepEqual(shown("music", ctx({ musicGeneration: true, audioType: "minimax_music3" })), [
    "music-description",
  ]);
  // Codec models had no instruction field and still have none.
  assert.deepEqual(shown("speak", ctx({ audioType: "snac" })), []);
  assert.deepEqual(shown("speak", ctx({ audioType: "csm" })), []);
  // A music model resident while Speak is open does not put its description on Speak.
  assert.deepEqual(shown("speak", ctx({ audioType: "audiocpp_music" })), []);
});

test("the panels keep the rail's exact copy", () => {
  assert.match(panels, /\? "Scene description"/);
  assert.match(panels, /"Voice or style description"/);
  assert.match(panels, /instructionsKind === "music"\s*\? musicNeedsDescription/);
  assert.match(panels, /id="audio-instructions"/);
  assert.match(panels, /id="audio-language"/);
  assert.match(registry, /kind === "style" \? \(\s*<MossLanguageField/);
});

test("a model that needs a description is held back without one", () => {
  assert.match(
    registry,
    /kind === "music" && ctx\.musicNeedsDescription && !value\.instructions\.trim\(\)/,
  );
  assert.match(
    registry,
    /"Add a music description\. This model needs one beside the lyrics\."/,
  );
});

test("PR 1 panels claim no spec option, so Advanced keeps all of them", () => {
  assert.match(registry, /claims: \[\],/);
  assert.deepEqual([...claimedOptionNames([{ claims: [] }, { claims: [] }])], []);
  assert.deepEqual([...claimedOptionNames([{ claims: ["a"] }, { claims: ["b", "a"] }])], ["a", "b"]);
});
