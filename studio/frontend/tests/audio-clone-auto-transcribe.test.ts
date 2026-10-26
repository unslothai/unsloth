// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  referenceLengthHint,
  shouldAutoTranscribeReference,
} from "../src/features/audio/clone-policy.ts";
import { readSrc } from "./helpers/kit.ts";

const ready = {
  mode: "required" as const,
  referenceText: "",
  referenceId: "ref-1",
  referenceBusy: false,
  referenceExpired: false,
  sttReady: true,
  doneFor: null,
};

test("a required transcript fills itself once per usable reference", () => {
  for (const [field, value, expected] of [
    ["mode", "required", true],
    ["mode", "optional", false],
    ["mode", "hidden", false],
    ["referenceText", "Hello there.", false],
    ["referenceText", "  ", true],
    ["referenceId", null, false],
    ["referenceBusy", true, false],
    ["referenceExpired", true, false],
    ["sttReady", false, false],
    ["doneFor", "ref-1", false],
    ["doneFor", "ref-0", true],
  ] as const) {
    const input = { ...ready, [field]: value };
    assert.equal(shouldAutoTranscribeReference(input), expected, field);
  }
});

test("the clone hook runs the gate with the page's downloaded STT model", () => {
  assert.match(
    readSrc("features/audio/hooks/use-clone-generation.ts"),
    /busy !== null \|\|\s*!shouldAutoTranscribeReference\(\{[\s\S]*doneFor: autoTranscribed\.current[\s\S]*autoTranscribed\.current = reference\.id;\s*void transcribeOnce\(reference\)/,
  );
  assert.match(
    readSrc("features/audio/audio-page.tsx"),
    /const cloneSttReady =\s*cloneSttRepo !== null &&\s*downloadedSttArtifacts\.some[\s\S]*sttReady: cloneSttReady/,
  );
});

test("the reference length hint", () => {
  const hint = (seconds: number | null) =>
    referenceLengthHint(seconds, [5, 15]);
  assert.equal(hint(4.21), "4.2 s. 5 to 15 s of one clear voice clones best.");
  for (const fine of [5, 15, null]) assert.equal(hint(fine), null);
  assert.equal(hint(22), "Longer than needed; 5 to 15 s works best.");
  assert.match(
    readSrc("features/audio/pages/clone-page.tsx"),
    /idealSeconds=\{\[5, 15\]\}/,
  );
});

test("auto-transcribe runs only while Audio is visible on the Clone page", () => {
  assert.match(
    readSrc("features/audio/hooks/use-clone-generation.ts"),
    /if \(\s*!onClonePage \|\|\s*!reference \|\|/,
  );
  assert.match(
    readSrc("features/audio/audio-page.tsx"),
    /onClonePage: active && pageWorkflow === "clone",/,
  );
});
