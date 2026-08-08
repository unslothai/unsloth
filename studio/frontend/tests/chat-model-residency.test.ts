// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The header used to read "loaded" off the picker selection alone. Loading an
// image or video model evicts the chat model (the GPU arbiter allows one owner),
// which leaves the selection untouched, so the header kept its tick and the next
// prompt came back a bare 400. These pin the rule the header now uses.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { chatModelLoaded } from "../src/features/chat/lib/chat-model-loaded.ts";

const PICKED = "unsloth/Qwen3.5-9B-GGUF";

test("a resident model reads as loaded", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: PICKED,
      modelLoading: false,
      isExternalModel: false,
      residentCheckpoint: PICKED,
    }),
    true,
  );
});

// The reported bug: the image load evicted it, the picker kept the name.
test("a model evicted for an image load does not read as loaded", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: PICKED,
      modelLoading: false,
      isExternalModel: false,
      residentCheckpoint: null,
    }),
    false,
  );
});

// Startup: assume loaded rather than flash "not loaded" on every launch.
test("residency not yet read is not treated as evicted", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: PICKED,
      modelLoading: false,
      isExternalModel: false,
      residentCheckpoint: undefined,
    }),
    true,
  );
});

// An API model has no local weights, so residency says nothing about it.
test("an external model is loaded whatever the backend holds", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: "openai:gpt-5",
      modelLoading: false,
      isExternalModel: true,
      residentCheckpoint: null,
    }),
    true,
  );
});

test("nothing picked is never loaded", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: "",
      modelLoading: false,
      isExternalModel: false,
      residentCheckpoint: PICKED,
    }),
    false,
  );
});

// The header tick is the selector's own isLoaded, which was `selected !== ""`.
// Reading the rule out of the source keeps the prop wired to the fix: the first
// attempt at this changed a different modelLoaded and the tick never moved.
test("the selector's tick asks the caller, and defaults to the old rule", () => {
  const source = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /const isLoaded = selected !== "" && \(loaded \?\? true\)/);
  const page = readFileSync(
    new URL("../src/features/chat/chat-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(page, /loaded=\{chatModelLoaded\(\{/, "the chat header must pass it");
  assert.match(page, /residentCheckpoint,/);
});

test("a model still loading is not loaded yet", () => {
  assert.equal(
    chatModelLoaded({
      checkpoint: PICKED,
      modelLoading: true,
      isExternalModel: false,
      residentCheckpoint: PICKED,
    }),
    false,
  );
});

// The trigger tick was only one of three places claiming "loaded", and all three
// read the picker selection. The dropdown's own green "Loaded" badge and the
// Model hub cards kept it after an eviction, which is the same lie in a second
// and third spot.
test("the picker's Loaded badge asks residency, not the selection", () => {
  const pickers = readFileSync(
    new URL(
      "../src/features/model-picker/components/model-selector/pickers.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(pickers, /const loadedModelId = chatModelLoaded\(\{/);
  assert.match(pickers, /residentCheckpoint,/);
  assert.doesNotMatch(
    pickers,
    /const loadedModelId = useChatRuntimeStore\(\(s\) => s\.params\.checkpoint\)/,
  );
});

test("the hub cards ask residency before saying Loaded", () => {
  const hub = readFileSync(
    new URL("../src/features/hub/hub-page.tsx", import.meta.url),
    "utf8",
  );
  assert.match(hub, /residentCheckpoint !== null\s*\n?\s*\? checkpoint/);
});

// Nothing in the chat runtime polls /status: refresh runs on mount and when the
// model lists change, never on a timer. So an eviction caused by the Images
// page was never observed and residentCheckpoint stayed undefined, which reads
// as loaded. The re-read has to be driven by the lifecycle event.
test("another runtime finishing a load re-reads the chat status", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/chat/hooks/use-chat-model-runtime.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(hook, /subscribeModelLifecycle\(\(\{ runtime, loading \}\) => \{/);
  assert.match(hook, /if \(loading \|\| runtime === "chat"\) return;/);
  assert.match(hook, /void refresh\(\{ includeLoras: false \}\)/);
  // And the branch it feeds still clears residency.
  assert.match(hook, /residentCheckpoint: null,/);
});
