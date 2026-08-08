// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The header used to read "loaded" off the picker selection alone. Loading an
// image or video model evicts the chat model (the GPU arbiter allows one owner),
// which leaves the selection untouched, so the header kept its tick and the next
// prompt came back a bare 400. These pin the rule the header now uses.

import assert from "node:assert/strict";
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
