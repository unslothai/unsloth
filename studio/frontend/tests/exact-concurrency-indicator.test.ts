// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The header indicator for llama.cpp's exact concurrency. "off" is the default and the common
// case, so the chip renders nothing for it, and "unavailable" has to name the server's refusal
// rather than read as a Studio fault. The wiring is checked at the source: the applier is one
// large object literal with no seam to call.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  type ExactConcurrencyState,
  exactConcurrencyChip,
  exactConcurrencyChipApplies,
  normalizeExactConcurrency,
} from "../src/features/chat/lib/exact-concurrency.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const read = (relative: string) =>
  readFileSync(path.join(HERE, "..", relative), "utf8");

const APPLIER = read(
  "src/features/chat/lib/apply-inference-status-to-store.ts",
);
const API_TYPES = read("src/features/chat/types/api.ts");
const STORE = read("src/features/chat/stores/chat-runtime-store.ts");
const CHIP = read("src/features/chat/components/exact-concurrency-chip.tsx");
const CHAT_PAGE = read("src/features/chat/chat-page.tsx");

test("the three reported states map to themselves", () => {
  assert.equal(normalizeExactConcurrency("on"), "on");
  assert.equal(normalizeExactConcurrency("off"), "off");
  assert.equal(normalizeExactConcurrency("unavailable"), "unavailable");
});

test("a backend that does not publish the field reads as off", () => {
  // undefined is "this server predates the switch", not "the guarantee holds".
  assert.equal(normalizeExactConcurrency(undefined), "off");
  assert.equal(normalizeExactConcurrency(null), "off");
  assert.equal(normalizeExactConcurrency("ON"), "off");
  assert.equal(normalizeExactConcurrency("auto"), "off");
  assert.equal(normalizeExactConcurrency(""), "off");
});

test("on shows Exact and explains what it buys", () => {
  const chip = exactConcurrencyChip("on");
  assert.ok(chip);
  assert.equal(chip.label, "Exact");
  assert.match(
    chip.title,
    /identical output regardless of other chats sharing this model/,
  );
});

test("unavailable says so, without blaming the server for a mode Studio may have withheld", () => {
  const chip = exactConcurrencyChip("unavailable");
  assert.ok(chip);
  assert.equal(chip.label, "Exact unavailable");
  assert.match(chip.title, /not running with it/);
  assert.doesNotMatch(chip.title, /refused/);
  assert.match(
    chip.title,
    /identical output regardless of other chats sharing this model/,
  );
});

test("off renders nothing at all", () => {
  assert.equal(exactConcurrencyChip("off"), null);
});

test("every state the normalizer can return has a decided chip", () => {
  const states: ExactConcurrencyState[] = ["on", "off", "unavailable"];
  for (const state of states) {
    const chip = exactConcurrencyChip(state);
    assert.equal(chip === null, state === "off");
  }
});

test("the status type carries what the running server does", () => {
  assert.match(API_TYPES, /exact_concurrency\?: string \| null;/);
  assert.match(API_TYPES, /requested_exact_concurrency\?: string \| null;/);
});

test("the store holds the state and starts off", () => {
  assert.match(STORE, /loadedExactConcurrency: ExactConcurrencyState;/);
  // Both the initial state and the unload reset.
  assert.equal(STORE.match(/loadedExactConcurrency: "off",/g)?.length, 2);
});

test("every status refresh republishes it, not just a seeded load", () => {
  // A tab opened onto an already-loaded model performs no load, so the chip would never
  // appear if this were gated on seedLoadParams.
  assert.match(
    APPLIER,
    /loadedExactConcurrency: normalizeExactConcurrency\(status\.exact_concurrency\),/,
  );
});

test("the chip reads the store and hangs the sentence off a title", () => {
  assert.match(
    CHIP,
    /useChatRuntimeStore\(\(s\) => s\.loadedExactConcurrency\)/,
  );
  assert.match(CHIP, /title=\{chip\.title\}/);
  assert.match(CHIP, /if \(!chip\) return null;/);
});

test("the chat header renders it beside the model selector, for the resident local model only", () => {
  // The state is the local llama-server's, which stays resident when a hosted model is picked
  // beside it, so unconditionally the chip put the local guarantee next to hosted output.
  assert.match(
    CHAT_PAGE,
    /triggerDataTour="chat-model-selector"[\s\S]{0,600}exactConcurrencyChipApplies\(\{[\s\S]{0,120}\}\) && <ExactConcurrencyChip \/>/,
  );
});

test("the chip applies to a resident local model and to nothing else", () => {
  const resident = { isExternalModel: false, residentCheckpoint: "qwen" };
  assert.equal(exactConcurrencyChipApplies(resident), true);
  assert.equal(
    exactConcurrencyChipApplies({ ...resident, isExternalModel: true }),
    false,
    "a hosted model selected beside the resident local one",
  );
  assert.equal(
    exactConcurrencyChipApplies({ ...resident, residentCheckpoint: null }),
    false,
    "nothing resident",
  );
  assert.equal(
    exactConcurrencyChipApplies({ ...resident, residentCheckpoint: undefined }),
    false,
    "residency unknown",
  );
  assert.equal(
    exactConcurrencyChipApplies({ ...resident, modelLoading: true }),
    false,
    "a load in flight describes the next server, not this one",
  );
});
