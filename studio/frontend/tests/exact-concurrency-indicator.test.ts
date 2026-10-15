// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// "off" renders nothing; "unavailable" names the server's refusal. Wiring is checked at the source.

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
const RUNTIME = read("src/features/chat/hooks/use-chat-model-runtime.ts");
const ADAPTER = read("src/features/chat/api/chat-adapter.ts");
const PROVIDER = read("src/features/chat/runtime-provider.tsx");

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
  assert.equal(STORE.match(/loadedExactConcurrency: "off",/g)?.length, 2);
});

test("a rollback resends the exact setting the previous load asked for", () => {
  // An omitted field would resolve to a since-changed stored `on` and fail a running model.
  const request = API_TYPES.slice(
    API_TYPES.indexOf("export interface LoadModelRequest"),
    API_TYPES.indexOf("export interface LoadModelResponse"),
  );
  assert.match(request, /exact_concurrency\?: string \| null;/);
  assert.match(STORE, /loadedRequestedExactConcurrency: string \| null;/);
  assert.equal(STORE.match(/loadedRequestedExactConcurrency: null,/g)?.length, 2);
  assert.match(
    APPLIER,
    /loadedRequestedExactConcurrency: status\.requested_exact_concurrency \?\? null,/,
  );
  assert.match(
    RUNTIME,
    /loadedRequestedExactConcurrency:\s*loadResponse\.requested_exact_concurrency \?\? null,/,
  );
  assert.match(
    RUNTIME,
    /loadedExactConcurrency: normalizeExactConcurrency\(\s*loadResponse\.exact_concurrency,?\s*\)/,
  );
  assert.match(
    RUNTIME,
    /exact_concurrency: rollbackState\.loadedRequestedExactConcurrency/,
  );
  // The rollback's own answer is published too: its `auto` can resolve differently.
  assert.match(
    RUNTIME,
    /loadedRequestedExactConcurrency:\s*rollbackResponse\.requested_exact_concurrency \?\? null,/,
  );
  assert.match(
    RUNTIME,
    /loadedExactConcurrency: normalizeExactConcurrency\(\s*rollbackResponse\.exact_concurrency,?\s*\)/,
  );
});

test("every status refresh republishes it, not just a seeded load", () => {
  // A tab opened onto an already-loaded model performs no load.
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
  // The local server stays resident beside a hosted model, so the chip must not show for hosted
  // output.
  assert.match(
    CHAT_PAGE,
    /triggerDataTour="chat-model-selector"[\s\S]{0,1500}exactConcurrencyChipApplies\(\{[\s\S]{0,200}\}\) && <ExactConcurrencyChip \/>/,
  );
});

test("the chip applies to a resident local model and to nothing else", () => {
  const resident = {
    isExternalModel: false,
    residentCheckpoint: "qwen",
    selectedCheckpoint: "qwen",
  };
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
    exactConcurrencyChipApplies({ ...resident, selectedCheckpoint: "gemma" }),
    false,
    "another local checkpoint picked while qwen is still resident",
  );
  assert.equal(
    exactConcurrencyChipApplies({ ...resident, modelLoading: true }),
    false,
    "a load in flight describes the next server, not this one",
  );
});

test("a recompute in this chat is named in the chip, whatever the load reports", () => {
  const on = exactConcurrencyChip("on", { recomputed: true });
  assert.ok(on);
  assert.match(on.title, /re-prefilled after a park the server could not hold/);
  assert.match(on.title, /not guaranteed byte-identical/);
  assert.notEqual(on.label, "Exact");
  const unavailable = exactConcurrencyChip("unavailable", { recomputed: true });
  assert.ok(unavailable);
  assert.match(unavailable.title, /re-prefilled after a park the server could not hold/);
});

test("without a recompute the chip reads exactly as it did", () => {
  assert.deepEqual(exactConcurrencyChip("on"), exactConcurrencyChip("on", {}));
  assert.equal(exactConcurrencyChip("on")?.label, "Exact");
  assert.doesNotMatch(exactConcurrencyChip("on")!.title, /re-prefilled/);
  assert.equal(exactConcurrencyChip("off", { recomputed: true }), null);
});

test("the recompute note is persisted with the answer and read from the answer on screen", () => {
  // The note rides message metadata, so reload, retry and branch switches agree.
  assert.equal(
    ADAPTER.match(/preemptRecomputed: sawPreemptRecompute \|\| undefined,/g)?.length,
    3,
    "the live, the final and the interrupted settlement metadata all carry it",
  );
  assert.doesNotMatch(ADAPTER, /notePreemptRecompute|clearPreemptRecompute/);
  assert.match(PROVIDER, /preemptRecomputed: true/);
  assert.match(PROVIDER, /function VisibleAnswerRecomputeSync\(/);
  assert.match(
    PROVIDER,
    /useAuiState\(\(\{ thread \}\) => \{[\s\S]{0,600}\?\.preemptRecomputed === true/,
  );
  assert.match(
    PROVIDER,
    /answerRecomputed &&\s+\(recomputedOn\.length > 0\s+\? recomputedOn\.includes\(checkpoint\)\s+: answerModelId === null \|\| answerModelId === checkpoint\)/,
  );
  assert.match(PROVIDER, /custom\?\.responseDetails as \{ modelId\?: unknown \}/);
  assert.match(PROVIDER, /typeof details\?\.modelId === "string" && details\.modelId\s+\? details\.modelId/);
  assert.equal(
    ADAPTER.match(
      /preemptRecomputedModelIds: recomputedOnField\(sawPreemptRecompute, preemptRecomputedOn\),/g,
    )?.length,
    3,
    "the live, the final and the interrupted settlement all name the recompute's model",
  );
  assert.match(PROVIDER, /const recomputedOn = recomputedModelIds\(custom\);/);
  assert.match(
    PROVIDER,
    /const recomputedOn = update\.run\.requestPayload\?\.model;[\s\S]{0,400}preemptRecomputedModelIds: withRecomputedModel\(\s+recomputedModelIds\(currentMetadata\),\s+recomputedOn,/,
  );
  // A fresh object per store read re-renders without end (React error 185).
  assert.doesNotMatch(PROVIDER, /useAuiState\(\(\{ thread \}\) => \{[\s\S]{0,400}return \{/);
  assert.match(PROVIDER, /setPreemptRecompute\(activeThreadId, recomputed\)/);
  assert.match(PROVIDER, /<VisibleAnswerRecomputeSync\s+enabled=\{modelType === "base" && !pairId && !backgrounded\}/);
  assert.doesNotMatch(PROVIDER, /notePreemptRecompute|clearPreemptRecompute/);
});

test("the chip reads the flag for the conversation on screen", () => {
  assert.match(STORE, /preemptRecomputedByThreadId: Record<string, boolean>/);
  assert.match(CHIP, /preemptRecomputedByThreadId\[s\.activeThreadId/);
});

test("the settings selector says what exact mode needs before offering it", () => {
  const section = read(
    "src/features/settings/components/llama-backend-section.tsx",
  );
  const api = read("src/features/settings/api/exact-concurrency.ts");
  assert.match(api, /parkingAvailable: value\.parking_available \?\? true/);
  assert.match(api, /parkingPrerequisite: value\.parking_prerequisite \?\? null/);
  assert.match(section, /disabled=\{!parkingAvailable && setting !== "off"\}/);
  assert.match(section, /exactConcurrency\.parkingRequired/);
  assert.match(section, /data-testid="exact-concurrency-parking-required"/);
});
