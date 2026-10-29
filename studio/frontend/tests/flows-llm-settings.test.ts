// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { specs } from "./flows-editor-fixtures.ts";
import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const s = await import("../src/features/flows/editor/llm-settings.ts");
const { defaultParams } = await import(
  "../src/features/flows/stores/catalog-store.ts"
);

const GENERATE = specs["llm.generate"];
const VISION = specs["llm.vision"];
const SAVED = {
  temperature: 0.3,
  topP: 0.7,
  topK: 40,
  minP: 0.05,
  minPMode: "custom" as const,
  repetitionPenalty: 1.1,
  presencePenalty: 0.5,
  maxTokens: 2048,
  systemPrompt: "Be brief.",
};

test("on, the saved settings win over the card's own values", () => {
  const card = { temperature: 1.5, system: "card", use_model_settings: true };
  const shown = s.effectiveParams(card, SAVED, GENERATE);
  assert.equal(shown.temperature, 0.3);
  assert.equal(shown.system, "Be brief.");
  assert.equal(shown.top_k, 40);
  assert.equal(shown.max_tokens, 2048);
});

test("off, the card's own values stand whatever is saved", () => {
  const card = { temperature: 1.5, system: "card", use_model_settings: false };
  assert.equal(s.effectiveParams(card, SAVED, GENERATE), card);
});

test("on with nothing saved, or only some keys, the card fills the gaps", () => {
  const card = { temperature: 1.5, top_p: 0.9, use_model_settings: true };
  assert.equal(s.effectiveParams(card, undefined, GENERATE).temperature, 1.5);
  const partial = s.effectiveParams(card, { temperature: 0.2 }, GENERATE);
  assert.equal(partial.temperature, 0.2);
  assert.equal(partial.top_p, 0.9);
});

test("server Min P, a Max output cap and out-of-range values follow the engine's rules", () => {
  const out = s.savedAsParams(
    { minP: 0.3, minPMode: "server-default", maxTokens: 2 ** 31 - 1 },
    GENERATE,
  );
  assert.equal("min_p" in out, false);
  assert.equal("max_tokens" in out, false);
  assert.equal(s.savedAsParams({ maxTokens: 20000 }, VISION).max_tokens, 8192);
});

test("editing while on forks the card onto its own copy of what it showed", () => {
  const card = { temperature: 1.5, use_model_settings: true };
  const patch = s.editPatch(card, SAVED, GENERATE, { top_p: 0.5 });
  assert.equal(patch.use_model_settings, false);
  assert.equal(patch.top_p, 0.5);
  assert.equal(patch.temperature, 0.3);
  assert.equal(patch.system, "Be brief.");
  assert.deepEqual(
    s.editPatch({ use_model_settings: false }, SAVED, GENERATE, { top_p: 0.5 }),
    { top_p: 0.5 },
  );
});

test("turning the toggle off keeps the values on screen; on drops nothing", () => {
  const off = s.togglePatch(SAVED, GENERATE, false);
  assert.equal(off.use_model_settings, false);
  assert.equal(off.temperature, 0.3);
  assert.deepEqual(s.togglePatch(SAVED, GENERATE, true), {
    use_model_settings: true,
  });
});

test("Load copies the saved values; Save writes only what the card sets", () => {
  assert.equal(s.loadPatch(SAVED, GENERATE).presence_penalty, 0.5);
  assert.deepEqual(
    s.paramsAsSaved({ temperature: 0.4, top_p: null, min_p: 0.1, system: "" }),
    { temperature: 0.4, minP: 0.1, minPMode: "custom" },
  );
});

test("the memory key is the card's model, else the loaded one, matched case-insensitively", () => {
  const picked = { model: { repo: "unsloth/Qwen3-4B-GGUF", variant: null } };
  assert.equal(s.modelKey(picked, "other"), "unsloth/Qwen3-4B-GGUF");
  assert.equal(s.modelKey({ model: null }, "loaded/one"), "loaded/one");
  assert.equal(s.modelKey({ model: null }, ""), null);
  assert.deepEqual(
    s.findSaved({ "Unsloth/Qwen3-4B-GGUF": { temperature: 0.1 } }, "unsloth/qwen3-4b-gguf"),
    { temperature: 0.1 },
  );
});

test("new cards start on; a flow saved before the setting stays off and keeps its values", () => {
  assert.equal(
    s.newCardParams("llm.generate", defaultParams(GENERATE)).use_model_settings,
    true,
  );
  assert.equal(
    s.newCardParams("text.template", { template: "" }).use_model_settings,
    undefined,
  );
  // An old llm.generate node: the engine's param names, no toggle, no sampling keys.
  const old = { system: "You extract.", temperature: 0.2, max_tokens: 300 };
  assert.equal(s.isOn(old), false);
  assert.equal(s.effectiveParams(old, SAVED, GENERATE), old);
  // The catalog keeps the engine's names, so the old keys still land on real params.
  for (const id of ["system", "temperature", "max_tokens", "json_schema", "seed"])
    assert.ok(GENERATE.params.some((p) => p.id === id), id);
  const toggle = GENERATE.params.find((p) => p.id === "use_model_settings");
  assert.equal(toggle?.default, false);
});

test("the inspector hides panel params from the generic rows, stably", () => {
  const trimmed = s.withoutPanelParams(GENERATE);
  assert.equal(trimmed, s.withoutPanelParams(GENERATE));
  assert.deepEqual(
    trimmed.params.map((p) => p.id),
    ["model", "prompt"],
  );
  assert.deepEqual(
    s.withoutPanelParams(VISION).params.map((p) => p.id),
    ["model", "question"],
  );
  const template = specs["text.template"];
  assert.equal(s.withoutPanelParams(template), template);
});

test("the panel reuses Chat's controls and none of its chat-only sections", () => {
  const src = readSrc("features/flows/editor/llm-settings-panel.tsx");
  assert.match(src, /ParamSlider/);
  assert.match(src, /setRememberedParamsForModel/);
  assert.match(src, /paramsByModel/);
  for (const chatOnly of ["Tools", "Retrieval", "Preset", "Prompt variables"])
    assert.doesNotMatch(src, new RegExp(`label="${chatOnly}"`));
});
