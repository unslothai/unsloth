// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { soleQuantRowState } from "../src/features/model-picker/components/model-selector/row-identity.ts";

const REPO = "unsloth/Qwen3-8B-GGUF";
const OTHER_REPO = "unsloth/Llama-3.1-8B-Instruct-GGUF";
const QUANT = "Q4_K_M";

/** The row shown for a repo holding only Q4_K_M on disk. */
const rowFor = (state: {
  pickerValue?: string | null;
  loadedModelId?: string | null;
  activeGgufVariant?: string | null;
}) =>
  soleQuantRowState({
    pickerValue: state.pickerValue ?? REPO,
    repoId: REPO,
    quant: QUANT,
    loadedModelId: state.loadedModelId ?? null,
    activeGgufVariant: state.activeGgufVariant ?? null,
  });

test("this quant running: selected and loaded", () => {
  assert.deepEqual(rowFor({ loadedModelId: REPO, activeGgufVariant: QUANT }), {
    selected: true,
    loaded: true,
  });
});

test("quant casing and padding still count as running", () => {
  assert.deepEqual(
    rowFor({ loadedModelId: REPO, activeGgufVariant: " q4_k_m " }),
    { selected: true, loaded: true },
  );
});

test("repo running a different quant: neither selected nor loaded", () => {
  // Q8 stayed resident while the active cache now holds only Q4_K_M.
  assert.deepEqual(rowFor({ loadedModelId: REPO, activeGgufVariant: "Q8_0" }), {
    selected: false,
    loaded: false,
  });
});

test("repo resident with no quant reported: not this row", () => {
  assert.deepEqual(rowFor({ loadedModelId: REPO }), {
    selected: false,
    loaded: false,
  });
});

test("another model resident: the picker's value still selects the row", () => {
  // Compare panes stage one model per pane while a single model is resident.
  assert.deepEqual(
    rowFor({ loadedModelId: OTHER_REPO, activeGgufVariant: "Q8_0" }),
    { selected: true, loaded: false },
  );
});

test("nothing resident: the picker's value still selects the row", () => {
  assert.deepEqual(rowFor({}), { selected: true, loaded: false });
});

test("the picker points elsewhere: not selected, even while running", () => {
  assert.deepEqual(
    rowFor({
      pickerValue: OTHER_REPO,
      loadedModelId: REPO,
      activeGgufVariant: QUANT,
    }),
    { selected: false, loaded: true },
  );
});

// Audio pages pass the runtime's loaded id and quant; the chat store must not stand in for either.
const FOLDER = "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-0.6B-Base-GGUF";
const audioRow = (
  loadedModelId: string | null,
  activeGgufVariant: string | null,
) =>
  soleQuantRowState({
    pickerValue: FOLDER,
    repoId: FOLDER,
    quant: "Q8_0",
    loadedModelId,
    activeGgufVariant,
  });

test("task override id and quant: the umbrella row is loaded", () => {
  assert.deepEqual(audioRow(FOLDER, "Q8_0"), { selected: true, loaded: true });
});

test("task override id without a quant: not loaded", () => {
  assert.deepEqual(audioRow(FOLDER, null), { selected: false, loaded: false });
});

test("empty task override id: nothing loaded, even with a quant", () => {
  assert.deepEqual(audioRow("", "Q8_0"), { selected: true, loaded: false });
});

test("the picker resolves the loaded quant from the task override first", async () => {
  const { readSrc } = await import("./helpers/kit.ts");
  const pickers = readSrc(
    "features/model-picker/components/model-selector/pickers.tsx",
  );
  assert.match(
    pickers,
    /const activeGgufVariant =\s*loadedGgufVariantOverride !== undefined\s*\? loadedGgufVariantOverride\s*: storeGgufVariant;/,
  );
  // `??` keeps an empty id override instead of falling back to the chat store.
  assert.match(
    pickers,
    /const loadedModelId = loadedModelIdOverride \?\? chatLoadedModelId;/,
  );
  assert.match(
    pickers,
    /useModelLoadTimes\(loadStampId\(value, valueLoaded\)\)/,
  );
  const selector = readSrc(
    "features/model-picker/components/model-selector.tsx",
  );
  assert.match(
    selector,
    /loadedGgufVariantOverride=\{loadedGgufVariantOverride\}\s*loraModels=\{fineTunedModels\}[\s\S]{0,200}?valueLoaded=\{loaded\}/,
  );
});
