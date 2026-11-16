// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";

import * as zustand from "zustand";
import * as languages from "../src/features/translate/languages.ts";
import * as models from "../src/features/translate/models.ts";
import * as history from "../src/features/translate/translate-history.ts";
import * as text from "../src/features/translate/translate-text.ts";
import { installLocalStorageFake } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { store: storage } = installLocalStorageFake();
Object.defineProperty(globalThis, "navigator", {
  value: { languages: ["en-US"] },
  configurable: true,
});

class TranslateRunError extends Error {}
let reply = (input: string) => `FR:${input}`;

type Store = {
  getState: () => Record<string, any>;
  setState: (state: Record<string, unknown>) => void;
};
const { useTranslateStore, AUTO_SOURCE } = loadWithStubs<{
  useTranslateStore: Store;
  AUTO_SOURCE: string;
}>(new URL("../src/features/translate/translate-store.ts", import.meta.url), {
  zustand,
  "./languages": languages,
  "./models": models,
  "./translate-history": history,
  "./translate-text": text,
  "./translate-runner": {
    TranslateRunError,
    runTranslation: async (options: { text: string }) => ({
      output: reply(options.text),
      model: "Hy-MT2 1.8B",
    }),
  },
});

const state = () => useTranslateStore.getState();

beforeEach(() => {
  storage.clear();
  reply = (input) => `FR:${input}`;
  useTranslateStore.setState({
    source: "",
    output: "",
    result: null,
    attempt: null,
    error: null,
    phase: null,
    history: [],
    readyModels: [],
  });
  state().setPrefs({ from: "en", target: "fr", saveHistory: true });
});

test("swap moves a current translation into the text box", async () => {
  state().setSource("Hello");
  await state().translate();
  state().swap();
  assert.equal(state().source, "FR:Hello");
  assert.equal(state().prefs.from, "fr");
  assert.equal(state().prefs.target, "en");
});

test("swap keeps an edited draft instead of the old translation", async () => {
  state().setSource("Hello");
  await state().translate();
  state().setSource("Hello, how are you?");
  state().swap();
  assert.equal(state().source, "Hello, how are you?");
  assert.equal(state().prefs.from, "fr");
});

test("translating the translation onward sets its language as the source", async () => {
  state().setPrefs({ from: AUTO_SOURCE });
  assert.equal(AUTO_SOURCE, "auto");
  state().setSource("Hello");
  await state().translate();
  state().outputToSource();
  assert.equal(state().source, "FR:Hello");
  assert.equal(state().prefs.from, "fr");
  assert.equal(state().output, "");
});

test("translating as you type keeps one history entry per draft", async () => {
  for (const draft of ["The quick", "The quick brown", "The quick brown fox"]) {
    state().setSource(draft);
    await state().translate({ auto: true });
  }
  assert.deepEqual(
    state().history.map((entry: { source: string }) => entry.source),
    ["The quick brown fox"],
  );
});

test("a new draft typed after another keeps both in the history", async () => {
  state().setSource("Good morning everyone");
  await state().translate({ auto: true });
  state().setSource("See you tomorrow");
  await state().translate({ auto: true });
  assert.equal(state().history.length, 2);
});

test("a Translate click is never replaced by a later typed draft", async () => {
  state().setSource("The quick");
  await state().translate();
  state().setSource("The quick brown");
  await state().translate({ auto: true });
  assert.equal(state().history.length, 2);
});

test("a model marked ready can be forgotten without touching the others", () => {
  state().setModelReady("hy-mt2-1.8b", true);
  state().setModelReady("chat", true);
  state().setModelReady("hy-mt2-1.8b", false);
  assert.deepEqual(state().readyModels, ["chat"]);
});

test("a full browser storage never wipes the saved history", async () => {
  state().setSource("First");
  await state().translate();
  const saved = storage.get("unsloth_translate_history");
  const setItem = globalThis.window.localStorage.setItem;
  globalThis.window.localStorage.setItem = () => {
    throw new Error("QuotaExceededError");
  };
  try {
    state().setSource("Second");
    await state().translate();
  } finally {
    globalThis.window.localStorage.setItem = setItem;
  }
  assert.equal(storage.get("unsloth_translate_history"), saved);
});

test("long text is cut into passages a small model translates whole", () => {
  const paragraph =
    "A sentence of about sixty characters to fill the passage. ".repeat(5);
  const source = Array.from({ length: 8 }, () => paragraph.trim()).join("\n\n");
  for (const chunk of text.splitForTranslation(source)) {
    assert.ok(chunk.text.length <= 1000);
  }
});
