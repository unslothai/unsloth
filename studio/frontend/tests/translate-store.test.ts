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
type RunOptions = {
  text: string;
  signal: AbortSignal;
  onOutput: (text: string) => void;
};
const answer = async (options: RunOptions) => ({
  output: reply(options.text),
  model: "Hy-MT2 1.8B",
});
let runner: (
  options: RunOptions,
) => Promise<{ output: string; model: string }> = answer;
const streaming = (options: RunOptions) => {
  options.onOutput("partial text");
  return new Promise<never>((_, reject) => {
    options.signal.addEventListener("abort", () =>
      reject(new DOMException("Aborted", "AbortError")),
    );
  });
};
const settle = () => new Promise((resolve) => setTimeout(resolve, 60));

type Store = {
  getState: () => Record<string, any>;
  setState: (state: Record<string, unknown>) => void;
};
const { useTranslateStore, AUTO_SOURCE, translationIsStale } = loadWithStubs<{
  useTranslateStore: Store;
  AUTO_SOURCE: string;
  translationIsStale: (state: Record<string, any>) => boolean;
}>(new URL("../src/features/translate/translate-store.ts", import.meta.url), {
  zustand,
  "./languages": languages,
  "./models": models,
  "./translate-history": history,
  "./translate-text": text,
  "./translate-runner": {
    TranslateRunError,
    runTranslation: (options: RunOptions) => runner(options),
  },
});

const state = () => useTranslateStore.getState();

beforeEach(() => {
  storage.clear();
  reply = (input) => `FR:${input}`;
  runner = answer;
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

test("language tabs keep their order when one is picked and take a new language at the front", () => {
  state().setPrefs({ target: "fr", recentTargets: ["fr", "de", "es"] });
  state().setTarget("es");
  assert.deepEqual(state().prefs.recentTargets, ["fr", "de", "es"]);
  state().setTarget("ja");
  assert.deepEqual(state().prefs.recentTargets, ["ja", "fr", "de"]);
  state().setPrefs({ from: AUTO_SOURCE, recentSources: [] });
  state().setFrom("en");
  state().setFrom(AUTO_SOURCE);
  assert.deepEqual(state().prefs.recentSources, ["en"]);
  assert.equal(state().prefs.from, AUTO_SOURCE);
});

test("picking Detect language keeps a source tab that was never saved as recent", () => {
  state().setPrefs({ from: "ja", recentSources: [] });
  state().setFrom(AUTO_SOURCE);
  assert.deepEqual(state().prefs.recentSources, ["ja"]);
});

test("swap keeps an edited draft instead of the old translation", async () => {
  state().setSource("Hello");
  await state().translate();
  state().setSource("Hello, how are you?");
  state().swap();
  assert.equal(state().source, "Hello, how are you?");
  assert.equal(state().prefs.from, "fr");
});

test("with an automatic source, swap uses the detected language", async () => {
  state().setPrefs({ from: AUTO_SOURCE });
  state().setSource("Hello");
  await state().translate();
  state().swap(null);
  assert.equal(state().prefs.from, AUTO_SOURCE);
  assert.equal(state().source, "Hello");
  // Already in the target language: nothing to swap to.
  state().swap("fr");
  assert.equal(state().prefs.from, AUTO_SOURCE);
  state().swap("en");
  assert.equal(state().source, "FR:Hello");
  assert.equal(state().prefs.from, "fr");
  assert.equal(state().prefs.target, "en");
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
    state().setSource("Where is the station?");
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

test("Stop keeps the text that had streamed in", async () => {
  runner = streaming;
  state().setSource("Hello");
  const run = state().translate();
  await Promise.resolve();
  state().stop();
  await run;
  await settle();
  assert.equal(state().output, "partial text");
  assert.equal(state().result, null);
});

test("Clear during a run leaves nothing behind", async () => {
  runner = streaming;
  state().setSource("Hello");
  const run = state().translate();
  await Promise.resolve();
  state().clear();
  await run;
  await settle();
  assert.equal(state().source, "");
  assert.equal(state().output, "");
});

test("restoring a history entry during a run shows that entry, not the stopped text", async () => {
  runner = streaming;
  state().setSource("Hello");
  const run = state().translate();
  await Promise.resolve();
  state().restore({
    id: "1",
    at: 1,
    source: "Old",
    output: "Vieux",
    target: "fr",
    from: "en",
    model: "m",
  });
  await run;
  await settle();
  assert.equal(state().output, "Vieux");
});

test("a typed draft too long for the history keeps the entry it would have replaced", async () => {
  const first = `${"word ".repeat(1500)}end`;
  state().setSource(first);
  await state().translate({ auto: true });
  assert.equal(state().history.length, 1);
  state().setSource(`${first} ${"more ".repeat(1500)}`);
  await state().translate({ auto: true });
  assert.deepEqual(
    state().history.map((entry: { source: string }) => entry.source),
    [first],
  );
});

test("Hy-MT2 never keeps a chat-only language; a chat model offers it again", async () => {
  const gemma = {
    id: "unsloth/gemma-4-E2B-it-GGUF",
    name: "Gemma 4",
    variant: "Q4_K_M",
    sizeBytes: null,
  };
  try {
    state().setPrefs({ modelId: "custom", customModel: gemma });
    state().setFrom("hmn");
    state().setTarget("de");
    assert.equal(state().prefs.from, "hmn");
    state().setFrom("en");
    state().setTarget("hmn");
    assert.equal(state().prefs.target, "hmn");

    state().setPrefs({ modelId: "hy-mt2-1.8b" });
    assert.equal(state().prefs.target, "de");
    state().setTarget("hmn");
    assert.equal(state().prefs.target, "de");
    state().setFrom("hmn");
    assert.equal(state().prefs.from, AUTO_SOURCE);
    // The tab stays for the next chat model.
    assert.ok(state().prefs.recentTargets.includes("hmn"));

    state().setPrefs({ modelId: "custom", customModel: gemma });
    state().setTarget("hmn");
    assert.equal(state().prefs.target, "hmn");
    state().setSource("Where is the station?");
    await state().translate();
    const entry = state().history[0];
    assert.equal(entry.target, "hmn");
    state().setPrefs({ modelId: "hy-mt2-1.8b" });
    state().restore(entry);
    assert.notEqual(state().prefs.target, "hmn");
    // The saved Hmong output is still labelled Hmong, so it reads aloud and moves on as Hmong.
    assert.equal(state().result.target, "hmn");
    // Translating as you type compares `attempt` with the settings: no rerun on opening it.
    assert.equal(state().attempt.target, state().prefs.target);
    assert.equal(
      translationIsStale({
        result: state().result,
        source: state().source,
        prefs: state().prefs,
      }),
      true,
    );

    state().setPrefs({ modelId: "custom", customModel: gemma });
    state().setTarget("hmn");
    // A Hy-MT2 GGUF picked as a custom model is still Hy-MT2.
    state().setPrefs({
      customModel: { ...gemma, id: "unsloth/Hy-MT2-7B-GGUF" },
    });
    assert.notEqual(state().prefs.target, "hmn");
  } finally {
    state().setPrefs({ modelId: "hy-mt2-1.8b", customModel: null });
  }
});
