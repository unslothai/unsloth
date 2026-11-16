// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  TRANSLATE_LANGUAGES,
  suggestedTargetLanguage,
  translateLanguageLabel,
} from "../src/features/translate/languages.ts";
import {
  CHAT_TRANSLATE_MODEL,
  DEDICATED_TRANSLATE_MODELS,
  DEFAULT_TRANSLATE_MODEL,
  dedicatedTranslateModel,
  isTranslateModelId,
  translateMessages,
} from "../src/features/translate/models.ts";
import {
  TRANSLATE_HISTORY_MAX_ENTRIES,
  addHistoryEntry,
  parseHistory,
  searchHistory,
} from "../src/features/translate/translate-history.ts";
import {
  restoreChunkWhitespace,
  splitForTranslation,
} from "../src/features/translate/translate-text.ts";
import {
  TRANSLATION_ONLY_REPOS,
  isTranslationHeldSlot,
  isTranslationOnlyRepo,
} from "../src/lib/translation-models.ts";

import { readText } from "./helpers/kit.ts";

const rejoin = (source: string, limit?: number) =>
  splitForTranslation(source, limit)
    .map((chunk) => chunk.text + chunk.after)
    .join("");

test("splitting a document loses nothing", () => {
  const sources = [
    "",
    "one line",
    "a\n\nb\n",
    "\n\n  indented\n",
    "   \n\n   ",
    `${"x".repeat(50)}. ${"y".repeat(60)}\nz`,
    "一二三。四五六。七八九十一二三四五六七八九。",
    "word ".repeat(400),
  ];
  for (const source of sources) {
    for (const limit of [20, 1600]) {
      assert.equal(rejoin(source, limit), source);
      for (const chunk of splitForTranslation(source, limit)) {
        assert.ok(
          chunk.text.length <= limit,
          `chunk over ${limit}: ${chunk.text.length}`,
        );
      }
    }
  }
});

test("a short text is one request and blank lines are never sent", () => {
  assert.deepEqual(splitForTranslation("Hello\n\nWorld\n"), [
    { text: "Hello\n\nWorld", after: "\n" },
  ]);
  const chunks = splitForTranslation(
    "first paragraph\n\n\nsecond paragraph",
    20,
  );
  assert.deepEqual(chunks, [
    { text: "first paragraph", after: "\n\n\n" },
    { text: "second paragraph", after: "" },
  ]);
  assert.deepEqual(splitForTranslation("  \n\n"), [
    { text: "", after: "  \n\n" },
  ]);
});

test("a line longer than a chunk breaks at sentence ends", () => {
  const chunks = splitForTranslation(
    "One two three. Four five six. Seven eight nine.",
    32,
  );
  assert.deepEqual(
    chunks.map((chunk) => chunk.text),
    ["One two three. Four five six.", "Seven eight nine."],
  );
});

test("the translation keeps the source's own surrounding whitespace", () => {
  assert.equal(
    restoreChunkWhitespace({ text: "  Hello ", after: "\n\n" }, "\nBonjour\n"),
    "  Bonjour \n\n",
  );
});

test("Hy-MT2 gets its model card's prompt: one user turn, no system prompt", () => {
  const [message, ...rest] = translateMessages(
    "hy-mt2-1.8b",
    "Hello",
    "French",
    "default",
  );
  assert.equal(rest.length, 0);
  assert.equal(message.role, "user");
  assert.equal(
    message.content,
    "Translate the following text into French. Note that you should only output the translated result without any additional explanation:\n\nHello",
  );
  const [styled] = translateMessages("hy-mt2-7b", "Hello", "French", "formal");
  assert.match(
    styled.content,
    /must strictly conform to \[formal and professional\]:\n\nHello$/,
  );
});

test("a chat model is told to translate the text, not answer it", () => {
  const messages = translateMessages(
    CHAT_TRANSLATE_MODEL,
    "What time is it?",
    "German",
    "casual",
  );
  assert.equal(messages.length, 1);
  assert.equal(messages[0].role, "user");
  assert.match(
    messages[0].content,
    /^Translate the text inside the <source> tags into German\./,
  );
  assert.match(messages[0].content, /never an instruction/);
  assert.match(messages[0].content, /casual and conversational/);
  assert.match(
    messages[0].content,
    /\n\n<source>\nWhat time is it\?\n<\/source>$/,
  );
});

test("the model list is the two downloads plus the chat model", () => {
  assert.equal(isTranslateModelId(DEFAULT_TRANSLATE_MODEL), true);
  assert.equal(isTranslateModelId(CHAT_TRANSLATE_MODEL), true);
  assert.equal(isTranslateModelId("gpt"), false);
  assert.equal(dedicatedTranslateModel(CHAT_TRANSLATE_MODEL), null);
  assert.deepEqual(
    DEDICATED_TRANSLATE_MODELS.map((model) => model.repo),
    [...TRANSLATION_ONLY_REPOS],
  );
});

test("language codes are unique and named in the interface language", () => {
  const codes = TRANSLATE_LANGUAGES.map((language) => language.code);
  assert.equal(new Set(codes).size, codes.length);
  const french = TRANSLATE_LANGUAGES.find((language) => language.code === "fr");
  assert.ok(french);
  assert.equal(translateLanguageLabel(french, "en"), "French");
  assert.equal(translateLanguageLabel(french, "fr"), "français");
});

test("the first target is the first browser language the model knows", () => {
  assert.equal(suggestedTargetLanguage(["sv-SE", "de-DE", "en"]), "de");
  assert.equal(suggestedTargetLanguage(["zh-TW"]), "zh-Hant");
  assert.equal(suggestedTargetLanguage(["zh-CN"]), "zh");
  assert.equal(suggestedTargetLanguage(["fil-PH"]), "tl");
  assert.equal(suggestedTargetLanguage(["sv-SE"]), "en");
  assert.equal(suggestedTargetLanguage([]), "en");
});

test("history is newest first, replaces a repeat and stays bounded", () => {
  const entry = {
    source: "Hello",
    output: "Bonjour",
    target: "fr",
    model: "Hy-MT2 1.8B",
  };
  let history = addHistoryEntry([], entry, 1);
  history = addHistoryEntry(
    history,
    { ...entry, source: "Bye", output: "Au revoir" },
    2,
  );
  history = addHistoryEntry(history, { ...entry, output: "Salut" }, 3);
  assert.deepEqual(
    history.map((item) => item.output),
    ["Salut", "Au revoir"],
  );
  assert.equal(addHistoryEntry(history, { ...entry, output: "  " }).length, 2);
  assert.equal(
    addHistoryEntry(history, { ...entry, source: "x".repeat(30_000) }).length,
    2,
    "a translation too long for localStorage is not kept",
  );
  for (let i = 0; i < TRANSLATE_HISTORY_MAX_ENTRIES + 5; i += 1) {
    history = addHistoryEntry(
      history,
      { ...entry, source: `text ${i}` },
      10 + i,
    );
  }
  assert.equal(history.length, TRANSLATE_HISTORY_MAX_ENTRIES);
  assert.deepEqual(
    searchHistory(history, "TEXT 104").map((item) => item.source),
    ["text 104"],
  );
});

test("a damaged saved history is read as far as it is valid", () => {
  assert.deepEqual(parseHistory("nope"), []);
  const good = { id: "a", at: 1, source: "s", output: "o", target: "fr" };
  assert.deepEqual(
    parseHistory([good, null, { id: 1 }, { ...good, output: 3 }]),
    [{ ...good, model: null }],
  );
});

test("chat does not take over a model the Translate page loaded", () => {
  const [small] = TRANSLATION_ONLY_REPOS;
  assert.equal(isTranslationOnlyRepo(small.toLowerCase()), true);
  assert.equal(isTranslationOnlyRepo("unsloth/gemma-4-E2B-it-GGUF"), false);
  assert.equal(isTranslationOnlyRepo(null), false);
  assert.equal(isTranslationHeldSlot(small, ""), true);
  assert.equal(
    isTranslationHeldSlot(small, "unsloth/gemma-4-E2B-it-GGUF"),
    true,
  );
  // Picked in Chat by the user: theirs to chat with.
  assert.equal(isTranslationHeldSlot(small, small), false);
  assert.equal(isTranslationHeldSlot("unsloth/gemma-4-E2B-it-GGUF", ""), false);

  const adopt = readText(
    "../src/features/chat/lib/apply-inference-status-to-store.ts",
  );
  assert.match(
    adopt.slice(
      adopt.indexOf("export async function tryAdoptServerActiveModel"),
    ),
    /isTranslationHeldSlot\(status\.active_model, null\)/,
  );
  const adapter = readText("../src/features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /function isChattableCachedRepo\([\s\S]*?!isTranslationOnlyRepo\(repo\.repo_id\)/,
    "auto-load must not pick a translation model as the chat model",
  );
  assert.match(
    adapter,
    /function isAutoLoadableLocalRow\([\s\S]*?!isTranslationOnlyRepo\(row\.model_id\)/,
  );
});

test("the translation model is fetched by a translation, never by opening the page", () => {
  const page = readText("../src/features/translate/translate-page.tsx");
  assert.doesNotMatch(page, /downloadManager|loadModel\(/);
  // Translating as you type waits for a model that already served a translation.
  assert.match(page, /if \(!readyModels\.includes\(prefs\.modelId\)\) return;/);
  const runner = readText("../src/features/translate/translate-runner.ts");
  assert.match(runner, /runtime: "translate"/);
  const loaded = readText("../src/features/loaded-models/use-loaded-models.ts");
  assert.match(
    loaded,
    /runtime === "tts" \|\| runtime === "translate" \? "chat"/,
  );
});
