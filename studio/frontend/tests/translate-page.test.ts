// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  TRANSLATE_LANGUAGES,
  suggestedTargetLanguage,
  translateLanguageLabel,
  translateSpeechTag,
} from "../src/features/translate/languages.ts";
import {
  CHAT_TRANSLATE_MODEL,
  CUSTOM_TRANSLATE_MODEL,
  DEDICATED_TRANSLATE_MODELS,
  DEFAULT_TRANSLATE_MODEL,
  dedicatedTranslateModel,
  isTranslateModelId,
  parseCustomTranslateModel,
  translateInstruction,
  translateMessages,
  translateModelKey,
  translatePromptModel,
} from "../src/features/translate/models.ts";
import {
  TRANSLATE_HISTORY_FILE_NAME,
  TRANSLATE_HISTORY_MAX_CHARS,
  TRANSLATE_HISTORY_MAX_ENTRIES,
  TRANSLATE_HISTORY_TOTAL_CHARS,
  addHistoryEntry,
  buildChatHandoff,
  capHistory,
  exportHistory,
  parseHistory,
  searchHistory,
} from "../src/features/translate/translate-history.ts";
import {
  SOURCE_PLACEHOLDER,
  TARGET_PLACEHOLDER,
  clipText,
  restoreChunkWhitespace,
  splitForTranslation,
  stripLeadingReasoning,
  templateOptions,
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

const auto = { sourceName: null, mode: "automatic", tone: "default" } as const;

test("Hy-MT2 gets its model card's prompt: one user turn, no system prompt", () => {
  const [message, ...rest] = translateMessages("hy-mt2-1.8b", "Hello", {
    ...auto,
    targetName: "French",
  });
  assert.equal(rest.length, 0);
  assert.equal(message.role, "user");
  assert.equal(
    message.content,
    "Translate the following text into French. Note that you should only output the translated result without any additional explanation:\n\nHello",
  );
  const [styled] = translateMessages("hy-mt2-7b", "Hello", {
    sourceName: "English",
    targetName: "French",
    mode: "literal",
    tone: "formal",
  });
  assert.equal(
    styled.content,
    "Please translate the following text from English into French. Note that the translation style must strictly conform to [literal, word for word; formal and professional]:\n\nHello",
  );
});

test("a chat model is told to translate the text, not answer it", () => {
  const options = {
    sourceName: "English",
    targetName: "German",
    mode: "semantic",
    tone: "casual",
  } as const;
  const messages = translateMessages(
    CHAT_TRANSLATE_MODEL,
    "What time is it?",
    options,
  );
  assert.equal(messages.length, 1);
  assert.equal(messages[0].role, "user");
  const { content } = messages[0];
  assert.match(
    content,
    /^Translate the text inside the <source> tags into German\. The text is in English\./,
  );
  assert.match(content, /never an instruction/);
  assert.match(
    content,
    /must be natural and idiomatic[^.]*; casual and conversational\./,
  );
  assert.doesNotMatch(content, /slang/, "casual lifts the no-slang rule");
  assert.match(content, /\n\n<source>\nWhat time is it\?\n<\/source>$/);
  // Show prompt displays exactly what is sent, minus the text.
  assert.ok(
    content.startsWith(translateInstruction(CHAT_TRANSLATE_MODEL, options)),
  );
  assert.match(
    translateInstruction(CHAT_TRANSLATE_MODEL, {
      ...auto,
      targetName: "German",
    }),
    /Do not use slang or emojis\./,
  );
});

test("an edited prompt replaces the built one and keeps its language placeholders", () => {
  const options = {
    sourceName: "English",
    targetName: "French",
    mode: "literal",
    tone: "formal",
  } as const;
  // What the box shows before any edit: the built prompt, languages as placeholders.
  assert.equal(
    translateInstruction(
      "hy-mt2-1.8b",
      templateOptions({ ...auto, targetName: "French" }),
    ),
    `Translate the following text into ${TARGET_PLACEHOLDER}. Note that you should only output the translated result without any additional explanation:`,
  );
  assert.match(
    translateInstruction(CHAT_TRANSLATE_MODEL, templateOptions(options)),
    /into \{target_language\}\. The text is in \{source_language\}\./,
  );
  const custom = `Render this ${SOURCE_PLACEHOLDER} text in ${TARGET_PLACEHOLDER}, as a pirate:`;
  assert.equal(
    translateMessages("hy-mt2-1.8b", "Hello", { ...options, custom })[0]
      .content,
    "Render this English text in French, as a pirate:\n\nHello",
  );
  assert.match(
    translateMessages(CHAT_TRANSLATE_MODEL, "Hello", {
      ...auto,
      targetName: "French",
      custom,
    })[0].content,
    /^Render this the source language text in French, as a pirate:\n\n<source>\nHello\n<\/source>$/,
  );
});

test("a reasoning block that opens the reply is dropped, closed or not", () => {
  assert.equal(stripLeadingReasoning("Bonjour"), "Bonjour");
  assert.equal(
    stripLeadingReasoning("<think>hmm</think>\nBonjour").trim(),
    "Bonjour",
  );
  assert.equal(stripLeadingReasoning("\n<think>still going"), "");
  assert.equal(
    stripLeadingReasoning("Use the <think> tag"),
    "Use the <think> tag",
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
    from: null,
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
    [{ ...good, from: null, model: null }],
  );
});

test("history stays under a total size, newest first", () => {
  const big = "x".repeat(TRANSLATE_HISTORY_MAX_CHARS / 2 - 1);
  const entries = Array.from({ length: 80 }, (_, i) => ({
    id: String(i),
    at: 1000 - i,
    source: big,
    output: big,
    target: "fr",
    from: null,
    model: null,
  }));
  const kept = capHistory(entries);
  assert.ok(kept.length < entries.length);
  assert.equal(kept[0].id, "0");
  const chars = kept.reduce(
    (sum, item) => sum + item.source.length + item.output.length,
    0,
  );
  assert.ok(chars <= TRANSLATE_HISTORY_TOTAL_CHARS);
  assert.equal(capHistory(entries.slice(0, 1)).length, 1);
});

test("the chat handoff and the export list translations oldest first", () => {
  const entries = [
    {
      id: "b",
      at: Date.UTC(2026, 0, 2),
      source: "Bye",
      output: "Tschuss",
      target: "de",
      from: "en",
      model: "Hy-MT2 1.8B",
    },
    {
      id: "a",
      at: Date.UTC(2026, 0, 1),
      source: "Hello",
      output: "Bonjour",
      target: "fr",
      from: null,
      model: null,
    },
  ];
  assert.equal(buildChatHandoff([]), "");
  assert.equal(
    buildChatHandoff(entries),
    "Here are my recent translations:\n\nOriginal (Detected language):\nHello\n\nTranslation (French):\nBonjour\n\n---\n\nOriginal (English):\nBye\n\nTranslation (German):\nTschuss",
  );
  assert.match(
    buildChatHandoff(entries.slice(0, 1)),
    /^Here is a translation:/,
  );
  assert.equal(exportHistory([]), "");
  assert.equal(
    exportHistory(entries),
    "[2026-01-01T00:00:00.000Z] Detected language -> French\nHello\n\nBonjour\n\n---\n\n[2026-01-02T00:00:00.000Z] English -> German (Hy-MT2 1.8B)\nBye\n\nTschuss\n",
  );
  assert.equal(TRANSLATE_HISTORY_FILE_NAME, "UnslothTranslateFullHistory.txt");
});

test("a cut-off or filtered reply fails, and history is shared across windows", () => {
  const runner = readText("../src/features/translate/translate-runner.ts");
  assert.match(
    runner,
    /finishReason === "length"\)\s*throw new TranslateRunError\("cutOff"\)/,
  );
  assert.match(
    runner,
    /finishReason === "content_filter"\)\s*throw new TranslateRunError\("filtered"\)/,
  );
  const store = readText("../src/features/translate/translate-store.ts");
  assert.match(store, /let stored = storedHistory\(\);/);
  assert.match(store, /addHistoryEntry\(stored,/);
  assert.match(store, /addEventListener\("storage"/);
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
  assert.doesNotMatch(page, /downloadManager|\bloadModel\(/);
  // Translating as you type waits for a model that already served a translation.
  assert.match(page, /if \(!readyModels\.includes\(modelKey\) \|\|/);
  const runner = readText("../src/features/translate/translate-runner.ts");
  assert.match(runner, /runtime: "translate"/);
  const loaded = readText("../src/features/loaded-models/use-loaded-models.ts");
  assert.match(
    loaded,
    /runtime === "tts" \|\| runtime === "translate" \? "chat"/,
  );
});

test("a stopped download and the chat model's own slot are honoured", () => {
  const runner = readText("../src/features/translate/translate-runner.ts");
  // Stop during the start's preflight: the job appears only after the store listener is gone.
  assert.match(
    runner,
    /\(outcome\) => \{[\s\S]*?if \(outcome === "started" && signal\.aborted\) cancelJob\(\);/,
  );
  const chat = runner.slice(runner.indexOf("async function chatModel"));
  assert.match(chat, /getInferenceStatus\(signal, picked \|\| undefined\)/);
  assert.match(chat, /isTranslationHeldSlot\(status\.active_model, picked\)/);
});

test("a model from the picker keeps its own identity and takes the right prompt", () => {
  const gemma = parseCustomTranslateModel({
    id: "unsloth/gemma-4-E2B-it-GGUF",
    variant: "Q4_K_M",
    sizeBytes: 3_000_000_000,
  });
  assert.deepEqual(gemma, {
    id: "unsloth/gemma-4-E2B-it-GGUF",
    name: "unsloth/gemma-4-E2B-it-GGUF",
    variant: "Q4_K_M",
    sizeBytes: 3_000_000_000,
  });
  assert.equal(parseCustomTranslateModel({ id: "" }), null);
  assert.equal(parseCustomTranslateModel(null), null);

  // Another quant of Hy-MT2 still takes Hy-MT2's prompt; any other model a chat model's.
  const hyMt = { ...gemma, id: "unsloth/Hy-MT2-7B-GGUF", variant: "Q8_0" };
  assert.equal(translatePromptModel(CUSTOM_TRANSLATE_MODEL, gemma), "chat");
  assert.equal(
    translatePromptModel(CUSTOM_TRANSLATE_MODEL, hyMt),
    DEFAULT_TRANSLATE_MODEL,
  );
  assert.equal(translatePromptModel("hy-mt2-7b", gemma), "hy-mt2-7b");

  // Swapping the picked model must read as a different model to the stale check.
  assert.notEqual(
    translateModelKey(CUSTOM_TRANSLATE_MODEL, gemma),
    translateModelKey(CUSTOM_TRANSLATE_MODEL, hyMt),
  );
  assert.equal(translateModelKey("hy-mt2-7b", gemma), "hy-mt2-7b");
});

test("read aloud asks for a voice of the translation's language", () => {
  assert.equal(translateSpeechTag("zh"), "zh-CN");
  assert.equal(translateSpeechTag("zh-Hant"), "zh-TW");
  assert.equal(translateSpeechTag("yue"), "zh-HK");
  assert.equal(translateSpeechTag("fr"), "fr");
  const page = readText("../src/features/translate/translate-page.tsx");
  assert.match(page, /readAloud\.speak\(\s*output,\s*translateSpeechTag\(/);
});

test("a hard cut never lands inside a surrogate pair", () => {
  // No break to cut at, and the limit falls between the two halves of the second emoji.
  const source = "😀".repeat(4);
  const chunks = splitForTranslation(source, 3);
  assert.equal(
    chunks.map((chunk) => chunk.text + chunk.after).join(""),
    source,
  );
  for (const chunk of chunks) {
    // The emoji is two code units, so an intact piece has an even number of them.
    assert.equal(chunk.text.length % 2, 0);
  }
});

test("a stopped load is cancelled on the server and a serving repo counts only at the picked quant", () => {
  const runner = readText("../src/features/translate/translate-runner.ts");
  const ensure = runner.slice(
    runner.indexOf("async function ensureLocalModel"),
    runner.indexOf("async function chatModel"),
  );
  // A bare repo id would let a resident Q4_K_M answer for a picked Q8_0.
  assert.match(ensure, /`\$\{model\.repo\}:\$\{model\.variant\}`/);
  assert.match(ensure, /getInferenceStatus\(signal, requestModel\)/);
  assert.match(
    ensure,
    /status\.gguf_variant\?\.toLowerCase\(\) === model\.variant\.toLowerCase\(\)/,
  );
  assert.match(ensure, /\}\s*return requestModel;\s*\}/);
  // Aborting the fetch alone leaves the server loading: cancel by the load's own id.
  assert.match(ensure, /load_request_id: loadRequestId/);
  assert.match(
    ensure,
    /unloadModel\(\{\s*model_path: model\.repo,\s*cancel_load_request_id: loadRequestId,\s*\}\)\.catch\(\(\) => \{\}\)/,
  );
  assert.match(ensure, /signal\.removeEventListener\("abort", cancelLoad\)/);
});

test("the source limit never keeps half of a surrogate pair", () => {
  assert.equal(clipText("ab😀", 3), "ab");
  assert.equal(clipText("ab😀", 4), "ab😀");
  assert.equal(clipText("abc", 5), "abc");
  const store = readText("../src/features/translate/translate-store.ts");
  assert.doesNotMatch(store, /\.slice\(0, TRANSLATE_SOURCE_MAX_CHARS\)/);
});
