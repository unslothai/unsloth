// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  TRANSLATE_HISTORY_FILE_NAME,
  TRANSLATE_LANGUAGES,
  type TranslationEntry,
  buildChatHandoff,
  buildTranslateSystemPrompt,
  cleanTranslation,
  exportTranslateHistory,
  isTranslateLanguage,
  isTranslateSource,
} from "../src/features/translate/translate-prompt.ts";

const entry = (over: Partial<TranslationEntry> = {}): TranslationEntry => ({
  id: "a",
  createdAt: Date.UTC(2026, 9, 28, 1, 2, 3),
  source: "fr",
  target: "en",
  mode: "automatic",
  tone: "balanced",
  input: "Bonjour",
  output: "Hello",
  model: "unsloth/Qwen3-1.7B-GGUF",
  ...over,
});

test("the picker offers exactly the issue's 24 target languages", () => {
  assert.deepEqual(
    TRANSLATE_LANGUAGES.map((language) => language.code),
    "en fr de es pt it ja ko ru zh ar fa id ms ne pl ro sr sv tr uk vi hi bn".split(
      " ",
    ),
  );
  assert.ok(isTranslateLanguage("bn"));
  assert.ok(!isTranslateLanguage("auto"));
  assert.ok(!isTranslateLanguage("xx"));
  assert.ok(isTranslateSource("auto"));
});

test("the system prompt names the languages and follows the mode and tone", () => {
  const prompt = buildTranslateSystemPrompt({
    source: "auto",
    target: "ja",
    mode: "literal",
    tone: "formal",
  });
  assert.match(prompt, /into Japanese\./);
  assert.match(prompt, /Detect the language of the text\./);
  assert.match(prompt, /word for word/);
  assert.match(prompt, /formal, polite register/);
  assert.match(prompt, /translation only/);
  assert.match(prompt, /Never invent or fabricate/);
  assert.match(prompt, /Do not use slang, emojis/);

  const semantic = buildTranslateSystemPrompt({
    source: "de",
    target: "en",
    mode: "semantic",
    tone: "casual",
  });
  assert.match(semantic, /The text is in German\./);
  assert.match(semantic, /idea for idea/);
  assert.match(semantic, /casual, conversational register/);
  // Casual is the one tone that may use slang, so the ban is dropped there.
  assert.doesNotMatch(semantic, /Do not use slang/);
});

test("reasoning never reaches the translation, finished or still streaming", () => {
  assert.equal(cleanTranslation("<think>hmm</think>\n\nHello\n"), "Hello");
  assert.equal(cleanTranslation("plan it</think>Hello"), "Hello");
  assert.equal(cleanTranslation("<think>still thinking"), "");
  assert.equal(
    cleanTranslation("  Line one\nLine two  "),
    "Line one\nLine two",
  );
});

test("the chat handoff carries every pair, oldest first", () => {
  const text = buildChatHandoff([
    entry({ id: "new", createdAt: 2, input: "Merci", output: "Thanks" }),
    entry({ id: "old", createdAt: 1 }),
  ]);
  assert.ok(text.indexOf("Bonjour") < text.indexOf("Merci"));
  assert.match(
    text,
    /Original \(French\):\nBonjour\n\nTranslation \(English\):\nHello/,
  );
  assert.ok(text.endsWith("\n\n"), "leaves room for the user's question");
  assert.equal(buildChatHandoff([]), "");
});

test("the exported history timestamps each entry", () => {
  const text = exportTranslateHistory([entry({ source: "auto" })]);
  assert.match(
    text,
    /^\[2026-10-28T01:02:03\.000Z\] Auto-detected -> English \(automatic, balanced, unsloth\/Qwen3-1\.7B-GGUF\)\nBonjour\n\nHello\n$/,
  );
  assert.equal(exportTranslateHistory([]), "");
  assert.equal(TRANSLATE_HISTORY_FILE_NAME, "UnslothTranslateFullHistory.txt");
});
