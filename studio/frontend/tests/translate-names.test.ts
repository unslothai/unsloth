// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

import type * as Runner from "../src/features/translate/translate-runner.ts";
import * as detect from "../src/features/translate/detect-language.ts";
import * as names from "../src/features/translate/translate-names.ts";
import * as text from "../src/features/translate/translate-text.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const { annotateNames, findNames, nameSpelling } = names;

test("names are runs of capitalised words, not a capital opening a sentence", () => {
  assert.deepEqual(
    findNames("Survivor contestant Devin Way was voted out on day 12."),
    ["Devin Way"],
  );
  assert.deepEqual(
    findNames("Yesterday, Siobhan Kowalczyk met Jürgen Klopp in Liverpool."),
    ["Siobhan Kowalczyk", "Jürgen Klopp", "Liverpool"],
  );
  assert.deepEqual(findNames("I'm fine. The cat sat.\nWe left."), []);
  assert.deepEqual(
    findNames("Send it to Marcus Oyelaran before Friday in May."),
    ["Marcus Oyelaran"],
  );
  assert.deepEqual(findNames("Jean-Luc Picard met Picard."), [
    "Jean-Luc Picard",
    "Picard",
  ]);
});

test("a standalone name translation is kept only when it is a different spelling", () => {
  assert.equal(nameSpelling("Devin Way", " डेविन वे।\n"), "डेविन वे");
  assert.equal(nameSpelling("Devin Way", "Devin Way"), null);
  assert.equal(nameSpelling("Devin Way", "डेविन\nवे"), null);
  assert.equal(nameSpelling("Way", "यह एक बहुत लंबा उत्तर है जो नाम नहीं है"), null);
});

test("the original goes after the first mention only, and never twice", () => {
  const spelled = [{ source: "Devin Way", target: "डेविन वे" }];
  const done = new Set<string>();
  const once = annotateNames("डेविन वे जीते। डेविन वे खुश हैं।", spelled, done);
  assert.equal(once, "डेविन वे (Devin Way) जीते। डेविन वे खुश हैं।");
  assert.equal(
    annotateNames("डेविन वे फिर आए।", spelled, done),
    "डेविन वे फिर आए।",
  );
  assert.equal(annotateNames(once, spelled, new Set()), once);
});

test("a surname alone is matched to its own mention, never inside the full name", () => {
  const spelled = [
    { source: "Picard", target: "Пикар" },
    { source: "Jean-Luc Picard", target: "Жан-Люк Пикар" },
  ];
  assert.equal(
    annotateNames(
      "Жан-Люк Пикар встретил Пикар. Жан-Люк Пикар ушёл.",
      spelled,
      new Set(),
    ),
    "Жан-Люк Пикар (Jean-Luc Picard) встретил Пикар (Picard). Жан-Люк Пикар ушёл.",
  );
  assert.equal(
    annotateNames("Жан-Люк Пикар ушёл.", spelled, new Set()),
    "Жан-Люк Пикар (Jean-Luc Picard) ушёл.",
  );
});

test("two names with one spelling are both left alone rather than guessed", () => {
  const spelled = [
    { source: "Sean", target: "Шоном" },
    { source: "Shawn", target: "Шоном" },
    { source: "Moscow", target: "Москве" },
  ];
  assert.equal(
    annotateNames("Я поговорил с Шоном и Шоном в Москве.", spelled, new Set()),
    "Я поговорил с Шоном и Шоном в Москве (Moscow).",
  );
});

test("a shared full-name spelling still keeps its surname out of it", () => {
  const spelled = [
    { source: "John Smith", target: "Джон Смит" },
    { source: "Jon Smith", target: "Джон Смит" },
    { source: "Smith", target: "Смит" },
  ];
  assert.equal(
    annotateNames("Джон Смит встретил Джон Смит и Смит.", spelled, new Set()),
    "Джон Смит встретил Джон Смит и Смит (Smith).",
  );
});

test("a spelling ending another word is not that name", () => {
  const mali = [{ source: "Mali", target: "马里" }];
  assert.equal(
    annotateNames("我住在索马里，而不是马里。", mali, new Set()),
    "我住在索马里，而不是马里。",
  );
  const li = [{ source: "Li", target: "ली" }];
  assert.equal(
    annotateNames("मैंने कली देखी। ली आए।", li, new Set()),
    "मैंने कली देखी। ली (Li) आए।",
  );
});

test("a spelling inside a longer word is skipped, an attached prefix is not", () => {
  const klopp = [{ source: "Jürgen Klopp", target: "يورغن كلوب" }];
  assert.equal(
    annotateNames("التقت بيورغن كلوب أمس", klopp, new Set()),
    "التقت بيورغن كلوب (Jürgen Klopp) أمس",
  );
  assert.equal(
    annotateNames("رأيت يورغن كلوبه", klopp, new Set()),
    "رأيت يورغن كلوبه",
  );
  const toronto = [{ source: "Toronto", target: "ቶሮንቶ" }];
  assert.equal(
    annotateNames("ከቶሮንቶ ተቀላቀለች።", toronto, new Set()),
    "ከቶሮንቶ (Toronto) ተቀላቀለች።",
  );
});

const GLOSSED = "सर्वाइवर प्रतियोगी डेविन वे को 12वें दिन बाहर कर दिया गया।";
const UNGLOSSED = "सर्वाइवर प्रतियोगी डेविन वई को 12वें दिन बाहर कर दिया गया।";

function fakeModel(sent: string[]) {
  return async function* (request: { messages: { content: string }[] }) {
    const content = request.messages[0].content;
    sent.push(content);
    const reply = content.includes("<source>\nDevin Way\n</source>")
      ? "डेविन वे"
      : content.includes("Devin Way as डेविन वे")
        ? GLOSSED
        : UNGLOSSED;
    yield { choices: [{ delta: { content: reply }, finish_reason: "stop" }] };
  };
}

function loadRunner(sent: string[]) {
  const generalMessages = text.generalMessages;
  return loadWithStubs<typeof Runner>(
    new URL("../src/features/translate/translate-runner.ts", import.meta.url),
    {
      "@/features/chat/api/chat-api": {
        streamChatCompletions: fakeModel(sent),
      },
      "@/features/chat/external-providers": {
        isExternalModelId: () => true,
        parseExternalModelId: () => null,
      },
      "@/features/chat/stores/chat-runtime-store": {},
      "@/features/chat/utils/chat-title": {
        buildTitleRequest: async (model: string) => ({
          model,
          messages: [],
          stream: true,
        }),
      },
      "@/features/hf-auth": {},
      "@/features/hub/download-manager/constants": {},
      "@/features/hub/download-manager/download-manager-controller": {},
      "@/features/hub/lib/model-identity": {},
      "@/features/hub/stores/hf-token-store": {},
      "@/lib/navigation-nonce": {},
      "@/lib/translation-models": { isTranslationOnlyRepo: () => false },
      "./detect-language": detect,
      "./models": {
        CUSTOM_TRANSLATE_MODEL: "custom",
        dedicatedTranslateModel: () => null,
        translatePromptModel: () => "custom",
        translateMessages: (
          _id: string,
          source: string,
          options: text.TranslatePromptOptions,
        ) => generalMessages(source, options),
      },
      "./quant-on-disk": {},
      "./translate-names": names,
      "./translate-subtitles": { parseSubtitles: () => null },
      "./translate-text": text,
    },
  );
}

async function run(
  source: string,
  keepNames: boolean,
  sourceCode: string | null = null,
) {
  const sent: string[] = [];
  const prompt = {
    sourceName: null,
    targetName: "Hindi",
    mode: "automatic" as const,
    tone: "default" as const,
    keepNames,
    sourceCode,
  };
  const { output } = await loadRunner(sent).runTranslation({
    text: source,
    prompt,
    modelId: "custom" as never,
    custom: { id: "external::p::m", name: "m" } as never,
    signal: new AbortController().signal,
    onPhase: () => {},
    onOutput: () => {},
  });
  return { output, sent, prompt };
}

const PROMPT = {
  sourceName: null,
  targetName: "Hindi",
  mode: "automatic" as const,
  tone: "default" as const,
};
const SPELLED = [{ source: "Devin Way", target: "डेविन वे" }];
const SOURCE = "Survivor contestant Devin Way was voted out on day 12.";

test("off: one request, byte-identical to the prompt without the option", async () => {
  const { output, sent, prompt } = await run(SOURCE, false);
  assert.equal(output, UNGLOSSED);
  assert.deepEqual(sent, [
    text.generalMessages(SOURCE, { ...prompt, keepNames: undefined })[0]
      .content,
  ]);
});

test("on: the name is spelled first, given as a reference, and its original added", async () => {
  const { output, sent } = await run(SOURCE, true);
  assert.equal(sent.length, 2);
  assert.match(
    sent[1],
    / Write these names exactly like this: Devin Way as डेविन वे\.\n\n<source>\n/,
  );
  assert.equal(
    text.hyMtMessages("x", { ...PROMPT, names: SPELLED })[0].content,
    `Reference the following translations:\nDevin Way translates to डेविन वे\n\n${text.hyMtInstruction(PROMPT)}\n\nx`,
  );
  assert.equal(
    output,
    "सर्वाइवर प्रतियोगी डेविन वे (Devin Way) को 12वें दिन बाहर कर दिया गया।",
  );
});

test("on, German source: capitals mark nouns, so nothing extra is asked", async () => {
  const german =
    "Der Hund und die Katze spielen heute im Garten, und das ist sehr schön.";
  assert.equal(detect.detectTranslateLanguage(german), "de");
  const { sent } = await run(german, true);
  assert.equal(sent.length, 1);
  assert.ok(!sent[0].startsWith("Reference"));
});

test("on, German picked: a sentence too short to detect still asks nothing extra", async () => {
  assert.equal(detect.detectTranslateLanguage("Ich brauche Hilfe."), null);
  const { sent } = await run("Ich brauche Hilfe.", true, "de");
  assert.equal(sent.length, 1);
});
