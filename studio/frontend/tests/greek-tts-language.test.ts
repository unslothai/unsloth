// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  SPEECH_LANGUAGES,
  resolveTtsLanguageName,
  ttsPreviewText,
} from "../src/features/settings/lib/speech-languages.ts";
import { readSrc } from "./helpers/kit.ts";
import { loadWithStubs } from "./helpers/module-stubs.ts";

const GREEK_PREVIEW = /Γεια σας/;
const TTS_STORE_READ = /getState\(\)\.ttsLanguage/;
const TTS_REQUEST_FIELD = /audio_language: audioLanguage/;
const SYSTEM_TTS_LANGUAGE =
  /utterance\.lang = requestedLanguage \?\? voice\?\.lang \?\? ""/;
const TTS_LANGUAGE_LABEL = /settings\.voice\.readAloud\.languageLabel/;
const TTS_LANGUAGE_VALUE = /value=\{ttsLanguage\}/;
const TTS_LANGUAGE_MERGE =
  /ttsLanguage: asString\(saved\?\.ttsLanguage, "auto"\)/;

test("Greek is offered as a speech and TTS language", () => {
  assert.deepEqual(
    SPEECH_LANGUAGES.find((language) => language.value === "el-GR"),
    { value: "el-GR", label: "Ελληνικά", ttsName: "Greek" },
  );
  assert.equal(resolveTtsLanguageName("el-GR"), "Greek");
  assert.equal(resolveTtsLanguageName("el_GR"), "Greek");
  assert.match(ttsPreviewText("el-GR"), GREEK_PREVIEW);
});

test("Studio read-aloud forwards its dedicated TTS language", () => {
  const adapter = readSrc(
    "features/chat/adapters/studio-speech-synthesis-adapter.ts",
  );
  const voiceTab = readSrc("features/settings/tabs/voice-tab.tsx");
  const store = readSrc("features/settings/stores/voice-settings-store.ts");

  assert.match(adapter, TTS_STORE_READ);
  assert.match(adapter, TTS_REQUEST_FIELD);
  assert.match(adapter, SYSTEM_TTS_LANGUAGE);
  assert.match(voiceTab, TTS_LANGUAGE_LABEL);
  assert.match(voiceTab, TTS_LANGUAGE_VALUE);
  assert.match(store, TTS_LANGUAGE_MERGE);
});

function loadAdapter(ttsLanguage: string, posted: RequestInit[] = []) {
  return loadWithStubs<{
    createConfiguredUtterance: (text: string) => SpeechSynthesisUtterance;
    generateStudioTtsAudio: (text: string) => Promise<string>;
  }>(
    new URL(
      "../src/features/chat/adapters/studio-speech-synthesis-adapter.ts",
      import.meta.url,
    ),
    {
      "@/features/auth": {
        authFetch: async (_url: string, init: RequestInit) => {
          posted.push(init);
          return {
            ok: true,
            json: async () => ({ audio: { data: "UklGRg==" } }),
          };
        },
      },
      "@/features/settings/lib/speech-languages": { resolveTtsLanguageName },
      "@/features/settings/stores/voice-settings-store": {
        useVoiceSettingsStore: {
          getState: () => ({
            ttsLanguage,
            dictationLanguage: "auto",
            ttsVoiceURI: "english",
            ttsRate: 1,
            ttsPitch: 1,
            ttsVolume: 1,
          }),
        },
      },
      "@/lib/toast": {},
      "../api/providers-api": {},
      "../external-providers": {},
      "../search-images/search-images": {},
      "../stores/external-providers-store": {},
    },
  );
}

test("Greek read-aloud sends the language hint, Auto preserves the old request", async () => {
  for (const language of ["el-GR", "auto"]) {
    const posted: RequestInit[] = [];
    const adapter = loadAdapter(language, posted);
    assert.equal(
      await adapter.generateStudioTtsAudio("Γεια σας"),
      "data:audio/wav;base64,UklGRg==",
    );
    assert.deepEqual(JSON.parse(String(posted[0].body)), {
      messages: [{ role: "user", content: "Γεια σας" }],
      stream: false,
      ...(language === "el-GR" ? { audio_language: "Greek" } : {}),
    });
  }
});

test("Greek system speech overrides an English voice, even beyond the curated cap", (t) => {
  const english = {
    voiceURI: "english",
    name: "English",
    lang: "en-US",
    default: true,
  };
  const greek = {
    voiceURI: "greek",
    name: "Greek",
    lang: "el-GR",
    default: false,
  };
  const british = {
    voiceURI: "british",
    name: "British English",
    lang: "en-GB",
    default: false,
  };
  let voices = [
    english,
    ...Array.from({ length: 25 }, (_, i) => ({
      ...english,
      voiceURI: `english-${i}`,
      name: `Premium English ${i}`,
    })),
    greek,
  ];
  for (const [key, value] of Object.entries({
    window: { speechSynthesis: { getVoices: () => voices } },
    SpeechSynthesisUtterance: class {},
  })) {
    const original = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, { configurable: true, value });
    t.after(() => {
      if (original) {
        Object.defineProperty(globalThis, key, original);
      } else {
        Reflect.deleteProperty(globalThis, key);
      }
    });
  }
  const selected = loadAdapter("el-GR").createConfiguredUtterance("Γεια σας");
  assert.equal(selected.lang, "el-GR");
  assert.equal(selected.voice, greek);
  assert.equal(
    loadAdapter("auto").createConfiguredUtterance("Hello").voice,
    english,
  );
  voices = [english, british];
  const regional = loadAdapter("en-GB").createConfiguredUtterance("Hello");
  assert.equal(regional.lang, "en-GB");
  assert.equal(regional.voice, british);
  voices = [english];
  const missing = loadAdapter("el-GR").createConfiguredUtterance("Γεια σας");
  assert.equal(missing.lang, "el-GR");
  assert.equal(missing.voice, undefined);
});
