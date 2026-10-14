// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import ts from "typescript";

import {
  installLocalStorageFake,
  readSrc,
  registerBundlerResolver,
} from "./helpers/kit.ts";

const { store } = installLocalStorageFake();
registerBundlerResolver();

const { useVoiceSettingsStore } = await import(
  "../src/features/settings/stores/voice-settings-store.ts"
);

// The shipped settle is lifted out of the adapter rather than copied: importing the
// module would drag in the auth pages for one function, and this runs it against the
// real voice-settings store.
const source = readSrc(
  "features/chat/adapters/studio-model-dictation-adapter.ts",
);
const start = source.indexOf("type HostSttModelSettle");
assert.ok(start >= 0, "the host settle is no longer defined in the adapter");
const end =
  source.indexOf(
    "\n}",
    source.indexOf("export function settleSttModelForHost("),
  ) + 2;
const settleSource = ts
  .transpileModule(source.slice(start, end), {
    compilerOptions: { target: ts.ScriptTarget.ES2020 },
  })
  .outputText.replace(/^export /gm, "");

// A host without audio.cpp: the default settles to its llama.cpp twin.
const NO_AUDIO_CPP = {
  transformers: { available: false },
  audiocpp: { available: false },
  mtmd: { available: true },
};

test("a default that returns later in the page is settled again", async () => {
  store.delete("unsloth_voice_settings");
  await useVoiceSettingsStore.persist.rehydrate();
  assert.equal(
    useVoiceSettingsStore.getState().sttModel,
    "audiocpp-qwen3-asr-0.6b",
  );

  let statusReads = 0;
  const settleSttModelForHost = new Function(
    "fetchSttStatus",
    "useVoiceSettingsStore",
    `${settleSource}; return settleSttModelForHost;`,
  )(async () => {
    statusReads++;
    return NO_AUDIO_CPP;
  }, useVoiceSettingsStore) as () => Promise<void>;

  await settleSttModelForHost();
  assert.equal(useVoiceSettingsStore.getState().sttModel, "qwen3-asr-0.6b");
  // The model it settled on is not read again.
  await settleSttModelForHost();
  assert.equal(statusReads, 1);

  // An English-only pick, then a language it cannot serve, resets to the default.
  useVoiceSettingsStore.getState().setSttModel("audiocpp-moonshine-tiny");
  useVoiceSettingsStore.getState().setDictationLanguage("ja-JP");
  assert.equal(
    useVoiceSettingsStore.getState().sttModel,
    "audiocpp-qwen3-asr-0.6b",
  );
  assert.equal(useVoiceSettingsStore.getState().sttModelSource, "default");

  await settleSttModelForHost();
  assert.equal(useVoiceSettingsStore.getState().sttModel, "qwen3-asr-0.6b");
  assert.equal(statusReads, 2);
});
