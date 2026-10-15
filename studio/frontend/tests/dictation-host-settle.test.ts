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

// A host without PyTorch: a retired whisper.cpp pick settles to llama.cpp Qwen3-ASR.
const NO_PYTORCH = {
  transformers: { available: false },
  audiocpp: { available: true },
  mtmd: { available: true },
};

test("only a retired whisper.cpp pick reads status, once per model", async () => {
  store.set(
    "unsloth_voice_settings",
    JSON.stringify({ state: { sttModel: "small" }, version: 1 }),
  );
  await useVoiceSettingsStore.persist.rehydrate();
  assert.equal(useVoiceSettingsStore.getState().sttModelSource, "whisper-cpp");

  let statusReads = 0;
  const settleSttModelForHost = new Function(
    "fetchSttStatus",
    "useVoiceSettingsStore",
    `${settleSource}; return settleSttModelForHost;`,
  )(async () => {
    statusReads++;
    return NO_PYTORCH;
  }, useVoiceSettingsStore) as () => Promise<void>;

  await settleSttModelForHost();
  assert.equal(useVoiceSettingsStore.getState().sttModel, "qwen3-asr-0.6b");
  assert.equal(useVoiceSettingsStore.getState().sttModelSource, "default");
  assert.equal(statusReads, 1);
  // The settled default, and any pick after it, needs no read.
  await settleSttModelForHost();
  useVoiceSettingsStore.getState().setSttModel("audiocpp-moonshine-tiny");
  await settleSttModelForHost();
  assert.equal(statusReads, 1);
});
