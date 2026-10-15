// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { sttDownloadedArtifacts } from "../src/features/audio/audio-page-policy.ts";
import {
  canonicalSttChoice,
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
} from "../src/features/audio/stt-artifacts.ts";
import { WHISPER_STT_MODELS } from "../src/features/settings/stores/stt-model-catalog.ts";

import { installLocalStorageFake, readSrc } from "./helpers/kit.ts";

installLocalStorageFake();
const { NOTIFICATION_PREF_KEYS } = await import(
  "../src/hooks/use-notification-frequency.ts"
);

test("no Whisper key or GGML repo routes to the whisper.cpp engine", () => {
  const adapter = readSrc("features/chat/adapters/studio-model-dictation-adapter.ts");
  assert.doesNotMatch(adapter, /return isCuratedSttModel\(model\) \? "gguf"/);
  assert.match(
    adapter,
    /if \(MTMD_STT_MODELS\.has\(id\)\) return "mtmd";\s*return "transformers";\s*\}/,
  );
  for (const key of WHISPER_STT_MODELS) {
    const ggml = `unslothai/whisper-${key}-GGUF`;
    assert.equal(sttEngineForRepoId(ggml), "transformers", key);
    assert.equal(sttRepoIdForSidecarKey(key, "gguf"), `unsloth/whisper-${key}`, key);
  }
});

test("whisper.cpp GGML caches are not offered as on-device models", () => {
  assert.deepEqual(
    sttDownloadedArtifacts(
      { gguf: { downloaded_models: ["small"] } },
      sttRepoIdForSidecarKey,
    ),
    [],
  );
  const page = readSrc("features/audio/audio-page.tsx");
  assert.doesNotMatch(page, /artifact\.engine === "gguf"/);
});

test("a saved Transcribe pick on whisper.cpp reads as its Transformers twin", () => {
  assert.deepEqual(canonicalSttChoice("unslothai/whisper-small-GGUF", "gguf"), {
    repo: "unsloth/whisper-small",
    engine: "transformers",
  });
  assert.deepEqual(canonicalSttChoice("unslothai/Whisper-Large-v3-GGUF", ""), {
    repo: "unsloth/whisper-large-v3",
    engine: "",
  });
  for (const [repo, engine] of [
    ["unsloth/whisper-small", "transformers"],
    ["unslothai/Qwen3-ASR-0.6B-GGUF", "mtmd"],
    ["someone/Speech-ASR", "audiocpp"],
    ["", ""],
  ]) {
    assert.deepEqual(canonicalSttChoice(repo, engine), { repo, engine });
  }
  const sidecar = readSrc("features/audio/hooks/use-stt-sidecar.ts");
  assert.match(
    sidecar,
    /const \{ repo: lastSttRepo, engine: lastSttEngine \} = canonicalSttChoice\(\s*savedSttRepo,\s*savedSttEngine,\s*\);/,
  );
  // the saved engine and quant maps are seeded from the canonical pick.
  assert.ok(
    sidecar.indexOf("canonicalSttChoice(") <
      sidecar.indexOf("lastSttRepo && isAudioSttEngine(lastSttEngine)"),
  );
});

test("dictation settles a retired whisper.cpp pick before it preloads or records", () => {
  const adapter = readSrc("features/chat/adapters/studio-model-dictation-adapter.ts");
  assert.match(
    adapter,
    /const hostSettle =\s*!usesExternalEndpoint && settings\.sttModelSource === "whisper-cpp"\s*\?\s*settleSttModelForHost\(\)/,
  );
  const listen = adapter.slice(adapter.indexOf("  listen(): DictationAdapter.Session {"));
  const awaited = listen.indexOf("await Promise.race([\n            hostSettle.then(");
  const repinned = listen.indexOf("          adoptSettledModel();\n");
  const preload = listen.indexOf("        if (settledInTime) preload();");
  const firstSegment = listen.indexOf("        startSegment();\n        session.status");
  const microphone = listen.indexOf("await navigator.mediaDevices.getUserMedia(");
  assert.ok(microphone > 0 && microphone < awaited, "settles after the microphone opens");
  assert.ok(awaited < repinned && repinned < preload && preload < firstSegment);
  // a slower read still switches later segments and preloads the settled model, never the stale pick.
  assert.match(
    listen,
    /if \(!settledInTime\) \{[\s\S]*?void hostSettle\.then\(\(\) => \{\s*if \(!ended && !cancelled && adoptSettledModel\(\)\) preload\(\);/,
  );
  assert.match(adapter, /HOST_SETTLE_WAIT_MS = 2_000/);
  // only a whisper.cpp pick reads status; the read is shared per model and retried after a failure.
  assert.match(adapter, /if \(sttModelSource !== "whisper-cpp"\) return Promise\.resolve\(\);/);
  assert.match(
    adapter,
    /if \(hostSttModelSettle\?\.model === model\) return hostSttModelSettle\.promise;[\s\S]*?adoptHostSttModel\(status\)[\s\S]*?hostSttModelSettle = null;/,
  );
  const offer = readSrc("features/chat/adapters/studio-dictation-adapter.tsx");
  assert.match(
    offer,
    /await settleSttModelForHost\(\)\.catch\(\(\) => \{\}\);\s*const \{ sttModel, sttGgufVariant, setDictationEngine \} =/,
  );
  const voiceTab = readSrc("features/settings/tabs/voice-tab.tsx");
  assert.match(
    voiceTab,
    /const status = await fetchSttStatus\(statusNonce, sttModel\);\s*if \(cancelled\) return;[\s\S]{0,240}adoptHostSttModel\(status\);/,
  );
  const upload = readSrc("features/chat/hooks/use-chat-audio-upload.ts");
  assert.match(upload, /adoptHostSttModel\(status\)/);
});

test("the Voice picker shows Whisper rows from their Transformers repos", () => {
  const voiceTab = readSrc("features/settings/tabs/voice-tab.tsx");
  assert.doesNotMatch(voiceTab, /unslothai\/whisper-\$\{model\}-GGUF/);
});

test("the whisper.cpp notification setting is gone; its stored keys still reset", () => {
  const general = readSrc("features/settings/tabs/general-tab.tsx");
  assert.doesNotMatch(general, /channel="whisper"/);
  assert.match(general, /channel="audio"/);
  const search = readSrc("features/settings/settings-search.ts");
  assert.doesNotMatch(search, /showWhisperUpdates/);
  const locales = readdirSync(new URL("../src/i18n/locales/", import.meta.url))
    .filter((name) => name.endsWith(".ts"));
  assert.equal(locales.length, 14);
  for (const name of locales) {
    const source = readSrc(`i18n/locales/${name}`);
    assert.doesNotMatch(source, /showWhisperUpdates/, name);
    assert.match(source, /showAudioCppUpdates/, name);
  }
  assert.ok(NOTIFICATION_PREF_KEYS.includes("unsloth_whisper_notification_frequency"));
});
