// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readdirSync } from "node:fs";
import test from "node:test";

import { sttDownloadedArtifacts } from "../src/features/audio/audio-page-policy.ts";
import {
  canonicalSttChoice,
  isRetiredWhisperChoice,
  retiredWhisperFallback,
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
    /if \(!settledInTime\) \{[\s\S]*?void hostSettle\.then\(\(\) => \{\s*if \(ended \|\| cancelled\) return;\s*adoptSettledModel\(\);\s*preload\(\);[\s\S]*?settleBeforeSend = Promise\.race\(\[\s*hostSettle,/,
  );
  // and no segment is sent before that slow read lands (bounded, so a hung read cannot stall it).
  assert.match(
    listen,
    /if \(settleBeforeSend\) await settleBeforeSend;\s*if \(cancelled\) return;\s*const text = await transcribeAudioBlob\(/,
  );
  assert.match(adapter, /HOST_SETTLE_SEND_WAIT_MS = 15_000/);
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

test("a saved Transcribe whisper.cpp pick moves to Qwen3-ASR only without PyTorch", () => {
  assert.equal(isRetiredWhisperChoice("unslothai/whisper-small-GGUF", ""), true);
  assert.equal(isRetiredWhisperChoice("unsloth/whisper-small", "gguf"), true);
  assert.equal(isRetiredWhisperChoice("unsloth/whisper-small", "transformers"), false);
  const host = (transformers?: boolean, mtmd?: boolean, audiocpp?: boolean) => ({
    transformers: transformers === undefined ? undefined : { available: transformers },
    mtmd: { available: Boolean(mtmd) },
    audiocpp: { available: Boolean(audiocpp) },
  });
  assert.equal(retiredWhisperFallback(host(true, true, true)), null);
  assert.deepEqual(retiredWhisperFallback(host(false, true, true)), {
    repo: "unslothai/Qwen3-ASR-0.6B-GGUF",
    engine: "mtmd",
  });
  assert.equal(retiredWhisperFallback(host(false, false, true))?.engine, "audiocpp");
  // nothing to run it on yet, or an older server: decide on a later read.
  assert.equal(retiredWhisperFallback(host(false, false, false)), undefined);
  assert.equal(retiredWhisperFallback(host(undefined, true, true)), undefined);
  const sidecar = readSrc("features/audio/hooks/use-stt-sidecar.ts");
  assert.match(
    sidecar,
    /if \(retiredWhisperPick\.current\) \{\s*const fallback = retiredWhisperFallback\(stt\);[\s\S]*?setLastSttRepoChoice\(fallback\.repo\);/,
  );
  // a pick the user made before the status read settles is theirs: the fallback must not replace it.
  assert.match(
    sidecar,
    /const setLastSttRepo = useCallback\(\s*\(repo: string\) => \{[\s\S]*?retiredWhisperPick\.current = false;\s*retiredWhisperRepo\.current = null;\s*setLastSttRepoChoice\(repo\);/,
  );
});

test("Audio page loads wait for a saved whisper.cpp pick to settle", () => {
  // the reference transcriber derives the engine from the settled repo id alone.
  for (const engine of [true, false]) {
    const fallback = retiredWhisperFallback({
      transformers: { available: false },
      mtmd: { available: engine },
      audiocpp: { available: true },
    });
    assert.ok(fallback);
    assert.equal(sttEngineForRepoId(fallback.repo), fallback.engine);
  }
  const sidecar = readSrc("features/audio/hooks/use-stt-sidecar.ts");
  // a read a newer one replaced still settles the pick, so a waiting load is not left on the twin.
  assert.ok(
    sidecar.indexOf("const fallback = retiredWhisperFallback(stt);") <
      sidecar.indexOf("if (generation !== sttStatusRefreshGeneration.current) return;"),
  );
  assert.match(
    sidecar,
    /const settleSavedSttRepo = useCallback\([\s\S]*?if \(!repo \|\| repo !== retiredWhisperRepo\.current\) return repo;\s*if \(retiredWhisperPick\.current\) await refreshSttStatus\(\);[\s\S]*?retiredWhisperSettled\.current \?\? repo/,
  );
  const transcription = readSrc("features/audio/hooks/use-transcription.ts");
  const prepare = transcription.slice(transcription.indexOf("const prepareTranscriptionModel"));
  assert.ok(
    prepare.indexOf("await settleSavedSttRepo(lastSttRepo)") >= 0 &&
      prepare.indexOf("await settleSavedSttRepo(lastSttRepo)") < prepare.indexOf("ensureSttLoaded("),
  );
  // a pick made while the host is read wins over the settled saved one.
  assert.match(prepare, /repo = selectedSttRepoRef\.current \?\? saved;/);
  const reference = readSrc("features/audio/hooks/use-reference-transcribe.ts");
  assert.ok(
    reference.indexOf("await settleSttRepo(sttRepo)") >= 0 &&
      reference.indexOf("await settleSttRepo(sttRepo)") < reference.indexOf("transcribeAudioInput("),
  );
  assert.doesNotMatch(reference, /sttEngineForRepoId\(sttRepo\)|sttSidecarKeyFor\(sttRepo\)/);
  for (const hook of ["use-clone-generation", "use-edit-generation", "use-convert-generation"]) {
    assert.match(
      readSrc(`features/audio/hooks/${hook}.ts`),
      /useReferenceTranscribe\(\{\s*sttRepo,\s*settleSttRepo,/,
    );
  }
  const page = readSrc("features/audio/audio-page.tsx");
  assert.equal(page.match(/settleSttRepo: settleSavedSttRepo,/g)?.length, 3);
});
