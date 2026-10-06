// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// GGUF audio models the backend serves on its audio runtime are ordinary GGUF repos to the app:
// a real Hub repo id, or a package folder of the shared repo ("<repo>/<Folder>"), named as the Hub
// names them. A short recommended list seeds the Audio pickers and dictation settings; saved
// dictation keys from before still resolve.

import assert from "node:assert/strict";
import test from "node:test";

import {
  AUDIO_CPP_DICTATION_MODELS,
  AUDIO_CPP_MODELS,
  AUDIO_CPP_MUSIC_MAX_SECONDS,
  AUDIO_CPP_MUSIC_MIN_SECONDS,
  AUDIO_CPP_REPO,
  AUDIO_CPP_STT_KEYS,
  AUDIO_CPP_UNOFFERED_FOLDERS,
  audioCppDictationModelFor,
  audioCppDisplayName,
  audioCppModelFor,
  audioCppSizeLabel,
  audioCppWorkflowsFor,
  isAudioCppFolderId,
} from "../src/features/audio/audio-cpp-catalog.ts";
import {
  MINIMAX_MUSIC_MAX_SECONDS,
  audioCppRuntimeProblem,
  audioCppRuntimeUpdate,
  audioSamplingControlsApply,
  isGgufTtsTarget,
  isTtsAudioType,
  musicDurationRange,
  musicLyricsOptional,
  musicNeedsDescription,
  nativeAudioInstructionsKind,
  resolveSttResidency,
  sttDownloadedArtifacts,
} from "../src/features/audio/audio-page-policy.ts";
import {
  isKnownSttArtifactRepoId,
  sttEngineForRepoId,
  sttRepoIdForSidecarKey,
  sttSidecarKeyFor,
} from "../src/features/audio/stt-artifacts.ts";
import {
  audioPickIsRoutable,
  audioPipelineTagFor,
  communityAudioRowIsRunnable,
  macTtsHubRowIsRunnable,
} from "../src/features/model-picker/components/model-selector/audio-picker-policy.ts";
import {
  AUDIO_CATALOG,
  catalogToModelOptions,
  groupForRepoId,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";
import {
  isGgufId,
  matchesFormatFilter,
} from "../src/features/model-picker/components/model-selector/recommended-fit.ts";
import { hasGgufRepoSuffix } from "../src/features/hub/lib/model-identifiers.ts";
import { shortModelLabel } from "../src/features/loaded-models/loaded-models-sources.ts";
import {
  AUDIO_CPP_STT_FOLDER_IDS,
  AUDIO_CPP_STT_MODELS,
  DEFAULT_STT_MODEL,
  MTMD_STT_MODELS,
  RECOMMENDED_STT_MODELS,
  STT_MODEL_LANGUAGES,
  STT_MODEL_REPOS,
  STT_MODELS,
  STT_PICKER_MODELS,
  sttModelName,
  sttModelSize,
} from "../src/features/settings/stores/stt-model-catalog.ts";

import {
  installLocalStorageFake,
  readSrc,
  readText,
  registerBundlerResolver,
} from "./helpers/kit.ts";
import { readAudioWorkspaceSource } from "./helpers/audio-workspace.ts";

registerBundlerResolver();
installLocalStorageFake();
const {
  ENGLISH_ONLY_STT_MODELS,
  isSttModelId,
  isSttModelLanguageCompatible,
  normalizeSttModel,
} = await import("../src/features/settings/stores/voice-settings-store.ts");
const {
  audioCapabilityLine,
  audioModelRequiresRemoteCode,
  audioTaskFor,
  isMusicGenerationModel,
  macTtsCatalogChoiceIsRunnable,
  musicGenerationRequiresCuda,
  usesNativeAudioRuntime,
} = await import("../src/features/audio/catalog.ts");

const KOKORO = `${AUDIO_CPP_REPO}/Kokoro-82M-GGUF`;
const ACE_STEP = `${AUDIO_CPP_REPO}/ACE-Step1.5-GGUF`;
const QWEN_ASR = `${AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF`;
const MOONSHINE = `${AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF`;
const MINIMAX_GGUF = "audio-cpp/MiniMax-Music3-GGUF";
const YUE2 = "audio-cpp/Yue2-3B-GGUF";

const BRAND = /audio\.cpp|audiocpp/i;

test("recommended ids are unique and name their Hub repo or package folder", () => {
  const ids = new Set(AUDIO_CPP_MODELS.map((model) => model.id.toLowerCase()));
  assert.equal(ids.size, AUDIO_CPP_MODELS.length);
  for (const model of AUDIO_CPP_MODELS) {
    assert.equal(audioCppModelFor(model.id), model);
    assert.equal(audioCppModelFor(model.id.toUpperCase()), model);
    assert.equal(audioCppModelFor(`${model.id}/`), model);
    // A folder id is the repo plus ONE top-level folder: sub-packages are variants.
    if (isAudioCppFolderId(model.id)) {
      assert.equal(model.id.split("/").length, 3, model.id);
    } else {
      assert.equal(model.id.split("/").length, 2, model.id);
    }
    assert.match(audioCppDisplayName(model.id), /-GGUF$/, model.id);
  }
  // MiniMax Music 3 and YuE2 ship as their own repos.
  assert.equal(audioCppModelFor(MINIMAX_GGUF)?.task, "music");
  assert.equal(audioCppModelFor(YUE2)?.task, "music");
  assert.equal(audioCppDisplayName(MINIMAX_GGUF), "MiniMax-Music3-GGUF");
  assert.equal(audioCppDisplayName(KOKORO), "Kokoro-82M-GGUF");
  assert.equal(audioCppDisplayName(`${ACE_STEP}/turbo`), "ACE-Step1.5-GGUF");
  assert.equal(isAudioCppFolderId(AUDIO_CPP_REPO), false);
  assert.equal(isAudioCppFolderId(`${AUDIO_CPP_REPO}/`), false);
  assert.equal(isAudioCppFolderId("unslothai/Qwen3-ASR-0.6B-GGUF"), false);
  assert.equal(audioCppModelFor("OpenMOSS-Team/MOSS-TTS-Nano-100M"), null);
  assert.equal(audioCppModelFor(""), null);
  assert.equal(audioCppSizeLabel(57.6 * 1024 * 1024), "58 MB");
  assert.equal(audioCppSizeLabel(2358.4 * 1024 * 1024), "2.3 GB");
});

test("folders of the shared repo the pickers leave out are never seeded and say why", () => {
  assert.equal(Object.keys(AUDIO_CPP_UNOFFERED_FOLDERS).length, 18);
  for (const [folder, reason] of Object.entries(AUDIO_CPP_UNOFFERED_FOLDERS)) {
    assert.equal(audioCppModelFor(`${AUDIO_CPP_REPO}/${folder}`), null, folder);
    assert.match(folder, /-GGUF$/);
    assert.ok(reason.trim(), folder);
  }
  // Transcription models with limited language coverage declare it.
  for (const [folder, languages] of [
    ["Citrinet-ASR-GGUF", ["en"]],
    [
      "Cohere-Transcribe-GGUF",
      ["en", "fr", "de", "es", "it", "pt", "nl", "pl", "el", "ar", "ja", "zh", "vi", "ko"],
    ],
    ["Fun-ASR-Nano-2512-GGUF", ["zh", "en", "ja"]],
    ["Granite-Speech-5.0-470M-TurboCTC-GGUF", ["en"]],
    ["Higgs-Audio-v3-STT-GGUF", ["en"]],
    ["Kroko-ASR-GGUF", ["en"]],
    ["Niagara-ASR-GGUF", ["en"]],
    ["Hviske-v5.3-GGUF", ["da"]],
    ["GigaAM-ASR-GGUF", ["ru", "en", "kk", "ky", "uz"]],
    [
      "Voxtral-Mini-4B-Realtime-2602-GGUF",
      ["en", "fr", "es", "de", "ru", "zh", "ja", "it", "pt", "nl", "ar", "hi", "ko"],
    ],
  ] as const) {
    assert.deepEqual(audioCppModelFor(`${AUDIO_CPP_REPO}/${folder}`)?.languages, languages, folder);
  }
});

test("recommended models are plain GGUF Audio rows named as on the Hub", () => {
  for (const model of AUDIO_CPP_MODELS) {
    const group = groupForRepoId(model.id, AUDIO_CATALOG);
    assert.equal(group?.canonicalId, model.id, model.id);
    assert.equal(group?.displayName, audioCppDisplayName(model.id));
    assert.equal(group?.task, model.task === "asr" ? "stt" : "tts", model.id);
    assert.equal(audioTaskFor(model.id), model.task === "asr" ? "stt" : "tts");
    assert.deepEqual(
      group?.artifacts.map((artifact) => [artifact.format, artifact.loadKind]),
      [["gguf", "gguf"]],
    );
    assert.doesNotMatch(`${group?.displayName} ${group?.description}`, BRAND);
  }
  // The folder ids do not steal the owner/name rows they resemble.
  assert.equal(
    groupForRepoId("unslothai/Qwen3-ASR-0.6B-GGUF", AUDIO_CATALOG)?.canonicalId,
    "unslothai/Qwen3-ASR-0.6B-GGUF",
  );
  assert.equal(groupForRepoId(AUDIO_CPP_REPO, AUDIO_CATALOG), null);
  for (const host of ["unknown", "gguf-only", "accelerated", "dense-quant"] as const) {
    const options = catalogToModelOptions(AUDIO_CATALOG, host);
    for (const model of AUDIO_CPP_MODELS) {
      const option = options.find((candidate) => candidate.id === model.id);
      assert.ok(option, `${host}: ${model.id}`);
      assert.equal(option.isGguf, true);
      assert.equal(option.descriptionSuffix, "GGUF");
    }
  }
});

test("voice conversion models are seeded with the pages they run on", () => {
  for (const [name, workflows, description] of [
    ["RVC-GGUF", ["convert"], "Voice conversion"],
    ["SeedVC-MLX-GGUF", ["convert"], "Voice conversion"],
    ["MeanVC2-GGUF", ["convert"], "Voice conversion"],
    ["Tone-Color-VC-GGUF", ["convert"], "Voice conversion"],
    ["Chatterbox-GGUF", ["clone", "convert"], "Voice cloning and conversion"],
    ["Vevo2-GGUF", ["clone", "edit", "convert"], "Voice cloning and conversion"],
    ["IndexTTS2-GGUF", ["clone"], "Voice cloning"],
  ] as const) {
    const id = `${AUDIO_CPP_REPO}/${name}`;
    const model = audioCppModelFor(id);
    assert.ok(model, name);
    assert.equal(model.task, "tts", name);
    assert.deepEqual(audioCppWorkflowsFor(model), workflows, name);
    assert.equal(
      groupForRepoId(id, AUDIO_CATALOG)?.description,
      description,
      name,
    );
  }
});

test("GGUF heuristics treat these rows like any llama.cpp GGUF", () => {
  for (const model of AUDIO_CPP_MODELS) {
    assert.equal(isGgufId(model.id), true, model.id);
    assert.equal(hasGgufRepoSuffix(model.id), true, model.id);
    assert.equal(matchesFormatFilter(model.id, false, "gguf"), true, model.id);
    assert.equal(isGgufTtsTarget({ repoId: model.id }), true, model.id);
  }
  assert.equal(isGgufTtsTarget({ repoId: KOKORO, isGguf: true }), true);
  // A loaded GGUF runtime model reads as speech even when the status calls it GGUF.
  for (const audioType of ["audiocpp_tts", "audiocpp_music"]) {
    assert.equal(isTtsAudioType(audioType, true), true, audioType);
    assert.equal(isTtsAudioType(audioType, false), true, audioType);
  }
  assert.equal(isTtsAudioType("csm", true), false);
});

test("speech and music load on the audio runtime without remote code", () => {
  for (const model of AUDIO_CPP_MODELS) {
    if (model.task === "asr") {
      assert.equal(usesNativeAudioRuntime(model.id), false);
      continue;
    }
    assert.equal(usesNativeAudioRuntime(model.id), true, model.id);
    assert.equal(audioModelRequiresRemoteCode(model.id), false, model.id);
    assert.equal(isMusicGenerationModel(model.id), model.task === "music");
    // Metal builds run speech and music alike; only the MiniMax pipeline needs CUDA.
    assert.equal(musicGenerationRequiresCuda(model.id), false);
    assert.equal(macTtsCatalogChoiceIsRunnable(model.id), true, model.id);
  }
  for (const audioType of ["audiocpp_tts", "audiocpp_music"]) {
    assert.equal(usesNativeAudioRuntime("someone/model", audioType), true);
    assert.equal(audioModelRequiresRemoteCode("someone/model", audioType), false);
    assert.equal(audioPipelineTagFor(audioType), "text-to-speech");
    assert.equal(
      macTtsHubRowIsRunnable({
        isMac: true,
        isTts: true,
        isGguf: false,
        hasRunnableGgufSibling: false,
        audioType,
      }),
      true,
    );
  }
  assert.equal(isMusicGenerationModel(null, "audiocpp_music"), true);
  assert.equal(isMusicGenerationModel(KOKORO, "audiocpp_tts"), false);
  assert.equal(isMusicGenerationModel(ACE_STEP), true);
  assert.equal(nativeAudioInstructionsKind("audiocpp_music"), "music");
  assert.equal(nativeAudioInstructionsKind("audiocpp_tts"), "voice");
  assert.equal(musicGenerationRequiresCuda("MiniMaxAI/MiniMax-Music3"), true);
  assert.equal(musicGenerationRequiresCuda(MINIMAX_GGUF), false);
  assert.equal(macTtsCatalogChoiceIsRunnable("MiniMaxAI/MiniMax-Music3"), false);
});

test("MiniMax Music 3 and YuE2 need a description beside the lyrics", () => {
  assert.equal(musicNeedsDescription("minimax_music3"), true);
  assert.equal(musicNeedsDescription("audiocpp_music", "minimax_music3"), true);
  assert.equal(musicNeedsDescription("audiocpp_music", "yue2"), true);
  for (const family of ["ace_step", "stable_audio", null]) {
    assert.equal(musicNeedsDescription("audiocpp_music", family), false, String(family));
  }
  assert.equal(musicNeedsDescription("audiocpp_tts", "yue2"), false);
  const page = readAudioWorkspaceSource();
  assert.match(
    page,
    /musicModelNeedsDescription\(status\?\.audio_type, status\?\.audio_family\)/,
  );
  assert.match(page, /\(musicNeedsDescription && !audioInstructions\.trim\(\)\)/);
});

test("Hub rows published for the audio runtime are runnable on the Audio page", () => {
  const row = {
    isGguf: true,
    id: "mistral-experimental/AudioCPP-Voxtral-Mini-4B-Realtime-2602-GGUF",
  };
  assert.equal(communityAudioRowIsRunnable({ ...row, isStt: true, isTts: false }), true);
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: false,
      isTts: true,
      isGguf: true,
      id: "someone/Speech-GGUF",
      tags: ["audio.cpp"],
    }),
    true,
  );
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: false,
      isTts: true,
      isGguf: true,
      id: "someone/Speech-GGUF",
      audioType: "audiocpp_tts",
    }),
    true,
  );
  // Without that evidence a GGUF ASR row still has no engine, and llama.cpp speech keeps its gate.
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: true,
      isTts: false,
      isGguf: true,
      id: "someone/whisper-small-GGUF",
    }),
    false,
  );
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: false,
      isTts: true,
      isGguf: true,
      id: "someone/Speech-GGUF",
    }),
    false,
  );
  // A downloaded GGUF classified by header routes from Chat like an Orpheus one.
  for (const audioType of ["audiocpp_tts", "audiocpp_music"]) {
    assert.equal(
      audioPickIsRoutable({
        id: KOKORO,
        task: "text-to-speech",
        isGguf: true,
        isCurated: false,
        taskFromGgufArch: true,
        audioType,
      }),
      true,
      audioType,
    );
  }
  const page = readAudioWorkspaceSource();
  assert.match(page, /speak: \["text-to-speech", "text-to-audio"\]/);
});

test("ASR repos and folders run on the audiocpp engine; saved keys still resolve", () => {
  assert.equal(sttEngineForRepoId(QWEN_ASR), "audiocpp");
  assert.equal(sttEngineForRepoId(MOONSHINE.toLowerCase()), "audiocpp");
  assert.equal(sttEngineForRepoId("someone/Speech-ASR-GGUF"), "audiocpp");
  assert.equal(sttEngineForRepoId("someone/speech-asr", true), "audiocpp");
  assert.equal(sttEngineForRepoId("openai/whisper-small"), "transformers");
  // The id is the sidecar key; an old key names its folder whatever engine the caller assumed.
  assert.equal(sttSidecarKeyFor(QWEN_ASR), QWEN_ASR);
  assert.equal(sttRepoIdForSidecarKey("audiocpp-qwen3-asr-0.6b", "audiocpp"), QWEN_ASR);
  assert.equal(sttRepoIdForSidecarKey("audiocpp-moonshine-tiny"), MOONSHINE);
  assert.equal(sttRepoIdForSidecarKey(QWEN_ASR, "audiocpp"), QWEN_ASR);
  assert.equal(sttEngineForRepoId("audiocpp-moonshine-small"), "audiocpp");
  assert.equal(isKnownSttArtifactRepoId(KOKORO), false);
  // The curated Whisper and Qwen3-ASR GGUFs keep their own engines and keys.
  assert.equal(sttEngineForRepoId("unslothai/Qwen3-ASR-0.6B-GGUF"), "mtmd");
  assert.equal(sttEngineForRepoId("unslothai/whisper-small-GGUF"), "gguf");
  assert.equal(sttRepoIdForSidecarKey("qwen3-asr-0.6b", "mtmd"), "unslothai/Qwen3-ASR-0.6B-GGUF");
});

test("audiocpp status blocks feed downloads and residency, by key or by id", () => {
  for (const reported of ["audiocpp-canary-180m-flash", `${AUDIO_CPP_REPO}/Canary-180M-Flash-GGUF`]) {
    const status = {
      audiocpp: { downloaded_models: [reported], loaded_model: reported },
    };
    assert.deepEqual(sttDownloadedArtifacts(status, sttRepoIdForSidecarKey), [
      {
        repoId: `${AUDIO_CPP_REPO}/Canary-180M-Flash-GGUF`,
        sidecarKey: reported,
        engine: "audiocpp",
      },
    ]);
    assert.deepEqual(resolveSttResidency(status, "audiocpp", true), {
      model: reported,
      engine: "audiocpp",
    });
  }
});

test("dictation settings list the saved keys by their Hub name", () => {
  assert.deepEqual(
    AUDIO_CPP_STT_KEYS,
    AUDIO_CPP_DICTATION_MODELS.map((model) => model.key),
  );
  assert.deepEqual(STT_MODELS.slice(-AUDIO_CPP_STT_KEYS.length), AUDIO_CPP_STT_KEYS);
  assert.equal(STT_MODELS[0], "qwen3-asr-0.6b");
  assert.equal(DEFAULT_STT_MODEL, "qwen3-asr-0.6b");
  for (const key of AUDIO_CPP_STT_KEYS) {
    const model = audioCppDictationModelFor(key);
    assert.ok(model, key);
    assert.ok(AUDIO_CPP_STT_MODELS.has(key));
    assert.ok(!MTMD_STT_MODELS.has(key));
    // Every saved key names a recommended ASR folder.
    assert.equal(audioCppModelFor(model.id)?.task, "asr", key);
    assert.equal(STT_MODEL_REPOS[key], model.id);
    assert.doesNotMatch(sttModelName(key), BRAND);
    assert.ok(sttModelName(key).startsWith(audioCppDisplayName(model.id)), key);
    assert.equal(sttModelSize(key), audioCppSizeLabel(model.sizeBytes));
  }
  assert.equal(sttModelName("audiocpp-qwen3-asr-0.6b"), "Qwen3-ASR-0.6B-GGUF");
  assert.equal(sttModelName("audiocpp-moonshine-tiny"), "Moonshine-Streaming-GGUF (tiny)");
  const voiceTab = readSrc("features/settings/tabs/voice-tab.tsx");
  assert.match(
    voiceTab,
    /if \(AUDIO_CPP_STT_MODELS\.has\(model\) \|\| isAudioCppFolderId\(model\)\) \{\s*return sttModelName\(model\);/,
  );
});

// Transcribe models Settings > Voice may leave out, each with the reason it cannot dictate.
const STT_PICKER_EXCLUDED: Readonly<Record<string, string>> = {};

test("Settings > Voice lists every ASR model the Transcribe page offers", () => {
  const transcribe = new Set(
    AUDIO_CATALOG.filter((group) => group.task === "stt").map((group) =>
      group.canonicalId.toLowerCase(),
    ),
  );
  const settings = new Set(
    STT_PICKER_MODELS.map((model) => {
      const keyed = audioCppDictationModelFor(model);
      if (keyed) return keyed.id.toLowerCase();
      if (isAudioCppFolderId(model)) return model.toLowerCase();
      const engine = MTMD_STT_MODELS.has(model) ? "mtmd" : "transformers";
      return sttRepoIdForSidecarKey(model, engine).toLowerCase();
    }),
  );
  for (const [id, reason] of Object.entries(STT_PICKER_EXCLUDED)) {
    assert.ok(reason.trim(), id);
    assert.ok(
      transcribe.delete(id.toLowerCase()),
      `${id} is not on Transcribe`,
    );
  }
  assert.deepEqual([...settings].sort(), [...transcribe].sort());
  // Every Moonshine size Transcribe offers has its own key.
  assert.deepEqual(
    AUDIO_CPP_DICTATION_MODELS.filter((model) =>
      model.id.endsWith("/Moonshine-Streaming-GGUF"),
    ).map((model) => model.variant),
    ["tiny", "small", "medium"],
  );
});

test("Transcribe's other ASR folders are listed by id, after the keyed models", () => {
  assert.equal(AUDIO_CPP_STT_FOLDER_IDS.length, 12);
  const keyedFolders = new Set(
    AUDIO_CPP_DICTATION_MODELS.map((model) => model.id),
  );
  for (const id of AUDIO_CPP_STT_FOLDER_IDS) {
    assert.ok(isAudioCppFolderId(id), id);
    assert.equal(audioCppModelFor(id)?.task, "asr", id);
    assert.ok(!keyedFolders.has(id), `${id} is listed twice`);
    assert.equal(sttEngineForRepoId(id), "audiocpp", id);
    // A saved folder id survives the store's id gate and a reload.
    assert.ok(isSttModelId(id), id);
    assert.equal(normalizeSttModel(id), id);
    assert.equal(sttModelName(id), audioCppDisplayName(id));
    assert.doesNotMatch(sttModelName(id), BRAND);
    // Their size comes from the variant listing, not a hand-kept table.
    assert.equal(sttModelSize(id), "");
  }
  assert.deepEqual(STT_PICKER_MODELS, [
    ...STT_MODELS,
    ...AUDIO_CPP_STT_FOLDER_IDS,
  ]);
  assert.ok(
    [...RECOMMENDED_STT_MODELS].every(
      (model, index) => STT_PICKER_MODELS[index] === model,
    ),
  );
  // The two largest sort last.
  assert.deepEqual(AUDIO_CPP_STT_FOLDER_IDS.slice(-2), [
    `${AUDIO_CPP_REPO}/VibeVoice-ASR-GGUF`,
    `${AUDIO_CPP_REPO}/Voxtral-Mini-4B-Realtime-2602-GGUF`,
  ]);
  const voiceTab = readSrc("features/settings/tabs/voice-tab.tsx");
  assert.match(
    voiceTab,
    /STT_PICKER_MODELS\.filter\(\(model\) =>\s*isSttModelLanguageCompatible\(model, language\)/,
  );
  // Searching finds them by name too, not only Hub Whisper repos.
  assert.match(
    voiceTab,
    /STT_PICKER_MODELS\.filter\(\s*\(model\) =>\s*\(sttModelName\(model\)\.toLowerCase\(\)\.includes\(needle\)/,
  );
  assert.match(voiceTab, /listGgufVariants\(id, hfApiToken\(hfToken\)\)/);
  assert.match(voiceTab, /downloadedModels\.has\(model\)/);
  assert.match(voiceTab, /\{ value: "da-DK", label: "Dansk" \}/);
});

test("an Audio-page ASR pick sends its quant; other engines and saved keys send none", () => {
  const adapter = readSrc("features/chat/adapters/studio-model-dictation-adapter.ts");
  assert.match(
    adapter,
    /return engine === "audiocpp" && ggufVariant\s*\?[\s\S]*\{ gguf_variant: ggufVariant \}\s*:\s*\{\};/,
  );
  assert.equal(adapter.match(/\.\.\.sttVariantBody\(resolvedEngine, ggufVariant\)/g)?.length, 2);
  const page = readAudioWorkspaceSource();
  assert.match(page, /sttGgufVariants\.current\.set\(id\.toLowerCase\(\), meta\.ggufVariant\)/);
  assert.match(
    page,
    /engine === "audiocpp"\s*\?\s*\(sttGgufVariants\.current\.get\(repoId\.toLowerCase\(\)\) \?\? null\)/,
  );
  assert.equal(
    page.match(/controller\.signal,\s*undefined,\s*ggufVariant,/g)?.length,
    2,
  );
  // Settings > Voice passes a saved key and no variant.
  const voiceTab = readSrc("features/settings/tabs/voice-tab.tsx");
  assert.match(voiceTab, /await startSttDownload\(sttModel, hfApiToken\(hfToken\)\);/);
});

test("sttEngineFor sends saved keys, folders and GGUF repos to the audiocpp engine", () => {
  const adapter = readSrc("features/chat/adapters/studio-model-dictation-adapter.ts");
  assert.match(adapter, /AUDIO_CPP_STT_MODELS\.has\(id\) \|\|\s*isAudioCppFolderId\(id\)/);
  assert.match(adapter, /if \(engine === "audiocpp"\) return status\.audiocpp;/);
});

test("English-only and partial-language ASR models are gated by language", () => {
  for (const key of [
    "audiocpp-moonshine-tiny",
    "audiocpp-moonshine-small",
    "audiocpp-moonshine-medium",
    "audiocpp-nemotron-3.5-asr-0.6b",
    `${AUDIO_CPP_REPO}/Citrinet-ASR-GGUF`,
    `${AUDIO_CPP_REPO}/Granite-Speech-5.0-470M-TurboCTC-GGUF`,
    `${AUDIO_CPP_REPO}/Higgs-Audio-v3-STT-GGUF`,
    `${AUDIO_CPP_REPO}/Kroko-ASR-GGUF`,
    `${AUDIO_CPP_REPO}/Niagara-ASR-GGUF`,
  ]) {
    assert.ok(ENGLISH_ONLY_STT_MODELS.has(key), key);
    assert.equal(isSttModelLanguageCompatible(key, "en-US"), true);
    assert.equal(isSttModelLanguageCompatible(key, "ja-JP"), false);
    assert.equal(isSttModelLanguageCompatible(key, "auto"), true);
  }
  const canary = "audiocpp-canary-180m-flash";
  assert.deepEqual(STT_MODEL_LANGUAGES.get(canary), ["en", "de", "es", "fr"]);
  assert.ok(!ENGLISH_ONLY_STT_MODELS.has(canary));
  for (const language of ["en-GB", "de-DE", "es-ES", "fr-FR", "auto"]) {
    assert.equal(isSttModelLanguageCompatible(canary, language), true, language);
  }
  assert.equal(isSttModelLanguageCompatible(canary, "ja-JP"), false);
  const hviske = `${AUDIO_CPP_REPO}/Hviske-v5.3-GGUF`;
  assert.equal(isSttModelLanguageCompatible(hviske, "da-DK"), true);
  assert.equal(isSttModelLanguageCompatible(hviske, "auto"), true);
  assert.equal(isSttModelLanguageCompatible(hviske, "en-US"), false);
  for (const [folder, languages, unsupported] of [
    ["Cohere-Transcribe-GGUF", ["ko-KR", "ar-SA", "pt-BR"], "ru-RU"],
    ["Fun-ASR-Nano-2512-GGUF", ["en-US", "zh-CN", "ja-JP"], "ko-KR"],
    ["GigaAM-ASR-GGUF", ["en-US", "ru-RU"], "ja-JP"],
    ["Voxtral-Mini-4B-Realtime-2602-GGUF", ["ko-KR", "hi-IN", "ru-RU"], "sv-SE"],
  ] as const) {
    const id = `${AUDIO_CPP_REPO}/${folder}`;
    for (const language of languages) {
      assert.equal(isSttModelLanguageCompatible(id, language), true, `${folder}: ${language}`);
    }
    assert.equal(isSttModelLanguageCompatible(id, unsupported), false, folder);
    assert.equal(isSttModelLanguageCompatible(id, "auto"), true, folder);
  }
  for (const key of [
    "audiocpp-parakeet-tdt-0.6b-v3",
    "audiocpp-qwen3-asr-0.6b",
    "qwen3-asr-0.6b",
    "small",
    `${AUDIO_CPP_REPO}/MOSS-Transcribe-Diarize-GGUF`,
    `${AUDIO_CPP_REPO}/VibeVoice-ASR-GGUF`,
  ]) {
    assert.equal(STT_MODEL_LANGUAGES.has(key), false, key);
    assert.equal(isSttModelLanguageCompatible(key, "ja-JP"), true, key);
  }
});

test("GGUF music length follows the backend clamp; the MiniMax pipeline keeps its own", () => {
  assert.deepEqual(musicDurationRange(false), {
    min: AUDIO_CPP_MUSIC_MIN_SECONDS,
    max: AUDIO_CPP_MUSIC_MAX_SECONDS,
  });
  assert.equal(AUDIO_CPP_MUSIC_MAX_SECONDS, 240);
  assert.deepEqual(musicDurationRange(true), { min: 1, max: MINIMAX_MUSIC_MAX_SECONDS });
  const page = readAudioWorkspaceSource();
  assert.match(page, /musicDurationRange\(cudaMusicGeneration\)/);
  assert.match(page, /min=\{musicRange\.min\}\s*max=\{musicRange\.max\}/);
  assert.match(page, /minimaxMusicFramesForSeconds\(musicSeconds\)/);
});

test("GGUF runtime speech trades sampling controls for the model's own options", () => {
  assert.equal(audioSamplingControlsApply("audiocpp_tts"), false);
  for (const audioType of ["audiocpp_music", "snac", "moss_tts_local", null]) {
    assert.equal(audioSamplingControlsApply(audioType), true, String(audioType));
  }
  const page = readAudioWorkspaceSource();
  assert.match(
    page,
    /\{musicGeneration \|\|\s*samplingControls \|\|\s*audioOptionSpecs\.length > 0 \? \(\s*<AdvancedDisclosure/,
  );
  assert.match(page, /!musicGeneration && samplingControls && temperatureEdited/);
  assert.match(page, /parseAudioOptions\(status\?\.audio_options\)/);
  assert.match(page, /\{ audio_options: requestOptions \}/);
  assert.match(page, /"Voice or style description"/);
});

test("pickers, toasts, downloads and loaded models show no engine name", () => {
  const page = readAudioWorkspaceSource();
  // Every user-visible string literal on the page, without comments.
  const withoutComments = page
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "");
  const literals = withoutComments.match(/(["'`])(?:\\.|(?!\1)[^\\])*\1/g) ?? [];
  const branded = literals.filter(
    (literal) =>
      /audio\.cpp/i.test(literal) && !/audio-cpp-catalog|audio_cpp_runtime/.test(literal),
  );
  assert.deepEqual(branded, []);
  assert.equal(shortModelLabel(KOKORO), "Kokoro-82M-GGUF");
  assert.equal(shortModelLabel("audiocpp-qwen3-asr-0.6b"), "Qwen3-ASR-0.6B-GGUF");
  assert.equal(shortModelLabel(MINIMAX_GGUF), MINIMAX_GGUF);
  const sources = readSrc("features/loaded-models/loaded-models-sources.ts");
  assert.match(sources, /audiocpp: "GGUF"/);
  const panel = readSrc("features/hub/download-manager/download-manager-panel.tsx");
  assert.match(panel, /isAudioCppFolderId\(repoId\) \? audioCppDisplayName\(repoId\) : repoId/);
  assert.match(panel, /\{job\.presentation\?\.label \?\? repoLabel\(job\.repoId\)\}/);
});

test("an Audio GGUF pick downloads its quant as the standard variant job", () => {
  const page = readAudioWorkspaceSource();
  assert.match(page, /ggufVariant: meta\.ggufVariant,/);
  const staged = readSrc("features/hub/download-manager/use-staged-download.ts");
  assert.match(staged, /current\.ggufVariant \?\? scopedVariant\(scopeId\)/);
  assert.match(staged, /variant: current\.ggufVariant,/);
});

test("recommended speech and music picks are refused when the runtime cannot run them", () => {
  const full = { available: true, espeak: true, backend: "cuda", release_tag: "v1" };
  const noEspeak = { ...full, espeak: false };
  const missing = { available: false, espeak: false, backend: null, release_tag: null };
  assert.deepEqual(
    AUDIO_CPP_MODELS.filter((model) => model.needsEspeak).map((model) =>
      audioCppDisplayName(model.id),
    ),
    ["Kokoro-82M-GGUF", "KittenTTS-GGUF", "Piper-TTS-GGUF", "Inflect-Micro-v2-GGUF"],
  );
  for (const model of AUDIO_CPP_MODELS) {
    assert.equal(audioCppRuntimeProblem(model.id, null), null, model.id);
    assert.equal(audioCppRuntimeProblem(model.id, undefined), null, model.id);
    assert.equal(audioCppRuntimeProblem(model.id, full), null, model.id);
    if (model.task === "asr") {
      assert.equal(audioCppRuntimeProblem(model.id, missing), null, model.id);
      continue;
    }
    const notInstalled = audioCppRuntimeProblem(model.id, missing) ?? "";
    assert.match(notInstalled, /audio runtime is not installed/);
    assert.doesNotMatch(notInstalled, BRAND);
    const problem = audioCppRuntimeProblem(model.id, noEspeak);
    if (model.needsEspeak) {
      assert.ok(
        problem?.startsWith(
          `${audioCppDisplayName(model.id)} needs an audio runtime built with eSpeak-ng`,
        ),
        model.id,
      );
      assert.doesNotMatch(problem ?? "", BRAND);
    } else {
      assert.equal(problem, null, model.id);
    }
  }
  assert.equal(audioCppRuntimeProblem("OpenMOSS-Team/MOSS-TTS-Nano-100M", missing), null);
  assert.equal(audioCppRuntimeProblem(null, missing), null);
  const route = readText("../../backend/routes/inference.py");
  assert.match(route, /"audio_cpp_runtime": _audio_cpp_runtime_status\(\)/);
  const page = readAudioWorkspaceSource();
  assert.match(page, /audioCppRuntime\.current = stt\.audio_cpp_runtime \?\? null;/);
  assert.match(
    page,
    /audioCppRuntimeProblem\(id, audioCppRuntime\.current\);\s*if \(runtimeProblem\) \{\s*toast\.error\(runtimeProblem, \{ duration: 7000 \}\);\s*return;/,
  );
});

test("an outdated managed runtime names both releases; anything else shows no notice", () => {
  const current = {
    available: true,
    espeak: true,
    backend: "cuda",
    release_tag: "v0.9.0-unsloth.1",
    expected_tag: "v0.9.0-unsloth.1",
    outdated: false,
  };
  const outdated = { ...current, release_tag: "v0.8.0-unsloth.1", outdated: true };
  assert.deepEqual(audioCppRuntimeUpdate(outdated), {
    installed: "v0.8.0-unsloth.1",
    expected: "v0.9.0-unsloth.1",
  });
  assert.equal(audioCppRuntimeUpdate(current), null);
  assert.equal(audioCppRuntimeUpdate(null), null);
  assert.equal(audioCppRuntimeUpdate(undefined), null);
  // support servers older than these fields or unable to name either release.
  assert.equal(
    audioCppRuntimeUpdate({ available: true, espeak: true, backend: null, release_tag: "v0.8.0" }),
    null,
  );
  assert.equal(audioCppRuntimeUpdate({ ...outdated, expected_tag: null }), null);
  assert.equal(audioCppRuntimeUpdate({ ...outdated, release_tag: null }), null);
  assert.equal(audioCppRuntimeUpdate({ ...outdated, available: false }), null);
  const route = readText("../../backend/routes/inference.py");
  assert.match(route, /"expected_tag": None,\s*"outdated": False,/);
  const page = readAudioWorkspaceSource();
  assert.match(page, /const nextUpdate = audioCppRuntimeUpdate\(audioCppRuntime\.current\);/);
  assert.match(page, /\{runtimeUpdate \? \(/);
  assert.match(
    page,
    /Stop Studio,\{" "\}\s*run\{" "\}\s*<code className="font-mono">unsloth studio update<\/code>,\s*then start Studio again\./,
  );
});

test("the capability line names GGUF audio and music, never the runtime's internal type", () => {
  assert.equal(audioCapabilityLine("tts", "audiocpp_tts"), "Text-to-speech · GGUF");
  assert.equal(audioCapabilityLine("music", "audiocpp_music"), "Music generation · GGUF");
  assert.equal(audioCapabilityLine("clone", "audiocpp_tts"), "Voice cloning · GGUF");
  assert.equal(audioCapabilityLine("convert", "audiocpp_tts"), "Voice conversion · GGUF");
  assert.equal(audioCapabilityLine("convert"), "Voice conversion");
  assert.equal(audioCapabilityLine("tts", "higgs_tts2"), "Text-to-speech · higgs_tts2");
  assert.equal(audioCapabilityLine("stt", "ready"), "Speech-to-text · ready");
});

test("YuE2 generates from a style description alone; other music still needs lyrics", () => {
  assert.equal(musicLyricsOptional("audiocpp_music", "yue2"), true);
  for (const family of ["ace_step", "minimax_music3", "stable_audio", null]) {
    assert.equal(musicLyricsOptional("audiocpp_music", family), false, String(family));
  }
  assert.equal(musicLyricsOptional("audiocpp_tts", "yue2"), false);
  const page = readAudioWorkspaceSource();
  assert.match(page, /\(!prompt\.trim\(\) && !lyricsOptional\)/);
  assert.match(page, /if \(!text && !lyricsOptional\) return;/);
});

test("a custom GGUF dictation repo skips the Whisper-only validator", () => {
  const tab = readSrc("features/settings/tabs/voice-tab.tsx");
  assert.match(tab, /!isCuratedSttModel\(model\) && sttEngineFor\(model\) !== "audiocpp"/);
});

test("the picked STT quant is remembered with the repo across a restart", () => {
  const page = readAudioWorkspaceSource();
  assert.match(page, /usePersistedChoice\("unsloth:audio:last-stt-variant", ""\)/);
  assert.match(page, /lastSttRepo && lastSttVariant \? \[\[lastSttRepo\.toLowerCase\(\), lastSttVariant\]\]/);
  assert.match(page, /setLastSttVariant\(sttGgufVariants\.current\.get\(repo\.toLowerCase\(\)\) \?\? ""\)/);
});
