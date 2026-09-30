// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// audio.cpp models are addressed by virtual three-segment ids (repo + package folder) and, for
// ASR, by short dictation keys. The frontend catalog mirrors the backend's table, and every
// surface derives from it: the Audio picker, the native-runtime sets, the dictation engine map.

import assert from "node:assert/strict";
import test from "node:test";

import {
  AUDIO_CPP_MODELS,
  AUDIO_CPP_MUSIC_MAX_SECONDS,
  AUDIO_CPP_MUSIC_MIN_SECONDS,
  AUDIO_CPP_REPO,
  AUDIO_CPP_STT_KEYS,
  audioCppModelFor,
  audioCppSizeLabel,
  isAudioCppModelId,
} from "../src/features/audio/audio-cpp-catalog.ts";
import {
  MINIMAX_MUSIC_MAX_SECONDS,
  audioSamplingControlsApply,
  isGgufTtsTarget,
  musicDurationRange,
  audioCppRuntimeProblem,
  isTtsAudioType,
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
  audioPipelineTagFor,
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
import {
  AUDIO_CPP_STT_MODELS,
  DEFAULT_STT_MODEL,
  MTMD_STT_MODELS,
  STT_MODEL_LANGUAGES,
  STT_MODEL_REPOS,
  STT_MODELS,
  sttModelName,
  sttModelSize,
} from "../src/features/settings/stores/stt-model-catalog.ts";

import {
  installLocalStorageFake,
  readSrc,
  readText,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();
const { ENGLISH_ONLY_STT_MODELS, isSttModelLanguageCompatible } = await import(
  "../src/features/settings/stores/voice-settings-store.ts"
);
const {
  audioModelRequiresRemoteCode,
  audioTaskFor,
  isMusicGenerationModel,
  macTtsCatalogChoiceIsRunnable,
  musicGenerationRequiresCuda,
  usesNativeAudioRuntime,
} = await import("../src/features/audio/catalog.ts");

const KOKORO = `${AUDIO_CPP_REPO}/Kokoro-82M-GGUF`;
const ACE_TURBO = `${AUDIO_CPP_REPO}/ACE-Step1.5-GGUF/turbo`;
const QWEN_ASR = `${AUDIO_CPP_REPO}/Qwen3-ASR-0.6B-GGUF`;
const MOONSHINE_TINY = `${AUDIO_CPP_REPO}/Moonshine-Streaming-GGUF/tiny`;

type BackendRow = {
  id: string;
  key: string;
  name: string;
  task: string;
  sizeBytes: number;
  needsEspeak: boolean;
};

/** The backend table, read from its `_m(...)` rows. */
function backendRows(): BackendRow[] {
  const source = readText("../../backend/core/inference/audio_cpp_models.py");
  const repo = /AUDIO_CPP_REPO = "([^"]+)"/.exec(source)?.[1];
  assert.equal(repo, AUDIO_CPP_REPO);
  const rows: BackendRow[] = [];
  const row =
    /_m\(\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)",\s*"[^"]+",\s*"([^"]+)",\s*\[[^\]]*\],\s*([\d.]+)/g;
  const matches = [...source.matchAll(row)];
  for (const [index, match] of matches.entries()) {
    // The rest of this _m(...) call, up to the next row or the end of the table.
    const start = match.index ?? 0;
    const end = matches[index + 1]?.index ?? source.indexOf("\n]", start);
    rows.push({
      id: `${repo}/${match[1]}`,
      key: match[2],
      name: match[3],
      task: match[4],
      sizeBytes: Math.trunc(Number(match[5]) * 1024 * 1024),
      needsEspeak: /needs_espeak\s*=\s*True/.test(source.slice(start, end)),
    });
  }
  return rows;
}

test("the frontend audio.cpp catalog mirrors the backend table", () => {
  const backend = backendRows();
  assert.ok(backend.length > 0, "no backend rows parsed");
  assert.deepEqual(
    AUDIO_CPP_MODELS.map((model) => ({
      id: model.id,
      key: model.key,
      name: model.displayName,
      task: model.task,
      sizeBytes: model.sizeBytes,
      needsEspeak: "needsEspeak" in model && model.needsEspeak === true,
    })),
    backend,
  );
});

test("ids and keys are unique and resolve both ways", () => {
  const ids = new Set(AUDIO_CPP_MODELS.map((model) => model.id.toLowerCase()));
  const keys = new Set(AUDIO_CPP_MODELS.map((model) => model.key));
  assert.equal(ids.size, AUDIO_CPP_MODELS.length);
  assert.equal(keys.size, AUDIO_CPP_MODELS.length);
  for (const model of AUDIO_CPP_MODELS) {
    // Three segments or more: never an owner/name repo.
    assert.ok(model.id.split("/").length >= 3, model.id);
    assert.equal(audioCppModelFor(model.id), model);
    assert.equal(audioCppModelFor(model.id.toUpperCase()), model);
    assert.equal(audioCppModelFor(`${model.id}/`), model);
    assert.equal(audioCppModelFor(model.key), model);
  }
  assert.equal(audioCppModelFor(AUDIO_CPP_REPO), null);
  assert.equal(audioCppModelFor("OpenMOSS-Team/MOSS-TTS-Nano-100M"), null);
  assert.equal(audioCppModelFor(""), null);
  assert.equal(audioCppSizeLabel(57.6 * 1024 * 1024), "58 MB");
  assert.equal(audioCppSizeLabel(2358.4 * 1024 * 1024), "2.3 GB");
});

test("every audio.cpp model is a curated Audio row, in the right mode", () => {
  for (const model of AUDIO_CPP_MODELS) {
    const group = groupForRepoId(model.id, AUDIO_CATALOG);
    assert.equal(group?.canonicalId, model.id, model.id);
    assert.equal(group?.task, model.task === "asr" ? "stt" : "tts", model.id);
    assert.equal(audioTaskFor(model.id), model.task === "asr" ? "stt" : "tts");
    assert.equal(group?.artifacts.length, 1);
    // A GGUF format would route the pick through llama.cpp with a gguf_variant.
    assert.notEqual(group?.artifacts[0].format, "gguf");
  }
  // The virtual ids do not steal the owner/name rows they resemble.
  assert.equal(
    groupForRepoId("OpenMOSS-Team/MOSS-TTS-Nano-100M", AUDIO_CATALOG)?.canonicalId,
    "OpenMOSS-Team/MOSS-TTS-Nano-100M",
  );
  assert.equal(
    groupForRepoId("unslothai/Qwen3-ASR-0.6B-GGUF", AUDIO_CATALOG)?.canonicalId,
    "unslothai/Qwen3-ASR-0.6B-GGUF",
  );
  assert.equal(groupForRepoId(AUDIO_CPP_REPO, AUDIO_CATALOG), null);
});

test("audio.cpp rows are offered on every host class and never as GGUF", () => {
  for (const host of ["unknown", "gguf-only", "accelerated", "dense-quant"] as const) {
    const options = catalogToModelOptions(AUDIO_CATALOG, host);
    for (const model of AUDIO_CPP_MODELS) {
      const option = options.find((candidate) => candidate.id === model.id);
      assert.ok(option, `${host}: ${model.id}`);
      assert.equal(option.isGguf, false);
      assert.equal(option.descriptionSuffix, "audio.cpp");
    }
  }
  assert.equal(isGgufTtsTarget({ repoId: KOKORO }), false);
  assert.equal(isGgufTtsTarget({ repoId: KOKORO, isGguf: true }), false);
  assert.equal(isGgufTtsTarget({ repoId: "unsloth/orpheus-3b-0.1-ft-GGUF" }), true);
});

test("speech and music load on the native runtime without remote code", () => {
  for (const model of AUDIO_CPP_MODELS) {
    if (model.task === "asr") {
      assert.equal(usesNativeAudioRuntime(model.id), false);
      continue;
    }
    assert.equal(usesNativeAudioRuntime(model.id), true, model.id);
    assert.equal(audioModelRequiresRemoteCode(model.id), false, model.id);
    assert.equal(isMusicGenerationModel(model.id), model.task === "music");
    // Metal builds of audio.cpp run speech and music alike; only MiniMax needs CUDA.
    assert.equal(musicGenerationRequiresCuda(model.id), false);
    assert.equal(macTtsCatalogChoiceIsRunnable(model.id), true, model.id);
  }
  for (const audioType of ["audiocpp_tts", "audiocpp_music"]) {
    assert.equal(usesNativeAudioRuntime("someone/model", audioType), true);
    assert.equal(audioModelRequiresRemoteCode("someone/model", audioType), false);
    assert.equal(isTtsAudioType(audioType), true);
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
  assert.equal(isMusicGenerationModel(ACE_TURBO), true);
  assert.equal(nativeAudioInstructionsKind("audiocpp_music"), "music");
  // Voice design and style families read it; the rest ignore it.
  assert.equal(nativeAudioInstructionsKind("audiocpp_tts"), "voice");
  // MiniMax keeps its CUDA and Mac gates.
  assert.equal(musicGenerationRequiresCuda("MiniMaxAI/MiniMax-Music3"), true);
  assert.equal(musicGenerationRequiresCuda(null, "minimax_music3"), true);
  assert.equal(macTtsCatalogChoiceIsRunnable("MiniMaxAI/MiniMax-Music3"), false);
});

test("the Audio route accepts the audio.cpp audio types", () => {
  const route = readSrc("app/routes/audio.tsx");
  assert.match(route, /\.\.\.AUDIO_CPP_AUDIO_TYPES/);
});

test("ASR ids map to their dictation keys on the audiocpp engine", () => {
  assert.equal(sttEngineForRepoId(QWEN_ASR), "audiocpp");
  assert.equal(sttEngineForRepoId(MOONSHINE_TINY.toLowerCase()), "audiocpp");
  assert.equal(sttSidecarKeyFor(QWEN_ASR), "audiocpp-qwen3-asr-0.6b");
  assert.equal(sttSidecarKeyFor(MOONSHINE_TINY), "audiocpp-moonshine-tiny");
  assert.equal(
    sttRepoIdForSidecarKey("audiocpp-qwen3-asr-0.6b", "audiocpp"),
    QWEN_ASR,
  );
  // A key names one artifact whatever engine the caller assumed.
  assert.equal(sttRepoIdForSidecarKey("audiocpp-moonshine-tiny"), MOONSHINE_TINY);
  assert.equal(isKnownSttArtifactRepoId(QWEN_ASR), true);
  assert.equal(isKnownSttArtifactRepoId(KOKORO), false);
  // The mtmd Qwen3-ASR keeps its own engine and key.
  assert.equal(sttEngineForRepoId("unslothai/Qwen3-ASR-0.6B-GGUF"), "mtmd");
  assert.equal(sttRepoIdForSidecarKey("qwen3-asr-0.6b", "mtmd"), "unslothai/Qwen3-ASR-0.6B-GGUF");
});

test("audiocpp status blocks feed downloads and residency", () => {
  const status = {
    audiocpp: {
      downloaded_models: ["audiocpp-canary-180m-flash"],
      loaded_model: "audiocpp-canary-180m-flash",
    },
  };
  assert.deepEqual(sttDownloadedArtifacts(status, sttRepoIdForSidecarKey), [
    {
      repoId: `${AUDIO_CPP_REPO}/Canary-180M-Flash-GGUF`,
      sidecarKey: "audiocpp-canary-180m-flash",
      engine: "audiocpp",
    },
  ]);
  assert.deepEqual(resolveSttResidency(status, null, false), {
    model: "audiocpp-canary-180m-flash",
    engine: "audiocpp",
  });
  assert.deepEqual(resolveSttResidency(status, "audiocpp", true), {
    model: "audiocpp-canary-180m-flash",
    engine: "audiocpp",
  });
});

test("dictation settings list the audio.cpp ASR keys after the existing models", () => {
  const keys = AUDIO_CPP_MODELS.filter((model) => model.task === "asr").map(
    (model) => model.key,
  );
  assert.deepEqual(AUDIO_CPP_STT_KEYS, keys);
  assert.deepEqual(STT_MODELS.slice(-keys.length), keys);
  assert.equal(STT_MODELS[0], "qwen3-asr-0.6b");
  assert.equal(DEFAULT_STT_MODEL, "qwen3-asr-0.6b");
  for (const key of keys) {
    const model = audioCppModelFor(key);
    assert.ok(AUDIO_CPP_STT_MODELS.has(key));
    assert.ok(!MTMD_STT_MODELS.has(key));
    assert.equal(STT_MODEL_REPOS[key], model?.id);
    assert.equal(sttModelName(key), `${model?.displayName} (audio.cpp)`);
    assert.equal(sttModelSize(key), audioCppSizeLabel(model?.sizeBytes ?? 0));
  }
});

test("sttEngineFor sends audio.cpp keys to the audiocpp engine", () => {
  const adapter = readSrc("features/chat/adapters/studio-model-dictation-adapter.ts");
  assert.match(
    adapter,
    /export function sttEngineFor\(model: string\): SttEngine \{\s*if \(AUDIO_CPP_STT_MODELS\.has\(model\.trim\(\)\)\) return "audiocpp";/,
  );
  assert.match(adapter, /if \(engine === "audiocpp"\) return status\.audiocpp;/);
});

test("English-only and partial-language audio.cpp ASR models are gated by language", () => {
  for (const key of [
    "audiocpp-moonshine-tiny",
    "audiocpp-moonshine-small",
    "audiocpp-nemotron-3.5-asr-0.6b",
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
  // Multilingual models stay open to every language.
  for (const key of [
    "audiocpp-parakeet-tdt-0.6b-v3",
    "audiocpp-qwen3-asr-0.6b",
    "qwen3-asr-0.6b",
    "small",
  ]) {
    assert.equal(STT_MODEL_LANGUAGES.has(key), false, key);
    assert.equal(isSttModelLanguageCompatible(key, "ja-JP"), true, key);
  }
});

test("audio.cpp music length follows the backend clamp; MiniMax keeps its own", () => {
  assert.deepEqual(musicDurationRange(false), {
    min: AUDIO_CPP_MUSIC_MIN_SECONDS,
    max: AUDIO_CPP_MUSIC_MAX_SECONDS,
  });
  assert.equal(AUDIO_CPP_MUSIC_MAX_SECONDS, 240);
  assert.deepEqual(musicDurationRange(true), { min: 1, max: MINIMAX_MUSIC_MAX_SECONDS });
  const backend = readText("../../backend/core/inference/audio_cpp_backend.py");
  assert.match(backend, new RegExp(`_MAX_MUSIC_SECONDS = ${AUDIO_CPP_MUSIC_MAX_SECONDS}\\.0`));
  assert.match(backend, new RegExp(`max\\(${AUDIO_CPP_MUSIC_MIN_SECONDS}\\.0, min\\(_MAX_MUSIC_SECONDS`));
  const page = readSrc("features/audio/audio-page.tsx");
  assert.match(page, /min=\{musicRange\.min\}\s*max=\{musicRange\.max\}/);
  assert.match(page, /minimaxMusicFramesForSeconds\(musicSeconds\)/);
});

test("audio.cpp speech hides sampling controls it ignores", () => {
  assert.equal(audioSamplingControlsApply("audiocpp_tts"), false);
  for (const audioType of ["audiocpp_music", "snac", "moss_tts_local", null]) {
    assert.equal(audioSamplingControlsApply(audioType), true, String(audioType));
  }
  const page = readSrc("features/audio/audio-page.tsx");
  assert.match(page, /\{musicGeneration \|\| samplingControls \? \(\s*<AdvancedDisclosure/);
  assert.match(page, /!musicGeneration && samplingControls && temperatureEdited/);
  // Only MiniMax blocks Generate on an empty description.
  assert.match(page, /\(musicNeedsDescription && !audioInstructions\.trim\(\)\)/);
  assert.match(page, /"Voice or style description"/);
});

test("format heuristics never read audio.cpp ids as llama.cpp GGUF", () => {
  for (const model of AUDIO_CPP_MODELS) {
    assert.ok(isAudioCppModelId(model.id));
    assert.equal(isGgufId(model.id), false, model.id);
    assert.equal(hasGgufRepoSuffix(model.id), false, model.id);
    assert.equal(matchesFormatFilter(model.id, false, "gguf"), false, model.id);
    assert.equal(matchesFormatFilter(model.id, false, "safetensors"), true, model.id);
  }
  // Keys and the umbrella repo are not model ids; real GGUF repos are untouched.
  assert.equal(isAudioCppModelId("audiocpp-kokoro-82m"), false);
  assert.equal(isAudioCppModelId(AUDIO_CPP_REPO), false);
  assert.equal(isGgufId("unsloth/orpheus-3b-0.1-ft-GGUF"), true);
  assert.equal(hasGgufRepoSuffix("unslothai/Qwen3-ASR-0.6B-GGUF"), true);
  assert.equal(matchesFormatFilter("unsloth/orpheus-3b-0.1-ft-GGUF", false, "gguf"), true);
});
test("audio.cpp speech and music picks are refused when the runtime cannot run them", () => {
  const full = { available: true, espeak: true, backend: "cuda", release_tag: "v1" };
  const noEspeak = { ...full, espeak: false };
  const missing = { available: false, espeak: false, backend: null, release_tag: null };
  assert.deepEqual(
    AUDIO_CPP_MODELS.filter((model) => "needsEspeak" in model && model.needsEspeak).map(
      (model) => model.displayName,
    ),
    ["Kokoro 82M", "KittenTTS Mini 0.8", "Piper (en-US Lessac)", "Inflect Micro v2"],
  );
  for (const model of AUDIO_CPP_MODELS) {
    const espeakOnly = "needsEspeak" in model && model.needsEspeak === true;
    // No status yet, or a server predating the block: the load decides.
    assert.equal(audioCppRuntimeProblem(model.id, null), null, model.id);
    assert.equal(audioCppRuntimeProblem(model.id, undefined), null, model.id);
    assert.equal(audioCppRuntimeProblem(model.id, full), null, model.id);
    if (model.task === "asr") {
      // Dictation reports its own availability per engine.
      assert.equal(audioCppRuntimeProblem(model.id, missing), null, model.id);
      continue;
    }
    assert.match(audioCppRuntimeProblem(model.id, missing) ?? "", /runtime is not installed/);
    const problem = audioCppRuntimeProblem(model.id, noEspeak);
    if (espeakOnly) {
      assert.ok(
        problem?.startsWith(`${model.displayName} needs an audio.cpp build with eSpeak-ng`),
        model.id,
      );
    } else {
      assert.equal(problem, null, model.id);
    }
  }
  // Only curated audio.cpp ids are gated.
  assert.equal(audioCppRuntimeProblem("OpenMOSS-Team/MOSS-TTS-Nano-100M", missing), null);
  assert.equal(audioCppRuntimeProblem(null, missing), null);
  // The backend reports the block and the page refuses the pick before any load.
  const route = readText("../../backend/routes/inference.py");
  assert.match(route, /"audio_cpp_runtime": _audio_cpp_runtime_status\(\)/);
  const page = readSrc("features/audio/audio-page.tsx");
  assert.match(page, /audioCppRuntime\.current = stt\.audio_cpp_runtime \?\? null;/);
  assert.match(
    page,
    /audioCppRuntimeProblem\(id, audioCppRuntime\.current\);\s*if \(runtimeProblem\) \{\s*toast\.error\(runtimeProblem, \{ duration: 7000 \}\);\s*return;/,
  );
});