// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  audioPipelineTagFor,
  nativeAudioCheckpointIsLoadable,
  audioPickIsRoutable,
  communityAudioRowIsRunnable,
  curatedAudioInventoryMatches,
  curatedAudioInventoryTask,
  filesystemRowsSupportedForTask,
  macTtsHubRowIsRunnable,
  localAudioRowIsUndecodableGguf,
  speechGgufIsUndecodable,
  shouldDiscoverCommunityModels,
  shouldRecommendCommunityModels,
  taskCatalogFormatMatches,
  taskForMediaPick,
  withPipelineTag,
} from "../src/features/model-picker/components/model-selector/audio-picker-policy.ts";

test("filesystem rows stay out of Transcribe while cached Hub ASR remains supported", () => {
  assert.equal(
    filesystemRowsSupportedForTask(["automatic-speech-recognition"]),
    false,
  );
  assert.equal(filesystemRowsSupportedForTask("automatic-speech-recognition"), false);
  assert.equal(filesystemRowsSupportedForTask(["text-to-speech"]), true);
  assert.equal(filesystemRowsSupportedForTask(undefined), true);
  assert.equal(
    filesystemRowsSupportedForTask(undefined, "automatic-speech-recognition"),
    false,
  );
  assert.equal(
    filesystemRowsSupportedForTask(undefined, "text-generation"),
    true,
  );
});

test("curated downloaded TTS artifacts override generic GGUF task metadata only in Speak", () => {
  assert.equal(
    curatedAudioInventoryMatches({
      isActiveCatalogArtifact: true,
      catalogScope: "audio",
      catalogTask: "tts",
      pickerTask: ["text-to-speech"],
    }),
    true,
  );
  assert.equal(
    curatedAudioInventoryMatches({
      isActiveCatalogArtifact: true,
      catalogScope: "audio",
      catalogTask: "tts",
      pickerTask: ["automatic-speech-recognition"],
    }),
    false,
  );
});

test("curated downloaded STT artifacts override generic metadata only in Transcribe", () => {
  assert.equal(
    curatedAudioInventoryMatches({
      isActiveCatalogArtifact: true,
      catalogScope: "audio",
      catalogTask: "stt",
      pickerTask: ["automatic-speech-recognition"],
    }),
    true,
  );
  assert.equal(
    curatedAudioInventoryMatches({
      isActiveCatalogArtifact: true,
      catalogScope: "audio",
      catalogTask: "stt",
      pickerTask: ["text-to-speech"],
    }),
    false,
  );
});

test("non-audio catalogs and unfiltered Chat get no downloaded-task exception", () => {
  assert.equal(
    curatedAudioInventoryMatches({
      isActiveCatalogArtifact: true,
      catalogScope: "image",
      catalogTask: "tts",
      pickerTask: ["text-to-speech"],
    }),
    false,
  );
  assert.equal(
    curatedAudioInventoryMatches({
      isActiveCatalogArtifact: true,
      catalogScope: "audio",
      catalogTask: "tts",
      pickerTask: undefined,
    }),
    false,
  );
  assert.equal(
    curatedAudioInventoryMatches({
      isActiveCatalogArtifact: false,
      catalogScope: "audio",
      catalogTask: "tts",
      pickerTask: ["text-to-speech"],
    }),
    false,
  );
});
import {
  AUDIO_CATALOG,
  artifactForRepoId,
  groupForRepoId,
} from "../src/features/model-picker/components/model-selector/model-catalog.ts";

const pickerSource = readFileSync(
  new URL(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("fresh Hub pipeline metadata routes media picks before stale inventory", () => {
  assert.equal(
    taskForMediaPick("automatic-speech-recognition", "text-to-speech"),
    "automatic-speech-recognition",
  );
  assert.equal(taskForMediaPick(null, "text-to-speech"), "text-to-speech");
});

test("an unbuildable diffusion GGUF keeps its on-device verdict over its Hub tag", () => {
  // gguf-org/stable-diffusion-v1-5-GGUF and friends: tagged text-to-image on the Hub,
  // image-diffusion-unsupported on device. Routing on the tag sends Run at an Images picker
  // that omits the row.
  assert.equal(
    taskForMediaPick("text-to-image", "image-diffusion-unsupported"),
    "image-diffusion-unsupported",
  );
  assert.equal(
    taskForMediaPick("image-to-image", "image-diffusion-unsupported"),
    "image-diffusion-unsupported",
  );
  assert.equal(
    taskForMediaPick(null, "image-diffusion-unsupported"),
    "image-diffusion-unsupported",
  );
  // Buildable diffusion rows are untouched.
  assert.equal(taskForMediaPick("text-to-image", "text-to-image"), "text-to-image");
});

test("generic cached GGUF metadata yields to the curated Audio task", () => {
  assert.equal(
    taskForMediaPick("text-generation", "automatic-speech-recognition"),
    "automatic-speech-recognition",
  );
  assert.equal(
    taskForMediaPick("text-generation", "text-to-speech"),
    "text-to-speech",
  );
});

test("exact cached Audio artifacts route from generic inventory to their Audio task", () => {
  const orpheus = artifactForRepoId(
    "unsloth/orpheus-3b-0.1-ft-GGUF",
    AUDIO_CATALOG,
  );
  assert.ok(orpheus);
  assert.equal(
    taskForMediaPick(
      null,
      curatedAudioInventoryTask({
        inventoryTask: "text-generation",
        isExactCatalogArtifact: true,
        catalogScope: orpheus.group.scope,
        catalogTask: orpheus.group.task,
      }),
    ),
    "text-to-speech",
  );

  const asr = artifactForRepoId("unslothai/Qwen3-ASR-0.6B-GGUF", AUDIO_CATALOG);
  assert.ok(asr);
  assert.equal(
    taskForMediaPick(
      null,
      curatedAudioInventoryTask({
        inventoryTask: "text-generation",
        isExactCatalogArtifact: true,
        catalogScope: asr.group.scope,
        catalogTask: asr.group.task,
      }),
    ),
    "automatic-speech-recognition",
  );
});

test("generic LLM and non-Audio artifacts keep their inventory task", () => {
  assert.equal(
    curatedAudioInventoryTask({
      inventoryTask: "text-generation",
      isExactCatalogArtifact: false,
      catalogScope: "audio",
      catalogTask: "tts",
    }),
    "text-generation",
  );
  assert.equal(
    curatedAudioInventoryTask({
      inventoryTask: "text-generation",
      isExactCatalogArtifact: true,
      catalogScope: "image",
      catalogTask: "tts",
    }),
    "text-generation",
  );
});

test("local Audio rows copy the curated task onto their clickable path alias", () => {
  assert.match(
    pickerSource,
    /const exactAudioArtifact = m\.model_id[\s\S]*artifactForRepoId\(m\.model_id, AUDIO_CATALOG\)[\s\S]*put\(m\.id, m\.task, exactAudioArtifact\)/,
  );
});

test("every filesystem inventory applies the runtime task gate", () => {
  for (const inventory of [
    "lmStudioModels",
    "localDirModels",
    "customFolderModels",
  ]) {
    assert.match(
      pickerSource,
      new RegExp(
        `${inventory}\\.filter\\([\\s\\S]*?filesystemRowsSupportedForTask\\(task, m\\.task\\)`,
      ),
    );
  }
});

test("curated task artifacts remain in All while explicit formats still filter", () => {
  assert.equal(taskCatalogFormatMatches("all", false), true);
  assert.equal(taskCatalogFormatMatches("safetensors", true), true);
  assert.equal(taskCatalogFormatMatches("gguf", false), false);
});

test("search-only Audio policy discovers community rows without recommending them", () => {
  assert.equal(shouldDiscoverCommunityModels("search-only"), true);
  assert.equal(shouldRecommendCommunityModels("search-only"), false);
  assert.equal(shouldRecommendCommunityModels("recommended"), true);
});

test("community ASR only offers checkpoints the Transformers Whisper sidecar can load", () => {
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: true,
      isTts: false,
      isGguf: false,
      id: "openai/whisper-large-v3",
      tags: ["transformers", "whisper", "automatic-speech-recognition"],
      libraryName: "transformers",
    }),
    true,
  );
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: true,
      isTts: false,
      isGguf: false,
      id: "community/speech-finetune",
      tags: ["whisper"],
      libraryName: "transformers",
    }),
    true,
  );
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: true,
      isTts: false,
      isGguf: false,
      id: "nvidia/parakeet-tdt-0.6b-v2",
      tags: ["automatic-speech-recognition"],
      libraryName: "transformers",
    }),
    false,
  );
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: true,
      isTts: false,
      isGguf: true,
      id: "community/whisper-GGUF",
    }),
    false,
  );
  // Llasa was in this list and should not have been. It speaks XCodec2, which
  // AudioCodecManager cannot decode and _AUDIO_TOKEN_PATTERNS cannot even recognise, so
  // probing a running Unsloth reports unsloth/Llasa-1B as is_audio=false. Admitting the row
  // produced a model that loaded and then failed at generation.
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: false,
      isTts: true,
      isGguf: false,
      id: "HKUSTAudio/Llasa-1B",
    }),
    false,
  );
  for (const id of [
    "canopylabs/orpheus-3b-0.1-ft",
    "sesame/csm-1b",
    "SparkAudio/Spark-TTS-0.5B",
    "OuteAI/Llama-OuteTTS-1.0-1B",
  ]) {
    assert.equal(
      communityAudioRowIsRunnable({
        isStt: false,
        isTts: true,
        isGguf: false,
        id,
      }),
      true,
    );
  }
  for (const id of [
    "suno/bark",
    "microsoft/speecht5_tts",
    "facebook/mms-tts-eng",
    "hexgrad/Kokoro-82M",
  ]) {
    assert.equal(
      communityAudioRowIsRunnable({
        isStt: false,
        isTts: true,
        isGguf: false,
        id,
      }),
      false,
    );
  }
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: false,
      isTts: true,
      isGguf: true,
      id: "community/csm-1b-GGUF",
    }),
    false,
  );
});

test("cached runnable community audio survives the Audio on-device trust gate", () => {
  assert.match(
    pickerSource,
    /c\.task === "automatic-speech-recognition" \|\|[\s\S]*c\.task === "text-to-speech"[\s\S]*communityAudioRowIsRunnable\(\{[\s\S]*isStt: c\.task === "automatic-speech-recognition",[\s\S]*isTts: c\.task === "text-to-speech",[\s\S]*id: c\.repo_id,[\s\S]*tags: c\.tags,[\s\S]*libraryName: c\.library_name/,
  );
  assert.match(
    pickerSource,
    /communityAudioRowIsRunnable\(\{[\s\S]*macTtsHubRowIsRunnable\(\{[\s\S]*isMac,[\s\S]*isTts: c\.task === "text-to-speech",[\s\S]*hasRunnableGgufSibling/,
  );
});

test("Chat-to-Audio handoff preserves the live Hub task", () => {
  // pickedTask, not meta.pipelineTag: a cached row carries no tag, so ASR would read as TTS.
  assert.match(
    pickerSource,
    /page === "audio"[\s\S]*task:\s*pickedTask \?\? undefined/,
  );
  assert.match(
    pickerSource,
    /page === "audio" &&\s*!audioPickIsRoutable\(\{[\s\S]*isCurated: artifactForRepoId\(id, AUDIO_CATALOG\) !== null/,
  );
  assert.match(
    pickerSource,
    /page === "audio"[\s\S]*ggufQuant:\s*meta\.ggufFilename[\s\S]*meta\.ggufVariant/,
  );
  assert.match(
    pickerSource,
    /const alreadyListed = new Set\([\s\S]*visibleCachedGguf[\s\S]*visibleCachedModels/,
  );
});

test("community Audio feeds participate in both infinite-scroll paths", () => {
  assert.match(pickerSource, /communityQuerySearch\.fetchMore\(\)/);
  assert.match(pickerSource, /communityBrowse\.fetchMore\(\)/);
  assert.match(pickerSource, /recommendedHasMore/);
  assert.match(
    pickerSource,
    /const unslothRequested = hasMore \? fetchMore\(\) : false;[\s\S]*const communityRequested =[\s\S]*communityQuerySearch\.fetchMore\(\)/,
  );
});

test("macOS TTS search only offers directly runnable or curated GGUF-backed rows", () => {
  assert.equal(
    macTtsHubRowIsRunnable({
      isMac: true,
      isTts: true,
      isGguf: false,
      hasRunnableGgufSibling: false,
    }),
    false,
  );
  assert.equal(
    macTtsHubRowIsRunnable({
      isMac: true,
      isTts: true,
      isGguf: true,
      hasRunnableGgufSibling: false,
    }),
    true,
  );
  assert.equal(
    macTtsHubRowIsRunnable({
      isMac: true,
      isTts: true,
      isGguf: false,
      hasRunnableGgufSibling: true,
    }),
    true,
  );
  assert.equal(
    macTtsHubRowIsRunnable({
      isMac: false,
      isTts: true,
      isGguf: false,
      hasRunnableGgufSibling: false,
    }),
    true,
  );
  const csmHasGguf = Boolean(
    groupForRepoId("unsloth/csm-1b", AUDIO_CATALOG)?.artifacts.some(
      (artifact) => artifact.format === "gguf",
    ),
  );
  assert.equal(csmHasGguf, false);
  assert.equal(
    macTtsHubRowIsRunnable({
      isMac: true,
      isTts: true,
      isGguf: false,
      hasRunnableGgufSibling: csmHasGguf,
    }),
    false,
  );
});

test("GGUF variant picks retain their Hub pipeline tag", () => {
  assert.deepEqual(
    withPipelineTag(
      { source: "hub", isLora: false, isGguf: true, ggufVariant: "Q4_K_M" },
      "automatic-speech-recognition",
    ),
    {
      source: "hub",
      isLora: false,
      isGguf: true,
      ggufVariant: "Q4_K_M",
      pipelineTag: "automatic-speech-recognition",
    },
  );
});

test("only speech repos the runtime can serve are routed to the Audio page", () => {
  const curated = {
    id: "unsloth/orpheus-3b-0.1-ft",
    task: "text-to-speech",
    isGguf: false,
    isCurated: true,
  };
  assert.equal(audioPickIsRoutable(curated), true);
  // The catalog, not the tag, is a curated id's runtime contract.
  assert.equal(audioPickIsRoutable({ ...curated, isCurated: false }), true);

  // Hub tags text-to-speech onto families the main-slot backend has no decoder for.
  assert.equal(
    audioPickIsRoutable({
      id: "suno/bark",
      task: "text-to-speech",
      isGguf: false,
      isCurated: false,
    }),
    false,
  );
  // Community ASR runs on the Transformers Whisper sidecar only.
  assert.equal(
    audioPickIsRoutable({
      id: "someone/whisper-small-fi",
      task: "automatic-speech-recognition",
      isGguf: false,
      isCurated: false,
    }),
    true,
  );
  assert.equal(
    audioPickIsRoutable({
      id: "someone/parakeet-tdt",
      task: "automatic-speech-recognition",
      isGguf: false,
      isCurated: false,
    }),
    false,
  );
  // A chat pick is not an audio pick, so the gate must not claim it.
  assert.equal(
    audioPickIsRoutable({
      id: "unsloth/Qwen3-8B",
      task: "text-generation",
      isGguf: false,
      isCurated: false,
    }),
    true,
  );
});

test("a cached speech GGUF no backend can decode is not listed at all", () => {
  assert.match(
    pickerSource,
    /cachedGguf\.filter\([\s\S]*passesTaskGate\([\s\S]*audioPickIsRoutable\(\{[\s\S]*isGguf: true/,
  );
  assert.equal(
    audioPickIsRoutable({
      id: "ggml-org/sesame-csm-1b-GGUF",
      task: "text-to-speech",
      isGguf: true,
      isCurated: false,
    }),
    false,
  );
});

test("a speech GGUF found on the filesystem is gated like a cached one", () => {
  // The backend tags llama-csm as text-to-speech wherever it is discovered, so every
  // filesystem list has to ask the same policy the cached GGUF list does.
  for (const rows of [
    "lmStudioModels",
    "localDirModels",
    "customFolderModels",
  ]) {
    assert.match(
      pickerSource,
      new RegExp(
        `${rows}\\.filter\\([\\s\\S]*?audioPickIsRoutable\\(\\{`,
      ),
      `${rows} must apply the speech gate`,
    );
  }
});

test("a local CSM GGUF is unroutable even though it was found on disk", () => {
  assert.equal(
    audioPickIsRoutable({
      id: "/models/sesame-csm-1b-GGUF/csm-1b-Q4_K_M.gguf",
      task: "text-to-speech",
      isGguf: true,
      isCurated: false,
      isLocalCheckpoint: true,
    }),
    false,
  );
  // A Transformers CSM checkpoint still runs, so local provenance keeps routing it.
  assert.equal(
    audioPickIsRoutable({
      id: "/outputs/csm-1b-finetune",
      task: "text-to-speech",
      isGguf: false,
      isCurated: false,
      isLocalCheckpoint: true,
    }),
    true,
  );
});

test("an unroutable speech pick is refused instead of loaded into chat", () => {
  assert.match(
    pickerSource,
    /!audioPickIsRoutable\(\{[\s\S]*\}\)\s*\) \{[\s\S]*toast\.error\([\s\S]*return;/,
  );
});

test("fine-tuned audio rows receive only runnable pipeline tags", () => {
  // The STT sidecar's resolve_model_id takes a curated key or an owner/model Hub id, so a
  // filesystem path 422s. Routing one from the picker advertised a row that cannot load.
  assert.equal(audioPipelineTagFor("whisper", true), undefined);
  assert.equal(audioPipelineTagFor("moss_tts_local", true), "text-to-speech");
  // Native runtimes reject adapter-only checkpoints; merged exports remain runnable.
  assert.equal(audioPipelineTagFor("moss_tts_local", true, true), undefined);
  assert.equal(audioPipelineTagFor("moss_tts_local", true, false), "text-to-speech");
  assert.match(
    pickerSource,
    /pipelineTag: audioPipelineTagFor\(adapter\.audioType, true, isLora\)/,
  );
});

test("adapter-only native audio checkpoints are hidden from the runnable picker", () => {
  assert.equal(nativeAudioCheckpointIsLoadable("moss_tts_local", "adapter"), false);
  assert.equal(nativeAudioCheckpointIsLoadable("higgs_tts2", "merged"), true);
  assert.equal(nativeAudioCheckpointIsLoadable("snac", "adapter"), true);
  assert.match(
    pickerSource,
    /fineTunedRows[\s\S]*nativeAudioCheckpointIsLoadable\(m\.audioType, m\.exportType\)/,
  );
});

test("an arch-tasked speech GGUF routes by detected codec", () => {
  const parkedUnderOrpheus = {
    id: "/models/orpheus/custom.gguf",
    task: "text-to-speech",
    isGguf: true,
    isCurated: false,
  };
  assert.equal(audioPickIsRoutable(parkedUnderOrpheus), true);
  assert.equal(
    audioPickIsRoutable({
      ...parkedUnderOrpheus,
      taskFromGgufArch: true,
      audioType: "csm",
    }),
    false,
  );
  assert.equal(
    audioPickIsRoutable({
      ...parkedUnderOrpheus,
      isCurated: true,
      taskFromGgufArch: true,
      audioType: "csm",
    }),
    false,
  );
  // The same speech task is runnable when the backend's GGUF classifier identifies
  // the ordinary-llama Orpheus build and records its SNAC decoder.
  assert.equal(
    audioPickIsRoutable({
      id: "someone/orpheus-3b-custom-GGUF",
      task: "text-to-speech",
      isGguf: true,
      isCurated: false,
      taskFromGgufArch: true,
      audioType: "snac",
    }),
    true,
  );
  // Older backends have no provenance field, so their speech GGUF rows remain fail-closed.
  assert.equal(
    audioPickIsRoutable({
      id: "/models/orpheus/custom.gguf",
      task: "text-to-speech",
      isGguf: true,
      isCurated: false,
      taskFromGgufArch: true,
    }),
    false,
  );
  assert.equal(
    audioPickIsRoutable({
      id: "unsloth/csm-1b",
      task: "text-to-speech",
      isGguf: false,
      isCurated: true,
      taskFromGgufArch: true,
    }),
    true,
  );
});

test("a renamed cached TTS checkpoint routes on its detected codec", () => {
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: false,
      isTts: true,
      isGguf: false,
      id: "someone/renamed-checkpoint",
      audioType: "snac",
    }),
    true,
  );
  assert.equal(
    communityAudioRowIsRunnable({
      isStt: false,
      isTts: true,
      isGguf: true,
      id: "someone/renamed-checkpoint",
      audioType: "csm",
    }),
    false,
  );
});

test("every filesystem list passes the arch-tasked flag, not just the policy call", () => {
  // The policy is only as good as its call sites: a list that asks without the flag gets
  // the name heuristic back and re-opens the hole this gate exists to close.
  const callSites = pickerSource.match(/taskFromGgufArch: true/g) ?? [];
  assert.equal(callSites.length, 4);
});

test("a Windows path is judged like its posix equivalent", () => {
  // The separator class carries a backslash: these predicates are handed local checkpoint
  // paths, and on Windows the same file arrives as C:\models\csm-1b\model.gguf, which a
  // posix-only class reads as one long segment and clears.
  for (const id of [
    "/models/csm-1b/model.gguf",
    "C:\\models\\csm-1b\\model.gguf",
    "C:\\Users\\me\\models\\csm\\q8.gguf",
  ]) {
    assert.equal(
      speechGgufIsUndecodable({ isGguf: true, id }),
      true,
      `${id} must be undecodable`,
    );
    assert.equal(
      audioPickIsRoutable({
        id,
        task: "text-to-speech",
        isGguf: true,
        isCurated: false,
        isLocalCheckpoint: true,
      }),
      false,
      `${id} must not route`,
    );
  }
  // Still no false positive on a name that merely contains the letters.
  assert.equal(
    speechGgufIsUndecodable({ isGguf: true, id: "C:\\models\\csmith-7b\\q8.gguf" }),
    false,
  );
});

test("a csm checkpoint exported to GGUF is not offered anywhere", () => {
  // audioType is read off the checkpoint by the backend, so it is authoritative even where
  // nothing in the path says csm. llama.cpp has no CSM decoder, whatever the container.
  assert.equal(
    localAudioRowIsUndecodableGguf({ audioType: "csm", exportType: "gguf" }),
    true,
  );
  assert.equal(
    localAudioRowIsUndecodableGguf({ audioType: "csm", isDirectGguf: true }),
    true,
  );
  assert.equal(
    localAudioRowIsUndecodableGguf({ audioType: "CSM", exportType: "gguf" }),
    true,
  );
  // A csm LoRA or merged safetensors checkpoint still runs on the Transformers path.
  assert.equal(
    localAudioRowIsUndecodableGguf({ audioType: "csm", exportType: "merged" }),
    false,
  );
  assert.equal(
    localAudioRowIsUndecodableGguf({ audioType: "csm", exportType: "lora" }),
    false,
  );
  // The codecs llama.cpp DOES decode keep their GGUF exports.
  for (const audioType of ["snac", "bicodec", "dac"]) {
    assert.equal(
      localAudioRowIsUndecodableGguf({ audioType, exportType: "gguf" }),
      false,
      audioType,
    );
  }
});

test("the fine-tuned section applies the undecodable-GGUF gate", () => {
  assert.match(
    pickerSource,
    /loraModels[\s\S]{0,400}?localAudioRowIsUndecodableGguf\(\{/,
  );
});

test("the audio page asks the GGUF-aware TTS predicate for trained rows", () => {
  // GGUF_TTS_AUDIO_TYPES leaves csm out because llama.cpp has no CSM decoder. Calling
  // isTtsAudioType without the flag answered off the wider Transformers list and offered
  // a csm GGUF export that fails at load.
  const source = readFileSync(
    new URL(
      "../src/features/audio/audio-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /isTtsAudioType\(\s*lora\.audio_type,\s*lora\.export_type === "gguf",?\s*\)/,
  );
});
