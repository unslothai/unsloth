// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  communityAudioRowIsRunnable,
  curatedAudioInventoryMatches,
  curatedAudioInventoryTask,
  filesystemRowsSupportedForTask,
  macTtsHubRowIsRunnable,
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
  for (const id of [
    "canopylabs/orpheus-3b-0.1-ft",
    "sesame/csm-1b",
    "SparkAudio/Spark-TTS-0.5B",
    "OuteAI/Llama-OuteTTS-1.0-1B",
    "HKUSTAudio/Llasa-1B",
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
  assert.match(
    pickerSource,
    /page === "audio"[\s\S]*task:\s*meta\.pipelineTag \?\? undefined/,
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
