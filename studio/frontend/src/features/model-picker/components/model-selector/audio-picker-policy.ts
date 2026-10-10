// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUDIO_CPP_AUDIO_TYPES,
  AUDIO_CPP_SEP_AUDIO_TYPE,
} from "../../../audio/audio-cpp-catalog.ts";
import type { FormatFilter } from "./recommended-fit";
import type { ModelSelectorChangeMeta } from "./types";

const NATIVE_AUDIO_TYPES = new Set([
  "higgs_tts2",
  "moss_tts_local",
  "moss_tts_nano",
  "higgs_tts3",
  "minimax_music3",
  ...AUDIO_CPP_AUDIO_TYPES,
]);

const TTS_CODECS = new Set(["snac", "csm", "bicodec", "dac"]);

/** Hub evidence that a GGUF targets the GGUF audio runtime rather than llama.cpp. */
const AUDIO_RUNTIME_EVIDENCE = /audio[-_.]?cpp/;

export type CommunityModelPolicy = "none" | "search-only" | "recommended";

export function shouldDiscoverCommunityModels(
  policy: CommunityModelPolicy,
): boolean {
  return policy !== "none";
}

export function shouldRecommendCommunityModels(
  policy: CommunityModelPolicy,
): boolean {
  return policy === "recommended";
}

export function audioPipelineTagFor(
  audioType?: string | null,
  isLocalCheckpoint = false,
  isLora = false,
): string | undefined {
  if (!audioType) return undefined;
  if (audioType === "whisper")
    return isLocalCheckpoint ? undefined : "automatic-speech-recognition";
  if (isLora && NATIVE_AUDIO_TYPES.has(audioType)) return undefined;
  return TTS_CODECS.has(audioType) || NATIVE_AUDIO_TYPES.has(audioType)
    ? "text-to-speech"
    : undefined;
}

export function nativeAudioCheckpointIsLoadable(
  audioType?: string | null,
  exportType?: string | null,
): boolean {
  return !audioType || !NATIVE_AUDIO_TYPES.has(audioType) || exportType === "merged";
}

/** Community ASR runs via the Transformers Whisper sidecar or the GGUF audio runtime. */
export function communityAudioRowIsRunnable({
  isStt,
  isTts,
  isGguf,
  id,
  baseModel,
  tags,
  libraryName,
  audioType,
}: {
  isStt: boolean;
  isTts: boolean;
  isGguf: boolean;
  id: string;
  baseModel?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
  audioType?: string | null;
}): boolean {
  if (!isStt && !isTts) {
    return true;
  }
  const evidence = [id, baseModel ?? "", ...(tags ?? [])].map((value) =>
    value.toLowerCase(),
  );
  if (
    isGguf &&
    isAudioRuntimeGgufEvidence({ id, baseModel, tags, libraryName, audioType })
  ) {
    return true;
  }
  if (isStt) {
    if (isGguf) return false;
    if (libraryName && libraryName.toLowerCase() !== "transformers")
      return false;
    return evidence.some((value) => value.includes("whisper"));
  }

  if (audioType && NATIVE_AUDIO_TYPES.has(audioType)) return true;

  // The main-slot TTS backend decodes only these codec families. Llasa is excluded: XCodec2 is not
  // decodable by AudioCodecManager. Keep in step with speechGgufIsUndecodable.
  const normalizedAudioType = (audioType ?? "").toLowerCase();
  if (["snac", "bicodec", "dac"].includes(normalizedAudioType)) return true;
  if (normalizedAudioType === "csm") return !isGguf;
  const family = evidence.find((value) =>
    /(?:^|[-_./])(orpheus|csm|spark-?tts|outetts|oute-?tts)(?:$|[-_./])/.test(value),
  );
  if (!family) return false;
  // llama.cpp intentionally has no CSM decoder; CSM is Transformers-only.
  return !(isGguf && /(?:^|[-_./])csm(?:$|[-_./])/.test(family));
}

function isAudioRuntimeGgufEvidence({
  id,
  baseModel,
  tags,
  libraryName,
  audioType,
}: {
  id: string;
  baseModel?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
  audioType?: string | null;
}): boolean {
  return (
    AUDIO_CPP_AUDIO_TYPES.has(audioType ?? "") ||
    [id, baseModel ?? "", ...(tags ?? []), libraryName ?? ""].some((value) =>
      AUDIO_RUNTIME_EVIDENCE.test(value.toLowerCase()),
    )
  );
}

/** A GGUF llama.cpp cannot decode, however it was found: CSM is Transformers-only, so it
 *  never loads in llama-server. Beside communityAudioRowIsRunnable's list to stay in step. */
export function speechGgufIsUndecodable({
  isGguf,
  id,
  baseModel,
  tags,
}: {
  isGguf: boolean;
  id: string;
  baseModel?: string | null;
  tags?: readonly string[] | null;
}): boolean {
  if (!isGguf) return false;
  return [id, baseModel ?? "", ...(tags ?? [])]
    .map((value) => value.toLowerCase())
    .some((value) => CSM_PATH_SEGMENT.test(value));
}

/** Separator class includes a backslash because Windows local paths arrive here. */
const CSM_PATH_SEGMENT = /(?:^|[-_./\\])csm(?:$|[-_./\\])/;

/** CSM checkpoint in a GGUF container; `audioType` comes from the checkpoint, not the path. */
export function localAudioRowIsUndecodableGguf({
  audioType,
  exportType,
  isDirectGguf = false,
}: {
  audioType?: string | null;
  exportType?: string | null;
  isDirectGguf?: boolean;
}): boolean {
  const isGguf = exportType === "gguf" || isDirectGguf;
  return isGguf && (audioType ?? "").toLowerCase() === "csm";
}

/** Whether a chat-picker audio pick may route to the Audio page, which applies the same
 *  runnable checks; curated ids always route. */
export function audioPickIsRoutable({
  id,
  task,
  isGguf,
  isCurated,
  isLocalCheckpoint = false,
  taskFromGgufArch = false,
  baseModel,
  tags,
  libraryName,
  audioType,
}: {
  id: string;
  task: string | null | undefined;
  isGguf: boolean;
  isCurated: boolean;
  isLocalCheckpoint?: boolean;
  /** Task came from reading the GGUF's own architecture. */
  taskFromGgufArch?: boolean;
  baseModel?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
  audioType?: string | null;
}): boolean {
  // Orpheus keeps the llama arch and runs on SNAC; CSM archs are unsupported; unknown rows fail closed.
  if (taskFromGgufArch && isGguf && task === "text-to-speech") {
    const codec = (audioType ?? "").toLowerCase();
    if (codec === "csm" || !codec) return false;
    return ["snac", "bicodec", "dac"].includes(codec) || AUDIO_CPP_AUDIO_TYPES.has(codec);
  }
  if (isCurated) return true;
  // Hub music / audio-to-audio tags also cover MusicGen, Stable Audio, codecs, enhancers: Audio runs none.
  if (task === "audio-to-audio") return audioType === AUDIO_CPP_SEP_AUDIO_TYPE;
  if (task === "text-to-audio") {
    if (NATIVE_AUDIO_TYPES.has(audioType ?? "")) return true;
    return (
      isGguf &&
      (taskFromGgufArch ||
        isLocalCheckpoint ||
        isAudioRuntimeGgufEvidence({ id, baseModel, tags, libraryName, audioType }))
    );
  }
  // A checkpoint from outputs/ has no Hub identity to judge, and the family-name heuristic would
  // reject it on its directory name. Its task came from the backend reading the checkpoint,
  // the stronger signal, and the Audio page lists it off that same tag.
  if (isLocalCheckpoint) {
    // A CSM GGUF on disk is as unrunnable as a cached one.
    if (speechGgufIsUndecodable({ isGguf, id, baseModel, tags })) return false;
    return (
      task === "text-to-speech" || task === "automatic-speech-recognition"
    );
  }
  // Same Hub evidence the Audio page's own lists judge on.
  return communityAudioRowIsRunnable({
    isStt: task === "automatic-speech-recognition",
    isTts: task === "text-to-speech",
    isGguf,
    id,
    baseModel,
    tags,
    libraryName,
    audioType,
  });
}

/** macOS TTS runs only via llama.cpp GGUF; curated rows may resolve to a GGUF sibling. */
export function macTtsHubRowIsRunnable({
  isMac,
  isTts,
  isGguf,
  hasRunnableGgufSibling,
  audioType,
}: {
  isMac: boolean;
  isTts: boolean;
  isGguf: boolean;
  hasRunnableGgufSibling: boolean;
  audioType?: string | null;
}): boolean {
  return (
    !isMac ||
    !isTts ||
    isGguf ||
    hasRunnableGgufSibling ||
    Boolean(
      audioType &&
      NATIVE_AUDIO_TYPES.has(audioType) &&
      audioType !== "minimax_music3",
    )
  );
}

export function taskCatalogFormatMatches(
  format: FormatFilter,
  matchesFormat: boolean,
): boolean {
  return format === "all" || matchesFormat;
}

export function taskPickerRowMatches({
  isCatalogSeed,
  isHidden = false,
  format,
  matchesFormat,
  matchesTask,
  isRecommendable,
}: {
  isCatalogSeed: boolean;
  isHidden?: boolean;
  format: FormatFilter;
  matchesFormat: boolean;
  matchesTask: boolean;
  isRecommendable: boolean;
}): boolean {
  if (isHidden && !isCatalogSeed) {
    return false;
  }
  if (isCatalogSeed) {
    return taskCatalogFormatMatches(format, matchesFormat);
  }
  if (!matchesTask) {
    return false;
  }
  return format === "all" ? isRecommendable : matchesFormat;
}

/** A downloaded GGUF often reports only its base arch (Orpheus says `llama`); an exact curated
 *  Audio artifact overrides that, but only for its own Audio mode. */
export function curatedAudioInventoryMatches({
  isActiveCatalogArtifact,
  catalogScope,
  catalogTask,
  pickerTask,
}: {
  isActiveCatalogArtifact: boolean;
  catalogScope: string | null | undefined;
  catalogTask: "tts" | "stt" | null | undefined;
  pickerTask: string | readonly string[] | null | undefined;
}): boolean {
  if (
    !isActiveCatalogArtifact ||
    catalogScope !== "audio" ||
    !catalogTask ||
    !pickerTask
  )
    return false;
  const expected =
    catalogTask === "tts" ? "text-to-speech" : "automatic-speech-recognition";
  return Array.isArray(pickerTask)
    ? pickerTask.includes(expected)
    : pickerTask === expected;
}

/** Only an exact catalog artifact may replace a text-generation fallback for a cached Audio GGUF. */
export function curatedAudioInventoryTask({
  inventoryTask,
  isExactCatalogArtifact,
  catalogScope,
  catalogTask,
}: {
  inventoryTask: string | null | undefined;
  isExactCatalogArtifact: boolean;
  catalogScope: string | null | undefined;
  catalogTask: "tts" | "stt" | null | undefined;
}): string | null {
  if (
    inventoryTask !== "text-generation" ||
    !isExactCatalogArtifact ||
    catalogScope !== "audio" ||
    !catalogTask
  ) {
    return inventoryTask ?? null;
  }
  return catalogTask === "tts"
    ? "text-to-speech"
    : "automatic-speech-recognition";
}

/** Hidden infrastructure rows stay hidden unless the active task page passed their exact
 *  normalized artifact id as an explicit runtime contract. */
export function allowedHiddenModelIdMatches(
  allowedHiddenModelIds: ReadonlySet<string> | undefined,
  ...modelIds: (string | null | undefined)[]
): boolean {
  return modelIds.some(
    (modelId) =>
      typeof modelId === "string" &&
      allowedHiddenModelIds?.has(modelId.trim().toLowerCase()),
  );
}

/** Backend tag for a diffusion GGUF the Images backend cannot assemble; both pickers hide these rows. */
const UNSUPPORTED_DIFFUSION_TASK = "image-diffusion-unsupported";

/** Fresh Hub rows carry their authoritative task in selection metadata, while
 *  downloaded/local rows retain their inventory task as a fallback. */
export function taskForMediaPick(
  pipelineTag: string | null | undefined,
  inventoryTask: string | null | undefined,
): string | null {
  // The on-device verdict is what the loader enforces, so it outranks the Hub tag.
  if (inventoryTask === UNSUPPORTED_DIFFUSION_TASK) return inventoryTask;
  // Cache inventory often reports Audio GGUFs as text-generation; the catalog task wins there.
  return pipelineTag && pipelineTag !== "text-generation"
    ? pipelineTag
    : (inventoryTask ?? pipelineTag ?? null);
}

/** STT sidecars cannot serve filesystem checkpoints yet (the API is Hub-only). */
export function filesystemRowsSupportedForTask(
  pickerTask: string | readonly string[] | null | undefined,
  rowTask?: string | null,
): boolean {
  const pickerIncludesStt = Array.isArray(pickerTask)
    ? pickerTask.includes("automatic-speech-recognition")
    : pickerTask === "automatic-speech-recognition";
  return !pickerIncludesStt && rowTask !== "automatic-speech-recognition";
}

export function withPipelineTag(
  meta: ModelSelectorChangeMeta,
  pipelineTag: string | null | undefined,
): ModelSelectorChangeMeta {
  return pipelineTag ? { ...meta, pipelineTag } : meta;
}
