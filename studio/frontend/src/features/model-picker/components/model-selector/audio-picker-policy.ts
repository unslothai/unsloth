// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FormatFilter } from "./recommended-fit";
import type { ModelSelectorChangeMeta } from "./types";

const NATIVE_AUDIO_TYPES = new Set([
  "higgs_tts2",
  "moss_tts_local",
  "moss_tts_nano",
  "higgs_tts3",
  "minimax_music3",
]);

const TTS_CODECS = new Set(["snac", "csm", "bicodec", "dac"]);

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

/** Maps detected audio runtime types to the media tag used by Chat routing. */
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

/** Community ASR runs through the Transformers Whisper sidecar. Curated GGUF/MTMD artifacts
 *  are handled by the catalog before this gate, so an uncurated row must identify a
 *  non-GGUF Whisper checkpoint. */
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
  if (isStt) {
    if (isGguf) return false;
    if (libraryName && libraryName.toLowerCase() !== "transformers")
      return false;
    return evidence.some((value) => value.includes("whisper"));
  }

  if (audioType && NATIVE_AUDIO_TYPES.has(audioType)) return true;

  // The main-slot TTS backend decodes only the four codec families below. Hub's
  // text-to-speech tag also covers Bark, VITS, SpeechT5 and others that load as language
  // models but cannot emit a WAV here. Llasa is excluded despite being well known: it speaks
  // XCodec2, which AudioCodecManager cannot decode, so it produced a row that loaded and
  // then failed at generation. This list and the comment above must stay in step.
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

/** `csm` as its own path or name segment. The separator class carries a BACKSLASH as well as
 *  a slash: local checkpoint paths arrive here, and on Windows
 *  `C:\models\csm-1b\model.gguf` reads as one segment to a posix-only class. */
const CSM_PATH_SEGMENT = /(?:^|[-_./\\])csm(?:$|[-_./\\])/;

/** Whether a fine-tuned or exported row is a CSM checkpoint in a GGUF container, which no
 *  runtime here decodes. `audioType` is read off the checkpoint by the backend, so it holds
 *  even where nothing in the path says "csm". */
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

/** Whether an audio pick from the chat picker may be routed to the Audio page. The page's own
 *  lists apply `communityAudioRowIsRunnable`, so routing a repo that fails it lands on a page
 *  that cannot show the row, and its `?model=` handoff evicts the chat model first. Curated
 *  ids always route: the catalog, not the tag, is their runtime contract. */
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
  /** Trained or exported here, so its codec was read off the checkpoint itself. */
  isLocalCheckpoint?: boolean;
  /** Filesystem inventory row: its task came from reading the GGUF's own architecture. */
  taskFromGgufArch?: boolean;
  baseModel?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
  audioType?: string | null;
}): boolean {
  // GGUF speech tasks have two provenances: Orpheus retains the ordinary llama architecture and
  // runs on Audio's SNAC path, while the dedicated CSM speech architectures are unsupported.
  // Unknown old-backend rows fail closed.
  if (taskFromGgufArch && isGguf && task === "text-to-speech") {
    const codec = (audioType ?? "").toLowerCase();
    if (codec === "csm" || !codec) return false;
    return ["snac", "bicodec", "dac"].includes(codec);
  }
  if (isCurated) return true;
  // A checkpoint from outputs/ has no Hub identity to judge, and the family-name heuristic would
  // reject it on its directory name. Its task came from the backend reading the checkpoint,
  // the stronger signal, and the Audio page lists it off that same tag.
  if (isLocalCheckpoint) {
    // Provenance says nothing about the decoder: a CSM GGUF found on disk is as unrunnable as a
    // cached one, and routing it hands Audio a row it cannot show after the handoff already
    // evicted the chat model.
    if (speechGgufIsUndecodable({ isGguf, id, baseModel, tags })) return false;
    return (
      task === "text-to-speech" || task === "automatic-speech-recognition"
    );
  }
  // The same Hub evidence the Audio page's own lists judge on. Passing the id alone rejected a
  // checkpoint whose family is in its tags or base model rather than its name, so the page
  // listed it but the chat picker refused to route.
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

/** The macOS audio runtime can execute TTS only through llama.cpp GGUF. Curated families may
 *  expose a non-GGUF canonical row because Audio resolves it to a published GGUF sibling. */
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

/** A downloaded GGUF often exposes only its base architecture (Orpheus reports `llama`), so
 *  generic inventory correctly calls it chat. An exact curated Audio artifact is a stronger
 *  contract, but only for its own active Audio mode. */
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

/** A cached Audio GGUF can be classified from its base architecture as text-generation. Only
 *  an exact catalog artifact may replace that fallback when Chat decides which page owns it. */
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
  // Such a GGUF still carries an ordinary text-to-image tag on the Hub. The on-device verdict is
  // what the loader enforces, so it outranks the tag: trusting the tag routes the pick at a
  // page whose picker omits the row and whose load would be refused.
  if (inventoryTask === UNSUPPORTED_DIFFUSION_TASK) return inventoryTask;
  // Cache inventory commonly reports Audio GGUFs as generic text-generation; an exact catalog
  // task is the stronger runtime contract there.
  return pipelineTag && pipelineTag !== "text-generation"
    ? pipelineTag
    : (inventoryTask ?? pipelineTag ?? null);
}

/** Filesystem checkpoints cannot be served by the STT sidecars yet. Keep cached Hub snapshots
 *  and curated artifacts visible, but not local-directory rows that would send an absolute
 *  path to the Hub-only API. */
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
