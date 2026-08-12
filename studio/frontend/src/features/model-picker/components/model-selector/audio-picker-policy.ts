// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FormatFilter } from "./recommended-fit";
import type { ModelSelectorChangeMeta } from "./types";

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

/** Community ASR runs through the Transformers Whisper sidecar. Curated GGUF/MTMD
 * artifacts are handled by the catalog before this gate, so an uncurated row must
 * identify a non-GGUF Whisper checkpoint. */
export function communityAudioRowIsRunnable({
  isStt,
  isTts,
  isGguf,
  id,
  baseModel,
  tags,
  libraryName,
}: {
  isStt: boolean;
  isTts: boolean;
  isGguf: boolean;
  id: string;
  baseModel?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
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

  // The main-slot TTS backend decodes only the four codec families below.
  // Hub's text-to-speech tag also covers Bark, VITS, SpeechT5, and many other
  // architectures that can load as language models but cannot emit a WAV here.
  // Llasa is NOT here despite being a well-known TTS family: it speaks XCodec2, which
  // AudioCodecManager cannot decode, so admitting it produced a row that loaded and then
  // failed at generation. The list and the comment above must stay in step.
  const family = evidence.find((value) =>
    /(?:^|[-_./])(orpheus|csm|spark-?tts|outetts|oute-?tts)(?:$|[-_./])/.test(value),
  );
  if (!family) return false;
  // llama.cpp intentionally has no CSM decoder; CSM is Transformers-only.
  return !(isGguf && /(?:^|[-_./])csm(?:$|[-_./])/.test(family));
}

/** Whether an audio pick from the chat picker may be routed to the Audio page.
 *
 * The page's own lists apply `communityAudioRowIsRunnable`, so routing a repo that fails
 * it lands on a page that cannot show the row, and its `?model=` handoff evicts the chat
 * model before reporting the repo unsupported. Curated ids always route: the catalog,
 * not the tag, is their runtime contract. */
export function audioPickIsRoutable({
  id,
  task,
  isGguf,
  isCurated,
  isLocalCheckpoint = false,
  baseModel,
  tags,
  libraryName,
}: {
  id: string;
  task: string | null | undefined;
  isGguf: boolean;
  isCurated: boolean;
  /** Trained or exported here, so its codec was read off the checkpoint itself. */
  isLocalCheckpoint?: boolean;
  baseModel?: string | null;
  tags?: readonly string[] | null;
  libraryName?: string | null;
}): boolean {
  if (isCurated) return true;
  // A checkpoint from outputs/ has no Hub identity for communityAudioRowIsRunnable to
  // judge, and the family-name heuristic it applies would reject it on its directory
  // name. Its task came from the backend reading the checkpoint, which is the stronger
  // signal, and the Audio page lists it off that same tag.
  if (isLocalCheckpoint) {
    return (
      task === "text-to-speech" || task === "automatic-speech-recognition"
    );
  }
  // The same Hub evidence the Audio page's own lists judge on. Passing the id alone
  // rejected a checkpoint whose family is in its tags or base model rather than its
  // name, so the page listed it but the chat picker refused to route to it.
  return communityAudioRowIsRunnable({
    isStt: task === "automatic-speech-recognition",
    isTts: task === "text-to-speech",
    isGguf,
    id,
    baseModel,
    tags,
    libraryName,
  });
}

/** The macOS audio runtime can execute TTS only through llama.cpp GGUF. Curated
 * families may expose a non-GGUF canonical row because Audio resolves it to a published
 * GGUF sibling before loading. */
export function macTtsHubRowIsRunnable({
  isMac,
  isTts,
  isGguf,
  hasRunnableGgufSibling,
}: {
  isMac: boolean;
  isTts: boolean;
  isGguf: boolean;
  hasRunnableGgufSibling: boolean;
}): boolean {
  return !isMac || !isTts || isGguf || hasRunnableGgufSibling;
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

/** A downloaded GGUF often exposes only its base architecture (Orpheus reports
 * `llama`), so generic inventory correctly calls it chat. An exact curated Audio
 * artifact is a stronger contract, but only for its own active Audio mode. */
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

/** A cached Audio GGUF can be classified from its base architecture as text-generation.
 * Only an exact catalog artifact may replace that fallback when Chat decides which task
 * page owns the pick. */
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

/** Hidden infrastructure rows remain hidden unless the active task page passed
 * their exact normalized artifact id as an explicit runtime contract. */
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

/** Backend tag for a diffusion GGUF the Images backend cannot assemble; both pickers hide
 * these rows. */
const UNSUPPORTED_DIFFUSION_TASK = "image-diffusion-unsupported";

/** Fresh Hub rows carry their authoritative task in selection metadata, while
 * downloaded/local rows retain their inventory task as a fallback. */
export function taskForMediaPick(
  pipelineTag: string | null | undefined,
  inventoryTask: string | null | undefined,
): string | null {
  // Such a GGUF still carries an ordinary text-to-image tag on the Hub. The on-device
  // verdict is the one the loader enforces, so it outranks the tag: trusting the tag routes
  // the pick at a page whose picker omits the row and whose load would be refused.
  if (inventoryTask === UNSUPPORTED_DIFFUSION_TASK) return inventoryTask;
  // Cache inventory commonly reports Audio GGUFs as generic text-generation;
  // an exact catalog task is the stronger runtime contract in that case.
  return pipelineTag && pipelineTag !== "text-generation"
    ? pipelineTag
    : (inventoryTask ?? pipelineTag ?? null);
}

/** Filesystem checkpoints cannot be served by the STT sidecars yet. Keep cached Hub
 * snapshots and curated artifacts visible, but not local-directory rows that would send
 * an absolute path to the Hub-only API. */
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
