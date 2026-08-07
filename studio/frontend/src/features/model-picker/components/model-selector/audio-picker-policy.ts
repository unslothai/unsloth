// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FormatFilter } from "./recommended-fit";
import type { ModelSelectorChangeMeta } from "./types";

/** Community audio checkpoints can be discovered explicitly without treating
 * every Hub result as a runtime-supported recommendation. */
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

/** The macOS audio runtime can execute TTS only through llama.cpp GGUF. Curated
 * families may expose a non-GGUF canonical row because Audio resolves that row
 * to its published GGUF sibling before loading. */
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

/** Curated task artifacts are explicit runtime contracts. The default All view
 * must show them on every platform; an explicit format choice still filters. */
export function taskCatalogFormatMatches(
  format: FormatFilter,
  matchesFormat: boolean,
): boolean {
  return format === "all" || matchesFormat;
}

/** Curated task seeds are already an explicit page/runtime contract. They do
 * not need a live Hub pipeline tag and default All must not device-hide them. */
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

/** A downloaded GGUF often exposes only its base architecture (for example,
 * Orpheus reports `llama`), so generic inventory correctly calls it chat. An
 * exact curated Audio artifact is a stronger page/runtime contract, but only
 * for its own active Audio mode. */
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

/** A cached/local Audio GGUF can be classified from its base architecture as
 * text-generation. Only an exact catalog artifact is allowed to replace that
 * generic fallback when Chat decides which task page should own the pick. */
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

/** Fresh Hub rows carry their authoritative task in selection metadata, while
 * downloaded/local rows retain their inventory task as a fallback. */
export function taskForMediaPick(
  pipelineTag: string | null | undefined,
  inventoryTask: string | null | undefined,
): string | null {
  return pipelineTag ?? inventoryTask ?? null;
}

/** GGUF variants are selected after the Hub row, so carry its task through the
 * expander instead of losing routing information at the quant click. */
export function withPipelineTag(
  meta: ModelSelectorChangeMeta,
  pipelineTag: string | null | undefined,
): ModelSelectorChangeMeta {
  return pipelineTag ? { ...meta, pipelineTag } : meta;
}
