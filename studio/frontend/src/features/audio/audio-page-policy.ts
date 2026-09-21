// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelSelectorChangeMeta } from "@/features/model-picker/components/model-selector/types";
import { nativeAudioCheckpointIsLoadable } from "../model-picker/components/model-selector/audio-picker-policy.ts";

export type AudioBusy =
  | "loading"
  | "unloading"
  | "generating"
  | "transcribing"
  | null;

export type AudioGenerationPhase =
  | "preparing"
  | "generating"
  | "stopping"
  | "finishing"
  | null;

export type AudioGenerationPresentation = {
  status: string;
  actionLabel: string;
  canStop: boolean;
};

/** Project request-lifetime phases into truthful UI copy. Audio generation has no
 *  browser-visible numeric progress, so these labels never imply a fraction or ETA. */
export function audioGenerationPresentation(
  phase: AudioGenerationPhase,
): AudioGenerationPresentation | null {
  switch (phase) {
    case "preparing":
      return {
        status: "Preparing audio…",
        actionLabel: "Preparing…",
        canStop: false,
      };
    case "generating":
      return {
        status: "Generating audio…",
        actionLabel: "Stop",
        canStop: true,
      };
    case "stopping":
      return {
        status: "Stopping audio…",
        actionLabel: "Stopping…",
        canStop: false,
      };
    case "finishing":
      return {
        status: "Finishing audio…",
        actionLabel: "Finishing…",
        canStop: false,
      };
    case null:
      return null;
  }
}

export type AudioPickTask = "tts" | "stt" | null;
export type AudioCreateMode = "speak" | "transcribe";
export type SttEngine = "transformers" | "gguf" | "mtmd";

const TTS_AUDIO_TYPES = new Set([
  "snac",
  "csm",
  "bicodec",
  "dac",
  "higgs_tts2",
  "moss_tts_local",
  "moss_tts_nano",
  "higgs_tts3",
  "minimax_music3",
]);
const GGUF_TTS_AUDIO_TYPES = new Set(["snac", "bicodec", "dac"]);
const NATIVE_TTS_AUDIO_TYPES = new Set([
  "higgs_tts2",
  "moss_tts_local",
  "moss_tts_nano",
  "higgs_tts3",
  "minimax_music3",
]);
export const MOSS_TTS_FRAMES_PER_SECOND = 12.5;
export const MOSS_TTS_DEFAULT_SECONDS = 15;
export const MOSS_TTS_MAX_FRAMES = 32768;
export const MOSS_TTS_MAX_SECONDS =
  MOSS_TTS_MAX_FRAMES / MOSS_TTS_FRAMES_PER_SECOND;
export const MINIMAX_MUSIC_FRAMES_PER_SECOND = 25;
export const MINIMAX_MUSIC_DEFAULT_SECONDS = 30;
export const MINIMAX_MUSIC_MAX_FRAMES = 9000;
export const MINIMAX_MUSIC_MAX_SECONDS =
  MINIMAX_MUSIC_MAX_FRAMES / MINIMAX_MUSIC_FRAMES_PER_SECOND;

export type NativeAudioInstructionsKind = "scene" | "style" | "music";

export function nativeAudioInstructionsKind(
  audioType?: string | null,
): NativeAudioInstructionsKind | null {
  if (audioType === "higgs_tts2") {
    return "scene";
  }
  if (audioType === "moss_tts_local") {
    return "style";
  }
  if (audioType === "minimax_music3") {
    return "music";
  }
  return null;
}

export function minimaxMusicFramesForSeconds(seconds: number): number {
  const exactFrames = seconds * MINIMAX_MUSIC_FRAMES_PER_SECOND;
  return Math.min(
    MINIMAX_MUSIC_MAX_FRAMES,
    Math.max(
      1,
      Math.floor(
        exactFrames + Number.EPSILON * Math.max(1, Math.abs(exactFrames)),
      ),
    ),
  );
}

export function mossTtsFramesForSeconds(
  seconds: number,
  maxFrames = MOSS_TTS_MAX_FRAMES,
): number {
  return Math.min(
    Math.max(1, Math.floor(maxFrames)),
    Math.max(1, Math.floor(seconds * MOSS_TTS_FRAMES_PER_SECOND)),
  );
}

export function mossTtsMaxFrames(
  audioType?: string | null,
  contextLength?: number | null,
): number | null {
  if (audioType !== "moss_tts_local" && audioType !== "moss_tts_nano") {
    return null;
  }
  const detected = Math.floor(Number(contextLength));
  return Number.isFinite(detected) && detected > 0
    ? detected
    : MOSS_TTS_MAX_FRAMES;
}

export function isTtsAudioType(
  audioType?: string | null,
  isGguf = false,
): boolean {
  return Boolean(
    audioType &&
      (isGguf
        ? GGUF_TTS_AUDIO_TYPES.has(audioType)
        : TTS_AUDIO_TYPES.has(audioType)),
  );
}

export function trainedTtsCheckpointIsLoadable(
  audioType?: string | null,
  exportType?: string | null,
): boolean {
  return nativeAudioCheckpointIsLoadable(audioType, exportType);
}

export function trainedTtsCheckpointIsRunnableOnMac(
  audioType?: string | null,
  exportType?: string | null,
): boolean {
  if (exportType === "gguf") return true;
  return Boolean(
    exportType === "merged" &&
      audioType &&
      NATIVE_TTS_AUDIO_TYPES.has(audioType) &&
      audioType !== "minimax_music3",
  );
}

type SttDownloadedStatus = {
  downloaded_models?: readonly string[];
  transformers?: { downloaded_models?: readonly string[] };
  gguf?: { downloaded_models?: readonly string[] };
  mtmd?: { downloaded_models?: readonly string[] };
};

export interface SttDownloadedArtifact {
  repoId: string;
  sidecarKey: string;
  engine: SttEngine;
}

/** Engine-qualified picker artifacts for every locally loadable checkpoint. Whisper uses one
 *  short key for distinct Transformers and whisper.cpp downloads, so engine provenance must
 *  survive this boundary. */
export function sttDownloadedArtifacts(
  status: SttDownloadedStatus,
  repoIdForSidecarKey: (sidecarKey: string, engine: SttEngine) => string,
): SttDownloadedArtifact[] {
  const seen = new Set<string>();
  const artifacts: SttDownloadedArtifact[] = [];
  const blocks: readonly [SttEngine, SttDownloadedStatus | undefined][] = [
    ["transformers", status],
    ["transformers", status.transformers],
    ["gguf", status.gguf],
    ["mtmd", status.mtmd],
  ];
  for (const [engine, block] of blocks) {
    for (const sidecarKey of block?.downloaded_models ?? []) {
      const repoId = repoIdForSidecarKey(sidecarKey, engine);
      const normalized = repoId.trim().toLowerCase();
      if (!normalized || seen.has(normalized)) continue;
      seen.add(normalized);
      artifacts.push({ repoId, sidecarKey, engine });
    }
  }
  return artifacts;
}

/** Catalog contracts win; uncurated ASR rows route from the inventory task in picker metadata.
 *  Unknown and TTS-tagged community repos keep the main-slot path for load-time validation. */
export function resolveAudioPickTask(
  catalogTask: AudioPickTask,
  pipelineTag?: string | null,
): AudioPickTask {
  return (
    catalogTask ??
    (pipelineTag === "automatic-speech-recognition" ? "stt" : null)
  );
}

/** Generation can be cancelled as part of a mode transition. Model lifecycle and transcription
 *  operations must settle before their controls disappear. */
export function canTransitionAudioMode(
  busy: AudioBusy,
  generationPhase: AudioGenerationPhase = busy === "generating"
    ? "generating"
    : null,
): boolean {
  return (
    busy === null || (busy === "generating" && generationPhase === "generating")
  );
}

/** A managed TTS completion owns auto-load only while the same staging generation is selected
 *  in Speak. Downloads continue globally after ownership changes, but their completion must
 *  not mutate the main slot. */
export function stagedTtsLoadIsOwned(
  pendingGeneration: number | null,
  currentGeneration: number,
  mode: AudioCreateMode,
): boolean {
  return (
    pendingGeneration !== null &&
    pendingGeneration === currentGeneration &&
    mode === "speak"
  );
}

/** Cached rows sometimes know only the quant label; remote staging always supplies the exact
 *  filename. Both are valid backend GGUF selectors. */
export function exactGgufLoadSelector(
  meta: Pick<ModelSelectorChangeMeta, "ggufFilename" | "ggufVariant">,
): string | null {
  return meta.ggufFilename ?? meta.ggufVariant ?? null;
}

/** Whether a TTS pick loads through llama.cpp.
 *
 * A direct .gguf file and a GGUF repo id carry no variant filename, so the selector
 * alone misses both. `meta.isGguf` wins where a caller has it; this covers the rest.
 */
export function isGgufTtsTarget({
  repoId,
  ggufFilename,
  loadId,
  isGguf,
}: {
  repoId: string;
  ggufFilename?: string | null;
  loadId?: string | null;
  /** The catalog's own answer, when the caller has one. The tests below are
   * name heuristics, blind to a GGUF repo whose ids do not spell it. */
  isGguf?: boolean | null;
}): boolean {
  const endsWithGguf = (value: string | null | undefined): boolean =>
    Boolean(value?.toLowerCase().endsWith(".gguf"));
  return Boolean(
    isGguf ||
      ggufFilename ||
      /(?:^|[-/])gguf(?:$|[-/])/i.test(repoId) ||
      endsWithGguf(repoId) ||
      endsWithGguf(loadId),
  );
}

export type MacTtsPickAction = "allow" | "use-gguf-sibling" | "reject";

/** MLX has no codec TTS decoder. Curated native PyTorch audio models bypass MLX; other Mac
 *  picks still need GGUF or a family GGUF sibling. */
export function macTtsPickAction({
  isMac,
  isGguf,
  ggufSibling,
  nativeRuntime = false,
}: {
  isMac: boolean;
  isGguf: boolean;
  ggufSibling: string | null;
  nativeRuntime?: boolean;
}): MacTtsPickAction {
  if (!isMac || isGguf || nativeRuntime) return "allow";
  return ggufSibling ? "use-gguf-sibling" : "reject";
}

export interface AutoGgufVariant {
  filename: string;
  quant: string;
  size_bytes: number;
  download_size_bytes?: number;
  downloaded?: boolean;
  partial?: boolean;
}

/** Prefer a complete cached quant, then the repo's declared default, then the first exact file:
 *  instant Mac fallback when a runnable sibling is present, deterministic otherwise. */
export function selectAutoGgufVariant<T extends AutoGgufVariant>(
  variants: readonly T[],
  defaultVariant: string | null | undefined,
): T | null {
  const exact = variants.filter(
    (variant) => variant.filename.trim().length > 0,
  );
  if (exact.length === 0) return null;
  const downloaded = exact.find(
    (variant) => variant.downloaded === true && variant.partial !== true,
  );
  if (downloaded) return downloaded;
  const normalizedDefault = defaultVariant?.trim().toLowerCase();
  if (normalizedDefault) {
    const preferred = exact.find(
      (variant) =>
        variant.quant.trim().toLowerCase() === normalizedDefault ||
        variant.filename.trim().toLowerCase() === normalizedDefault,
    );
    if (preferred) return preferred;
  }
  return exact[0];
}

export function expectedGgufDownloadBytes(variant: AutoGgufVariant): number {
  const downloadBytes = variant.download_size_bytes;
  return typeof downloadBytes === "number" &&
    Number.isFinite(downloadBytes) &&
    downloadBytes > 0
    ? downloadBytes
    : variant.size_bytes;
}

/** Fold a freshly fetched first page into the list already on screen. The page is authoritative
 *  for the newest `page.length` clips and any scrollback below it is kept; replacing outright
 *  collapsed a paginated History on every delete and reselected a different clip.
 *  `removedId` drops a clip this client just deleted; `hasMore` is the server's own report. */
export function mergeGalleryPage<T extends { id: string }>(
  page: readonly T[],
  cached: readonly T[],
  removedId?: string,
  hasMore?: boolean,
): { clips: T[]; stitched: boolean } {
  const inPage = new Set(page.map((clip) => clip.id));
  // An empty page means the server holds nothing: a clear from anywhere, not scrollback.
  if (page.length === 0) return { clips: [], stitched: false };
  // A complete first page IS everything the server holds, so there is no scrollback to keep: a
  // cached clip below it was deleted by another client or pruned by the size cap, and
  // stitching it back rendered a row that could never be played again.
  if (hasMore === false) return { clips: [...page], stitched: false };
  // The page is authoritative over the window it covers, so a cached clip inside that window and
  // absent from the page was deleted by another client and must go. Only what sits BELOW the
  // page's oldest entry is scrollback, keyed on that entry's position.
  const oldestInPage = cached.findIndex(
    (clip) => clip.id === page[page.length - 1].id,
  );
  // Without that boundary the cache cannot prove where safe scrollback begins: an external
  // archive can shift one unseen row into the page while every earlier row still overlaps.
  if (oldestInPage === -1 && cached.length > 0) {
    return { clips: [...page], stitched: false };
  }
  const scrollback = oldestInPage === -1 ? cached : cached.slice(oldestInPage + 1);
  const tail = scrollback.filter(
    (clip) => !inPage.has(clip.id) && clip.id !== removedId,
  );
  return { clips: [...page, ...tail], stitched: tail.length > 0 };
}

/** Match the gallery record returned by this generation, never another client's concurrently persisted clip. */
export function persistedClipForGeneration<T extends { id: string }>(
  clipId: string | null | undefined,
  refreshed: readonly T[],
): T | null {
  return clipId ? (refreshed.find((clip) => clip.id === clipId) ?? null) : null;
}

export function sttSelectionReady(
  selectedRepo: string | null,
  loadedModel: string | null,
  sidecarKeyFor: (repoId: string) => string,
  selectedEngine?: SttEngine | null,
  loadedEngine?: SttEngine | null,
): boolean {
  return Boolean(
    selectedRepo &&
      loadedModel === sidecarKeyFor(selectedRepo) &&
      (!selectedEngine || !loadedEngine || selectedEngine === loadedEngine),
  );
}

type SttEngineResidency = {
  loaded_model?: string | null;
  loading?: boolean;
  available?: boolean;
};

type SttResidencyStatus = SttEngineResidency & {
  transformers?: SttEngineResidency;
  gguf?: SttEngineResidency;
  mtmd?: SttEngineResidency;
};

/** Resolve the resident model from the engine-aware status shape. The legacy top-level fields
 *  mirror Transformers only, so reading them for Qwen3-ASR or whisper.cpp clears a model that
 *  is actually ready. While the selected engine is pending, an older model on another engine
 *  must not steal the selector. */
export function resolveSttLoadedModel(
  status: SttResidencyStatus,
  selectedEngine: SttEngine | null,
  preserveSelected: boolean,
): string | null {
  return (
    resolveSttResidency(status, selectedEngine, preserveSelected)?.model ?? null
  );
}

export interface SttResidency {
  model: string;
  engine: SttEngine;
}

/** Resolve model and owning engine together; equal Whisper short keys do not identify which runtime is resident. */
export function resolveSttResidency(
  status: SttResidencyStatus,
  selectedEngine: SttEngine | null,
  preserveSelected: boolean,
): SttResidency | null {
  // A whisper.cpp pick on a host without whisper-server is deliberately served through
  // Transformers, so its residency lives in that block. The engine reported stays the selected
  // one, since that is what the user picked and what the backend routes; the same
  // sttEngineStatusFor fallback applies. Without this the refresh completing the load found
  // nothing, since it runs while preserveSelected is true, and Transcribe stayed disabled.
  const selectedStatus =
    selectedEngine === "transformers" ||
    (selectedEngine === "gguf" && status.gguf?.available === false)
      ? (status.transformers ?? status)
      : selectedEngine
        ? status[selectedEngine]
        : undefined;
  if (selectedStatus?.loaded_model && selectedEngine) {
    return { model: selectedStatus.loaded_model, engine: selectedEngine };
  }
  if (selectedEngine && (preserveSelected || selectedStatus?.loading)) {
    return null;
  }
  const transformersModel =
    status.transformers?.loaded_model ?? status.loaded_model;
  if (transformersModel)
    return { model: transformersModel, engine: "transformers" };
  if (status.gguf?.loaded_model) {
    return { model: status.gguf.loaded_model, engine: "gguf" };
  }
  if (status.mtmd?.loaded_model) {
    return { model: status.mtmd.loaded_model, engine: "mtmd" };
  }
  return null;
}

/** Reconcile the picker selection with the sidecar's authoritative status. Preserve a selection
 *  only while its load/download is genuinely pending. */
export function reconcileSttSelection({
  selectedRepo,
  loadedModel,
  loadedEngine,
  preservePending,
  sidecarKeyFor,
  repoIdForSidecarKey,
  engineForRepo,
}: {
  selectedRepo: string | null;
  loadedModel: string | null;
  loadedEngine?: SttEngine | null;
  preservePending: boolean;
  sidecarKeyFor: (repoId: string) => string;
  repoIdForSidecarKey: (sidecarKey: string, engine?: SttEngine) => string;
  engineForRepo?: (repoId: string) => SttEngine;
}): string | null {
  if (loadedModel) {
    if (
      selectedRepo &&
      sidecarKeyFor(selectedRepo) === loadedModel &&
      (!loadedEngine ||
        !engineForRepo ||
        engineForRepo(selectedRepo) === loadedEngine)
    ) {
      return selectedRepo;
    }
    return repoIdForSidecarKey(loadedModel, loadedEngine ?? "transformers");
  }
  return preservePending ? selectedRepo : null;
}

/** Permission prompts cannot be aborted, so freshness is checked immediately after
 *  getUserMedia resolves and stale streams are stopped before recording. */
export function micStreamRequestIsCurrent(
  requestGeneration: number,
  currentGeneration: number,
  active: boolean,
): boolean {
  return active && requestGeneration === currentGeneration;
}
