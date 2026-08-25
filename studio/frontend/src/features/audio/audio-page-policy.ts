// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelSelectorChangeMeta } from "@/features/model-picker/components/model-selector/types";

export type AudioBusy =
  | "loading"
  | "unloading"
  | "generating"
  | "transcribing"
  | null;

export type AudioPickTask = "tts" | "stt" | null;
export type AudioCreateMode = "speak" | "transcribe";
export type SttEngine = "transformers" | "gguf" | "mtmd";

const TTS_AUDIO_TYPES = new Set(["snac", "csm", "bicodec", "dac"]);
const GGUF_TTS_AUDIO_TYPES = new Set(["snac", "bicodec", "dac"]);

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

/** Engine-qualified picker artifacts for every locally loadable checkpoint. Whisper
 * uses one short key for distinct Transformers and whisper.cpp downloads, so engine
 * provenance must survive this boundary. */
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

/** Catalog contracts win; uncurated ASR rows route from the inventory task in picker
 * metadata. Unknown and TTS-tagged community repos keep the main-slot path for
 * load-time capability validation. */
export function resolveAudioPickTask(
  catalogTask: AudioPickTask,
  pipelineTag?: string | null,
): AudioPickTask {
  return (
    catalogTask ??
    (pipelineTag === "automatic-speech-recognition" ? "stt" : null)
  );
}

/** Generation can be cancelled as part of a mode transition. Model lifecycle
 * and transcription operations must settle before their controls disappear. */
export function canTransitionAudioMode(busy: AudioBusy): boolean {
  return busy === null || busy === "generating";
}

/** A managed TTS completion owns auto-load only while the same staging generation is
 * selected in Speak. Downloads continue globally after ownership changes, but their
 * completion must not mutate the main slot. */
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

/** Cached rows sometimes know only the quant label; remote staging always
 * supplies the exact filename. Both are valid backend GGUF selectors. */
export function exactGgufLoadSelector(
  meta: Pick<ModelSelectorChangeMeta, "ggufFilename" | "ggufVariant">,
): string | null {
  return meta.ggufFilename ?? meta.ggufVariant ?? null;
}

export type MacTtsPickAction = "allow" | "use-gguf-sibling" | "reject";

/** MLX has no TTS decoder. A Mac TTS pick is runnable only when it already is
 * GGUF or its curated family publishes a GGUF sibling. */
export function macTtsPickAction({
  isMac,
  isGguf,
  ggufSibling,
}: {
  isMac: boolean;
  isGguf: boolean;
  ggufSibling: string | null;
}): MacTtsPickAction {
  if (!isMac || isGguf) return "allow";
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

/** Prefer a complete cached quant, then the repo's declared default, then the first
 * exact file: instant Mac fallback when a runnable sibling is present, deterministic
 * otherwise. */
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

/** Fold a freshly fetched first page into the list already on screen.
 *
 * The page is authoritative for the newest `page.length` clips and any scrollback below
 * it is kept; replacing outright collapsed a paginated History on every delete and
 * generate, and reselected a different clip. `removedId` drops a clip this client just
 * deleted, which the page can no longer report. `hasMore` is the page's own report of
 * whether the server holds anything older. */
export function mergeGalleryPage<T extends { id: string }>(
  page: readonly T[],
  cached: readonly T[],
  removedId?: string,
  hasMore?: boolean,
): { clips: T[]; stitched: boolean } {
  const inPage = new Set(page.map((clip) => clip.id));
  // An empty page means the server holds nothing: a clear from anywhere, not scrollback.
  if (page.length === 0) return { clips: [], stitched: false };
  // A complete first page IS everything the server holds, so there is no scrollback to
  // keep: a cached clip below it was deleted by another client or pruned by the size cap,
  // and stitching it back rendered a row that could never be played again.
  if (hasMore === false) return { clips: [...page], stitched: false };
  // The page is authoritative over the window it covers, so a cached clip inside that
  // window and absent from the page was deleted by another client and must go. Only what
  // sits BELOW the page's oldest entry is scrollback. Keying on the position of that
  // entry, since a clip record carries no cursor of its own.
  const oldestInPage = cached.findIndex(
    (clip) => clip.id === page[page.length - 1].id,
  );
  // Without that boundary the cache cannot prove where safe scrollback begins: an external archive
  // can shift one unseen row into the page while every earlier row still overlaps.
  if (oldestInPage === -1 && cached.length > 0) {
    return { clips: [...page], stitched: false };
  }
  const scrollback = oldestInPage === -1 ? cached : cached.slice(oldestInPage + 1);
  const tail = scrollback.filter(
    (clip) => !inPage.has(clip.id) && clip.id !== removedId,
  );
  return { clips: [...page, ...tail], stitched: tail.length > 0 };
}

/** Match the gallery record returned by this generation, never another
 * client's concurrently persisted clip. */
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

/** Resolve the resident model from the engine-aware status shape. The legacy top-level
 * fields mirror Transformers only, so reading them for Qwen3-ASR or whisper.cpp clears a
 * model that is actually ready. While the selected engine is pending, do not let an
 * older model on another engine steal the selector. */
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

/** Resolve model and owning engine together; equal Whisper short keys do not
 * identify which runtime is resident. */
export function resolveSttResidency(
  status: SttResidencyStatus,
  selectedEngine: SttEngine | null,
  preserveSelected: boolean,
): SttResidency | null {
  // A whisper.cpp pick on a host without whisper-server is deliberately served, and
  // loaded, through Transformers, so its residency lives in that block. Same fallback
  // sttEngineStatusFor applies; the engine reported stays the selected one, because that
  // is what the user picked and what the backend routes. Without this the refresh that
  // completes the load found nothing (it runs while preserveSelected is true) and the
  // Transcribe controls stayed disabled until the page was left and revisited.
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

/** Reconcile the picker selection with the sidecar's authoritative status.
 * Preserve a selection only while its load/download is genuinely pending. */
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

/** Permission prompts cannot be aborted, so freshness is checked immediately
 * after getUserMedia resolves and stale streams are stopped before recording. */
export function micStreamRequestIsCurrent(
  requestGeneration: number,
  currentGeneration: number,
  active: boolean,
): boolean {
  return active && requestGeneration === currentGeneration;
}
