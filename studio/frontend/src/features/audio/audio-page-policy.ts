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

/** Engine-qualified picker artifacts for every checkpoint the sidecars can
 * load locally. Whisper uses the same short key for distinct Transformers and
 * whisper.cpp downloads, so engine provenance must survive this boundary. */
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

/** Catalog contracts win; uncurated on-device/community ASR rows route from
 * the inventory task preserved in picker metadata. Unknown and TTS-tagged
 * community repos keep the main-slot path for load-time capability validation. */
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

/** A managed TTS completion still owns auto-load only while the same staging
 * generation remains selected in Speak. Downloads may continue globally after
 * ownership changes, but their completion must not mutate the main slot. */
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

/** Prefer a complete cached quant, then the repo's declared default, then the
 * first exact file. This keeps Mac fallback instant when any runnable sibling
 * is already present while still making an uncached choice deterministic. */
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
};

type SttResidencyStatus = SttEngineResidency & {
  transformers?: SttEngineResidency;
  gguf?: SttEngineResidency;
  mtmd?: SttEngineResidency;
};

/** Resolve the resident model from the engine-aware status shape. The legacy
 * top-level fields mirror Transformers only, so reading them for Qwen3-ASR or
 * whisper.cpp clears a model that is actually ready. While the selected engine
 * is pending, do not let an older model on another engine steal the selector. */
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
  const selectedStatus =
    selectedEngine === "transformers"
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
