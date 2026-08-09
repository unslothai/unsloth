// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Chat, images, video and dictation each own a runtime and their own /status, and
// nothing publishes "everything resident" in one call. These map each payload to a
// common row shape. Pure and React-free so node --test can import them; the types
// come from each feature's own module because the indexes re-export their pages.

import type { InferenceStatusResponse } from "@/features/chat/types/api";
import type { DiffusionStatus } from "@/features/images/api";
import type { VideoStatus } from "@/features/video/api";

/** What the row is, for the icon and label. */
export type LoadedModelKind = "text" | "tts" | "image" | "video" | "stt";

/** Which runtime holds the weights, so which endpoint releases them. */
export type LoadedModelSource = "chat" | "image" | "video" | "stt";

/** The dictation sidecars, as /audio/stt/status names them. */
export type SttEngine = "transformers" | "mtmd" | "gguf";

export type LoadedModelEntry = {
  /** Stable across polls, so a row does not remount mid-eject. */
  id: string;
  kind: LoadedModelKind;
  source: LoadedModelSource;
  /** The id the backend reports; also what the eject call names. */
  name: string;
  /** One short line: quantisation, family, device. */
  detail: string;
  /** Only on `source: "stt"`: its unload takes an engine, not a model id. */
  sttEngine?: SttEngine;
  /** Cached by the chat runtime but not active, so no status flags describe it. */
  inactive?: boolean;
  /** Still loading: shown with a spinner and no eject, since there is nothing
   *  resident to release yet. Matches the load toast rather than trailing it. */
  loading?: boolean;
};

/** The per-engine half of /api/inference/audio/stt/status. */
export type SttEngineStatus = {
  loaded_model?: string | null;
  device?: string | null;
  /** The sidecar is starting: reported per engine by /audio/stt/status. */
  loading?: boolean;
};

export type SttStatusResponse = SttEngineStatus & {
  transformers?: SttEngineStatus | null;
  mtmd?: SttEngineStatus | null;
  gguf?: SttEngineStatus | null;
};

const STT_ENGINE_LABELS: Record<SttEngine, string> = {
  transformers: "Transformers",
  mtmd: "llama.cpp",
  gguf: "whisper.cpp",
};

/**
 * Where a row's model is actually used, so clicking it goes there. Keyed on the
 * runtime holding the weights rather than the kind: a Whisper checkpoint in the
 * chat slot belongs to Chat, while the dictation sidecars belong to Voice.
 *
 * Dictation has no page of its own yet, so it opens the settings tab that
 * drives it. Point it at the Audio page once that lands.
 */
export type LoadedModelTarget =
  | { open: "route"; to: "/chat" | "/images" | "/video"; label: string }
  | { open: "settings"; tab: "voice"; label: string };

export function loadedModelTarget(source: LoadedModelSource): LoadedModelTarget {
  switch (source) {
    case "image":
      return { open: "route", to: "/images", label: "Images" };
    case "video":
      return { open: "route", to: "/video", label: "Video" };
    case "stt":
      return { open: "settings", tab: "voice", label: "Voice settings" };
    default:
      return { open: "route", to: "/chat", label: "Chat" };
  }
}

export const LOADED_MODEL_KIND_LABELS: Record<LoadedModelKind, string> = {
  text: "Chat",
  tts: "Speech",
  image: "Image",
  video: "Video",
  stt: "Dictation",
};

/**
 * Join the known parts, so no row shows a stray separator, and drop repeats:
 * the llama.cpp and whisper.cpp dictation sidecars report their engine name as
 * the device, which would otherwise print twice.
 */
function joinDetail(...parts: (string | null | undefined)[]): string {
  const seen = new Set<string>();
  const kept: string[] = [];
  for (const part of parts) {
    if (!part) continue;
    const key = part.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    kept.push(part);
  }
  return kept.join(" · ");
}

/** Keep a path's last two segments; a repo id is already short. */
export function shortModelLabel(name: string): string {
  const normalized = name.replace(/[\\/]+$/, "");
  const segments = normalized.split(/[\\/]+/).filter(Boolean);
  if (segments.length <= 2) return normalized;
  return segments.slice(-2).join("/");
}

/** Rows for whatever the chat runtime holds. `audio_type` splits TTS and whisper
 *  out of the chat row: same picker, but neither answers prompts. */
export function describeInferenceStatus(
  status: InferenceStatusResponse | null,
): LoadedModelEntry[] {
  if (!status) return [];
  const entries: LoadedModelEntry[] = [];
  const active = status.active_model;
  if (active) {
    const audioType = status.audio_type ?? null;
    // is_audio means TTS here, as mlx_inference documents ("audio_vlm (omni
    // audio input; is_audio stays False -- it means TTS and redirects in the
    // chat route)"). So the audio types split three ways, not two: whisper is
    // the ASR sidecar, audio_vlm is a chat model that happens to listen, and
    // the rest speak. Only the third kind belongs under Speech.
    const isTts =
      Boolean(status.is_audio) &&
      audioType !== "whisper" &&
      audioType !== "audio_vlm";
    const isStt = Boolean(status.is_audio) && audioType === "whisper";
    const runtime = status.is_gguf
      ? "GGUF"
      : status.is_mlx
        ? "MLX"
        : "Transformers";
    entries.push({
      id: `chat:${active}`,
      kind: isTts ? "tts" : isStt ? "stt" : "text",
      source: "chat",
      name: active,
      detail: joinDetail(
        runtime,
        status.gguf_variant,
        status.is_vision ? "Vision" : null,
      ),
    });
  }
  // Reported by /status for the whole load, so a load started in another tab or
  // before this page opened still shows, not only one driven from here.
  for (const name of status.loading ?? []) {
    if (entries.some((entry) => entry.name === name)) continue;
    entries.push({
      id: `chat:${name}`,
      kind: "text",
      source: "chat",
      name,
      detail: "Loading",
      loading: true,
    });
  }
  // Only the Transformers backend caches past the active model, but it is memory
  // nothing else surfaces or releases.
  for (const name of status.loaded ?? []) {
    if (name === active) continue;
    if (entries.some((entry) => entry.name === name)) continue;
    entries.push({
      id: `chat:${name}`,
      kind: "text",
      source: "chat",
      name,
      detail: "Still in memory",
      inactive: true,
    });
  }
  return entries;
}

/**
 * The precision a pipeline actually loaded at, short enough for the row. Images
 * report a torch dtype, video a resolved quantisation ("none" being plain
 * bf16). Anything unrecognised is shown as the backend spelled it, upper-cased.
 */
export function precisionLabel(value: string | null | undefined): string | null {
  if (!value || value === "none") return null;
  const known: Record<string, string> = {
    bfloat16: "BF16",
    float16: "FP16",
    float32: "FP32",
  };
  return known[value] ?? value.toUpperCase();
}

export function describeDiffusionStatus(
  status: DiffusionStatus | null,
): LoadedModelEntry[] {
  if (!status?.loaded || !status.repo_id) return [];
  return [
    {
      id: `image:${status.repo_id}`,
      kind: "image",
      source: "image",
      name: status.repo_id,
      detail: joinDetail(
        status.family,
        status.model_kind === "gguf" ? "GGUF" : null,
        // A GGUF load reports "gguf" here too; joinDetail drops the repeat.
        precisionLabel(status.dtype),
        status.device,
      ),
    },
  ];
}

export function describeVideoStatus(
  status: VideoStatus | null,
): LoadedModelEntry[] {
  if (!status?.loaded || !status.repo_id) return [];
  return [
    {
      id: `video:${status.repo_id}`,
      kind: "video",
      source: "video",
      name: status.repo_id,
      detail: joinDetail(
        status.family,
        status.model_kind === "gguf" ? "GGUF" : null,
        // The dense transformer's own precision when one engaged, since that is
        // what distinguishes the build; otherwise the pipeline dtype.
        precisionLabel(status.transformer_quant) ??
          precisionLabel(status.dtype),
        status.device,
      ),
    },
  ];
}

/** A server predating the engine split reports the resident Transformers model
 *  only at the top level, so fall back to it rather than showing no row. */
export function sttEngineStatus(
  status: SttStatusResponse,
  engine: SttEngine,
): SttEngineStatus | null {
  const block = status[engine];
  if (block) return block;
  return engine === "transformers"
    ? { loaded_model: status.loaded_model, device: status.device }
    : null;
}

export function describeSttStatus(
  status: SttStatusResponse | null,
): LoadedModelEntry[] {
  if (!status) return [];
  const engines: SttEngine[] = ["transformers", "mtmd", "gguf"];
  const entries: LoadedModelEntry[] = [];
  for (const engine of engines) {
    const block = sttEngineStatus(status, engine);
    if (block?.loading && !block.loaded_model) {
      entries.push({
        id: `stt:${engine}`,
        kind: "stt",
        source: "stt",
        name: STT_ENGINE_LABELS[engine],
        detail: "Loading",
        sttEngine: engine,
        loading: true,
      });
      continue;
    }
    if (!block?.loaded_model) continue;
    entries.push({
      id: `stt:${engine}`,
      kind: "stt",
      source: "stt",
      name: block.loaded_model,
      detail: joinDetail(STT_ENGINE_LABELS[engine], block.device),
      sttEngine: engine,
    });
  }
  return entries;
}

/**
 * What a row's model turned out to be when the eject re-read its runtime.
 * "replaced" is the one that matters: /images/unload, /video/unload and the STT
 * unload carry no model id and release whatever the runtime holds, so a row up
 * to one poll old must not be trusted to name it.
 */
export type ResidentVerdict = "match" | "gone" | "replaced";

export function verifyResident(
  entryName: string,
  resident: string | null | undefined,
  matches: (left: string, right: string) => boolean,
): ResidentVerdict {
  if (!resident) return "gone";
  return matches(entryName, resident) ? "match" : "replaced";
}

/** One list in a fixed group order, so rows never jump between polls. Ids are
 *  deduped defensively; two runtimes holding one repo id are two copies. */
export function mergeLoadedModels(
  groups: LoadedModelEntry[][],
): LoadedModelEntry[] {
  const seen = new Set<string>();
  const merged: LoadedModelEntry[] = [];
  for (const group of groups) {
    for (const entry of group) {
      if (seen.has(entry.id)) continue;
      seen.add(entry.id);
      merged.push(entry);
    }
  }
  return merged;
}

/**
 * Fold the loads announced by `withModelLoadNotice` into the polled rows.
 *
 * The poll is 5s, so a load that has only just started shows nothing while its
 * toast already says "loading". These rows come from the load call itself, and
 * a status row for the same runtime replaces them the moment one arrives: the
 * backend's answer always wins over the optimistic one.
 */
export function withPendingLoads(
  rows: LoadedModelEntry[],
  pending: Map<LoadedModelSource, string | null>,
): LoadedModelEntry[] {
  if (pending.size === 0) return rows;
  const extra: LoadedModelEntry[] = [];
  for (const [source, model] of pending) {
    // A status row wins only when it describes the same load. Images and video
    // keep the OLD pipeline resident while the replacement downloads, freeing it
    // only at the commit, so a source-only test hid the incoming model for the
    // whole pull: the card showed the model being replaced and no sign of the
    // one arriving. A row that is itself loading always wins, which is what
    // keeps chat and dictation from announcing the same load twice when the
    // backend spells its name differently.
    if (
      rows.some(
        (row) =>
          row.source === source &&
          (row.loading === true || model == null || row.name === model),
      )
    ) {
      continue;
    }
    extra.push({
      id: `${source}:pending`,
      kind: PENDING_KINDS[source],
      source,
      name: model ?? "Loading model",
      detail: "Loading",
      loading: true,
    });
  }
  return extra.length > 0 ? [...rows, ...extra] : rows;
}

const PENDING_KINDS: Record<LoadedModelSource, LoadedModelKind> = {
  chat: "text",
  image: "image",
  video: "video",
  stt: "stt",
};
