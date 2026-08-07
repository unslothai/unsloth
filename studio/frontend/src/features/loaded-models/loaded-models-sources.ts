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
};

/** The per-engine half of /api/inference/audio/stt/status. */
export type SttEngineStatus = {
  loaded_model?: string | null;
  device?: string | null;
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

export const LOADED_MODEL_KIND_LABELS: Record<LoadedModelKind, string> = {
  text: "Chat",
  tts: "Speech",
  image: "Image",
  video: "Video",
  stt: "Dictation",
};

/** Join only the known parts, so no row shows a stray separator. */
function joinDetail(...parts: (string | null | undefined)[]): string {
  return parts.filter((part): part is string => Boolean(part)).join(" · ");
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
    const isTts = Boolean(status.is_audio) && audioType !== "whisper";
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
        status.device,
      ),
    },
  ];
}

/** A server predating the engine split reports the resident Transformers model
 *  only at the top level, so fall back to it rather than showing no row. */
function sttEngineStatus(
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
