// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  TRANSLATION_ONLY_REPOS,
  isTranslationOnlyRepo,
} from "../../lib/translation-models.ts";
import {
  type TranslateMessage,
  type TranslatePromptOptions,
  generalInstruction,
  generalMessages,
  hyMtInstruction,
  hyMtMessages,
} from "./translate-text.ts";

export type TranslateSampling = {
  temperature: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
};

export type DedicatedTranslateModel = {
  id: "hy-mt2-1.8b" | "hy-mt2-7b";
  label: string;
  repo: string;
  variant: string;
  /** Size of that quant on the Hub, shown before the download starts. */
  sizeBytes: number;
  sampling: TranslateSampling;
};

// Sampling is the Hy-MT2 model card's recommendation for the 1.8B and 7B models.
export const HY_MT2_SAMPLING: TranslateSampling = {
  temperature: 0.7,
  top_p: 0.6,
  top_k: 20,
  repetition_penalty: 1.05,
};

export const DEDICATED_TRANSLATE_MODELS: readonly DedicatedTranslateModel[] = [
  {
    id: "hy-mt2-1.8b",
    label: "Hy-MT2 1.8B",
    repo: TRANSLATION_ONLY_REPOS[0],
    variant: "Q4_K_M",
    sizeBytes: 1_133_081_568,
    sampling: HY_MT2_SAMPLING,
  },
  {
    id: "hy-mt2-7b",
    label: "Hy-MT2 7B",
    repo: TRANSLATION_ONLY_REPOS[1],
    variant: "Q4_K_M",
    sizeBytes: 4_624_650_016,
    sampling: HY_MT2_SAMPLING,
  },
];

/** "chat" translates with whatever model Chat has loaded and downloads nothing. */
export const CHAT_TRANSLATE_MODEL = "chat";
/** "custom" is the one other model the user picked in the model picker. */
export const CUSTOM_TRANSLATE_MODEL = "custom";
export type TranslateModelId =
  | DedicatedTranslateModel["id"]
  | typeof CHAT_TRANSLATE_MODEL
  | typeof CUSTOM_TRANSLATE_MODEL;

/** A model from the picker: a Hub repo, a local file, or a connected provider's model. */
export type CustomTranslateModel = {
  id: string;
  name: string;
  /** The GGUF quant to fetch and load; null when the model has none. */
  variant: string | null;
  sizeBytes: number | null;
};

export function parseCustomTranslateModel(
  value: unknown,
): CustomTranslateModel | null {
  if (!value || typeof value !== "object") return null;
  const { id, name, variant, sizeBytes } = value as Record<string, unknown>;
  if (typeof id !== "string" || id.trim() === "") return null;
  return {
    id,
    name: typeof name === "string" && name ? name : id,
    variant: typeof variant === "string" && variant ? variant : null,
    sizeBytes:
      typeof sizeBytes === "number" && sizeBytes > 0 ? sizeBytes : null,
  };
}

/** What tells one selection from another: the custom slot holds a different model over time. */
export function translateModelKey(
  id: TranslateModelId,
  custom: CustomTranslateModel | null,
): string {
  return id === CUSTOM_TRANSLATE_MODEL
    ? `${id}:${custom?.id ?? ""}:${custom?.variant ?? ""}`
    : id;
}

/** The built-in model whose prompt a selection takes: Hy-MT2's for any Hy-MT2 quant, else a chat model's. */
export function translatePromptModel(
  id: TranslateModelId,
  custom: CustomTranslateModel | null,
): Exclude<TranslateModelId, typeof CUSTOM_TRANSLATE_MODEL> {
  if (id !== CUSTOM_TRANSLATE_MODEL) return id;
  return isTranslationOnlyRepo(custom?.id)
    ? DEFAULT_TRANSLATE_MODEL
    : CHAT_TRANSLATE_MODEL;
}

export const DEFAULT_TRANSLATE_MODEL = "hy-mt2-1.8b" satisfies TranslateModelId;

/** Both passages of a chunk fit beside the instruction, with room to spare. */
export const TRANSLATE_CONTEXT_TOKENS = 8192;
export const TRANSLATE_MAX_TOKENS = 4096;

export function isTranslateModelId(value: unknown): value is TranslateModelId {
  return (
    value === CHAT_TRANSLATE_MODEL ||
    value === CUSTOM_TRANSLATE_MODEL ||
    DEDICATED_TRANSLATE_MODELS.some((model) => model.id === value)
  );
}

export function dedicatedTranslateModel(
  id: TranslateModelId,
): DedicatedTranslateModel | null {
  return DEDICATED_TRANSLATE_MODELS.find((model) => model.id === id) ?? null;
}

export function translateMessages(
  id: TranslateModelId,
  text: string,
  options: TranslatePromptOptions,
): TranslateMessage[] {
  return id === CHAT_TRANSLATE_MODEL || id === CUSTOM_TRANSLATE_MODEL
    ? generalMessages(text, options)
    : hyMtMessages(text, options);
}

/** The two kinds of model take different instructions, so each keeps its own edited prompt. */
export type TranslatePromptKind = "dedicated" | "chat";

export function translatePromptKind(id: TranslateModelId): TranslatePromptKind {
  return id === CHAT_TRANSLATE_MODEL || id === CUSTOM_TRANSLATE_MODEL
    ? "chat"
    : "dedicated";
}

/** The instructions as sent, without the text, for the Show prompt option. */
export function translateInstruction(
  id: TranslateModelId,
  options: TranslatePromptOptions,
): string {
  return id === CHAT_TRANSLATE_MODEL || id === CUSTOM_TRANSLATE_MODEL
    ? generalInstruction(options)
    : hyMtInstruction(options);
}

export function formatModelBytes(bytes: number): string {
  if (!(bytes > 0)) return "";
  const gb = bytes / 1000 ** 3;
  return gb >= 1
    ? `${gb.toFixed(1)} GB`
    : `${Math.max(1, Math.round(bytes / 1000 ** 2))} MB`;
}

/** A downloaded model offered under "Your models": a complete GGUF that generates text. */
export type DownloadedTranslateModel = {
  id: string;
  name: string;
  sizeBytes: number | null;
  /** In the Hugging Face cache, so the quant on disk has to be looked up before loading it. */
  fromHub: boolean;
};

type InventoryRow = {
  id: string;
  model_id?: string | null;
  display_name: string;
  source: string;
  model_format?: string | null;
  runtime?: string | null;
  size_bytes?: number;
  partial?: boolean;
  task?: string | null;
  audio_type?: string | null;
  updated_at?: number | null;
};

export const DOWNLOADED_MODELS_SHOWN = 6;

export function downloadedTranslateModels(
  rows: readonly InventoryRow[],
): DownloadedTranslateModel[] {
  const seen = new Set<string>();
  return rows
    .filter(
      (row) =>
        !row.partial &&
        (row.model_format === "gguf" || row.runtime === "llama_cpp") &&
        !row.audio_type &&
        (!row.task || row.task === "text-generation") &&
        !isTranslationOnlyRepo(row.model_id ?? row.id),
    )
    .sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0))
    .map((row) => ({
      id: row.model_id ?? row.id,
      name: row.display_name.replace(/-GGUF$/i, ""),
      sizeBytes: row.size_bytes && row.size_bytes > 0 ? row.size_bytes : null,
      fromHub: row.source === "hf_cache",
    }))
    .filter((model) => !seen.has(model.id) && seen.add(model.id))
    .slice(0, DOWNLOADED_MODELS_SHOWN);
}

/** The quant to load from what is on disk: Q4_K_M when there is a choice. */
export function downloadedQuant(
  variants: readonly {
    quant: string;
    size_bytes: number;
    downloaded?: boolean;
    partial?: boolean;
  }[],
): { quant: string; sizeBytes: number } | null {
  const onDisk = variants.filter((v) => v.downloaded === true && !v.partial);
  const best =
    onDisk.find((v) => v.quant.toUpperCase() === "Q4_K_M") ?? onDisk[0];
  return best ? { quant: best.quant, sizeBytes: best.size_bytes } : null;
}

/** Downloaded models that are not text models (image, video, audio), by lowercased id. */
export function nonTextModelIds(rows: readonly InventoryRow[]): Set<string> {
  const ids = new Set<string>();
  for (const row of rows) {
    if (row.audio_type || (row.task && row.task !== "text-generation")) {
      ids.add(row.id.toLowerCase());
      if (row.model_id) ids.add(row.model_id.toLowerCase());
    }
  }
  return ids;
}
