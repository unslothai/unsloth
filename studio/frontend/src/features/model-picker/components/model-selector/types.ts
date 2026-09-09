// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
import type { PerModelConfig } from "../../model-config/per-model-config";

export interface ModelOption {
  id: string;
  name: string;
  description?: string;
  icon?: ReactNode;
  isGguf?: boolean;
  /** Fixed quant used by a specialized on-device runtime. Generic Hub GGUF rows discover their
   *  variants dynamically instead. */
  deviceQuant?: string;
  /** Fallback metadata for task-owned caches that an older generic inventory cannot describe.
   *  Current servers normally provide a full cached row. */
  deviceSize?: string;
  deviceSizeBytes?: number;
  deviceLoaded?: boolean;
  /** Detected local audio architecture, retained by task-owned on-device rows. */
  audioType?: string | null;
}

export interface LoraModelOption extends ModelOption {
  baseModel?: string;
  updatedAt?: number;
  source?: "training" | "exported" | "local";

  /** This local GGUF is one directly loadable artifact, not a repo whose quant variants must be listed first. */
  isDirectGguf?: boolean;
  exportType?: "lora" | "merged" | "gguf";
  /** Codec when the checkpoint fine-tunes an audio model, else null. */
  audioType?: string | null;
}

export interface ExternalModelOption extends ModelOption {
  providerId: string;
  providerName: string;
  /** Registry key (e.g. openai, gemini) for provider branding. */
  providerType: string;
}

export interface ModelSelectorChangeMeta {
  source: "hub" | "lora" | "exported" | "local" | "external";
  isLora: boolean;
  ggufVariant?: string;
  /** Exact GGUF filename for the picked quant; filenames do not always follow the repo name
   *  (FLUX.1-schnell -> flux1-schnell-*.gguf). */
  ggufFilename?: string;
  isDownloaded?: boolean;
  expectedBytes?: number;
  /** Native GGUF context, threaded so a staged pick can seed the slider. */
  contextLength?: number | null;
  /** Direct local .gguf file picked without a variant (custom folder / LM Unsloth). Marks it as a
   *  GGUF source for the deferred-load staging flow. */
  isGguf?: boolean;
  /** Staged metadata confirmed the separate DiffusionGemma runner. */
  isDiffusion?: boolean;
  config?: PerModelConfig;
  forceReload?: boolean;
  /** model_path to send when the pick loads from elsewhere, e.g. a pinned snapshot dir. */
  loadId?: string | null;
  /** Native path token so an active-model reload can reopen a file-picked GGUF. */
  nativePathToken?: string;
  /** Hub pipeline tag for an uncurated pick, so a task page can tell which task the repo does when
   *  it is not in the page's catalog. */
  pipelineTag?: string | null;
  /** Detected local audio architecture, used when a filesystem path has no Hub id. */
  audioType?: string | null;
  nativePathExpiresAtMs?: number | null;
}

/** Full on-disk requirement for a model pick, including its checkpoint and companion assets
 *  (text encoders, VAE, tokenizer/config files). */
export interface ModelDownloadFootprint {
  requiredBytes: number;
  checkpointBytes: number;
}

export type ModelDownloadFootprintResolver = (
  id: string,
  meta: ModelSelectorChangeMeta,
) => Promise<ModelDownloadFootprint | null>;

export interface ModelPickTarget {
  id: string;
  displayName: string;
  ggufVariant?: string | null;
  isGguf: boolean;
  /** Whether an OpenAI-compatible request can actually load this model. Not the same as isGguf:
   *  local_model_resolver skips Ollama's scanner. Defaults to isGguf when unknown. */
  apiLoadable?: boolean;
  /** Identity the saved settings are keyed by, when that is not what loads: a repo cached outside
   *  the active HF cache loads by snapshot path while its settings key on the repo id. Probes
   *  that must open the model keep using `id`. */
  configId?: string;
  meta: ModelSelectorChangeMeta;
}

export interface DeletedModelRef {
  id: string;
  ggufVariant?: string;
}
