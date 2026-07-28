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
}

export interface LoraModelOption extends ModelOption {
  baseModel?: string;
  updatedAt?: number;
  source?: "training" | "exported" | "local";
  exportType?: "lora" | "merged" | "gguf";
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
  isDownloaded?: boolean;
  expectedBytes?: number;
  /** Native GGUF context, threaded so a staged pick can seed the slider. */
  contextLength?: number | null;
  /** Direct local .gguf file picked without a variant (custom folder / LM
   *  Studio). Marks it as a GGUF source for the deferred-load staging flow. */
  isGguf?: boolean;
  config?: PerModelConfig;
  forceReload?: boolean;
  /** Native path token so an active-model reload can reopen a file-picked GGUF. */
  nativePathToken?: string;
  nativePathExpiresAtMs?: number | null;
}

export interface ModelPickTarget {
  id: string;
  displayName: string;
  ggufVariant?: string | null;
  isGguf: boolean;
  /**
   * Whether an OpenAI-compatible request can actually load this model.
   *
   * Not the same as isGguf: local_model_resolver skips Ollama's scanner, so an
   * Ollama GGUF is never in the auto-switch index and no API request can resolve
   * it. Mirroring its settings would advertise a load that cannot happen.
   * Defaults to isGguf where a caller does not know.
   */
  apiLoadable?: boolean;
  /**
   * Identity the saved settings are keyed by, when that is not what loads.
   *
   * A repo cached outside the active HF cache loads by snapshot path, while the
   * picker and the auto-switch index keep its settings under the repo id.
   * Keying the save by the path would strand the settings where no load looks
   * for them. Probes that need something openable (the chat template, the GGUF
   * header) keep using `id`. Defaults to `id`.
   */
  configId?: string;
  meta: ModelSelectorChangeMeta;
}

export interface DeletedModelRef {
  id: string;
  ggufVariant?: string;
}
