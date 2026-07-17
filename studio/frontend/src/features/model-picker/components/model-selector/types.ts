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
  meta: ModelSelectorChangeMeta;
}

export interface DeletedModelRef {
  id: string;
  ggufVariant?: string;
}
