// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
import type { PerModelConfig } from "@/features/chat";
import type { ModelInventoryFormat } from "@/features/inventory";

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
  runDisplayName?: string;
  trainingMethod?: string;
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
  exportType?: "lora" | "merged" | "gguf";
  ggufVariant?: string;
  modelFormat?: ModelInventoryFormat | null;
  isDownloaded?: boolean;
  isPartial?: boolean;
  preferLocalCache?: boolean;
  localPath?: string | null;
  expectedBytes?: number;
  config?: PerModelConfig;
}

export interface ModelPickTarget {
  id: string;
  displayName: string;
  meta: ModelSelectorChangeMeta;
  isGguf: boolean;
  supportsTrustRemoteCode: boolean;
}

export interface DeletedModelRef {
  id: string;
  ggufVariant?: string;
  deletedRunIds?: string[];
}
