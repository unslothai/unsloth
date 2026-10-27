// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
import type { PerModelConfig } from "../../model-config/per-model-config";

export interface ModelOption {
  id: string;
  name: string;
  description?: string;
  descriptionSuffix?: string;
  icon?: ReactNode;
  isGguf?: boolean;
  deviceQuant?: string;
  deviceSize?: string;
  deviceSizeBytes?: number;
  deviceLoaded?: boolean;
  audioType?: string | null;
}

export interface LoraModelOption extends ModelOption {
  baseModel?: string;
  updatedAt?: number;
  source?: "training" | "exported" | "local";

  isDirectGguf?: boolean;
  exportType?: "lora" | "merged" | "gguf";
  sizeBytes?: number | null;
  audioType?: string | null;
}

export interface ExternalModelOption extends ModelOption {
  providerId: string;
  providerName: string;
  providerType: string;
}

export interface ModelSelectorChangeMeta {
  source: "hub" | "lora" | "exported" | "local" | "external";
  isLora: boolean;
  ggufVariant?: string;
  /** Filenames do not always follow the repo name (FLUX.1-schnell -> flux1-schnell-*.gguf). */
  ggufFilename?: string;
  isDownloaded?: boolean;
  expectedBytes?: number;
  downloadPresentation?: {
    label: string;
    filename: string;
    expectedBytes: number;
  };
  contextLength?: number | null;
  isGguf?: boolean;
  /** Undefined means unknown, not text-only. */
  isVision?: boolean;
  isDiffusion?: boolean;
  config?: PerModelConfig;
  forceReload?: boolean;
  loadId?: string | null;
  nativePathToken?: string;
  pipelineTag?: string | null;
  familyOverrideRequired?: boolean;
  audioType?: string | null;
  nativePathExpiresAtMs?: number | null;
}

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
  apiLoadable?: boolean;
  /** Settings key when it differs from what loads (snapshot path vs repo id). Probes use `id`. */
  configId?: string;
  meta: ModelSelectorChangeMeta;
}

export interface DeletedModelRef {
  id: string;
  ggufVariant?: string;
}
