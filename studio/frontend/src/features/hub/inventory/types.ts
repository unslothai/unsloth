// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BaseModelSource, LocalArtifactKind } from "./api";
import type { InventoryHintKind, LocalSource } from "./constants";

export type ModelInventoryFormat =
  | "gguf"
  | "safetensors"
  | "adapter"
  | "checkpoint"
  | "unknown";
export interface ModelInventoryCapabilities {
  canTrain: boolean;
  canChat: boolean;
  canDelete: boolean;
  canDownload: boolean;
  requiresVariant: boolean;
  supportsLora: boolean;
  supportsVision: boolean;
}

export interface InventoryHint {
  kind: InventoryHintKind;
  repoId: string;
  bytes?: number;
  startedAt?: number;
  createdAt?: number;
}

export interface CachedInventoryRow {
  kind: "cache";
  id: string;
  loadId: string;
  repoId: string;
  owner: string;
  repo: string;
  isGguf: boolean;
  modelFormat: ModelInventoryFormat;
  artifact: LocalArtifactKind;
  formatVariant?: string | null;
  capabilities: ModelInventoryCapabilities;
  bytes: number;
  cachePath?: string | null;
  loadCachePath?: string | null;
  lastModified?: number | null;
  partial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  downloading?: boolean;
  /** Moves on a sibling cancel, which changes neither bytes nor mtime. */
  hasVariantState?: boolean;
  pipelineTag?: string | null;
  task?: string | null;
  audioType?: string | null;
  // Loadable only via from_single_file, so pickers must not offer it as a pipeline load.
  singleFile?: boolean;
  // VAE / text encoders without a denoiser: listed so they can be deleted, never picked.
  companion?: boolean;
  companionPrefetch?: boolean;
  tags?: string[];
  libraryName?: string | null;
  quantMethod?: string | null;
  liveDownload?: boolean;
  optimistic?: boolean;
}

export interface LocalInventoryRow {
  kind: "local";
  id: string;
  loadId: string;
  repoId: string | null;
  owner: string;
  title: string;
  source: LocalSource;
  sourceLabel: string;
  datasetSource?: "recipe" | "upload";
  modelId?: string | null;
  displayName?: string;
  path: string;
  isGguf: boolean;
  modelFormat: ModelInventoryFormat;
  artifact: LocalArtifactKind;
  formatVariant?: string | null;
  capabilities: ModelInventoryCapabilities;
  baseModel?: string | null;
  baseModelSource?: BaseModelSource | null;
  baseModelHubId?: string | null;
  adapterType?: string | null;
  trainingMethod?: string | null;
  task?: string | null;
  audioType?: string | null;
  updatedAt: number | null;
  partial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  downloading?: boolean;
  companionPrefetch?: boolean;
  activeCache?: boolean | null;
  pipelineTag?: string | null;
  tags?: string[];
  libraryName?: string | null;
  quantMethod?: string | null;
}

export type InventoryRow = CachedInventoryRow | LocalInventoryRow;
