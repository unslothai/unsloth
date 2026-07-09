// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { BaseModelSource } from "./api";
import type { InventoryHintKind, LocalSource } from "./constants";

export type ModelInventoryFormat =
  | "gguf"
  | "safetensors"
  | "adapter"
  | "checkpoint"
  | "unknown";
export type ModelInventoryRuntime =
  | "llama_cpp"
  | "transformers"
  | "adapter"
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
  runtime: ModelInventoryRuntime;
  formatVariant?: string | null;
  capabilities: ModelInventoryCapabilities;
  bytes: number;
  cachePath?: string | null;
  lastModified?: number | null;
  partial?: boolean;
  partialTransport?: string | null;
  pipelineTag?: string | null;
  tags?: string[];
  libraryName?: string | null;
  quantMethod?: string | null;
  liveDownload?: boolean;
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
  modelId?: string | null;
  displayName?: string;
  path: string;
  isGguf: boolean;
  modelFormat: ModelInventoryFormat;
  runtime: ModelInventoryRuntime;
  formatVariant?: string | null;
  capabilities: ModelInventoryCapabilities;
  baseModel?: string | null;
  baseModelSource?: BaseModelSource | null;
  baseModelHubId?: string | null;
  adapterType?: string | null;
  trainingMethod?: string | null;
  updatedAt: number | null;
  partial?: boolean;
  partialTransport?: string | null;
  pipelineTag?: string | null;
  tags?: string[];
  libraryName?: string | null;
  quantMethod?: string | null;
}

export type InventoryRow = CachedInventoryRow | LocalInventoryRow;
