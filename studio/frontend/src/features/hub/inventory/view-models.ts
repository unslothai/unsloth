// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ownerOf, repoOf } from "../lib/format";
import { looksLikeLocalPath } from "../lib/local-path";
import { isGgufLike } from "../lib/model-identifiers";
import type {
  BackendModelCapabilities,
  LocalModelInfo,
  ModelInventoryFormat,
  ModelInventoryRuntime,
} from "./api";
import type {
  ModelInventoryCapabilities,
  CachedInventoryRow,
  LocalInventoryRow,
} from "./types";

export function localSourceLabel(source: LocalModelInfo["source"]): string {
  switch (source) {
    case "lmstudio":
      return "LM Studio";
    case "ollama":
      return "Ollama";
    case "custom":
      return "Custom folder";
    case "models_dir":
      return "Models folder";
    case "hf_cache":
      return "HF Cache";
    default:
      return "Local model";
  }
}

export function normalizeTimestamp(value?: number | null): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return value < 10_000_000_000 ? value * 1000 : value;
}

export function formatLocalUpdated(value?: number | null): string {
  const normalized = normalizeTimestamp(value);
  if (!normalized) return "Unknown update";
  const formatter = new Intl.DateTimeFormat("en", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  return formatter.format(new Date(normalized));
}

function getLocalHubId(model: LocalModelInfo): string | null {
  if (model.source !== "hf_cache") return null;
  const candidate = model.model_id?.trim();
  if (candidate?.toLowerCase().startsWith("ollama/")) {
    return null;
  }
  if (candidate && candidate.includes("/") && !looksLikeLocalPath(candidate)) {
    return candidate;
  }
  return null;
}

function getBaseModelHubId(model: LocalModelInfo): string | null {
  const baseModel = model.base_model?.trim();
  return model.base_model_source === "huggingface" && baseModel
    ? baseModel
    : null;
}

function isOllamaInventoryModel(model: LocalModelInfo): boolean {
  return (
    model.source === "ollama" ||
    model.id.toLowerCase().startsWith("ollama-manifest:") ||
    (model.model_id?.toLowerCase().startsWith("ollama/") ?? false)
  );
}

export function normalizeModelFormat(
  value: string | null | undefined,
  fallback: ModelInventoryFormat = "unknown",
): ModelInventoryFormat {
  if (
    value === "gguf" ||
    value === "safetensors" ||
    value === "adapter" ||
    value === "checkpoint" ||
    value === "unknown"
  ) {
    return value;
  }
  return fallback;
}

export function normalizeRuntime(
  value: string | null | undefined,
  modelFormat: ModelInventoryFormat,
): ModelInventoryRuntime {
  if (
    value === "llama_cpp" ||
    value === "transformers" ||
    value === "adapter" ||
    value === "unknown"
  ) {
    return value;
  }
  if (modelFormat === "gguf") return "llama_cpp";
  if (modelFormat === "adapter") return "adapter";
  if (modelFormat === "safetensors" || modelFormat === "checkpoint") {
    return "transformers";
  }
  return "unknown";
}

export function defaultCapabilities(
  modelFormat: ModelInventoryFormat,
  partial = false,
  requiresVariant = false,
): ModelInventoryCapabilities {
  const complete = !partial;
  const canChat =
    complete &&
    (modelFormat === "gguf" ||
      modelFormat === "safetensors" ||
      modelFormat === "adapter" ||
      modelFormat === "checkpoint");
  return {
    canTrain:
      complete &&
      (modelFormat === "safetensors" || modelFormat === "checkpoint"),
    canChat,
    canDelete: false,
    canDownload: false,
    requiresVariant,
    supportsLora:
      complete &&
      (modelFormat === "safetensors" || modelFormat === "checkpoint"),
    supportsVision: false,
  };
}

export function normalizeCapabilities(
  value: BackendModelCapabilities | null | undefined,
  modelFormat: ModelInventoryFormat,
  partial = false,
  requiresVariant = false,
): ModelInventoryCapabilities {
  const fallback = defaultCapabilities(modelFormat, partial, requiresVariant);
  return {
    canTrain: value?.can_train ?? fallback.canTrain,
    canChat: value?.can_chat ?? fallback.canChat,
    canDelete: value?.can_delete ?? fallback.canDelete,
    canDownload: value?.can_download ?? fallback.canDownload,
    requiresVariant: value?.requires_variant ?? fallback.requiresVariant,
    supportsLora: value?.supports_lora ?? fallback.supportsLora,
    supportsVision: value?.supports_vision ?? fallback.supportsVision,
  };
}

export function buildCachedInventoryRow(
  row: {
    repo_id: string;
    size_bytes: number;
    cache_path?: string;
    partial?: boolean;
    partial_transport?: string | null;
    pipeline_tag?: string | null;
    tags?: string[];
    library_name?: string | null;
    quant_method?: string | null;
    inventory_id?: string | null;
    load_id?: string | null;
    model_format?: ModelInventoryFormat | null;
    runtime?: string | null;
    format_variant?: string | null;
    capabilities?: BackendModelCapabilities | null;
    last_modified?: number | null;
    optimistic?: boolean;
  },
  fallbackFormat: ModelInventoryFormat,
): CachedInventoryRow {
  const rawModelFormat = normalizeModelFormat(row.model_format, "unknown");
  const modelFormat =
    rawModelFormat === "unknown" ? fallbackFormat : rawModelFormat;
  const inferredFromEndpoint =
    rawModelFormat === "unknown" && modelFormat !== "unknown";
  const requiresVariant = modelFormat === "gguf";
  const capabilities = normalizeCapabilities(
    inferredFromEndpoint ? null : row.capabilities,
    modelFormat,
    row.partial ?? false,
    requiresVariant,
  );
  if (row.optimistic) {
    capabilities.canChat = false;
  }
  return {
    kind: "cache",
    id:
      !inferredFromEndpoint && row.inventory_id
        ? row.inventory_id
        : `cache:${modelFormat}:${row.repo_id}`,
    loadId: row.load_id ?? row.repo_id,
    repoId: row.repo_id,
    owner: row.repo_id.includes("/") ? ownerOf(row.repo_id) : "Hub",
    repo: row.repo_id.includes("/") ? repoOf(row.repo_id) : row.repo_id,
    isGguf: modelFormat === "gguf",
    modelFormat,
    runtime: normalizeRuntime(
      inferredFromEndpoint ? null : row.runtime,
      modelFormat,
    ),
    formatVariant: row.format_variant ?? null,
    capabilities,
    bytes: row.size_bytes,
    cachePath: row.cache_path ?? null,
    lastModified:
      typeof row.last_modified === "number" &&
      Number.isFinite(row.last_modified) &&
      row.last_modified > 0
        ? row.last_modified
        : null,
    partial: row.partial ?? false,
    partialTransport: row.partial_transport ?? null,
    pipelineTag: row.pipeline_tag ?? null,
    tags: row.tags,
    libraryName: row.library_name ?? null,
    quantMethod: row.quant_method ?? null,
    optimistic: row.optimistic,
  };
}

function sourceSortWeight(source: LocalModelInfo["source"]): number {
  switch (source) {
    case "models_dir":
      return 0;
    case "custom":
      return 1;
    case "lmstudio":
      return 2;
    case "ollama":
      return 3;
    case "hf_cache":
      return 4;
    default:
      return 5;
  }
}

export function buildLocalInventoryRows(
  models: LocalModelInfo[],
): LocalInventoryRow[] {
  return models
    .map((model) => {
      const repoId = getLocalHubId(model);
      const baseModel = model.base_model?.trim() || null;
      const owner = repoId ? ownerOf(repoId) : localSourceLabel(model.source);
      const title = repoId ? repoOf(repoId) : model.display_name;
      const fallbackIsGguf =
        isOllamaInventoryModel(model) ||
        isGgufLike(model.path) ||
        isGgufLike(model.id) ||
        isGgufLike(model.model_id) ||
        isGgufLike(model.display_name);
      const modelFormat = normalizeModelFormat(
        model.model_format,
        fallbackIsGguf ? "gguf" : "unknown",
      );
      const fallbackRequiresVariant =
        modelFormat === "gguf" && !model.path.toLowerCase().endsWith(".gguf");
      const capabilities = normalizeCapabilities(
        model.capabilities,
        modelFormat,
        model.partial ?? false,
        fallbackRequiresVariant,
      );
      return {
        kind: "local" as const,
        id: model.inventory_id ?? model.id,
        loadId: model.load_id ?? model.id,
        repoId,
        owner,
        title,
        source: model.source,
        sourceLabel: localSourceLabel(model.source),
        modelId: model.model_id ?? null,
        displayName: model.display_name,
        path: model.path,
        isGguf: modelFormat === "gguf",
        modelFormat,
        runtime: normalizeRuntime(model.runtime, modelFormat),
        formatVariant: model.format_variant ?? null,
        capabilities,
        baseModel,
        baseModelSource: model.base_model_source ?? null,
        baseModelHubId: getBaseModelHubId(model),
        adapterType: model.adapter_type ?? null,
        trainingMethod: model.training_method ?? null,
        updatedAt: normalizeTimestamp(model.updated_at),
        partial: model.partial ?? false,
        partialTransport: model.partial_transport ?? null,
        activeCache: model.active_cache ?? null,
        pipelineTag: model.pipeline_tag ?? null,
        tags: model.tags,
        libraryName: model.library_name ?? null,
        quantMethod: model.quant_method ?? null,
      };
    })
    .sort((a, b) => {
      if (Boolean(a.partial) !== Boolean(b.partial)) return a.partial ? 1 : -1;
      const sourceWeight = sourceSortWeight(a.source) - sourceSortWeight(b.source);
      if (sourceWeight !== 0) return sourceWeight;
      if (a.updatedAt && b.updatedAt && a.updatedAt !== b.updatedAt) {
        return b.updatedAt - a.updatedAt;
      }
      return a.title.localeCompare(b.title);
    });
}
