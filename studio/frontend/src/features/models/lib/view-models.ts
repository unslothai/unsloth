// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalModelInfo } from "@/features/chat/api/chat-api";
import type { HfModelResult } from "@/hooks";
import type {
  CapabilityFilter,
  DiscoverRow,
  LocalInventoryRow,
  ModelFormatFilter,
} from "../types";
import {
  detectCapabilities,
  type CapabilityKey,
} from "./capabilities";
import { ownerOf, repoOf } from "./format";

const EXCLUDED_TAGS = new Set([
  "gptq",
  "awq",
  "exl2",
  "mlx",
  "onnx",
  "openvino",
  "coreml",
  "tflite",
  "ctranslate2",
]);

const EMBEDDING_TAGS = new Set([
  "sentence-transformers",
  "feature-extraction",
]);

const DTYPE_BYTES: Record<string, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  NF4: 0.5,
  FP4: 0.5,
  INT4: 0.5,
  GPTQ: 0.5,
};

export const CAPABILITY_FILTER_OPTIONS: ReadonlyArray<{
  value: CapabilityFilter;
  label: string;
}> = [
  { value: "all", label: "All capabilities" },
  { value: "reasoning", label: "Reasoning" },
  { value: "code", label: "Code" },
  { value: "vision", label: "Vision" },
  { value: "audio", label: "Audio" },
  { value: "tools", label: "Tools" },
  { value: "embedding", label: "Embeddings" },
  { value: "multilingual", label: "Multilingual" },
];

export const FORMAT_FILTER_OPTIONS: ReadonlyArray<{
  value: ModelFormatFilter;
  label: string;
}> = [
  { value: "all", label: "All formats" },
  { value: "gguf", label: "GGUF" },
  { value: "checkpoint", label: "Checkpoints" },
];

export const RESOURCE_TYPE_OPTIONS: ReadonlyArray<{
  value: "models" | "datasets";
  label: string;
}> = [
  { value: "models", label: "Models" },
  { value: "datasets", label: "Datasets" },
];

function estimateSizeFromDtypes(
  params: Record<string, number> | undefined,
): number | undefined {
  if (!params) return undefined;
  let total = 0;
  for (const [dtype, count] of Object.entries(params)) {
    const bytesPerParam = DTYPE_BYTES[dtype.toUpperCase()] ?? 2;
    total += count * bytesPerParam;
  }
  return total > 0 ? total : undefined;
}

export function formatPipelineTag(tag: string | undefined): string | null {
  if (!tag) return null;
  return tag
    .split(/[-_]/)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatLibrary(name: string | undefined): string | null {
  if (!name) return null;
  if (name === "transformers") return "Transformers";
  return name;
}

export function buildSummary(result: HfModelResult): string {
  const parts: string[] = [];
  const pipeline = formatPipelineTag(result.pipelineTag);
  if (pipeline) parts.push(pipeline);
  if (result.totalParams) parts.push(`${compactNumber(result.totalParams)} params`);
  if (result.isGguf) {
    parts.push("GGUF");
  } else {
    const library = formatLibrary(result.libraryName);
    if (library) parts.push(library);
  }
  return parts.join(" · ") || "Hugging Face model";
}

function compactNumber(value: number): string {
  return new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: value < 10 ? 1 : 0,
  }).format(value);
}

export function toHfModelResult(raw: unknown): HfModelResult | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const model = raw as {
    name?: string;
    downloads?: number;
    likes?: number;
    task?: string;
    pipeline_tag?: string;
    library_name?: string;
    updatedAt?: Date | string;
    safetensors?: { total?: number; parameters?: Record<string, number> };
    gguf?: { total?: number; architecture?: string };
    tags?: string[];
  };

  if (!model.name) return null;
  const isEmbedding = model.tags?.some((tag) => EMBEDDING_TAGS.has(tag));
  if (!isEmbedding && model.tags?.some((tag) => EXCLUDED_TAGS.has(tag))) {
    return null;
  }

  const isGguf =
    Boolean(model.tags?.some((tag) => tag.toLowerCase() === "gguf")) ||
    isGgufLike(model.name);
  const updatedAt =
    model.updatedAt instanceof Date
      ? model.updatedAt.toISOString()
      : typeof model.updatedAt === "string"
        ? model.updatedAt
        : undefined;

  return {
    id: model.name,
    downloads: model.downloads ?? 0,
    likes: model.likes ?? 0,
    totalParams: model.safetensors?.total ?? model.gguf?.total,
    estimatedSizeBytes: estimateSizeFromDtypes(model.safetensors?.parameters),
    isGguf,
    tags: model.tags,
    pipelineTag: model.task ?? model.pipeline_tag,
    updatedAt,
    libraryName: model.library_name,
  };
}

export function buildDiscoverRows(
  results: HfModelResult[],
  availableSet: Set<string>,
): DiscoverRow[] {
  return results.map((result) => ({
    id: result.id,
    owner: ownerOf(result.id),
    repo: repoOf(result.id),
    result,
    isAvailableOnDevice: availableSet.has(result.id.toLowerCase()),
    summary: buildSummary(result),
    capabilities: detectCapabilities(
      result.tags,
      result.pipelineTag,
      result.id,
    ),
  }));
}

export function matchesFormat(
  isGguf: boolean,
  formatFilter: ModelFormatFilter,
): boolean {
  if (formatFilter === "all") return true;
  return formatFilter === "gguf" ? isGguf : !isGguf;
}

export function matchesCapability(
  capabilities: Array<{ key: CapabilityKey }>,
  capabilityFilter: CapabilityFilter,
): boolean {
  if (capabilityFilter === "all") return true;
  return capabilities.some((capability) => capability.key === capabilityFilter);
}

export function localSourceLabel(source: LocalModelInfo["source"]): string {
  switch (source) {
    case "lmstudio":
      return "External";
    case "custom":
      return "Custom folder";
    case "models_dir":
      return "Models folder";
    case "hf_cache":
      return "HF cache";
    default:
      return "Local model";
  }
}

function normalizeTimestamp(value?: number | null): number | null {
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

export function isGgufLike(value: string | undefined | null): boolean {
  if (!value) return false;
  return value.toLowerCase().endsWith(".gguf") || /-GGUF(?:$|-)/i.test(value);
}

function getLocalHubId(model: LocalModelInfo): string | null {
  const candidate = model.model_id?.trim();
  if (candidate && candidate.includes("/") && !isPathLike(candidate)) {
    return candidate;
  }
  return null;
}

function isPathLike(value: string): boolean {
  return /^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)/.test(value);
}

export function buildLocalInventoryRows(
  models: LocalModelInfo[],
): LocalInventoryRow[] {
  return models
    .filter((model) => model.source !== "hf_cache")
    .map((model) => {
      const repoId = getLocalHubId(model);
      const owner = repoId ? ownerOf(repoId) : localSourceLabel(model.source);
      const title = repoId ? repoOf(repoId) : model.display_name;
      return {
        kind: "local" as const,
        id: model.id,
        repoId,
        owner,
        title,
        source: model.source,
        sourceLabel: localSourceLabel(model.source),
        path: model.path,
        isGguf:
          isGgufLike(model.path) ||
          isGgufLike(model.id) ||
          isGgufLike(model.model_id) ||
          isGgufLike(model.display_name),
        updatedAt: normalizeTimestamp(model.updated_at),
      };
    })
    .sort((a, b) => {
      const sourceWeight = sourceSortWeight(a.source) - sourceSortWeight(b.source);
      if (sourceWeight !== 0) return sourceWeight;
      if (a.updatedAt && b.updatedAt && a.updatedAt !== b.updatedAt) {
        return b.updatedAt - a.updatedAt;
      }
      return a.title.localeCompare(b.title);
    });
}

function sourceSortWeight(source: LocalModelInfo["source"]): number {
  switch (source) {
    case "models_dir":
      return 0;
    case "custom":
      return 1;
    case "lmstudio":
      return 2;
    case "hf_cache":
      return 3;
    default:
      return 4;
  }
}
