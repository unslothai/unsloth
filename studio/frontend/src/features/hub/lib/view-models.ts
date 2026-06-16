// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HfModelResult } from "@/features/hub/hooks/use-hub-model-search";
import { resolveInventoryResource } from "@/features/hub/inventory/resource-resolver";
import type {
  CachedInventoryRow,
  LocalInventoryRow,
} from "@/features/hub/inventory/types";
import type {
  CapabilityFilter,
  DiscoverRow,
  ModelFormatFilter,
} from "../types";
import {
  detectBaseModel,
  detectCapabilities,
  type CapabilityKey,
} from "./model-capabilities";
import { ownerOf, repoOf } from "@/features/hub/lib/format";
import { estimateSizeFromDtypes, isGgufLike } from "./hf-model-meta";
export {
  detectResultFormat,
  isUnslothFinetunable,
  matchesFormat,
} from "./format-filters";
export {
  formatLocalUpdated,
  localSourceLabel,
} from "@/features/hub/inventory";
export { isGgufLike } from "./hf-model-meta";

export const CAPABILITY_FILTER_OPTIONS: ReadonlyArray<{
  value: CapabilityFilter;
  label: string;
}> = [
  { value: "all", label: "All capabilities" },
  { value: "reasoning", label: "Reasoning" },
  { value: "vision", label: "Vision" },
  { value: "audio", label: "Audio" },
  { value: "embedding", label: "Embeddings" },
];

export const FORMAT_FILTER_OPTIONS: ReadonlyArray<{
  value: ModelFormatFilter;
  label: string;
}> = [
  { value: "all", label: "All formats" },
  { value: "gguf", label: "GGUF" },
  { value: "checkpoint", label: "Safetensors" },
  { value: "mlx", label: "MLX" },
];

const BILLION = 1_000_000_000;

export function formatParamCount(totalParams: number | undefined): string {
  if (!totalParams || totalParams <= 0) return "N/A";
  const billions = totalParams / BILLION;
  if (billions >= 1) {
    return `${billions >= 100 ? Math.round(billions) : Number(billions.toFixed(billions >= 10 ? 0 : 1))}B`;
  }
  const millions = totalParams / 1_000_000;
  return `${Math.round(millions)}M`;
}

const PARAM_IN_ID_RE = /(\d+(?:\.\d+)?)\s*[bB](?=$|[^a-z0-9])/gi;

export function paramLabelFromId(repo: string): string | null {
  let bestLabel: string | null = null;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (const match of repo.matchAll(PARAM_IN_ID_RE)) {
    const value = Number.parseFloat(match[1]);
    if (Number.isFinite(value) && value > bestValue) {
      bestValue = value;
      bestLabel = match[1];
    }
  }
  return bestLabel ? `${bestLabel}B` : null;
}

export function formatModelParamLabel(
  repo: string,
  totalParams: number | undefined,
): string {
  return paramLabelFromId(repo) ?? formatParamCount(totalParams);
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
  if (result.totalParams)
    parts.push(`${compactNumber(result.totalParams)} params`);
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

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

export function toHfModelResult(raw: unknown): HfModelResult | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const model = raw as {
    name?: string;
    downloads?: number;
    likes?: number;
    private?: boolean;
    gated?: false | "auto" | "manual";
    task?: string;
    pipeline_tag?: string;
    library_name?: string;
    updatedAt?: Date | string;
    createdAt?: Date | string;
    downloadsAllTime?: number;
    safetensors?: { total?: number; parameters?: Record<string, number> };
    gguf?: { total?: number; architecture?: string };
    tags?: string[];
    config?: { quantization_config?: { quant_method?: string } };
  };

  if (!model.name) return null;

  const isGguf =
    Boolean(model.tags?.some((tag) => tag.toLowerCase() === "gguf")) ||
    isGgufLike(model.name);
  const updatedAt =
    model.updatedAt instanceof Date
      ? model.updatedAt.toISOString()
      : typeof model.updatedAt === "string"
        ? model.updatedAt
        : undefined;
  const createdAt =
    model.createdAt instanceof Date
      ? model.createdAt.toISOString()
      : typeof model.createdAt === "string"
        ? model.createdAt
        : undefined;

  return {
    id: model.name,
    downloads: finiteNumber(model.downloads) ?? 0,
    likes: finiteNumber(model.likes) ?? 0,
    private: model.private,
    gated: model.gated,
    totalParams:
      finiteNumber(model.safetensors?.total) ?? finiteNumber(model.gguf?.total),
    estimatedSizeBytes: estimateSizeFromDtypes(model.safetensors?.parameters),
    isGguf,
    baseModel: detectBaseModel(model.tags),
    tags: model.tags,
    pipelineTag: model.task ?? model.pipeline_tag,
    updatedAt,
    createdAt,
    downloadsAllTime: finiteNumber(model.downloadsAllTime),
    libraryName: model.library_name,
    quantMethod: model.config?.quantization_config?.quant_method,
  };
}

export function buildDiscoverRows(
  results: HfModelResult[],
  cachedRows: CachedInventoryRow[],
  localRows: LocalInventoryRow[],
): DiscoverRow[] {
  return results.map((result) => {
    const resource = resolveInventoryResource({
      repoId: result.id,
      cachedRows,
      localRows,
      formatHint: result.isGguf ? "gguf" : "non-gguf",
    });
    const partial = Boolean(
      resource.cachedRow?.partial ?? resource.localRow?.partial ?? false,
    );
    return {
      id: result.id,
      owner: ownerOf(result.id),
      repo: repoOf(result.id),
      result,
      isAvailableOnDevice: Boolean(resource.cachedRow || resource.localRow),
      isPartialOnDevice: partial,
      summary: buildSummary(result),
      capabilities: detectCapabilities(
        result.tags,
        result.pipelineTag,
        result.id,
      ),
    };
  });
}

export function matchesCapability(
  capabilities: Array<{ key: CapabilityKey }>,
  capabilityFilter: CapabilityFilter,
): boolean {
  if (capabilityFilter === "all") return true;
  return capabilities.some((capability) => capability.key === capabilityFilter);
}

const LANGUAGE_TAG_PREFIX = "language:";

export function parseLanguageTags(
  tags: readonly string[] | null | undefined,
): string[] {
  const langs: string[] = [];
  for (const tag of tags ?? []) {
    if (!tag.startsWith(LANGUAGE_TAG_PREFIX)) continue;
    const code = tag.slice(LANGUAGE_TAG_PREFIX.length);
    if (code && !langs.includes(code)) langs.push(code);
  }
  return langs;
}
