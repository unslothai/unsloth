// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ModelCatalogItem {
  id: string;
  name: string;
  type: string;
  maxTokens: number | null;
  quantization: string | null;
  tags: string[];
  isFeatured: boolean;
  hasFlex: boolean;
  hasPriority: boolean;
  hasDiscount: boolean;
  discountPercent: number | null;
  centsPerInputToken: number | null;
  centsPerOutputToken: number | null;
  pricePerMillionInput: number | null; // in $/Mt
  pricePerMillionOutput: number | null; // in $/Mt
  description?: string;
}

export interface DeepInfraRawModel {
  model_name: string;
  type?: string;
  description?: string;
  tags?: string[];
  featured?: boolean;
  pricing?: {
    cents_per_input_token?: number | null;
    cents_per_output_token?: number | null;
    rate_per_service_tier_flex?: number | null;
    discount?: number | null;
    discount_ends_at?: string | null;
  } | null;
  max_tokens?: number | null;
  quantization?: string | null;
  deprecated?: boolean | number | null;
}

export function formatPricePerMillion(
  centsPerToken: number | null | undefined,
): string {
  if (centsPerToken == null) return "—";
  if (centsPerToken === 0) return "Free";
  const dollarsPerMillion = centsPerToken * 10000;
  if (dollarsPerMillion < 0.01) {
    return `$${dollarsPerMillion.toFixed(4)}/Mt`;
  }
  return `$${dollarsPerMillion.toFixed(2)}/Mt`;
}

export function formatContextTokens(tokens: number | null | undefined): string {
  if (tokens == null || tokens <= 0) return "—";
  if (tokens === 1048576) return "1024k";
  if (tokens >= 1_000_000) {
    return `${(tokens / 1_000_000).toFixed(tokens % 1_000_000 === 0 ? 0 : 1)}M`;
  }
  if (tokens >= 1_000) {
    return `${Math.round(tokens / 1024)}k`;
  }
  return `${tokens}`;
}

export function parseModelIds(value: string): string[] {
  return [
    ...new Set(
      value
        .split(/[,\n]/)
        .map((item) => item.trim())
        .filter((item) => item.length > 0),
    ),
  ];
}

export function parseDeepInfraCatalog(
  rawList: DeepInfraRawModel[],
): ModelCatalogItem[] {
  return rawList
    .filter(
      (item) =>
        !item.deprecated &&
        !item.model_name.toLowerCase().startsWith("anthropic/"),
    )
    .map((item) => {
      const tags = Array.isArray(item.tags) ? item.tags : [];
      const isFeatured = item.featured === true || tags.includes("featured");
      const hasFlex = tags.includes("flex");
      const hasPriority = tags.includes("priority");
      const discountVal = item.pricing?.discount;
      const hasDiscount =
        (typeof discountVal === "number" && discountVal > 0) ||
        tags.includes("discount");
      const discountPercent =
        typeof discountVal === "number" && discountVal > 0
          ? Math.round(discountVal * 100)
          : null;
      const centsIn = item.pricing?.cents_per_input_token ?? null;
      const centsOut = item.pricing?.cents_per_output_token ?? null;
      return {
        id: item.model_name,
        name: item.model_name,
        type: item.type || "text-generation",
        maxTokens: item.max_tokens ?? null,
        quantization: item.quantization ?? null,
        tags,
        isFeatured,
        hasFlex,
        hasPriority,
        hasDiscount,
        discountPercent,
        centsPerInputToken: centsIn,
        centsPerOutputToken: centsOut,
        pricePerMillionInput: centsIn != null ? centsIn * 10000 : null,
        pricePerMillionOutput: centsOut != null ? centsOut * 10000 : null,
        description: item.description,
      };
    });
}

export type SortField = "name" | "context" | "quant" | "priceIn" | "priceOut";
export type SortDirection = "asc" | "desc";

export interface ModelFilterOptions {
  searchQuery?: string;
  selectedType?: string;
  selectedQuant?: string;
  minContext?: number;
  featuredOnly?: boolean;
  discountOnly?: boolean;
  priorityOnly?: boolean;
  flexOnly?: boolean;
}

export function filterModels(
  models: ModelCatalogItem[],
  options: ModelFilterOptions,
): ModelCatalogItem[] {
  const query = (options.searchQuery ?? "").trim().toLowerCase();
  const selectedType = options.selectedType ?? "all";
  const selectedQuant = options.selectedQuant ?? "all";
  const minContext = options.minContext ?? 0;

  return models.filter((model) => {
    if (selectedType !== "all" && model.type !== selectedType) {
      return false;
    }
    if (selectedQuant !== "all" && model.quantization !== selectedQuant) {
      return false;
    }
    if (options.featuredOnly && !model.isFeatured) {
      return false;
    }
    if (options.discountOnly && !model.hasDiscount) {
      return false;
    }
    if (options.priorityOnly && !model.hasPriority) {
      return false;
    }
    if (options.flexOnly && !model.hasFlex) {
      return false;
    }
    if (
      minContext > 0 &&
      (model.maxTokens == null || model.maxTokens < minContext)
    ) {
      return false;
    }
    if (query.length > 0) {
      const matchName = model.name.toLowerCase().includes(query);
      const matchId = model.id.toLowerCase().includes(query);
      const matchDesc = model.description?.toLowerCase().includes(query);
      if (!matchName && !matchId && !matchDesc) {
        return false;
      }
    }
    return true;
  });
}

export function sortModels(
  models: ModelCatalogItem[],
  sortField: SortField,
  sortDirection: SortDirection,
): ModelCatalogItem[] {
  const list = [...models];
  list.sort((a, b) => {
    let cmp = 0;
    if (sortField === "name") {
      cmp = a.name.localeCompare(b.name);
    } else if (sortField === "context") {
      const aVal = a.maxTokens;
      const bVal = b.maxTokens;
      if (aVal == null && bVal == null) cmp = 0;
      else if (aVal == null) return 1;
      else if (bVal == null) return -1;
      else cmp = aVal - bVal;
    } else if (sortField === "quant") {
      const aVal = a.quantization ?? "";
      const bVal = b.quantization ?? "";
      if (!aVal && bVal) return 1;
      else if (aVal && !bVal) return -1;
      else cmp = aVal.localeCompare(bVal);
    } else if (sortField === "priceIn") {
      const aVal = a.centsPerInputToken;
      const bVal = b.centsPerInputToken;
      if (aVal == null && bVal == null) cmp = 0;
      else if (aVal == null) return 1;
      else if (bVal == null) return -1;
      else cmp = aVal - bVal;
    } else if (sortField === "priceOut") {
      const aVal = a.centsPerOutputToken;
      const bVal = b.centsPerOutputToken;
      if (aVal == null && bVal == null) cmp = 0;
      else if (aVal == null) return 1;
      else if (bVal == null) return -1;
      else cmp = aVal - bVal;
    }
    return sortDirection === "asc" ? cmp : -cmp;
  });
  return list;
}
