// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  SeedConfig,
  SeedSamplingStrategy,
  SeedSelectionType,
  SeedSourceType,
} from "../../../types";
import { isRecord, readNumberString, readString } from "../helpers";

function normalizeSampling(value: unknown): SeedSamplingStrategy {
  const raw = readString(value);
  if (raw === "shuffle") return "shuffle";
  return "ordered";
}

function makeDefaultSeedConfig(id: string): SeedConfig {
  return {
    id,
    kind: "seed",
    name: "seed",
    drop: false,
    seed_drop_columns: [],
    seed_source_type: "hf",
    hf_repo_id: "",
    hf_subset: "",
    hf_split: "",
    hf_path: "",
    hf_token: "",
    hf_endpoint: "https://huggingface.co",
    local_file_name: "",
    unstructured_file_ids: [],
    unstructured_file_names: [],
    unstructured_file_sizes: [],
    seed_preview_rows: [],
    unstructured_chunk_size: "1200",
    unstructured_chunk_overlap: "200",
    seed_splits: [],
    seed_globs_by_split: {},
    seed_columns: [],
    sampling_strategy: "ordered",
    selection_type: "none",
    selection_start: "0",
    selection_end: "10",
    selection_index: "0",
    selection_num_partitions: "1",
  };
}

function inferRepoIdFromSeedPath(path: string): string {
  const trimmed = path.trim();
  if (!trimmed) return "";
  const parts = trimmed.split("/").filter(Boolean);
  if (parts.length >= 3 && parts[0] === "datasets") {
    return `${parts[1]}/${parts[2]}`;
  }
  if (parts.length >= 2) {
    return `${parts[0]}/${parts[1]}`;
  }
  return "";
}

function parseSeedSettings(seedConfigRaw: unknown): Partial<SeedConfig> {
  if (!isRecord(seedConfigRaw)) {
    return {};
  }

  const sampling_strategy = normalizeSampling(seedConfigRaw.sampling_strategy);

  let seed_source_type: SeedSourceType = "hf";
  let hf_path = "";
  let hf_token = "";
  let hf_endpoint = "https://huggingface.co";
  let hf_repo_id = "";
  let local_file_name = "";
  let unstructuredFileIds: string[] = [];
  let unstructuredFileNames: string[] = [];
  let unstructuredFileSizes: number[] = [];
  let resolved_paths: string[] = [];
  let unstructured_chunk_size = "1200";
  let unstructured_chunk_overlap = "200";
  let github_repo_slug = "";
  let github_token = "";
  let github_limit = "100";
  let github_item_types: ("issues" | "pulls" | "commits")[] = ["issues", "pulls"];
  let github_include_comments = true;
  let github_max_comments_per_item = "30";
  const sourceRaw = seedConfigRaw.source;
  if (isRecord(sourceRaw)) {
    const seedType = readString(sourceRaw.seed_type);
    const sourcePath = readString(sourceRaw.path) ?? "";
    if (seedType === "hf") {
      seed_source_type = "hf";
      hf_path = sourcePath;
      hf_token = readString(sourceRaw.token) ?? "";
      hf_endpoint = readString(sourceRaw.endpoint) ?? hf_endpoint;
      hf_repo_id = inferRepoIdFromSeedPath(hf_path);
    } else if (seedType === "local") {
      seed_source_type = "local";
      hf_path = sourcePath;
      local_file_name = sourcePath.split("/").pop() ?? sourcePath;
    } else if (seedType === "unstructured") {
      seed_source_type = "unstructured";
      const paths = Array.isArray(sourceRaw.paths) ? sourceRaw.paths : [];
      const stringPaths = paths.filter((p): p is string => typeof p === "string");
      if (stringPaths.length === 0 && sourcePath) {
        stringPaths.push(sourcePath);
      }
      hf_path = stringPaths[0] ?? sourcePath;
      resolved_paths = stringPaths;
      unstructuredFileIds = [];
      unstructuredFileNames = [];
      unstructured_chunk_size = readNumberString(sourceRaw.chunk_size) || "1200";
      unstructured_chunk_overlap = readNumberString(sourceRaw.chunk_overlap) || "200";
    } else if (seedType === "github_repo") {
      seed_source_type = "github_repo";
      const rawRepos = Array.isArray(sourceRaw.repos) ? sourceRaw.repos : [];
      const repos = rawRepos.filter((r): r is string => typeof r === "string");
      github_repo_slug = repos.join("\n");
      github_token = readString(sourceRaw.token) ?? "";
      github_limit = readNumberString(sourceRaw.limit) || "100";
      const rawItems = Array.isArray(sourceRaw.item_types) ? sourceRaw.item_types : [];
      const validItems = rawItems.filter(
        (t): t is "issues" | "pulls" | "commits" =>
          t === "issues" || t === "pulls" || t === "commits",
      );
      if (validItems.length > 0) {
        github_item_types = validItems;
      }
      if (typeof sourceRaw.include_comments === "boolean") {
        github_include_comments = sourceRaw.include_comments;
      }
      github_max_comments_per_item =
        readNumberString(sourceRaw.max_comments_per_item) || "30";
    }
  }

  let selection_type: SeedSelectionType = "none";
  let selection_start = "0";
  let selection_end = "10";
  let selection_index = "0";
  let selection_num_partitions = "1";
  const selectionRaw = seedConfigRaw.selection_strategy;
  if (isRecord(selectionRaw)) {
    if (
      typeof selectionRaw.start === "number" &&
      typeof selectionRaw.end === "number"
    ) {
      selection_type = "index_range";
      selection_start = String(selectionRaw.start);
      selection_end = String(selectionRaw.end);
    } else if (
      typeof selectionRaw.index === "number" &&
      typeof selectionRaw.num_partitions === "number"
    ) {
      selection_type = "partition_block";
      selection_index = String(selectionRaw.index);
      selection_num_partitions = String(selectionRaw.num_partitions);
    }
  }

  return {
    seed_source_type,
    hf_repo_id,
    hf_path,
    hf_token,
    hf_endpoint,
    local_file_name,
    unstructured_file_ids: unstructuredFileIds,
    unstructured_file_names: unstructuredFileNames,
    unstructured_file_sizes: unstructuredFileSizes,
    resolved_paths,
    unstructured_chunk_size,
    unstructured_chunk_overlap,
    github_repo_slug,
    github_token,
    github_limit,
    github_item_types,
    github_include_comments,
    github_max_comments_per_item,
    sampling_strategy,
    selection_type,
    selection_start,
    selection_end,
    selection_index,
    selection_num_partitions,
  };
}

export function parseSeedConfig(
  seedConfigRaw: unknown,
  id: string,
  options?: {
    preferredSourceType?: SeedSourceType;
    seed_columns?: string[];
    seed_drop_columns?: string[];
    seed_preview_rows?: Record<string, unknown>[];
    local_file_name?: string;
    unstructuredFileIds?: string[];
    unstructuredFileNames?: string[];
    unstructuredFileSizes?: number[];
    unstructured_chunk_size?: string;
    unstructured_chunk_overlap?: string;
  },
): SeedConfig | null {
  if (!seedConfigRaw) {
    return null;
  }
  const parsed = parseSeedSettings(seedConfigRaw);
  let sourceType: SeedSourceType = "hf";
  if (parsed.seed_source_type === "hf") {
    sourceType = "hf";
  } else if (options?.preferredSourceType) {
    sourceType = options.preferredSourceType;
  } else if (parsed.seed_source_type) {
    sourceType = parsed.seed_source_type;
  }
  return {
    ...makeDefaultSeedConfig(id),
    ...parsed, // payload-only fields override ui defaults
    seed_source_type: sourceType,
    ...(options?.seed_columns ? { seed_columns: options.seed_columns } : {}),
    ...(options?.seed_drop_columns
      ? { seed_drop_columns: options.seed_drop_columns }
      : {}),
    ...(options?.seed_preview_rows
      ? { seed_preview_rows: options.seed_preview_rows }
      : {}),
    ...(options?.local_file_name !== undefined
      ? { local_file_name: options.local_file_name }
      : {}),
    ...(options?.unstructuredFileIds !== undefined
      ? { unstructured_file_ids: options.unstructuredFileIds }
      : {}),
    ...(options?.unstructuredFileNames !== undefined
      ? { unstructured_file_names: options.unstructuredFileNames }
      : {}),
    ...(options?.unstructuredFileSizes !== undefined
      ? { unstructured_file_sizes: options.unstructuredFileSizes }
      : {}),
    ...(options?.unstructured_chunk_size !== undefined
      ? { unstructured_chunk_size: options.unstructured_chunk_size }
      : {}),
    ...(options?.unstructured_chunk_overlap !== undefined
      ? { unstructured_chunk_overlap: options.unstructured_chunk_overlap }
      : {}),
  };
}
