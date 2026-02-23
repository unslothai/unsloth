import type {
  SeedConfig,
  SeedSamplingStrategy,
  SeedSelectionType,
  SeedSourceType,
} from "../../../types";
import { isRecord, readString } from "../helpers";

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
    seed_source_type: "hf",
    hf_repo_id: "",
    hf_subset: "",
    hf_split: "",
    hf_path: "",
    hf_token: "",
    hf_endpoint: "https://huggingface.co",
    local_file_name: "",
    unstructured_file_name: "",
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
  let unstructured_file_name = "";
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
    unstructured_file_name,
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
): SeedConfig | null {
  if (!seedConfigRaw) {
    return null;
  }
  return {
    ...makeDefaultSeedConfig(id),
    ...parseSeedSettings(seedConfigRaw), // payload-only fields override ui defaults
  };
}
