import type { NodeConfig, SeedConfig } from "../../types";

function parseIntStrict(value: string | undefined): number | null {
  const trimmed = value?.trim();
  if (!trimmed) return null;
  const num = Number(value);
  if (!Number.isFinite(num) || !Number.isInteger(num)) return null;
  return num;
}

export function buildSeedConfig(
  config: SeedConfig,
  errors: string[],
): Record<string, unknown> | undefined {
  const seedSourceType = config.seed_source_type ?? "hf";
  const path = config.hf_path.trim();
  if (!path) {
    return undefined;
  }

  const endpoint = config.hf_endpoint?.trim() || "https://huggingface.co";
  const token = config.hf_token?.trim() || null;

  let selectionStrategy: Record<string, unknown> | null = null;
  if (config.selection_type === "index_range") {
    const start = parseIntStrict(config.selection_start);
    const end = parseIntStrict(config.selection_end);
    if (start === null || end === null) {
      errors.push(`Seed ${config.name}: selection index range invalid.`);
      return undefined;
    }
    selectionStrategy = { start, end };
  } else if (config.selection_type === "partition_block") {
    const index = parseIntStrict(config.selection_index);
    const numPartitions = parseIntStrict(config.selection_num_partitions);
    if (index === null || numPartitions === null) {
      errors.push(`Seed ${config.name}: selection partition invalid.`);
      return undefined;
    }
    // biome-ignore lint/style/useNamingConvention: api schema
    selectionStrategy = { index, num_partitions: numPartitions };
  }

  const source =
    seedSourceType === "hf"
      ? {
          // biome-ignore lint/style/useNamingConvention: api schema
          seed_type: "hf",
          path,
          token,
          endpoint,
        }
      : {
          // biome-ignore lint/style/useNamingConvention: api schema
          seed_type: "local",
          path,
        };

  return {
    source,
    // biome-ignore lint/style/useNamingConvention: api schema
    sampling_strategy: config.sampling_strategy,
    // biome-ignore lint/style/useNamingConvention: api schema
    selection_strategy: selectionStrategy,
  };
}

export function pickFirstSeedConfig(
  configs: Record<string, NodeConfig>,
): SeedConfig | null {
  for (const config of Object.values(configs)) {
    if (config.kind === "seed") {
      return config;
    }
  }
  return null;
}

export function buildSeedDropProcessor(
  config: SeedConfig,
  errors: string[],
): Record<string, unknown> | null {
  if (!config.drop) {
    return null;
  }
  const cols = (config.seed_columns ?? []).map((c) => c.trim()).filter(Boolean);
  if (cols.length === 0) {
    errors.push(`Seed ${config.name}: drop enabled but no seed columns loaded.`);
    return null;
  }
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    processor_type: "drop_columns",
    name: "drop_seed_columns",
    // biome-ignore lint/style/useNamingConvention: api schema
    build_stage: "post_batch",
    // biome-ignore lint/style/useNamingConvention: api schema
    column_names: cols,
  };
}
