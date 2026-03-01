import type { ValidatorConfig } from "../../types";

function parseBatchSize(value: string): number {
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 1) {
    return 10;
  }
  return parsed;
}

export function buildValidatorColumn(
  config: ValidatorConfig,
  errors: string[],
): Record<string, unknown> {
  const targetColumns = (config.target_columns ?? [])
    .map((value) => value.trim())
    .filter(Boolean);
  if (targetColumns.length === 0) {
    errors.push(`Validator ${config.name}: target code column required.`);
  }
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    column_type: "validation",
    name: config.name,
    drop: config.drop ?? false,
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: targetColumns,
    // biome-ignore lint/style/useNamingConvention: api schema
    validator_type: "code",
    // biome-ignore lint/style/useNamingConvention: api schema
    validator_params: {
      // biome-ignore lint/style/useNamingConvention: api schema
      code_lang: config.code_lang,
    },
    // biome-ignore lint/style/useNamingConvention: api schema
    batch_size: parseBatchSize(config.batch_size),
  };
}
