import type { ValidatorConfig } from "../../types";

const OXC_VALIDATION_FN_MARKER = "unsloth_oxc_validator";

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
  if (config.validator_type === "oxc") {
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      column_type: "validation",
      name: config.name,
      drop: config.drop ?? false,
      // biome-ignore lint/style/useNamingConvention: api schema
      target_columns: targetColumns,
      // biome-ignore lint/style/useNamingConvention: api schema
      validator_type: "local_callable",
      // biome-ignore lint/style/useNamingConvention: api schema
      validator_params: {
        // backend resolves this marker to a real callable.
        // biome-ignore lint/style/useNamingConvention: api schema
        validation_function: `${OXC_VALIDATION_FN_MARKER}:${config.code_lang}`,
      },
      // biome-ignore lint/style/useNamingConvention: api schema
      batch_size: parseBatchSize(config.batch_size),
    };
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
