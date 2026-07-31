// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig, ValidatorConfig } from "../../types";
import { isValidatorCodeLang } from "../validators/code-lang";

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
  nameToConfig?: Map<string, NodeConfig>,
): Record<string, unknown> {
  const targetColumns = (config.target_columns ?? [])
    .map((value) => value.trim())
    .filter(Boolean);
  if (targetColumns.length === 0) {
    errors.push(`Validator ${config.name}: target code column required.`);
  }
  if (config.validator_type === "oxc") {
    const targetName = targetColumns[0] ?? "";
    const targetConfig = targetName ? nameToConfig?.get(targetName) : null;
    let codeLang = config.code_lang;
    if (
      targetConfig &&
      targetConfig.kind === "llm" &&
      targetConfig.llm_type === "code"
    ) {
      const targetLang = (targetConfig.code_lang ?? "").trim();
      if (isValidatorCodeLang(targetLang)) {
        codeLang = targetLang;
      }
    }
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
        validation_function: `${OXC_VALIDATION_FN_MARKER}:${codeLang}:${config.oxc_validation_mode}:${config.oxc_code_shape ?? "auto"}`,
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
