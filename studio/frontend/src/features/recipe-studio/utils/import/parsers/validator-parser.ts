import type { ValidatorConfig } from "../../../types";
import { readNumberString } from "../helpers";
import { normalizeValidatorCodeLang } from "../../validators/code-lang";

export function parseValidator(
  column: Record<string, unknown>,
  name: string,
  id: string,
): ValidatorConfig {
  const targetColumns = Array.isArray(column.target_columns)
    ? column.target_columns
        .filter((value): value is string => typeof value === "string")
        .map((value) => value.trim())
        .filter(Boolean)
    : [];
  const params =
    column.validator_params && typeof column.validator_params === "object"
      ? (column.validator_params as Record<string, unknown>)
      : {};
  return {
    id,
    kind: "validator",
    name,
    drop: column.drop === true,
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: targetColumns,
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: normalizeValidatorCodeLang(params.code_lang),
    batch_size: readNumberString(column.batch_size) || "10",
  };
}
