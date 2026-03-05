import type { ValidatorConfig } from "../../../types";
import { readNumberString } from "../helpers";
import { normalizeValidatorCodeLang } from "../../validators/code-lang";

const OXC_VALIDATION_FN_MARKER = "unsloth_oxc_validator";

function parseOxcCodeLang(validationFunctionRaw: string): string {
  if (!validationFunctionRaw.startsWith(OXC_VALIDATION_FN_MARKER)) {
    return "";
  }
  const suffix = validationFunctionRaw
    .slice(OXC_VALIDATION_FN_MARKER.length)
    .trim();
  if (!suffix) {
    return "";
  }
  const normalizedSuffix = suffix.startsWith(":") || suffix.startsWith("_")
    ? suffix.slice(1).trim().toLowerCase()
    : suffix.toLowerCase();
  if (normalizedSuffix === "js") {
    return "javascript";
  }
  if (normalizedSuffix === "ts") {
    return "typescript";
  }
  if (normalizedSuffix === "jsx" || normalizedSuffix === "tsx") {
    return normalizedSuffix;
  }
  if (normalizedSuffix === "javascript" || normalizedSuffix === "typescript") {
    return normalizedSuffix;
  }
  return "";
}

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
  const validationFunctionRaw =
    typeof params.validation_function === "string"
      ? params.validation_function.trim()
      : "";
  const isOxc =
    String(column.validator_type ?? "").trim() === "local_callable" &&
    validationFunctionRaw.startsWith(OXC_VALIDATION_FN_MARKER);
  const oxcLang = isOxc ? parseOxcCodeLang(validationFunctionRaw) : "";
  return {
    id,
    kind: "validator",
    name,
    drop: column.drop === true,
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: targetColumns,
    validator_type: isOxc ? "oxc" : "code",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: normalizeValidatorCodeLang(
      isOxc ? oxcLang || "javascript" : params.code_lang,
    ),
    batch_size: readNumberString(column.batch_size) || "10",
  };
}
