import type { ValidatorConfig } from "../../../types";
import { readNumberString } from "../helpers";
import { normalizeValidatorCodeLang } from "../../validators/code-lang";
import { normalizeOxcValidationMode } from "../../validators/oxc-mode";

const OXC_VALIDATION_FN_MARKER = "unsloth_oxc_validator";

function parseOxcValidationMarker(
  validationFunctionRaw: string,
): { codeLang: string; mode: string } {
  const marker = `${OXC_VALIDATION_FN_MARKER}:`;
  if (!validationFunctionRaw.startsWith(marker)) {
    return { codeLang: "", mode: "syntax" };
  }
  const parts = validationFunctionRaw
    .slice(marker.length)
    .split(":")
    .map((value) => value.trim())
    .filter(Boolean);
  if (parts.length < 2) {
    return { codeLang: "", mode: "syntax" };
  }
  return {
    codeLang: parts[0],
    mode: parts[1],
  };
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
  const marker = isOxc
    ? parseOxcValidationMarker(validationFunctionRaw)
    : { codeLang: "", mode: "syntax" };
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
      isOxc ? marker.codeLang || "javascript" : params.code_lang,
    ),
    oxc_validation_mode: isOxc
      ? normalizeOxcValidationMode(marker.mode)
      : "syntax",
    batch_size: readNumberString(column.batch_size) || "10",
  };
}
