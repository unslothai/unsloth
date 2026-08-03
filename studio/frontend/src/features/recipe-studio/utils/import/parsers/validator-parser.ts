// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ValidatorConfig } from "../../../types";
import { readNumberString } from "../helpers";
import { normalizeValidatorCodeLang } from "../../validators/code-lang";
import { normalizeOxcCodeShape } from "../../validators/oxc-code-shape";
import { normalizeOxcValidationMode } from "../../validators/oxc-mode";
import {
  CUSTOM_VALIDATION_FN_MARKER,
  decodeCustomSource,
  decodeToolSpec,
  TOOL_VALIDATION_FN_MARKER,
  TOOL_OUTPUT_MAX_KIB_DEFAULT,
  TOOL_SCAFFOLD_FILE_MAX_KIB_DEFAULT,
} from "../../validators/validation-markers";

const OXC_VALIDATION_FN_MARKER = "unsloth_oxc_validator";

function parseOxcValidationMarker(
  validationFunctionRaw: string,
): { codeLang: string; mode: string; codeShape: string } {
  const marker = `${OXC_VALIDATION_FN_MARKER}:`;
  if (!validationFunctionRaw.startsWith(marker)) {
    // Legacy bare marker ("unsloth_oxc_validator" with no colon suffix): the
    // backend still accepts it and treats it as a default JS syntax check.
    return { codeLang: "javascript", mode: "syntax", codeShape: "auto" };
  }
  const parts = validationFunctionRaw
    .slice(marker.length)
    .split(":")
    .map((value) => value.trim())
    .filter(Boolean);
  if (parts.length < 2) {
    return { codeLang: "javascript", mode: "syntax", codeShape: "auto" };
  }
  return {
    codeLang: parts[0],
    mode: parts[1],
    codeShape: parts[2] ?? "auto",
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
  const isLocalCallable =
    String(column.validator_type ?? "").trim() === "local_callable";
  const isOxc =
    isLocalCallable && validationFunctionRaw.startsWith(OXC_VALIDATION_FN_MARKER);
  const isTool =
    isLocalCallable && validationFunctionRaw.startsWith(`${TOOL_VALIDATION_FN_MARKER}:`);
  const isCustom =
    isLocalCallable &&
    validationFunctionRaw.startsWith(`${CUSTOM_VALIDATION_FN_MARKER}:`);
  const marker = isOxc
    ? parseOxcValidationMarker(validationFunctionRaw)
    : { codeLang: "", mode: "syntax", codeShape: "auto" };
  const toolSpec = isTool
    ? decodeToolSpec(
        validationFunctionRaw.slice(TOOL_VALIDATION_FN_MARKER.length + 1),
      )
    : null;
  const customSource = isCustom
    ? decodeCustomSource(
        validationFunctionRaw.slice(CUSTOM_VALIDATION_FN_MARKER.length + 1),
      )
    : "";
  return {
    id,
    kind: "validator",
    name,
    drop: column.drop === true,
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: targetColumns,
    validator_type: isOxc ? "oxc" : isTool ? "tool" : isCustom ? "custom" : "code",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: normalizeValidatorCodeLang(
      isOxc ? marker.codeLang || "javascript" : params.code_lang,
    ),
    oxc_validation_mode: isOxc
      ? normalizeOxcValidationMode(marker.mode)
      : "syntax",
    oxc_code_shape: isOxc
      ? normalizeOxcCodeShape(marker.codeShape)
      : "auto",
    tool_command: isTool ? toolSpec?.command ?? "" : undefined,
    tool_ext: isTool ? toolSpec?.ext ?? "" : undefined,
    tool_scaffold:
      isTool && toolSpec && (toolSpec.scaffold?.length ?? 0) > 0
        ? toolSpec.scaffold
        : undefined,
    tool_output_max_kib: isTool
      ? String((toolSpec?.output_max_chars ?? TOOL_OUTPUT_MAX_KIB_DEFAULT * 1024) / 1024)
      : undefined,
    tool_scaffold_file_max_kib: isTool
      ? String(
          (toolSpec?.source_file_max_chars ?? TOOL_SCAFFOLD_FILE_MAX_KIB_DEFAULT * 1024) /
            1024,
        )
      : undefined,
    // Importing a recipe does not count as the current user's consent: an
    // imported marker can hide an arbitrary command or Python body, so the
    // user must acknowledge the local-execution warning before a run starts.
    tool_acknowledged: isTool ? false : undefined,
    custom_source: isCustom ? customSource : undefined,
    custom_acknowledged: isCustom ? false : undefined,
    batch_size: readNumberString(column.batch_size) || "10",
  };
}
