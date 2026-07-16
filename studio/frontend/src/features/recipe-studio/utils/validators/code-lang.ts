// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ValidatorCodeLang } from "../../types";

export const VALIDATOR_OXC_CODE_LANGS: ValidatorCodeLang[] = [
  "javascript",
  "typescript",
  "jsx",
  "tsx",
];

export const VALIDATOR_SQL_CODE_LANGS: ValidatorCodeLang[] = [
  "sql:sqlite",
  "sql:postgres",
  "sql:mysql",
  "sql:tsql",
  "sql:bigquery",
  "sql:ansi",
];

const VALIDATOR_CODE_LANG_SET = new Set<ValidatorCodeLang>([
  ...VALIDATOR_OXC_CODE_LANGS,
  "python",
  ...VALIDATOR_SQL_CODE_LANGS,
]);

export function isValidatorCodeLang(value: string): value is ValidatorCodeLang {
  return VALIDATOR_CODE_LANG_SET.has(value as ValidatorCodeLang);
}

export function normalizeValidatorCodeLang(
  value: unknown,
): ValidatorCodeLang {
  const raw = typeof value === "string" ? value.trim() : "";
  if (!raw) {
    return "python";
  }
  if (VALIDATOR_OXC_CODE_LANGS.includes(raw as ValidatorCodeLang)) {
    return raw as ValidatorCodeLang;
  }
  if (raw === "python") {
    return "python";
  }
  if (raw.startsWith("sql:")) {
    if (VALIDATOR_SQL_CODE_LANGS.includes(raw as ValidatorCodeLang)) {
      return raw as ValidatorCodeLang;
    }
    return "sql:sqlite";
  }
  return "python";
}
