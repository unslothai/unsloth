// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { makeValidatorConfig } = await import(
  "../src/features/recipe-studio/utils/config-factories.ts"
);
const { buildValidatorColumn } = await import(
  "../src/features/recipe-studio/utils/payload/builders-validator.ts"
);
const { parseValidator } = await import(
  "../src/features/recipe-studio/utils/import/parsers/validator-parser.ts"
);

for (const [kind, marker] of [
  ["json", "unsloth_json_validator"],
  ["markdown", "unsloth_markdown_validator"],
] as const) {
  test(`a ${kind} check survives a save and reopen`, () => {
    const config = {
      ...makeValidatorConfig("v1", kind, kind, []),
      // biome-ignore lint/style/useNamingConvention: api schema
      target_columns: ["answer"],
    };
    const errors: string[] = [];
    const column = buildValidatorColumn(config, errors);
    assert.deepEqual(errors, []);
    assert.equal(column.validator_type, "local_callable");
    assert.deepEqual(column.validator_params, {
      // biome-ignore lint/style/useNamingConvention: api schema
      validation_function: marker,
    });

    const parsed = parseValidator(column, config.name, config.id);
    assert.equal(parsed.validator_type, kind);
    assert.equal(parsed.code_lang, kind);
    assert.deepEqual(parsed.target_columns, ["answer"]);
  });
}
