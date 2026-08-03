// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";
import type { ValidatorConfig } from "../src/features/recipe-studio/types/index.ts";

registerBundlerResolver();

const markers = await import(
  "../src/features/recipe-studio/utils/validators/validation-markers.ts"
);
const { buildValidatorColumn } = await import(
  "../src/features/recipe-studio/utils/payload/builders-validator.ts"
);
const { parseValidator } = await import(
  "../src/features/recipe-studio/utils/import/parsers/validator-parser.ts"
);
const { getConfigErrors } = await import(
  "../src/features/recipe-studio/utils/validation.ts"
);

const TOOL_MARKER = "unsloth_tool_validator";
const CUSTOM_MARKER = "unsloth_custom_validator";

function toolConfig(overrides: Partial<ValidatorConfig> = {}): ValidatorConfig {
  return {
    id: "n1",
    kind: "validator",
    name: "go_check",
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: ["code"],
    validator_type: "tool",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: "python",
    oxc_validation_mode: "syntax",
    oxc_code_shape: "auto",
    tool_command: "go vet ./...",
    tool_ext: "go",
    tool_acknowledged: true,
    batch_size: "10",
    ...overrides,
  };
}

function customConfig(overrides: Partial<ValidatorConfig> = {}): ValidatorConfig {
  return {
    id: "n2",
    kind: "validator",
    name: "py_check",
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: ["code"],
    validator_type: "custom",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: "python",
    oxc_validation_mode: "syntax",
    oxc_code_shape: "auto",
    custom_source: "def validate(df):\n    df['is_valid'] = df.iloc[:, 0].str.len() > 0\n    return df",
    custom_acknowledged: true,
    batch_size: "10",
    ...overrides,
  };
}

test("tool marker encodes and decodes", () => {
  const encoded = markers.encodeToolSpec({ ext: "go", command: "go vet ./..." });
  assert.ok(!encoded.includes(":"));
  assert.deepEqual(markers.decodeToolSpec(encoded), {
    ext: "go",
    command: "go vet ./...",
  });
});

test("decodeToolSpec rejects unsafe extensions", () => {
  const encoded = markers.encodeToolSpec({ ext: "../x", command: "go vet" });
  assert.equal(markers.decodeToolSpec(encoded), null);
  assert.equal(markers.decodeToolSpec("!!not base64!!"), null);
});

test("custom source encodes and decodes unicode", () => {
  const source =
    "def validate(df):\n    df['is_valid'] = df['code'].str.contains('好')\n    return df";
  const encoded = markers.encodeCustomSource(source);
  assert.ok(!encoded.includes(":"));
  assert.equal(markers.decodeCustomSource(encoded), source);
});

test("validationFunctionFromConfig builds a tool marker", () => {
  const marker = markers.validationFunctionFromConfig(toolConfig());
  assert.ok(marker?.startsWith(`${TOOL_MARKER}:`));
  const spec = markers.decodeToolSpec(marker!.slice(TOOL_MARKER.length + 1));
  assert.deepEqual(spec, { ext: "go", command: "go vet ./..." });
});

test("validationFunctionFromConfig builds a custom marker", () => {
  const marker = markers.validationFunctionFromConfig(customConfig());
  assert.ok(marker?.startsWith(`${CUSTOM_MARKER}:`));
  const source = markers.decodeCustomSource(marker!.slice(CUSTOM_MARKER.length + 1));
  assert.equal(source, customConfig().custom_source);
});

test("buildValidatorColumn emits tool local_callable", () => {
  const errors: string[] = [];
  const column = buildValidatorColumn(toolConfig(), errors);
  const params = column.validator_params as Record<string, unknown>;
  assert.equal(column.validator_type, "local_callable");
  assert.equal(
    String(params.validation_function).split(":")[0],
    TOOL_MARKER,
  );
  assert.deepEqual(errors, []);
});

test("buildValidatorColumn emits custom local_callable", () => {
  const errors: string[] = [];
  const column = buildValidatorColumn(customConfig(), errors);
  const params = column.validator_params as Record<string, unknown>;
  assert.equal(column.validator_type, "local_callable");
  assert.equal(
    String(params.validation_function).split(":")[0],
    CUSTOM_MARKER,
  );
  assert.deepEqual(errors, []);
});

test("buildValidatorColumn flags missing tool command", () => {
  const errors: string[] = [];
  buildValidatorColumn(
    toolConfig({ tool_command: "  ", tool_acknowledged: false }),
    errors,
  );
  assert.ok(errors.some((message) => message.includes("tool command")));
});

test("parseValidator reconstructs a tool config", () => {
  const marker = markers.validationFunctionFromConfig(toolConfig());
  const config = parseValidator(
    {
      column_type: "validation",
      name: "go_check",
      drop: false,
      target_columns: ["code"],
      validator_type: "local_callable",
      validator_params: { validation_function: marker },
      batch_size: 10,
    },
    "go_check",
    "n1",
  );
  assert.equal(config.validator_type, "tool");
  assert.equal(config.tool_command, "go vet ./...");
  assert.equal(config.tool_ext, "go");
  assert.equal(config.tool_acknowledged, true);
});

test("parseValidator reconstructs a custom config", () => {
  const marker = markers.validationFunctionFromConfig(customConfig());
  const config = parseValidator(
    {
      column_type: "validation",
      name: "py_check",
      drop: false,
      target_columns: ["code"],
      validator_type: "local_callable",
      validator_params: { validation_function: marker },
      batch_size: 10,
    },
    "py_check",
    "n2",
  );
  assert.equal(config.validator_type, "custom");
  assert.equal(config.custom_source, customConfig().custom_source);
  assert.equal(config.custom_acknowledged, true);
});

test("getConfigErrors requires tool acknowledgement", () => {
  const unacked = getConfigErrors(toolConfig({ tool_acknowledged: false }));
  assert.ok(unacked.some((message) => message.includes("arbitrary commands")));
  const acked = getConfigErrors(toolConfig());
  assert.ok(!acked.some((message) => message.includes("arbitrary commands")));
});

test("getConfigErrors requires custom acknowledgement", () => {
  const unacked = getConfigErrors(customConfig({ custom_acknowledged: false }));
  assert.ok(unacked.some((message) => message.includes("arbitrary Python")));
  const acked = getConfigErrors(customConfig());
  assert.ok(!acked.some((message) => message.includes("arbitrary Python")));
});

test("text-to-go learning recipe round-trips the tool validator", async () => {
  const recipeJson = await import(
    "../src/features/data-recipes/learning-recipes/text-to-go.json",
    { with: { type: "json" } }
  );
  const payload = recipeJson.default;
  const columns = payload.recipe.columns;
  const goCheck = columns.find((column) => column.name === "go_check");
  assert.ok(goCheck, "go_check column exists");
  assert.equal(goCheck.validator_type, "local_callable");
  const config = parseValidator(goCheck, "go_check", "n9");
  assert.equal(config.validator_type, "tool");
  assert.equal(config.tool_ext, "go");
  assert.equal(
    config.tool_command,
    "go vet ./... && go build ./...",
  );
  assert.equal(config.tool_acknowledged, true);
});
