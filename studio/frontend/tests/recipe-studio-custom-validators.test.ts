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
const B64_PLUS_RE = /\+/g;
const B64_SLASH_RE = /\//g;
const B64_PADDING_RE = /=+$/;

function toBase64Url(input: string): string {
  return btoa(input).replace(B64_PLUS_RE, "-").replace(B64_SLASH_RE, "_").replace(B64_PADDING_RE, "");
}

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

const GO_SCAFFOLD = [
  { path: "go.mod", content: "module example.com/check\n\ngo 1.21\n" },
  { path: "main.go", content: "{source}" },
];

test("encodeToolSpec round-trips scaffold rows", () => {
  const encoded = markers.encodeToolSpec({
    ext: "rs",
    command: "cargo check",
    scaffold: [
      { path: "Cargo.toml", content: '[package]\nname = "check"\n' },
      { path: "src/main.rs", content: "{source}" },
    ],
  });
  assert.ok(!encoded.includes(":"));
  assert.deepEqual(markers.decodeToolSpec(encoded), {
    ext: "rs",
    command: "cargo check",
    scaffold: [
      { path: "Cargo.toml", content: '[package]\nname = "check"\n' },
      { path: "src/main.rs", content: "{source}" },
    ],
  });
});

test("encodeToolSpec omits empty scaffold", () => {
  const encoded = markers.encodeToolSpec({
    ext: "sql",
    command: "sqlfluff lint {file}",
    scaffold: [],
  });
  assert.deepEqual(markers.decodeToolSpec(encoded), {
    ext: "sql",
    command: "sqlfluff lint {file}",
  });
});

test("decodeToolSpec rejects unsafe scaffold paths", () => {
  const spec = {
    ext: "txt",
    command: "cat {file}",
    scaffold: [{ path: "../evil.txt", content: "x" }],
  };
  const encoded = toBase64Url(JSON.stringify(spec));
  assert.equal(markers.decodeToolSpec(encoded), null);
});

test("firstInvalidToolScaffoldPath reports bad rows", () => {
  assert.equal(markers.firstInvalidToolScaffoldPath(undefined), null);
  assert.equal(
    markers.firstInvalidToolScaffoldPath([
      { path: "go.mod", content: "x" },
      { path: "../evil.txt", content: "x" },
    ]),
    "../evil.txt",
  );
  assert.equal(
    markers.firstInvalidToolScaffoldPath([
      { path: "src/main.rs", content: "{source}" },
    ]),
    null,
  );
});

test("toolScaffoldLimitError reports oversized scaffolds", () => {
  assert.equal(markers.toolScaffoldLimitError(undefined), null);
  assert.equal(markers.toolScaffoldLimitError(GO_SCAFFOLD), null);
  const tooManyRows = Array.from({ length: 11 }, (_, index) => ({
    path: `file${index}.txt`,
    content: "x",
  }));
  assert.ok(markers.toolScaffoldLimitError(tooManyRows)?.includes("max 10"));
  const tooMuchContent = [{ path: "big.txt", content: "x".repeat(32 * 1024) }];
  assert.ok(markers.toolScaffoldLimitError(tooMuchContent)?.includes("32 KiB"));
  // normalizeToolScaffold keeps the same guard for serialization.
  assert.deepEqual(markers.normalizeToolScaffold(tooManyRows), []);
  assert.deepEqual(markers.normalizeToolScaffold(tooMuchContent), []);
});

test("getConfigErrors flags oversized scaffolds", () => {
  const tooManyRows = Array.from({ length: 11 }, (_, index) => ({
    path: `file${index}.txt`,
    content: "x",
  }));
  const errors = getConfigErrors(
    toolConfig({ tool_scaffold: tooManyRows, tool_acknowledged: true }),
  );
  assert.ok(errors.some((message) => message.includes("Scaffold: Too many")));
});

test("getConfigErrors flags invalid scaffold paths", () => {
  const errors = getConfigErrors(
    toolConfig({
      tool_scaffold: [{ path: "../evil.txt", content: "x" }],
      tool_acknowledged: true,
    }),
  );
  assert.ok(errors.some((message) => message.includes("../evil.txt")));
});

test("validationFunctionFromConfig includes scaffold in the tool marker", () => {
  const marker = markers.validationFunctionFromConfig(
    toolConfig({ tool_scaffold: GO_SCAFFOLD }),
  );
  assert.ok(marker?.startsWith(`${TOOL_MARKER}:`));
  const spec = markers.decodeToolSpec(marker!.slice(TOOL_MARKER.length + 1));
  assert.deepEqual(spec, { ext: "go", command: "go vet ./...", scaffold: GO_SCAFFOLD });
});

test("parseValidator reconstructs tool_scaffold", () => {
  const marker = markers.validationFunctionFromConfig(
    toolConfig({ tool_scaffold: GO_SCAFFOLD }),
  );
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
  assert.deepEqual(config.tool_scaffold, GO_SCAFFOLD);
  assert.equal(config.tool_acknowledged, false);
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
  assert.equal(config.tool_acknowledged, false);
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
  assert.equal(config.custom_acknowledged, false);
});

test("parseValidator keeps accepting legacy bare OXC markers", () => {
  const config = parseValidator(
    {
      column_type: "validation",
      name: "js_check",
      drop: false,
      target_columns: ["code"],
      validator_type: "local_callable",
      validator_params: { validation_function: "unsloth_oxc_validator" },
      batch_size: 10,
    },
    "js_check",
    "n1",
  );
  // The backend treats a bare marker as a default JS syntax check; the
  // importer must reconstruct an OXC validator, not a Python code validator.
  assert.equal(config.validator_type, "oxc");
  assert.equal(config.code_lang, "javascript");
  assert.equal(config.oxc_validation_mode, "syntax");
  assert.equal(config.oxc_code_shape, "auto");
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
  // Importing a recipe does not count as the current user's consent.
  assert.equal(config.tool_acknowledged, false);
});

test("text-to-go learning recipe carries the go scaffold", async () => {
  const recipeJson = await import(
    "../src/features/data-recipes/learning-recipes/text-to-go.json",
    { with: { type: "json" } }
  );
  const payload = recipeJson.default;
  const goCheck = payload.recipe.columns.find(
    (column) => column.name === "go_check",
  );
  assert.ok(goCheck, "go_check column exists");
  const params = goCheck.validator_params as Record<string, unknown>;
  const marker = String(params.validation_function);
  const spec = markers.decodeToolSpec(marker.slice(TOOL_MARKER.length + 1));
  assert.ok(spec?.scaffold, "go_check marker carries scaffold rows");
  assert.deepEqual(spec.scaffold, [
    { path: "go.mod", content: "module example.com/check\n\ngo 1.21\n" },
    { path: "main.go", content: "{source}" },
  ]);
});

test("text-to-rust learning recipe round-trips the custom validator", async () => {
  const recipeJson = await import(
    "../src/features/data-recipes/learning-recipes/text-to-rust.json",
    { with: { type: "json" } }
  );
  const payload = recipeJson.default;
  const columns = payload.recipe.columns;
  const rustCheck = columns.find((column) => column.name === "rust_check");
  assert.ok(rustCheck, "rust_check column exists");
  assert.equal(rustCheck.validator_type, "local_callable");
  const config = parseValidator(rustCheck, "rust_check", "n9");
  assert.equal(config.validator_type, "custom");
  assert.ok((config.custom_source ?? "").includes("cargo check"));
  assert.ok((config.custom_source ?? "").includes("Cargo.toml"));
  // Importing a recipe does not count as the current user's consent.
  assert.equal(config.custom_acknowledged, false);
});
