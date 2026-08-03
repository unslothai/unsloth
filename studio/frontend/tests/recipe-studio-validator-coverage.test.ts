// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";
import type { NodeConfig, ValidatorConfig } from "../src/features/recipe-studio/types/index.ts";

registerBundlerResolver();

const { addToolScaffoldRow, updateToolScaffoldRow, removeToolScaffoldRow } =
  await import(
    "../src/features/recipe-studio/utils/validators/tool-scaffold.ts"
  );
const { isValidatorConsentRequired } = await import(
  "../src/features/recipe-studio/utils/validators/consent.ts"
);
const {
  makeValidatorConfig,
  makeLlmConfig,
  DEFAULT_TOOL_COMMAND,
  DEFAULT_TOOL_EXT,
  DEFAULT_TOOL_SCAFFOLD,
  DEFAULT_CUSTOM_VALIDATOR_SOURCE,
} = await import(
  "../src/features/recipe-studio/utils/config-factories.ts"
);
const { nodeDataFromConfig } = await import(
  "../src/features/recipe-studio/utils/node-data.ts"
);
const {
  getBlocksForKind,
  getBlockDefinitionForConfig,
} = await import(
  "../src/features/recipe-studio/blocks/definitions.ts"
);
const { applyRecipeConnection } = await import(
  "../src/features/recipe-studio/utils/graph/recipe-graph-connection.ts"
);
const { validateValidatorConfigs } = await import(
  "../src/features/recipe-studio/utils/payload/validate.ts"
);
const markers = await import(
  "../src/features/recipe-studio/utils/validators/validation-markers.ts"
);
const { buildValidatorColumn } = await import(
  "../src/features/recipe-studio/utils/payload/builders-validator.ts"
);
const { LEARNING_RECIPES } = await import(
  "../src/features/data-recipes/learning-recipes/index.ts"
);

const TOOL_MARKER = "unsloth_tool_validator";

const GO_SCAFFOLD = [
  { path: "go.mod", content: "module example.com/check\n\ngo 1.21\n" },
  { path: "main.go", content: "{source}" },
];

function toolConfig(overrides: Partial<ValidatorConfig> = {}): ValidatorConfig {
  return {
    id: "n1",
    kind: "validator",
    name: "go_check",
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: ["code_col"],
    validator_type: "tool",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: "python",
    oxc_validation_mode: "syntax",
    oxc_code_shape: "auto",
    tool_command: "go vet ./...",
    tool_ext: "go",
    tool_scaffold: GO_SCAFFOLD.map((file) => ({ ...file })),
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
    target_columns: ["code_col"],
    validator_type: "custom",
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: "python",
    oxc_validation_mode: "syntax",
    oxc_code_shape: "auto",
    custom_source:
      "def validate(df):\n    df['is_valid'] = df.iloc[:, 0].str.len() > 0\n    return df",
    custom_acknowledged: true,
    batch_size: "10",
    ...overrides,
  };
}

function llmCodeNode(name = "code_col", codeLang = "go"): NodeConfig {
  return {
    ...makeLlmConfig(`id-${name}`, "code", []),
    name,
    // biome-ignore lint/style/useNamingConvention: api schema
    code_lang: codeLang,
  };
}

function validatorIdMap(
  validator: ValidatorConfig,
  target: NodeConfig,
): { configs: Record<string, NodeConfig>; nameToConfig: Map<string, NodeConfig> } {
  const configs: Record<string, NodeConfig> = {
    [validator.id]: validator,
    [target.id]: target,
  };
  const nameToConfig = new Map<string, NodeConfig>([
    [target.name, target],
    [validator.name, validator],
  ]);
  return { configs, nameToConfig };
}

test("addToolScaffoldRow appends an empty row", () => {
  const rows = [{ path: "go.mod", content: "x" }];
  const next = addToolScaffoldRow(rows);
  assert.deepEqual(next, [
    { path: "go.mod", content: "x" },
    { path: "", content: "" },
  ]);
  assert.deepEqual(rows, [{ path: "go.mod", content: "x" }]);
});

test("addToolScaffoldRow handles undefined rows", () => {
  assert.deepEqual(addToolScaffoldRow(undefined), [{ path: "", content: "" }]);
});

test("updateToolScaffoldRow replaces only the target row", () => {
  const rows = [
    { path: "go.mod", content: "old-mod" },
    { path: "main.go", content: "{source}" },
  ];
  const next = updateToolScaffoldRow(rows, 0, {
    path: "go.mod",
    content: "new-mod",
  });
  assert.deepEqual(next, [
    { path: "go.mod", content: "new-mod" },
    { path: "main.go", content: "{source}" },
  ]);
  assert.deepEqual(rows[0], { path: "go.mod", content: "old-mod" });
});

test("updateToolScaffoldRow handles undefined rows and out-of-range index", () => {
  assert.deepEqual(updateToolScaffoldRow(undefined, 0, { path: "a", content: "b" }), []);
  const rows = [{ path: "a.txt", content: "x" }];
  assert.deepEqual(updateToolScaffoldRow(rows, 3, { path: "b", content: "c" }), rows);
});

test("removeToolScaffoldRow removes first, middle, and last rows", () => {
  const rows = [
    { path: "a.txt", content: "1" },
    { path: "b.txt", content: "2" },
    { path: "c.txt", content: "3" },
  ];
  assert.deepEqual(removeToolScaffoldRow(rows, 0).map((row) => row.path), ["b.txt", "c.txt"]);
  assert.deepEqual(removeToolScaffoldRow(rows, 1).map((row) => row.path), ["a.txt", "c.txt"]);
  assert.deepEqual(removeToolScaffoldRow(rows, 2).map((row) => row.path), ["a.txt", "b.txt"]);
  assert.deepEqual(rows.length, 3);
});

test("removeToolScaffoldRow handles undefined rows", () => {
  assert.deepEqual(removeToolScaffoldRow(undefined, 0), []);
});

test("isValidatorConsentRequired flags only unacknowledged tool/custom checks", () => {
  assert.equal(isValidatorConsentRequired(toolConfig({ tool_acknowledged: false })), true);
  assert.equal(isValidatorConsentRequired(toolConfig()), false);
  assert.equal(isValidatorConsentRequired(customConfig({ custom_acknowledged: false })), true);
  assert.equal(isValidatorConsentRequired(customConfig()), false);
  assert.equal(
    isValidatorConsentRequired({
      ...customConfig(),
      validator_type: "code",
      // biome-ignore lint/style/useNamingConvention: api schema
      code_lang: "python",
    }),
    false,
  );
});

test("makeValidatorConfig builds the tool flavor with defaults", () => {
  const config = makeValidatorConfig("n1", "tool", "python", []);
  assert.equal(config.validator_type, "tool");
  assert.equal(config.name, "validator_tool_1");
  assert.equal(config.tool_command, DEFAULT_TOOL_COMMAND);
  assert.equal(config.tool_ext, DEFAULT_TOOL_EXT);
  assert.deepEqual(config.tool_scaffold, DEFAULT_TOOL_SCAFFOLD);
  assert.equal(config.tool_acknowledged, false);
  assert.deepEqual(config.target_columns, []);
});

test("makeValidatorConfig deep-copies the default scaffold", () => {
  const first = makeValidatorConfig("n1", "tool", "python", []);
  assert.ok(first.tool_scaffold);
  first.tool_scaffold[0].content = "mutated";
  const second = makeValidatorConfig("n2", "tool", "python", []);
  assert.equal(second.tool_scaffold?.[0]?.content, DEFAULT_TOOL_SCAFFOLD[0].content);
  assert.equal(DEFAULT_TOOL_SCAFFOLD[0].content, "module example.com/check\n\ngo 1.21\n");
});

test("makeValidatorConfig builds the custom flavor with defaults", () => {
  const config = makeValidatorConfig("n1", "custom", "python", []);
  assert.equal(config.validator_type, "custom");
  assert.equal(config.name, "validator_custom_1");
  assert.equal(config.custom_source, DEFAULT_CUSTOM_VALIDATOR_SOURCE);
  assert.equal(config.custom_acknowledged, false);
  assert.ok(!("tool_scaffold" in config));
});

test("makeValidatorConfig keeps preset flavor names", () => {
  assert.equal(makeValidatorConfig("n1", "code", "sql:postgres", []).name, "validator_sql_1");
  assert.equal(makeValidatorConfig("n2", "code", "python", []).name, "validator_python_1");
  assert.equal(makeValidatorConfig("n3", "oxc", "javascript", []).name, "validator_oxc_1");
});

test("nodeDataFromConfig maps validator subtypes and block types", () => {
  const toolData = nodeDataFromConfig(toolConfig());
  assert.equal(toolData.blockType, "validator_tool");
  assert.equal(toolData.subtype, "Custom");
  const customData = nodeDataFromConfig(customConfig());
  assert.equal(customData.blockType, "validator_custom");
  assert.equal(customData.subtype, "Advanced");
  const sqlData = nodeDataFromConfig(
    makeValidatorConfig("n1", "code", "sql:sqlite", []),
  );
  assert.equal(sqlData.blockType, "validator_sql");
  assert.equal(sqlData.subtype, "SQL");
  const pythonData = nodeDataFromConfig(
    makeValidatorConfig("n2", "code", "python", []),
  );
  assert.equal(pythonData.blockType, "validator_python");
  assert.equal(pythonData.subtype, "Python");
  const oxcData = nodeDataFromConfig(
    makeValidatorConfig("n3", "oxc", "javascript", []),
  );
  assert.equal(oxcData.blockType, "validator_oxc");
  assert.equal(oxcData.subtype, "OXC");
});

test("getBlockDefinitionForConfig maps tool and custom validators", () => {
  assert.equal(getBlockDefinitionForConfig(toolConfig())?.type, "validator_tool");
  assert.equal(getBlockDefinitionForConfig(toolConfig())?.title, "Custom check");
  assert.equal(
    getBlockDefinitionForConfig(customConfig())?.type,
    "validator_custom",
  );
  assert.equal(
    getBlockDefinitionForConfig(customConfig())?.title,
    "Advanced custom check",
  );
  assert.equal(
    getBlockDefinitionForConfig(makeValidatorConfig("n1", "code", "sql:sqlite", []))?.type,
    "validator_sql",
  );
  assert.equal(
    getBlockDefinitionForConfig(makeValidatorConfig("n2", "code", "python", []))?.type,
    "validator_python",
  );
  assert.equal(getBlockDefinitionForConfig(null), null);
});

test("validator block definitions expose all flavors with working factories", () => {
  const validatorBlocks = getBlocksForKind("validator");
  const types = validatorBlocks.map((block) => block.type).sort();
  assert.deepEqual(types, [
    "validator_custom",
    "validator_oxc",
    "validator_python",
    "validator_sql",
    "validator_tool",
  ]);
  assert.ok(
    validatorBlocks.every(
      (block) => block.dialogKey === "validator" && block.kind === "validator",
    ),
  );
  const toolBlock = validatorBlocks.find((block) => block.type === "validator_tool");
  assert.equal(
    (toolBlock?.createConfig("n1", []) as ValidatorConfig).validator_type,
    "tool",
  );
  const customBlock = validatorBlocks.find(
    (block) => block.type === "validator_custom",
  );
  assert.equal(
    (customBlock?.createConfig("n1", []) as ValidatorConfig).validator_type,
    "custom",
  );
});

function connectValidator(validator: ValidatorConfig, codeLang: string) {
  const target = llmCodeNode("code_col", codeLang);
  const { configs } = validatorIdMap(validator, target);
  const result = applyRecipeConnection(
    {
      source: target.id,
      target: validator.id,
      sourceHandle: "data-out",
      targetHandle: "data-in",
    },
    configs,
    [],
  );
  return { result, target };
}

test("applyRecipeConnection links a tool validator to any code language", () => {
  for (const codeLang of ["go", "rust", "python", "sql:postgres", "kotlin"]) {
    const { result } = connectValidator(toolConfig(), codeLang);
    const next = result.configs?.["n1"] as ValidatorConfig | undefined;
    assert.ok(next, `tool validator updated for ${codeLang}`);
    assert.deepEqual(next.target_columns, ["code_col"]);
    assert.equal(next.code_lang, codeLang);
  }
});

test("applyRecipeConnection links a custom validator to any code language", () => {
  const { result } = connectValidator(customConfig(), "go");
  const next = result.configs?.["n2"] as ValidatorConfig | undefined;
  assert.deepEqual(next?.target_columns, ["code_col"]);
  assert.equal(next?.code_lang, "go");
});

test("applyRecipeConnection keeps oxc restricted to JS-family languages", () => {
  const oxc = makeValidatorConfig("n3", "oxc", "javascript", []);
  const jsTarget = llmCodeNode("code_col", "typescript");
  const jsResult = applyRecipeConnection(
    { source: jsTarget.id, target: oxc.id, sourceHandle: "data-out", targetHandle: "data-in" },
    { n3: oxc, [jsTarget.id]: jsTarget },
    [],
  );
  assert.equal((jsResult.configs?.["n3"] as ValidatorConfig).code_lang, "typescript");

  const pythonTarget = llmCodeNode("code_col", "python");
  const pyResult = applyRecipeConnection(
    { source: pythonTarget.id, target: oxc.id, sourceHandle: "data-out", targetHandle: "data-in" },
    { n3: oxc, [pythonTarget.id]: pythonTarget },
    [],
  );
  assert.equal((pyResult.configs?.["n3"] as ValidatorConfig).code_lang, "javascript");
});

test("applyRecipeConnection applies SQL dialects for sql validators only", () => {
  const sql = makeValidatorConfig("n4", "code", "sql:sqlite", []);
  const pgTarget = llmCodeNode("code_col", "sql:postgres");
  const pgResult = applyRecipeConnection(
    { source: pgTarget.id, target: sql.id, sourceHandle: "data-out", targetHandle: "data-in" },
    { n4: sql, [pgTarget.id]: pgTarget },
    [],
  );
  assert.equal((pgResult.configs?.["n4"] as ValidatorConfig).code_lang, "sql:postgres");

  const goTarget = llmCodeNode("code_col", "go");
  const goResult = applyRecipeConnection(
    { source: goTarget.id, target: sql.id, sourceHandle: "data-out", targetHandle: "data-in" },
    { n4: sql, [goTarget.id]: goTarget },
    [],
  );
  assert.equal((goResult.configs?.["n4"] as ValidatorConfig).code_lang, "sql:sqlite");
});

test("validateValidatorConfigs reports missing and wrong targets", () => {
  const errors: string[] = [];
  const orphan = toolConfig({ target_columns: ["missing_col"] });
  const { nameToConfig } = validatorIdMap(orphan, llmCodeNode("code_col", "go"));
  const configs = { [orphan.id]: orphan };
  validateValidatorConfigs(configs, nameToConfig, errors);
  assert.ok(errors.some((message) => message.includes("missing_col")));

  const errors2: string[] = [];
  const textTarget = makeLlmConfig("t1", "text", []);
  const configs2 = { v1: toolConfig() };
  validateValidatorConfigs(
    configs2,
    new Map([["code_col", textTarget as unknown as NodeConfig]]),
    errors2,
  );
  assert.ok(errors2.some((message) => message.includes("must be LLM Code")));
});

test("validateValidatorConfigs reports tool command, extension, consent, and scaffold errors", () => {
  const errors: string[] = [];
  const { configs, nameToConfig } = validatorIdMap(
    toolConfig({
      tool_command: "  ",
      tool_ext: "  ",
      tool_acknowledged: false,
      tool_scaffold: [{ path: "../evil.txt", content: "x" }],
    }),
    llmCodeNode("code_col", "go"),
  );
  validateValidatorConfigs(configs, nameToConfig, errors);
  assert.ok(errors.some((message) => message.includes("tool command required")));
  assert.ok(errors.some((message) => message.includes("tool extension required")));
  assert.ok(errors.some((message) => message.includes("arbitrary commands")));
  assert.ok(errors.some((message) => message.includes("../evil.txt")));
});

test("validateValidatorConfigs reports custom source and consent errors", () => {
  const errors: string[] = [];
  const { configs, nameToConfig } = validatorIdMap(
    customConfig({ custom_source: "", custom_acknowledged: false }),
    llmCodeNode("code_col", "go"),
  );
  validateValidatorConfigs(configs, nameToConfig, errors);
  assert.ok(errors.some((message) => message.includes("source required")));
  assert.ok(errors.some((message) => message.includes("arbitrary Python")));
});

test("validateValidatorConfigs passes valid tool and custom configs", () => {
  const tool = toolConfig();
  const custom = customConfig();
  const target = llmCodeNode("code_col", "go");
  const errors: string[] = [];
  validateValidatorConfigs(
    { [tool.id]: tool, [custom.id]: custom, [target.id]: target },
    new Map([
      [target.name, target],
      [tool.name, tool],
      [custom.name, custom],
    ]),
    errors,
  );
  assert.deepEqual(errors, []);
});

test("buildValidatorColumn embeds scaffold rows in the tool marker", () => {
  const errors: string[] = [];
  const column = buildValidatorColumn(toolConfig(), errors);
  const marker = String(
    (column.validator_params as Record<string, unknown>).validation_function,
  );
  const spec = markers.decodeToolSpec(marker.slice(TOOL_MARKER.length + 1));
  assert.deepEqual(spec?.scaffold, GO_SCAFFOLD);
  assert.deepEqual(errors, []);
});

test("buildValidatorColumn omits scaffold when no rows are configured", () => {
  const errors: string[] = [];
  const column = buildValidatorColumn(
    toolConfig({ tool_scaffold: [] }),
    errors,
  );
  const marker = String(
    (column.validator_params as Record<string, unknown>).validation_function,
  );
  const spec = markers.decodeToolSpec(marker.slice(TOOL_MARKER.length + 1));
  assert.equal("scaffold" in (spec ?? {}), false);
  assert.deepEqual(errors, []);
});

test("LEARNING_RECIPES registers the text-to-go and text-to-rust templates", () => {
  const ids = LEARNING_RECIPES.map((recipe) => recipe.id);
  assert.ok(ids.includes("text-to-go"));
  assert.ok(ids.includes("text-to-rust"));
  const rust = LEARNING_RECIPES.find((recipe) => recipe.id === "text-to-rust");
  assert.equal(rust?.title, "Text to Rust");
  assert.equal(typeof rust?.loadPayload, "function");
  const go = LEARNING_RECIPES.find((recipe) => recipe.id === "text-to-go");
  assert.equal(go?.title, "Text to Go");
  assert.equal(typeof go?.loadPayload, "function");
});
