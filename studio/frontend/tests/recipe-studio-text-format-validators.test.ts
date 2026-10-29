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

const { makeLlmConfig, makeSamplerConfig } = await import(
  "../src/features/recipe-studio/utils/config-factories.ts"
);
const { applyRecipeConnection } = await import(
  "../src/features/recipe-studio/utils/graph/recipe-graph-connection.ts"
);
const { HANDLE_IDS } = await import(
  "../src/features/recipe-studio/utils/handles.ts"
);

for (const [label, source] of [
  ["an AI text", makeLlmConfig("src", "text", [])],
  ["a sampler", makeSamplerConfig("src", "category", [])],
] as const) {
  test(`dragging ${label} output onto a JSON check's semantic input sets its target`, () => {
    const validator = makeValidatorConfig("v1", "json", "json", []);
    const configs = { [source.id]: source, [validator.id]: validator };
    const result = applyRecipeConnection(
      {
        source: source.id,
        target: validator.id,
        sourceHandle: HANDLE_IDS.dataOut,
        targetHandle: HANDLE_IDS.semanticIn,
      },
      configs,
      [],
    );
    assert.equal(result.edges.length, 1);
    assert.equal(result.edges[0].type, "semantic");
    const next = result.configs?.[validator.id];
    assert.ok(next && next.kind === "validator");
    assert.deepEqual(next.target_columns, [source.name]);
  });
}

for (const downstreamType of ["text", "code"] as const) {
  test(`a JSON check's result can feed a later AI ${downstreamType} step without retargeting the check`, () => {
    const source = makeLlmConfig("src", "text", []);
    const validator = {
      ...makeValidatorConfig("v1", "json", "json", []),
      // biome-ignore lint/style/useNamingConvention: api schema
      target_columns: [source.name],
    };
    const downstream = makeLlmConfig("down", downstreamType, [source]);
    const configs = {
      [source.id]: source,
      [validator.id]: validator,
      [downstream.id]: downstream,
    };
    const result = applyRecipeConnection(
      {
        source: validator.id,
        target: downstream.id,
        sourceHandle: HANDLE_IDS.dataOut,
        targetHandle: HANDLE_IDS.dataIn,
      },
      configs,
      [],
    );
    assert.equal(result.edges.length, 1);
    assert.equal(result.edges[0].source, validator.id);
    assert.notEqual(result.edges[0].type, "semantic");
    const next = result.configs?.[validator.id] ?? validator;
    assert.ok(next.kind === "validator");
    assert.deepEqual(next.target_columns, [source.name]);
  });
}

const { syncEdgesForConfigPatch } = await import(
  "../src/features/recipe-studio/stores/helpers/edge-sync.ts"
);

test("changing a JSON check's field keeps the edge to a step reading its result", () => {
  const first = makeLlmConfig("a", "text", []);
  const second = makeLlmConfig("b", "text", [first]);
  const downstream = makeLlmConfig("c", "text", [first, second]);
  const validator = {
    ...makeValidatorConfig("v1", "json", "json", []),
    // biome-ignore lint/style/useNamingConvention: api schema
    target_columns: [first.name],
  };
  const configs = {
    [first.id]: first,
    [second.id]: second,
    [downstream.id]: downstream,
    [validator.id]: validator,
  };
  const edges = [
    { id: "in", source: first.id, target: validator.id, type: "semantic" },
    { id: "out", source: validator.id, target: downstream.id, type: "canvas" },
  ];
  const next = syncEdgesForConfigPatch(
    validator,
    // biome-ignore lint/style/useNamingConvention: api schema
    { target_columns: [second.name] },
    configs,
    edges,
    "LR",
  );
  const pairs = next.map((edge) => `${edge.source}->${edge.target}`).sort();
  assert.deepEqual(pairs, [
    `${second.id}->${validator.id}`,
    `${validator.id}->${downstream.id}`,
  ]);
});
