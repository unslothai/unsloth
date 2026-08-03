// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";
import type { RecipePayload } from "../src/features/recipe-studio/utils/payload/types.ts";
import type { RecipeRunSettings } from "../src/features/recipe-studio/stores/recipe-executions.ts";

registerBundlerResolver();

const { buildExecutionPayload } = await import(
  "../src/features/recipe-studio/executions/run-settings.ts"
);

const TOOL_MARKER = "unsloth_tool_validator";
const CUSTOM_MARKER = "unsloth_custom_validator";

const DEFAULT_SETTINGS: RecipeRunSettings = {
  batchSize: 1000,
  batchEnabled: false,
  mergeBatches: false,
  llmParallelRequests: null,
  nonInferenceWorkers: 4,
  maxConversationRestarts: 5,
  maxConversationCorrectionSteps: 0,
  disableEarlyShutdown: false,
  shutdownErrorRate: 0.5,
  shutdownErrorWindow: 60,
};

function basePayload(columns: Record<string, unknown>[]): RecipePayload {
  return {
    recipe: {
      // biome-ignore lint/style/useNamingConvention: api schema
      model_providers: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      mcp_providers: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      model_configs: [],
      // biome-ignore lint/style/useNamingConvention: api schema
      tool_configs: [],
      columns,
      processors: [],
    },
    run: { rows: 5, preview: true, output_formats: ["jsonl"] },
    ui: { nodes: [], edges: [] },
  };
}

function validationColumn(validationFunction: string): Record<string, unknown> {
  return {
    column_type: "validation",
    name: "check",
    target_columns: ["code"],
    validator_type: "local_callable",
    validator_params: { validation_function: validationFunction },
    batch_size: 10,
  };
}

function buildRun(payload: RecipePayload): Record<string, unknown> {
  return buildExecutionPayload({
    payload,
    kind: "preview",
    rows: 5,
    settings: DEFAULT_SETTINGS,
  }).run as Record<string, unknown>;
}

test("execution payload attests consent when tool/custom markers exist", () => {
  const toolRun = buildRun(
    basePayload([validationColumn(`${TOOL_MARKER}:eyJleHQiOiJ0eHQiLCJjb21tYW5kIjoiZWNobyB7ZmlsZX0ifQ`)]),
  );
  assert.equal(toolRun.local_execution_consent, true);

  const customRun = buildRun(
    basePayload([validationColumn(`${CUSTOM_MARKER}:ZGVmIHZhbGlkYXRlKGRmKTogcmV0dXJuIGRm`)]),
  );
  assert.equal(customRun.local_execution_consent, true);
});

test("execution payload omits consent for recipes without local checks", () => {
  const codeRun = buildRun(
    basePayload([
      {
        column_type: "validation",
        name: "code_check",
        target_columns: ["code"],
        validator_type: "code",
        validator_params: { code_lang: "python" },
        batch_size: 10,
      },
    ]),
  );
  assert.equal(codeRun.local_execution_consent, undefined);

  const oxcRun = buildRun(
    basePayload([
      validationColumn("unsloth_oxc_validator:javascript:syntax:auto"),
    ]),
  );
  assert.equal(oxcRun.local_execution_consent, undefined);

  const emptyRun = buildRun(basePayload([]));
  assert.equal(emptyRun.local_execution_consent, undefined);
});
