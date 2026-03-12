// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeConfig } from "../types";

export type AvailableVariableSource = "column" | "seed";

export type AvailableVariableEntry = {
  name: string;
  source: AvailableVariableSource;
};

function getStructuredRefs(llmName: string, outputFormat: string): string[] {
  try {
    const schema = JSON.parse(outputFormat);
    if (!(schema?.properties && typeof schema.properties === "object")) {
      return [];
    }
    return Object.keys(schema.properties).map((key) => `${llmName}.${key}`);
  } catch {
    return [];
  }
}

export function getAvailableVariableEntries(
  configs: Record<string, NodeConfig>,
  currentId: string,
): AvailableVariableEntry[] {
  const vars: AvailableVariableEntry[] = [];

  for (const config of Object.values(configs)) {
    if (config.id === currentId) {
      continue;
    }
    if (
      config.kind === "model_provider" ||
      config.kind === "model_config" ||
      config.kind === "tool_config"
    ) {
      continue;
    }

    if (config.kind === "sampler") {
      vars.push({ name: config.name, source: "column" });
      continue;
    }

    if (config.kind === "expression") {
      vars.push({ name: config.name, source: "column" });
      continue;
    }

    if (config.kind === "validator") {
      vars.push({ name: config.name, source: "column" });
      continue;
    }

    if (config.kind === "seed") {
      for (const col of config.seed_columns ?? []) {
        const name = col.trim();
        if (!name) continue;
        vars.push({ name, source: "seed" });
      }
      continue;
    }

    if (config.kind !== "llm") {
      continue;
    }

    vars.push({ name: config.name, source: "column" });
    if (config.llm_type !== "structured" || !config.output_format) {
      continue;
    }
    vars.push(
      ...getStructuredRefs(config.name, config.output_format).map((name) => ({
        name,
        source: "column" as const,
      })),
    );
  }

  return vars;
}

export function getAvailableVariables(
  configs: Record<string, NodeConfig>,
  currentId: string,
): string[] {
  return getAvailableVariableEntries(configs, currentId).map((entry) => entry.name);
}
