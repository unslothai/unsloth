// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  LlmConfig,
  ModelConfig,
  NodeConfig,
  SamplerConfig,
} from "../../types";
import { removeRef, replaceRef } from "../../utils/refs";

function updateTemplateFields(
  config: NodeConfig,
  updater: (value: string) => string,
): NodeConfig {
  if (config.kind === "llm") {
    const nextPrompt = updater(config.prompt);
    const nextSystem = updater(config.system_prompt);
    const nextOutput =
      typeof config.output_format === "string"
        ? updater(config.output_format)
        : config.output_format;
    if (
      nextPrompt === config.prompt &&
      nextSystem === config.system_prompt &&
      nextOutput === config.output_format
    ) {
      return config;
    }
    return {
      ...config,
      prompt: nextPrompt,
      // biome-ignore lint/style/useNamingConvention: api schema
      system_prompt: nextSystem,
      // biome-ignore lint/style/useNamingConvention: api schema
      output_format: nextOutput,
    };
  }
  if (config.kind === "expression") {
    const nextExpr = updater(config.expr);
    if (nextExpr === config.expr) {
      return config;
    }
    return { ...config, expr: nextExpr };
  }
  return config;
}

export function applyRenameToConfig(
  config: NodeConfig,
  from: string,
  to: string,
): NodeConfig {
  let next = updateTemplateFields(config, (value) =>
    replaceRef(value, from, to),
  );
  if (
    config.kind === "sampler" &&
    config.sampler_type === "subcategory" &&
    config.subcategory_parent === from
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: to,
    };
  }
  if (
    config.kind === "sampler" &&
    config.sampler_type === "timedelta" &&
    config.reference_column_name === from
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column_name: to,
    };
  }
  if (config.kind === "model_config" && config.provider === from) {
    const base = next as ModelConfig;
    next = { ...base, provider: to };
  }
  if (config.kind === "llm" && config.model_alias === from) {
    const base = next as LlmConfig;
    next = { ...base, model_alias: to };
  }
  if (config.kind === "llm" && config.tool_alias === from) {
    const base = next as LlmConfig;
    next = { ...base, tool_alias: to };
  }
  if (config.kind === "validator") {
    const targets = config.target_columns ?? [];
    if (targets.includes(from)) {
      const base = next as typeof config;
      next = {
        ...base,
        // biome-ignore lint/style/useNamingConvention: api schema
        target_columns: targets.map((target) => (target === from ? to : target)),
      };
    }
  }
  return next;
}

export function applyRemovalToConfig(
  config: NodeConfig,
  ref: string,
): NodeConfig {
  let next = updateTemplateFields(config, (value) => removeRef(value, ref));
  if (
    config.kind === "sampler" &&
    config.sampler_type === "subcategory" &&
    config.subcategory_parent === ref
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_parent: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      subcategory_mapping: {},
    };
  }
  if (
    config.kind === "sampler" &&
    config.sampler_type === "timedelta" &&
    config.reference_column_name === ref
  ) {
    const base = next as SamplerConfig;
    next = {
      ...base,
      // biome-ignore lint/style/useNamingConvention: api schema
      reference_column_name: "",
    };
  }
  if (config.kind === "model_config" && config.provider === ref) {
    const base = next as ModelConfig;
    // Clear the synthetic "local" placeholder when the provider that was
    // a local provider is removed; otherwise the stale placeholder would
    // pass validation against a future external provider and then fail
    // at runtime against a real API ("model not found").
    next = {
      ...base,
      provider: "",
      model: base.model === "local" ? "" : base.model,
    };
  }
  if (config.kind === "llm" && config.model_alias === ref) {
    const base = next as LlmConfig;
    next = { ...base, model_alias: "" };
  }
  if (config.kind === "llm" && config.tool_alias === ref) {
    const base = next as LlmConfig;
    next = { ...base, tool_alias: "" };
  }
  if (config.kind === "validator") {
    const targets = (config.target_columns ?? []).filter((target) => target !== ref);
    if (targets.length !== (config.target_columns ?? []).length) {
      const base = next as typeof config;
      next = {
        ...base,
        // biome-ignore lint/style/useNamingConvention: api schema
        target_columns: targets,
      };
    }
  }
  return next;
}

function applyConfigTransform(
  configs: Record<string, NodeConfig>,
  transform: (config: NodeConfig) => NodeConfig,
): Record<string, NodeConfig> {
  let next = configs;
  for (const [id, config] of Object.entries(configs)) {
    const updated = transform(config);
    if (updated !== config) {
      if (next === configs) {
        next = { ...configs };
      }
      next[id] = updated;
    }
  }
  return next;
}

export function applyRenameToConfigs(
  configs: Record<string, NodeConfig>,
  from: string,
  to: string,
): Record<string, NodeConfig> {
  if (!from || from === to) {
    return configs;
  }
  return applyConfigTransform(configs, (config) =>
    applyRenameToConfig(config, from, to),
  );
}

export function applyRemovalToConfigs(
  configs: Record<string, NodeConfig>,
  ref: string,
): Record<string, NodeConfig> {
  if (!ref) {
    return configs;
  }
  return applyConfigTransform(configs, (config) => applyRemovalToConfig(config, ref));
}
