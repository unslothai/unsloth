// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ModelConfig,
  ModelProviderConfig,
  NodeConfig,
  ValidatorCodeLang,
  ValidatorConfig,
} from "../../types";
import { VALIDATOR_OXC_CODE_LANGS } from "../validators/code-lang";
import { isOxcCodeShape } from "../validators/oxc-code-shape";
import { isOxcValidationMode } from "../validators/oxc-mode";

export function validateSubcategoryConfigs(
  configs: Record<string, NodeConfig>,
  nameToConfig: Map<string, NodeConfig>,
  errors: string[],
): void {
  for (const config of Object.values(configs)) {
    if (config.kind !== "sampler" || config.sampler_type !== "subcategory") {
      continue;
    }
    const parentName = config.subcategory_parent;
    if (!parentName) {
      errors.push(`Subcategory ${config.name}: parent category required.`);
      continue;
    }
    const parent = nameToConfig.get(parentName);
    const parentValues =
      parent && parent.kind === "sampler" && parent.sampler_type === "category"
        ? (parent.values ?? [])
        : [];
    const mapping = config.subcategory_mapping ?? {};
    for (const value of parentValues) {
      const list = mapping[value];
      if (!list || list.length === 0) {
        errors.push(
          `Subcategory ${config.name}: '${value}' needs at least 1 subcategory.`,
        );
      }
    }
  }
}

export function validateTimedeltaConfigs(
  configs: Record<string, NodeConfig>,
  nameToConfig: Map<string, NodeConfig>,
  errors: string[],
): void {
  for (const config of Object.values(configs)) {
    if (config.kind !== "sampler" || config.sampler_type !== "timedelta") {
      continue;
    }
    const reference = config.reference_column_name?.trim() ?? "";
    if (!reference) {
      errors.push(`Timedelta ${config.name}: reference datetime column required.`);
      continue;
    }
    const parent = nameToConfig.get(reference);
    if (
      !parent ||
      parent.kind !== "sampler" ||
      parent.sampler_type !== "datetime"
    ) {
      errors.push(`Timedelta ${config.name}: reference '${reference}' must be datetime.`);
    }
  }
}

export function validateModelAliasLinks(
  modelAliases: Set<string>,
  modelConfigConfigs: ModelConfig[],
  errors: string[],
): void {
  for (const alias of modelAliases) {
    if (!modelConfigConfigs.some((config) => config.name === alias)) {
      errors.push(`LLM model_alias ${alias}: missing model config.`);
    }
  }
}

export function validateModelConfigProviders(
  modelConfigConfigs: ModelConfig[],
  modelAliases: Set<string>,
  modelProviderNames: Set<string>,
  localProviderNames: Set<string>,
  errors: string[],
): void {
  for (const config of modelConfigConfigs) {
    const provider = config.provider.trim();
    const alias = config.name;
    const isLocal = localProviderNames.has(provider);
    // Local providers do not require a real model id - the loaded Chat
    // model is used regardless of what gets sent in the payload.
    if (!isLocal && modelAliases.has(alias) && !config.model.trim()) {
      errors.push(`Model config ${alias}: model is required.`);
    }
    if (provider && !modelProviderNames.has(provider)) {
      errors.push(`Model config ${alias}: provider ${provider} not found.`);
    }
  }
}

export function validateUsedProviders(
  modelProviderConfigs: ModelProviderConfig[],
  modelConfigConfigs: ModelConfig[],
  errors: string[],
): void {
  const usedProviders = new Set(
    modelConfigConfigs.map((config) => config.provider.trim()).filter(Boolean),
  );
  for (const provider of modelProviderConfigs) {
    if (!usedProviders.has(provider.name)) {
      continue;
    }
    if (provider.is_local) {
      continue;
    }
    if (!provider.endpoint.trim()) {
      errors.push(`Model provider ${provider.name}: endpoint is required.`);
    }
    if (!provider.provider_type.trim()) {
      errors.push(`Model provider ${provider.name}: provider_type is required.`);
    }
  }
}

export function validateValidatorConfigs(
  configs: Record<string, NodeConfig>,
  nameToConfig: Map<string, NodeConfig>,
  errors: string[],
): void {
  for (const config of Object.values(configs)) {
    if (config.kind !== "validator") {
      continue;
    }
    const target = (config as ValidatorConfig).target_columns[0]?.trim();
    if (!target) {
      continue;
    }
    const targetConfig = nameToConfig.get(target);
    if (!targetConfig) {
      errors.push(`Validator ${config.name}: target '${target}' not found.`);
      continue;
    }
    if (targetConfig.kind !== "llm" || targetConfig.llm_type !== "code") {
      errors.push(`Validator ${config.name}: target '${target}' must be LLM Code.`);
      continue;
    }
    if (
      config.validator_type === "oxc" &&
      !VALIDATOR_OXC_CODE_LANGS.includes(
        (targetConfig.code_lang ?? "").trim() as ValidatorCodeLang,
      )
    ) {
      errors.push(
        `Validator ${config.name}: target '${target}' must use javascript/typescript/jsx/tsx.`,
      );
      continue;
    }
    if (
      config.validator_type === "oxc" &&
      !isOxcValidationMode(config.oxc_validation_mode)
    ) {
      errors.push(
        `Validator ${config.name}: oxc_validation_mode '${config.oxc_validation_mode}' is invalid.`,
      );
      continue;
    }
    if (
      config.validator_type === "oxc" &&
      !isOxcCodeShape(config.oxc_code_shape)
    ) {
      errors.push(
        `Validator ${config.name}: oxc_code_shape '${config.oxc_code_shape}' is invalid.`,
      );
      continue;
    }
    if (
      config.validator_type !== "oxc" &&
      (targetConfig.code_lang ?? "").trim() !== config.code_lang.trim()
    ) {
      errors.push(
        `Validator ${config.name}: code_lang '${config.code_lang}' must match target '${target}' (${targetConfig.code_lang ?? "unknown"}).`,
      );
    }
  }
}
