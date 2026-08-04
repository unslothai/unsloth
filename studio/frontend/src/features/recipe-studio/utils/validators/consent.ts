// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ValidatorConfig } from "../../types";

const CONTENT_KEYS_BY_TYPE: Record<string, readonly string[]> = {
  tool: ["tool_command", "tool_ext", "tool_scaffold"],
  custom: ["custom_source"],
};

export function isValidatorConsentRequired(config: ValidatorConfig): boolean {
  return (
    (config.validator_type === "tool" && config.tool_acknowledged !== true) ||
    (config.validator_type === "custom" && config.custom_acknowledged !== true)
  );
}

function contentKeysFor(config: ValidatorConfig): readonly string[] {
  return CONTENT_KEYS_BY_TYPE[config.validator_type] ?? [];
}

export function consentInvalidatedByEdit(
  config: ValidatorConfig,
  patch: Partial<ValidatorConfig>,
): boolean {
  const current = config as Record<string, unknown>;
  const next = patch as Record<string, unknown>;
  return contentKeysFor(config).some(
    (key) => key in next && current[key] !== next[key],
  );
}

export function applyValidatorContentEdit(
  config: ValidatorConfig,
  patch: Partial<ValidatorConfig>,
): Partial<ValidatorConfig> {
  if (!consentInvalidatedByEdit(config, patch)) {
    return patch;
  }
  // Editing the command/source/scaffold invalidates the previous
  // acknowledgement: the new (possibly pasted) content must not run without a
  // fresh opt-in. Both flags are cleared; only the type-appropriate one is
  // ever read.
  return {
    ...patch,
    // biome-ignore lint/style/useNamingConvention: api schema (ui state)
    tool_acknowledged: false,
    // biome-ignore lint/style/useNamingConvention: api schema (ui state)
    custom_acknowledged: false,
  };
}
