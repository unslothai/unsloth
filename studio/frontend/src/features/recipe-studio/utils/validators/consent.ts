// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ValidatorConfig } from "../../types";

const TOOL_CONTENT_KEYS = [
  "tool_command",
  "tool_ext",
  "tool_scaffold",
] as const;
const CUSTOM_CONTENT_KEYS = ["custom_source"] as const;

export function isValidatorConsentRequired(config: ValidatorConfig): boolean {
  return (
    (config.validator_type === "tool" && config.tool_acknowledged !== true) ||
    (config.validator_type === "custom" && config.custom_acknowledged !== true)
  );
}

function contentKeysFor(config: ValidatorConfig): readonly string[] {
  if (config.validator_type === "tool") {
    return TOOL_CONTENT_KEYS;
  }
  if (config.validator_type === "custom") {
    return CUSTOM_CONTENT_KEYS;
  }
  return [];
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
