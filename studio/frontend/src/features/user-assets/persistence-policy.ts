// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import policy from "../../../../user_assets_persistence_policy.json";

export const DENIED_SECRET_KEYS = new Set(policy.deniedSecretKeys);
export const SAFE_SECRET_LOOKING_KEYS = new Set(policy.safeSecretLookingKeys);
export const MCP_ENV_DENIED_KEY_PARTS = policy.mcpEnvDeniedKeyParts;
export const MCP_ENV_DENIED_KEY_SUFFIXES = policy.mcpEnvDeniedKeySuffixes;
export const MCP_ENV_DENIED_EXACT_KEYS = new Set(
  policy.mcpEnvDeniedExactKeys,
);
export const MAX_RECIPE_JSON_BYTES = policy.maxRecipeJsonBytes;
export const MAX_EXECUTION_JSON_BYTES = policy.maxExecutionJsonBytes;
export const MAX_LEGACY_BATCH_JSON_BYTES = policy.maxLegacyBatchJsonBytes;
