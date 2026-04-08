// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelConfig, ModelProviderConfig } from "../../types";
import { parseJsonObject } from "./parse";

export function buildModelProvider(
  config: ModelProviderConfig,
  errors: string[],
): Record<string, unknown> {
  // Local providers do not use any of the advanced request overrides -
  // the backend overrides endpoint/api_key/provider_type and strips the
  // extra fields in _inject_local_providers. Skip parsing the hidden
  // JSON inputs here so imported or hydrated recipes with stale extra
  // headers/body cannot block the client-side validation step.
  if (config.is_local === true) {
    return {
      name: config.name,
      endpoint: "",
      // biome-ignore lint/style/useNamingConvention: api schema
      provider_type: "openai",
      // biome-ignore lint/style/useNamingConvention: api schema
      extra_headers: {},
      // biome-ignore lint/style/useNamingConvention: api schema
      extra_body: {},
      // biome-ignore lint/style/useNamingConvention: api schema
      is_local: true,
    };
  }

  const extraHeaders = parseJsonObject(
    config.extra_headers,
    `Provider ${config.name} extra_headers`,
    errors,
  );
  const extraBody = parseJsonObject(
    config.extra_body,
    `Provider ${config.name} extra_body`,
    errors,
  );
  return {
    name: config.name,
    endpoint: config.endpoint,
    // biome-ignore lint/style/useNamingConvention: api schema
    provider_type: "openai",
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key_env: config.api_key_env?.trim() || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key: config.api_key?.trim() || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    extra_headers: extraHeaders ?? {},
    // biome-ignore lint/style/useNamingConvention: api schema
    extra_body: extraBody ?? {},
  };
}

export function buildModelConfig(
  config: ModelConfig,
  errors: string[],
): Record<string, unknown> {
  const inference: Record<string, unknown> = {};
  const temp = config.inference_temperature?.trim();
  const topP = config.inference_top_p?.trim();
  const maxTokens = config.inference_max_tokens?.trim();
  const timeout = config.inference_timeout?.trim();
  const extraBody = parseJsonObject(
    config.inference_extra_body,
    `Model ${config.name} inference extra_body`,
    errors,
  );

  if (temp) {
    const parsed = Number(temp);
    if (Number.isFinite(parsed)) {
      inference.temperature = parsed;
    }
  }
  if (topP) {
    const parsed = Number(topP);
    if (Number.isFinite(parsed)) {
      // biome-ignore lint/style/useNamingConvention: api schema
      inference.top_p = parsed;
    }
  }
  if (maxTokens) {
    const parsed = Number(maxTokens);
    if (Number.isFinite(parsed)) {
      // biome-ignore lint/style/useNamingConvention: api schema
      inference.max_tokens = parsed;
    }
  }
  if (timeout) {
    const parsed = Number(timeout);
    if (Number.isFinite(parsed)) {
      inference.timeout = Math.trunc(parsed);
    }
  }
  if (extraBody) {
    // biome-ignore lint/style/useNamingConvention: api schema
    inference.extra_body = extraBody;
  }

  return {
    alias: config.name,
    model: config.model,
    provider: config.provider || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_parameters:
      Object.keys(inference).length > 0 ? inference : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    skip_health_check: config.skip_health_check || undefined,
  };
}
