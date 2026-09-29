// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelConfig, ModelProviderConfig } from "../../types";
import { parseJsonObject } from "./parse";

export function buildModelProvider(
  config: ModelProviderConfig,
  errors: string[],
): Record<string, unknown> {
  // Local providers ignore advanced request overrides: the backend overrides
  // endpoint/api_key/provider_type and strips extras in _inject_local_providers.
  // Skip parsing hidden JSON inputs so stale headers/body in imported recipes
  // can't block client-side validation.
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

  const providerType = config.provider_type.trim().toLowerCase() || "openai";
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
    provider_type: providerType,
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

function assignFiniteNumber(
  target: Record<string, unknown>,
  key: string,
  rawValue: string | undefined,
  transform: (value: number) => number = (value) => value,
): void {
  const trimmed = rawValue?.trim();
  if (!trimmed) {
    return;
  }

  const parsed = Number(trimmed);
  if (Number.isFinite(parsed)) {
    target[key] = transform(parsed);
  }
}

function buildInferenceParameters(
  config: ModelConfig,
  errors: string[],
): Record<string, unknown> {
  const inference: Record<string, unknown> = {};
  assignFiniteNumber(inference, "temperature", config.inference_temperature);
  assignFiniteNumber(inference, "top_p", config.inference_top_p);
  assignFiniteNumber(inference, "max_tokens", config.inference_max_tokens);
  assignFiniteNumber(
    inference,
    "timeout",
    config.inference_timeout,
    Math.trunc,
  );

  const extraBody = parseJsonObject(
    config.inference_extra_body,
    `Model ${config.name} inference extra_body`,
    errors,
  );
  if (extraBody) {
    inference.extra_body = extraBody;
  }

  return inference;
}

export function buildModelConfig(
  config: ModelConfig,
  errors: string[],
): Record<string, unknown> {
  const inference = buildInferenceParameters(config, errors);
  const ggufVariant = config.gguf_variant?.trim();

  return {
    alias: config.name,
    model: config.model,
    // biome-ignore lint/style/useNamingConvention: api schema
    gguf_variant: ggufVariant || undefined,
    provider: config.provider || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_parameters:
      Object.keys(inference).length > 0 ? inference : undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    skip_health_check: config.skip_health_check || undefined,
  };
}
