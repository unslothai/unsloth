// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  ModelConfig,
  ModelProviderConfig,
} from "../../../types";
import {
  isRecord,
  readNumberString,
  readString,
} from "../helpers";

export function parseModelProvider(
  provider: Record<string, unknown>,
  name: string,
  id: string,
): ModelProviderConfig {
  return {
    id,
    kind: "model_provider",
    name,
    endpoint: readString(provider.endpoint) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    provider_type: readString(provider.provider_type) ?? "openai",
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key_env: readString(provider.api_key_env) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key: readString(provider.api_key) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    extra_headers: isRecord(provider.extra_headers)
      ? JSON.stringify(provider.extra_headers, null, 2)
      : "",
    // biome-ignore lint/style/useNamingConvention: api schema
    extra_body: isRecord(provider.extra_body)
      ? JSON.stringify(provider.extra_body, null, 2)
      : "",
    // biome-ignore lint/style/useNamingConvention: api schema
    is_local: provider.is_local === true,
  };
}

export function parseModelConfig(
  model: Record<string, unknown>,
  name: string,
  id: string,
): ModelConfig {
  const inference = isRecord(model.inference_parameters)
    ? (model.inference_parameters as Record<string, unknown>)
    : {};
  return {
    id,
    kind: "model_config",
    name,
    model: readString(model.model) ?? "",
    provider: readString(model.provider) ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_temperature: readNumberString(inference.temperature),
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_top_p: readNumberString(inference.top_p),
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_max_tokens: readNumberString(inference.max_tokens),
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_timeout: readNumberString(inference.timeout),
    // biome-ignore lint/style/useNamingConvention: api schema
    inference_extra_body: isRecord(inference.extra_body)
      ? JSON.stringify(inference.extra_body, null, 2)
      : "",
    // biome-ignore lint/style/useNamingConvention: api schema
    skip_health_check:
      typeof model.skip_health_check === "boolean"
        ? model.skip_health_check
        : false,
  };
}
