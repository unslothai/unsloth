// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LlmMcpProviderConfig } from "../../types";

export function createMcpProviderId(prefix: string, index: number): string {
  return `${prefix}-mcp-${Date.now()}-${index + 1}`;
}

export function addUnique(items: string[], value: string): string[] {
  const trimmed = value.trim();
  if (!trimmed || items.includes(trimmed)) {
    return items;
  }
  return [...items, trimmed];
}

export function collectToolSuggestions(
  providerNames: string[],
  toolsByProvider: Record<string, string[]>,
): string[] {
  return Array.from(
    new Set(
      providerNames.flatMap(
        (providerName) => toolsByProvider[providerName.trim()] ?? [],
      ),
    ),
  );
}

export function isProviderReadyForToolFetch(
  provider: LlmMcpProviderConfig,
): boolean {
  const hasName = provider.name.trim().length > 0;
  if (!hasName) {
    return false;
  }
  if (provider.provider_type === "stdio") {
    return (provider.command?.trim().length ?? 0) > 0;
  }
  return (provider.endpoint?.trim().length ?? 0) > 0;
}

export function toApiProvider(
  provider: LlmMcpProviderConfig,
): Record<string, unknown> {
  if (provider.provider_type === "stdio") {
    const env = Object.fromEntries(
      (provider.env ?? [])
        .map((item) => [item.key.trim(), item.value.trim()] as const)
        .filter(([key, value]) => key && value),
    );
    return {
      // biome-ignore lint/style/useNamingConvention: api schema
      provider_type: "stdio",
      name: provider.name.trim(),
      command: provider.command?.trim() ?? "",
      args: (provider.args ?? []).map((value) => value.trim()).filter(Boolean),
      env,
    };
  }
  return {
    // biome-ignore lint/style/useNamingConvention: api schema
    provider_type: "streamable_http",
    name: provider.name.trim(),
    endpoint: provider.endpoint?.trim() ?? "",
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key: provider.api_key?.trim() || undefined,
    // biome-ignore lint/style/useNamingConvention: api schema
    api_key_env: provider.api_key_env?.trim() || undefined,
  };
}
