// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { PROVIDER_LOGOS } from "../hub/lib/provider-logos";

// API provider names differ from the model hub's publisher IDs.
const HUB_PROVIDER_IDS = new Map([
  ["openai", "openai"],
  ["openai_codex", "openai"],
  ["mistral", "mistralai"],
  ["deepseek", "deepseek-ai"],
  ["huggingface", "huggingface"],
  ["kimi", "moonshotai"],
  ["qwen", "qwen"],
]);

const HUB_LOGO_PATHS = new Map(
  PROVIDER_LOGOS.map(({ id, logoPath }) => [id, logoPath]),
);

// Gemini keeps its own mark; the hub uses Google's logo for Gemma models.
const CONNECTION_LOGO_PATHS = new Map([
  ["gemini", "/provider-logos/gemini.svg"],
  ["anthropic", "/provider-logos/anthropic.svg"],
  ["openrouter", "/provider-logos/openrouter.svg"],
  ["vllm", "/provider-logos/vllm.svg"],
  ["ollama", "/provider-logos/ollama.svg"],
  ["llama_cpp", "/provider-logos/llama_cpp.svg"],
]);

/** Public asset path shared by Connections, agent icons, and the model picker. */
export function providerLogoPath(
  providerType: string | undefined | null,
): string | undefined {
  if (!providerType) return undefined;
  const hubId = HUB_PROVIDER_IDS.get(providerType);
  return hubId
    ? HUB_LOGO_PATHS.get(hubId)
    : CONNECTION_LOGO_PATHS.get(providerType);
}
