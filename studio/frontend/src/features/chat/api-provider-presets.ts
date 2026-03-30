// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Built-in provider targets; the backend proxy resolves routing — users do not type a base URL for these. */
export interface ApiProviderPreset {
  id: string;
  label: string;
  description: string;
}

export const API_PROVIDER_PRESETS: readonly ApiProviderPreset[] = [
  {
    id: "openai",
    label: "OpenAI",
    description: "GPT models (OpenAI API)",
  },
  {
    id: "anthropic",
    label: "Anthropic",
    description: "Claude models",
  },
  {
    id: "google",
    label: "Google AI",
    description: "Gemini models",
  },
  {
    id: "groq",
    label: "Groq",
    description: "Groq inference",
  },
  {
    id: "together",
    label: "Together AI",
    description: "Together API",
  },
] as const;

export function getApiProviderPreset(presetId: string): ApiProviderPreset | undefined {
  return API_PROVIDER_PRESETS.find((p) => p.id === presetId);
}
