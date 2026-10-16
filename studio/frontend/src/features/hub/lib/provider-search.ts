// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A search for a model maker ("meta", "alibaba", "zai", "google") also lists Unsloth's uploads of
 * that maker's models: the ones the Hub shows with the maker's avatar (provider-logos.ts).
 */

import { PROVIDER_LOGOS, type ProviderLogo } from "./provider-logos.ts";

/** Names people search by beyond a provider's id, display name and Hub orgs. */
const QUERY_ALIASES: Readonly<Record<string, readonly string[]>> = {
  "meta-llama": ["meta", "facebook"],
  qwen: ["alibaba", "alibaba cloud", "qwen"],
  "zai-org": ["zai", "z.ai", "zhipu", "zhipu ai", "thudm"],
  google: ["google", "deepmind", "google deepmind"],
  "deepseek-ai": ["deepseek"],
  moonshotai: ["moonshot"],
  mistralai: ["mistral"],
  "xai-org": ["xai"],
  minimax: ["minimax"],
  huggingface: ["hf"],
  cohere: ["cohere", "coherelabs"],
};

/** The maker's own Hub orgs, where the registry lists none. */
const HUB_ORGS: Readonly<Record<string, readonly string[]>> = {
  qwen: ["Qwen"],
  "zai-org": ["zai-org"],
  google: ["google"],
  nvidia: ["nvidia"],
  "deepseek-ai": ["deepseek-ai"],
  microsoft: ["microsoft"],
  moonshotai: ["moonshotai"],
  mistralai: ["mistralai"],
  "xai-org": ["xai-org"],
  minimax: ["MiniMaxAI"],
  huggingface: ["HuggingFaceTB"],
  ibm: ["ibm-granite"],
  cohere: ["CohereLabs"],
  openai: ["openai"],
};

/** Hub orgs the maker uploads under, so "alibaba" reaches Qwen's own repos. */
export function providerHubOrgs(provider: ProviderLogo): readonly string[] {
  return provider.owners ?? HUB_ORGS[provider.id] ?? [];
}

const normalize = (value: string) => value.toLowerCase().replace(/[\s._-]+/g, "");

/** The provider a whole query names, or null. Exact names only, so "gem" or "metal" match nothing. */
export function matchProviderByQuery(query: string): ProviderLogo | null {
  const needle = normalize(query);
  if (!needle) {
    return null;
  }
  for (const provider of PROVIDER_LOGOS) {
    const names = [
      provider.id,
      provider.name,
      ...(provider.owners ?? []),
      ...(QUERY_ALIASES[provider.id] ?? []),
    ];
    if (names.some((name) => normalize(name) === needle)) {
      return provider;
    }
  }
  return null;
}

/**
 * Hub search terms that reach the provider's repo names: its prefixes and stems without the
 * trailing dash, dropping any term that already contains a shorter one (the Hub matches substrings).
 */
export function providerSearchTerms(provider: ProviderLogo): string[] {
  const terms = [...provider.prefixes, ...(provider.stems ?? [])]
    .map((term) => term.replace(/-+$/, "").toLowerCase())
    .filter(Boolean);
  const unique = [...new Set(terms)].sort((a, b) => a.length - b.length);
  return unique.filter(
    (term, i) => !unique.slice(0, i).some((shorter) => term.includes(shorter)),
  );
}
