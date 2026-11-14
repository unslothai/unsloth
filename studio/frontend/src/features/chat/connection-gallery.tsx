// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";
import { ApiProviderLogo } from "./api-provider-logo";
import type { ProviderRegistryEntry } from "./api/providers-api";
import {
  CUSTOM_PROVIDER_DISPLAY_NAME,
  CUSTOM_PROVIDER_PRESETS,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  connectionPickerValue,
  isDecisionConnection,
} from "./external-providers";

const POPULAR_PROVIDER_TYPES = [
  "openai_codex",
  "openai",
  "anthropic",
  "gemini",
  "openrouter",
];

const PROVIDER_DESCRIPTIONS: Record<string, string> = {
  openai_codex: "Sign in with your ChatGPT plan. No API key.",
  openai: "GPT models with an API key.",
  anthropic: "Claude models with an API key.",
  gemini: "Gemini models from Google AI Studio.",
  openrouter: "One key for hundreds of models.",
  deepseek: "DeepSeek chat and reasoning models.",
  zai: "GLM models.",
  kimi: "Kimi models from Moonshot AI.",
  mistral: "Mistral, Codestral and Devstral.",
  minimax: "MiniMax M-series models.",
  opencode: "OpenCode Console and Go plans.",
  qwencloud: "Qwen models, pay as you go.",
  huggingface: "Open models through Inference Providers.",
  ollama: "Models served by Ollama, local or cloud.",
  llama_cpp: "A running llama-server.",
  vllm: "A self-hosted vLLM server.",
  [LEGACY_CUSTOM_PROVIDER_TYPE]: "Any OpenAI-compatible endpoint.",
};

const DECISION_DESCRIPTION = "Decision models for the Decision API.";

export interface ConnectionGalleryItem {
  providerType: string;
  name: string;
  description: string;
}

export interface ConnectionGalleryGroup {
  id: "popular" | "local" | "more";
  label: string;
  items: ConnectionGalleryItem[];
}

/** Punctuation-insensitive, so "llamacpp" finds llama.cpp and "open ai" finds OpenAI. */
function searchKey(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

export function buildConnectionGallery(
  registry: readonly ProviderRegistryEntry[],
  query = "",
): ConnectionGalleryGroup[] {
  const byType = new Map(registry.map((entry) => [entry.provider_type, entry]));
  const fromRegistry = (
    entry: ProviderRegistryEntry,
  ): ConnectionGalleryItem => ({
    providerType: entry.provider_type,
    name: entry.provider_type === "opencode" ? "OpenCode" : entry.display_name,
    description:
      PROVIDER_DESCRIPTIONS[entry.provider_type] ??
      (isDecisionConnection({ providerType: entry.provider_type })
        ? DECISION_DESCRIPTION
        : ""),
  });
  const popular = POPULAR_PROVIDER_TYPES.flatMap((type) => {
    const entry = byType.get(type);
    return entry ? [fromRegistry(entry)] : [];
  });
  const popularTypes = new Set(popular.map((item) => item.providerType));
  const local: ConnectionGalleryItem[] = [
    ...CUSTOM_PROVIDER_PRESETS.map((preset) => ({
      providerType: preset.providerType as string,
      name: preset.displayName as string,
      description: PROVIDER_DESCRIPTIONS[preset.providerType] ?? "",
    })),
    {
      providerType: LEGACY_CUSTOM_PROVIDER_TYPE,
      name: CUSTOM_PROVIDER_DISPLAY_NAME,
      description: PROVIDER_DESCRIPTIONS[LEGACY_CUSTOM_PROVIDER_TYPE],
    },
  ];
  const more = registry
    .filter((entry) => !popularTypes.has(entry.provider_type))
    .map(fromRegistry);
  const needle = searchKey(query);
  const matches = (item: ConnectionGalleryItem) =>
    !needle ||
    searchKey(item.name).includes(needle) ||
    searchKey(item.providerType).includes(needle) ||
    searchKey(item.description).includes(needle);
  const groups: ConnectionGalleryGroup[] = [
    { id: "popular", label: "Popular", items: popular.filter(matches) },
    { id: "local", label: "Local and custom", items: local.filter(matches) },
    { id: "more", label: "More providers", items: more.filter(matches) },
  ];
  return groups.filter((group) => group.items.length > 0);
}

export function ConnectionGallery({
  registry,
  connectedTypes,
  loading,
  onPick,
}: {
  registry: readonly ProviderRegistryEntry[];
  connectedTypes: ReadonlySet<string>;
  loading: boolean;
  onPick: (providerType: string) => void;
}) {
  const [query, setQuery] = useState("");
  const groups = useMemo(
    () => buildConnectionGallery(registry, query),
    [registry, query],
  );
  return (
    <section
      aria-label="Add a connection"
      className="overflow-hidden rounded-xl border border-border/70 bg-muted/[0.12]"
    >
      <div className="flex items-center justify-between gap-3 border-border/60 border-b px-3 py-2.5">
        <div className="flex min-w-0 flex-col gap-0.5">
          <span className="text-sm font-medium text-foreground">
            Add a connection
          </span>
          <span className="text-xs leading-snug text-muted-foreground">
            Pick a provider, then paste a key or point at your own server.
          </span>
        </div>
        <Input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search providers"
          aria-label="Search providers"
          className="h-8 w-44 shrink-0 bg-background/55 text-xs placeholder:text-muted-foreground/65 focus-visible:border-border focus-visible:ring-0 max-sm:w-32"
        />
      </div>
      {groups.length === 0 ? (
        <p className="px-3 py-4 text-xs text-muted-foreground">
          {loading
            ? "Loading providers"
            : "No provider matches. Custom works with any OpenAI-compatible endpoint."}
        </p>
      ) : (
        <div className="flex flex-col gap-2 p-2">
          {groups.map((group) => (
            <div key={group.id} className="flex flex-col gap-0.5">
              <span className="px-1.5 pt-1 text-ui-11 font-medium text-muted-foreground/80">
                {group.label}
              </span>
              <div className="grid grid-cols-2 gap-0.5 max-sm:grid-cols-1">
                {group.items.map((item) => {
                  const connected = connectedTypes.has(
                    connectionPickerValue(item.providerType),
                  );
                  return (
                    <button
                      key={item.providerType}
                      type="button"
                      onClick={() => onPick(item.providerType)}
                      data-provider-type={item.providerType}
                      className="group/tile flex min-w-0 items-center gap-2.5 rounded-lg px-1.5 py-1.5 text-left transition-colors hover:bg-muted/45 focus-visible:bg-muted/45 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    >
                      <span className="flex size-8 shrink-0 items-center justify-center rounded-sm border border-border/70 bg-background/80">
                        <ApiProviderLogo
                          providerType={item.providerType}
                          className="size-5"
                          title={item.name}
                        />
                      </span>
                      <span className="flex min-w-0 flex-1 flex-col">
                        <span className="truncate text-sm font-medium text-foreground">
                          {item.name}
                        </span>
                        {item.description ? (
                          <span className="truncate text-xs text-muted-foreground">
                            {item.description}
                          </span>
                        ) : null}
                      </span>
                      <span
                        className={cn(
                          "flex shrink-0 items-center gap-1 pr-1 text-ui-11 text-muted-foreground",
                          !connected &&
                            "opacity-0 transition-opacity group-hover/tile:opacity-100 group-focus-visible/tile:opacity-100",
                        )}
                      >
                        <HugeiconsIcon
                          icon={connected ? Tick02Icon : PlusSignIcon}
                          className="size-3.5"
                        />
                        {connected ? "Added" : null}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
