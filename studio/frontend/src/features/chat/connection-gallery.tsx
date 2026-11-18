// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Input } from "@/components/ui/input";
import { Tick02Icon } from "@/lib/tick-icon";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { PlusSignIcon, Search01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useState } from "react";
import { SettingsSection } from "@/features/settings/components/settings-section";
import { ApiProviderLogo } from "./api-provider-logo";
import type { ProviderRegistryEntry } from "./api/providers-api";
import {
  CUSTOM_PROVIDER_DISPLAY_NAME,
  CUSTOM_PROVIDER_PRESETS,
  LEGACY_CUSTOM_PROVIDER_TYPE,
  connectionPickerValue,
  isCloudPresetProviderType,
  isDecisionConnection,
} from "./external-providers";

const POPULAR_PROVIDER_TYPES = [
  "openai_codex",
  "openai",
  "anthropic",
  "gemini",
  "openrouter",
  "deepseek",
];

// Niche or setup-heavy providers, folded away until asked for or searched.
const EVEN_MORE_PROVIDER_TYPES = new Set([
  "cohere",
  "perplexity",
  "azure",
  "vertex",
  "typesafe",
  "liquid",
]);

export interface ConnectionGalleryItem {
  providerType: string;
  name: string;
}

export interface ConnectionGalleryGroup {
  id: "popular" | "local" | "more" | "even-more";
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
  });
  const popular = POPULAR_PROVIDER_TYPES.flatMap((type) => {
    const entry = byType.get(type);
    return entry ? [fromRegistry(entry)] : [];
  });
  const popularTypes = new Set(popular.map((item) => item.providerType));
  const presets: ConnectionGalleryItem[] = CUSTOM_PROVIDER_PRESETS.map(
    (preset) => ({
      providerType: preset.providerType as string,
      name: preset.displayName as string,
    }),
  );
  // Custom sits with the popular ones: most self-hosted setups end up on it.
  popular.push({
    providerType: LEGACY_CUSTOM_PROVIDER_TYPE,
    name: CUSTOM_PROVIDER_DISPLAY_NAME,
  });
  const local = presets.filter(
    (item) => !isCloudPresetProviderType(item.providerType),
  );
  const chat = registry.filter(
    (entry) =>
      !popularTypes.has(entry.provider_type) &&
      !isDecisionConnection({ providerType: entry.provider_type }),
  );
  const decision = registry.filter((entry) =>
    isDecisionConnection({ providerType: entry.provider_type }),
  );
  // Cloud platforms sit with the other hosted providers, ahead of the decision-only ones.
  const hosted = [
    ...chat.map(fromRegistry),
    ...presets.filter((item) => isCloudPresetProviderType(item.providerType)),
    ...decision.map(fromRegistry),
  ];
  const needle = searchKey(query);
  const matches = (item: ConnectionGalleryItem) =>
    !needle ||
    searchKey(item.name).includes(needle) ||
    searchKey(item.providerType).includes(needle);
  const folded = (item: ConnectionGalleryItem) =>
    EVEN_MORE_PROVIDER_TYPES.has(item.providerType);
  const groups: ConnectionGalleryGroup[] = [
    { id: "popular", label: "Popular", items: popular.filter(matches) },
    { id: "local", label: "Local", items: local.filter(matches) },
    {
      id: "more",
      label: "More providers",
      items: hosted.filter((item) => !folded(item) && matches(item)),
    },
    {
      id: "even-more",
      label: "Even more",
      items: hosted.filter((item) => folded(item) && matches(item)),
    },
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
  const [showEvenMore, setShowEvenMore] = useState(false);
  const groups = useMemo(
    () => buildConnectionGallery(registry, query),
    [registry, query],
  );
  return (
    <SettingsSection
      title="Add a connection"
      action={
        <div className="relative w-80 max-sm:w-44">
          <HugeiconsIcon
            icon={Search01Icon}
            className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-muted-foreground"
            aria-hidden="true"
          />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search providers"
            aria-label="Search providers"
            className="h-control pl-9"
          />
        </div>
      }
    >
      {groups.length === 0 ? (
        <p className="py-3 text-xs text-muted-foreground">
          {loading
            ? "Loading providers"
            : "No provider matches. Custom works with any OpenAI-compatible endpoint."}
        </p>
      ) : (
        <div className="-mx-3 flex flex-col gap-4 pt-3">
          {groups.map((group) => {
            // A search shows its matches wherever they live.
            const collapsed =
              group.id === "even-more" && !showEvenMore && !query.trim();
            return (
              <div key={group.id} className="flex flex-col gap-0.5">
                {group.id === "even-more" ? (
                  <button
                    type="button"
                    aria-expanded={!collapsed}
                    onClick={() => setShowEvenMore((open) => !open)}
                    className="flex items-center gap-1 self-start rounded-md px-3 pb-1 text-ui-11 font-medium text-muted-foreground/80 transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    {group.label}
                    <HugeiconsIcon
                      icon={ChevronDownStandardIcon}
                      className={cn(
                        "size-3 transition-transform",
                        collapsed && "-rotate-90",
                      )}
                    />
                  </button>
                ) : (
                  <span className="px-3 pb-1 text-ui-11 font-medium text-muted-foreground/80">
                    {group.label}
                  </span>
                )}
                {collapsed ? null : (
                  <div className="grid grid-cols-2 gap-x-2 gap-y-1 max-sm:grid-cols-1">
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
                          className="group/tile flex min-w-0 items-center gap-3 rounded-full px-3 py-2.5 text-left transition-colors hover:bg-accent focus-visible:bg-accent focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                        >
                          <span className="flex size-5 shrink-0 items-center justify-center">
                            <ApiProviderLogo
                              providerType={item.providerType}
                              className="size-5"
                              title={item.name}
                              name={
                                item.providerType ===
                                LEGACY_CUSTOM_PROVIDER_TYPE
                                  ? undefined
                                  : item.name
                              }
                            />
                          </span>
                          <span className="flex min-w-0 flex-1 flex-col">
                            <span className="truncate text-sm font-medium text-foreground">
                              {item.name}
                            </span>
                          </span>
                          <span
                            className={cn(
                              "flex h-5 min-w-5 shrink-0 items-center justify-center gap-1 text-ui-11 text-muted-foreground",
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
                )}
              </div>
            );
          })}
        </div>
      )}
    </SettingsSection>
  );
}
