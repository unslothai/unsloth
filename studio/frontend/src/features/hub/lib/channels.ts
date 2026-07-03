// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HfSortKey } from "@/features/hub/hooks/use-hub-model-search";
import {
  NewReleasesIcon,
  SlidersHorizontalIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { ModelFormatFilter } from "../types";

export type ChannelId =
  | "unsloth-trending"
  | "unsloth-latest"
  | "unsloth-safetensors";

export interface ChannelPreset {
  id: ChannelId;
  label: string;
  icon: IconSvgElement;
  hint: string;
  owner?: string;
  tags?: readonly string[];
  query?: string;
  idSuffix?: string;
  format: ModelFormatFilter;
  sort: HfSortKey;
  // Keep only formats Unsloth can fine-tune (drops fp8, nvfp4, w4a16, etc.).
  finetunableOnly?: boolean;
}

export const CHANNEL_PRESETS: readonly ChannelPreset[] = [
  {
    id: "unsloth-trending",
    label: "Unsloth Trending",
    icon: SparklesIcon,
    hint: "Most trending models published by Unsloth.",
    owner: "unsloth",
    tags: ["gguf"],
    format: "gguf",
    sort: "trendingScore",
  },
  {
    id: "unsloth-latest",
    label: "Latest Unsloth",
    icon: NewReleasesIcon,
    hint: "Freshly released models from the Unsloth channel.",
    owner: "unsloth",
    format: "all",
    // Newest by creation date so the feed shows freshly released models, not
    // ones merely re-touched (lastModified).
    sort: "createdAt",
  },
  {
    id: "unsloth-safetensors",
    label: "Fine-tune ready",
    icon: SlidersHorizontalIcon,
    hint: "Checkpoints ready to fine-tune.",
    owner: "unsloth",
    format: "checkpoint",
    sort: "lastModified",
    finetunableOnly: true,
  },
];

export function findChannel(id: ChannelId | null): ChannelPreset | null {
  if (!id) return null;
  return CHANNEL_PRESETS.find((preset) => preset.id === id) ?? null;
}

export type HubSection = "trending" | "latest" | "finetune";

export const HUB_SECTION_ORDER: readonly HubSection[] = [
  "trending",
  "latest",
  "finetune",
];

export const SECTION_TO_CHANNEL: Record<HubSection, ChannelId> = {
  trending: "unsloth-trending",
  latest: "unsloth-latest",
  finetune: "unsloth-safetensors",
};

export const CHANNEL_TO_SECTION: Record<ChannelId, HubSection> = {
  "unsloth-trending": "trending",
  "unsloth-latest": "latest",
  "unsloth-safetensors": "finetune",
};

export const HUB_SECTION_TITLE: Record<HubSection, string> = {
  trending: "Trending Now",
  latest: "Latest Unsloth Models",
  finetune: "Fine-tune Ready",
};

export function isHubSection(value: unknown): value is HubSection {
  return value === "trending" || value === "latest" || value === "finetune";
}
