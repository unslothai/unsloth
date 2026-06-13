// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  NewReleasesIcon,
  SlidersHorizontalIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { HfSortKey } from "@/features/hub/hooks/use-hub-model-search";
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
}

export const CHANNEL_PRESETS: readonly ChannelPreset[] = [
  {
    id: "unsloth-trending",
    label: "Unsloth Trending",
    icon: SparklesIcon,
    hint: "Trending Unsloth models.",
    owner: "unsloth",
    format: "gguf",
    sort: "trendingScore",
  },
  {
    id: "unsloth-latest",
    label: "Latest Unsloth",
    icon: NewReleasesIcon,
    hint: "Newest Unsloth releases.",
    owner: "unsloth",
    format: "all",
    sort: "lastModified",
  },
  {
    id: "unsloth-safetensors",
    label: "Fine-tune ready",
    icon: SlidersHorizontalIcon,
    hint: "Checkpoints ready to fine-tune.",
    owner: "unsloth",
    format: "checkpoint",
    sort: "lastModified",
  },
];

export function findChannel(id: ChannelId | null): ChannelPreset | null {
  if (!id) return null;
  return CHANNEL_PRESETS.find((preset) => preset.id === id) ?? null;
}
