// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { classifyUnslothSupport } from "@/features/hub/hooks/use-hub-model-search";
import { ownerPaletteColor } from "@/features/hub/lib/avatar-theme";
import { buildAdaptiveCardAccentStyle } from "@/features/hub/lib/card-accent";
import { useDominantColor } from "@/features/hub/lib/use-dominant-color";
import { formatModelParamLabel } from "@/features/hub/lib/view-models";
import { formatCompact } from "@/lib/utils";
import { Download01Icon, FavouriteIcon } from "@hugeicons/core-free-icons";
import { type CSSProperties, memo, useMemo } from "react";
import type { DiscoverRow } from "../types";
import { StatChip, buildRowStatusTooltip } from "./models-catalog-rows";
import { OwnerAvatar, useAvatarImageUrl } from "./owner-avatar";
import { AccessGlyphs, CapabilityPill } from "./shared";

export const MODEL_CARD_WIDTH_PX = 204;
export const MODEL_CARD_HEIGHT_PX = 128;

export type TrendingCardStyle = CSSProperties & {
  "--hub-card-art-accent"?: string;
  "--hub-trending-glow"?: string;
  "--hub-trending-glow-x"?: string;
  "--hub-trending-glow-y"?: string;
  "--hub-trending-glow-size"?: string;
  "--hub-trending-sheen-angle"?: string;
  "--hub-trending-scrim-angle"?: string;
  "--hub-trending-secondary-glow"?: string;
  "--hub-trending-secondary-x"?: string;
  "--hub-trending-secondary-y"?: string;
  "--hub-trending-secondary-size"?: string;
};

const DEFAULT_TRENDING_AURA_LAYOUT = {
  x: 24,
  y: 16,
  size: 48,
  secondaryX: 82,
  secondaryY: 12,
  secondarySize: 24,
  sheenAngle: 145,
  scrimAngle: 320,
} as const;

const TRENDING_AURA_LAYOUTS = [
  DEFAULT_TRENDING_AURA_LAYOUT,
  {
    x: 72,
    y: 15,
    size: 47,
    secondaryX: 18,
    secondaryY: 28,
    secondarySize: 25,
    sheenAngle: 220,
    scrimAngle: 45,
  },
  {
    x: 50,
    y: 8,
    size: 45,
    secondaryX: 78,
    secondaryY: 36,
    secondarySize: 23,
    sheenAngle: 260,
    scrimAngle: 110,
  },
  {
    x: 18,
    y: 36,
    size: 52,
    secondaryX: 72,
    secondaryY: 10,
    secondarySize: 24,
    sheenAngle: 115,
    scrimAngle: 285,
  },
  {
    x: 84,
    y: 34,
    size: 50,
    secondaryX: 22,
    secondaryY: 12,
    secondarySize: 24,
    sheenAngle: 245,
    scrimAngle: 70,
  },
  {
    x: 36,
    y: 50,
    size: 52,
    secondaryX: 82,
    secondaryY: 18,
    secondarySize: 23,
    sheenAngle: 35,
    scrimAngle: 205,
  },
  {
    x: 66,
    y: 48,
    size: 51,
    secondaryX: 18,
    secondaryY: 16,
    secondarySize: 24,
    sheenAngle: 325,
    scrimAngle: 145,
  },
  {
    x: 42,
    y: 22,
    size: 46,
    secondaryX: 86,
    secondaryY: 44,
    secondarySize: 22,
    sheenAngle: 75,
    scrimAngle: 255,
  },
] as const;

function hashString(value: string): number {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) | 0;
  }
  return hash >>> 0;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function resolveTrendingGlow(row: DiscoverRow): Pick<
  TrendingCardStyle,
  "--hub-trending-glow" | "--hub-trending-secondary-glow"
> {
  const signature = `${row.owner} ${row.repo} ${row.result.id}`.toLowerCase();
  if (signature.includes("gemma") || signature.includes("google")) {
    return {
      "--hub-trending-glow": "rgba(62, 125, 255, 0.52)",
      "--hub-trending-secondary-glow": "rgba(120, 180, 255, 0.18)",
    };
  }
  if (signature.includes("qwen") || signature.includes("qwq")) {
    return {
      "--hub-trending-glow": "rgba(150, 92, 255, 0.52)",
      "--hub-trending-secondary-glow": "rgba(196, 140, 255, 0.18)",
    };
  }
  if (signature.includes("llama") || signature.includes("meta")) {
    return {
      "--hub-trending-glow": "rgba(20, 184, 166, 0.5)",
      "--hub-trending-secondary-glow": "rgba(72, 222, 196, 0.16)",
    };
  }
  if (signature.includes("mistral") || signature.includes("mixtral")) {
    return {
      "--hub-trending-glow": "rgba(255, 138, 64, 0.5)",
      "--hub-trending-secondary-glow": "rgba(255, 206, 92, 0.16)",
    };
  }
  if (signature.includes("deepseek")) {
    return {
      "--hub-trending-glow": "rgba(104, 96, 255, 0.52)",
      "--hub-trending-secondary-glow": "rgba(150, 130, 255, 0.16)",
    };
  }
  if (signature.includes("phi") || signature.includes("microsoft")) {
    return {
      "--hub-trending-glow": "rgba(34, 176, 224, 0.5)",
      "--hub-trending-secondary-glow": "rgba(96, 214, 244, 0.16)",
    };
  }
  if (
    signature.includes("unsloth") ||
    signature.includes("lfm") ||
    signature.includes("liquid")
  ) {
    return {
      "--hub-trending-glow": "rgba(0, 198, 150, 0.48)",
      "--hub-trending-secondary-glow": "rgba(64, 226, 196, 0.16)",
    };
  }
  return {
    "--hub-trending-glow":
      "color-mix(in srgb, var(--hub-card-art-accent) 46%, transparent)",
    "--hub-trending-secondary-glow":
      "color-mix(in srgb, var(--hub-card-art-accent) 14%, transparent)",
  };
}

export function buildTrendingAuraStyle(row: DiscoverRow): TrendingCardStyle {
  const hash = hashString(`${row.id}:${row.owner}:${row.repo}`);
  const layout =
    TRENDING_AURA_LAYOUTS[hash % TRENDING_AURA_LAYOUTS.length] ??
    DEFAULT_TRENDING_AURA_LAYOUT;
  const glowX = clamp(layout.x + ((hash >>> 4) % 11) - 5, 12, 88);
  const glowY = clamp(layout.y + ((hash >>> 8) % 9) - 4, 6, 56);
  const secondaryX = clamp(
    layout.secondaryX + ((hash >>> 12) % 11) - 5,
    10,
    90,
  );
  const secondaryY = clamp(
    layout.secondaryY + ((hash >>> 16) % 9) - 4,
    6,
    58,
  );
  const sheenAngle = layout.sheenAngle + ((hash >>> 20) % 17) - 8;
  const scrimAngle = layout.scrimAngle + ((hash >>> 24) % 19) - 9;
  return {
    ...resolveTrendingGlow(row),
    "--hub-trending-glow-x": `${glowX}%`,
    "--hub-trending-glow-y": `${glowY}%`,
    "--hub-trending-glow-size": `${layout.size}%`,
    "--hub-trending-sheen-angle": `${sheenAngle}deg`,
    "--hub-trending-scrim-angle": `${scrimAngle}deg`,
    "--hub-trending-secondary-x": `${secondaryX}%`,
    "--hub-trending-secondary-y": `${secondaryY}%`,
    "--hub-trending-secondary-size": `${layout.secondarySize}%`,
  };
}

export const ModelCard = memo(function ModelCard({
  row,
  deviceType,
  isDataset,
  onSelect,
}: {
  row: DiscoverRow;
  deviceType: string | null;
  isDataset: boolean;
  onSelect: (id: string) => void;
}) {
  const support = useMemo(
    () =>
      isDataset
        ? null
        : classifyUnslothSupport({
            modelId: row.id,
            pipelineTag: row.result.pipelineTag,
            tags: row.result.tags,
            libraryName: row.result.libraryName,
            deviceType,
            quantMethod: row.result.quantMethod,
          }),
    [isDataset, row.id, row.result, deviceType],
  );
  const unsupported = support?.status === "unsupported";
  const partial = row.isAvailableOnDevice && row.isPartialOnDevice;
  const onDevice = row.isAvailableOnDevice && !row.isPartialOnDevice;
  const topCapability = row.capabilities[0] ?? null;
  const sizeLabel = formatModelParamLabel(row.repo, row.result.totalParams);
  const hasSize = sizeLabel !== "N/A";
  const tip = buildRowStatusTooltip({
    partialRepoId: partial ? row.result.id : undefined,
    unsupported,
    unsupportedReason: support?.reason ?? null,
    resourceLabel: isDataset ? "dataset" : "model",
  });
  const iconUrl = useAvatarImageUrl(row.owner, row.repo);
  const dominant = useDominantColor(iconUrl);
  const cardAccentStyle = useMemo(
    () => ({
      "--hub-card-art-accent": ownerPaletteColor(`${row.owner}/${row.repo}`),
      ...buildTrendingAuraStyle(row),
      ...buildAdaptiveCardAccentStyle(dominant),
    }),
    [dominant, row],
  );
  const card = (
    <button
      type="button"
      aria-label={row.repo}
      onClick={() => onSelect(row.id)}
      style={cardAccentStyle}
      className="hub-model-card hub-trending-card group/card flex h-full w-full cursor-pointer flex-col overflow-hidden px-3 py-3.5 text-left outline-none focus-visible:ring-1 focus-visible:ring-ring focus-visible:ring-inset"
    >
      <div className="flex min-w-0 items-start gap-2.5">
        <OwnerAvatar
          owner={row.owner}
          repoName={row.repo}
          className="size-11 shrink-0 rounded-[14px] text-[1.0625rem] ring-1 ring-white/10"
          remote={false}
        />
        <div className="min-w-0 flex-1 space-y-0.5">
          <p className="hub-trending-title line-clamp-2 text-[0.84375rem] font-semibold leading-[1rem] text-foreground">
            {row.repo}
          </p>
          <span className="hub-trending-owner flex min-w-0 items-center gap-1 text-[0.71875rem] leading-[0.9375rem] text-muted-foreground/80">
            <span className="truncate">{row.owner}</span>
            {row.owner.toLowerCase() === "unsloth" && (
              <span
                aria-label="Verified Unsloth"
                className="hub-verified-badge size-3.5 shrink-0 text-verified"
              />
            )}
          </span>
        </div>
        <div className="mt-[3px] flex shrink-0 items-center gap-1">
          <AccessGlyphs
            gated={row.result.gated}
            isPrivate={row.result.private}
          />
          {partial && (
            <span
              role="img"
              aria-label="Partial download"
              className="inline-block size-[5px] rounded-full bg-status-warning"
            />
          )}
          {unsupported && (
            <span
              role="img"
              aria-label="May not be supported yet"
              className="inline-block size-[5px] rounded-full bg-status-danger"
            />
          )}
          {onDevice && (
            <span
              role="img"
              aria-label="On device"
              className="inline-block size-[5px] rounded-full bg-status-success"
            />
          )}
        </div>
      </div>
      <div className="mt-auto flex items-end justify-between gap-2 pt-2.5">
        <div className="hub-trending-stats flex min-w-0 items-center gap-2.5 pb-[3px]">
          <StatChip
            icon={Download01Icon}
            value={formatCompact(row.result.downloads)}
          />
          <StatChip
            icon={FavouriteIcon}
            value={formatCompact(row.result.likes)}
          />
        </div>
        {hasSize ? (
          <span className="hub-chip shrink-0">{sizeLabel}</span>
        ) : topCapability ? (
          <CapabilityPill capability={topCapability} iconOnly={true} />
        ) : null}
      </div>
    </button>
  );

  if (!tip) {
    return card;
  }
  return (
    <Tooltip>
      <TooltipTrigger asChild={true}>{card}</TooltipTrigger>
      <TooltipContent side="top" className="tooltip-compact max-w-[240px]">
        {tip}
      </TooltipContent>
    </Tooltip>
  );
});
