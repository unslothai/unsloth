// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Skeleton } from "@/components/ui/skeleton";
import { useUiSpaceScale } from "@/hooks/use-ui-space-scale";
import { memo } from "react";
import type { DiscoverRow } from "../types";
import { CardCarousel } from "./card-carousel";
import {
  MODEL_CARD_HEIGHT_PX,
  MODEL_CARD_WIDTH_PX,
  ModelCard,
} from "./model-card";
import {
  ChevronRightIcon,
} from "lucide-react";

const SKELETON_KEYS = ["s0", "s1", "s2", "s3", "s4"] as const;

function HubSectionRowSkeleton({
  width,
  height,
}: {
  width: number;
  height: number;
}) {
  return (
    <div className="flex gap-4 overflow-hidden pb-4 pt-2">
      {SKELETON_KEYS.map((key) => (
        <Skeleton
          key={key}
          className="shrink-0 rounded-[20px]"
          style={{ width, height }}
        />
      ))}
    </div>
  );
}

export const HubSectionRow = memo(function HubSectionRow({
  title,
  rows,
  onSelect,
  onOpenList,
  deviceType,
  isDataset,
  isLoading,
}: {
  title: string;
  rows: DiscoverRow[];
  onSelect: (id: string) => void;
  onOpenList: () => void;
  deviceType: string | null;
  isDataset: boolean;
  isLoading: boolean;
}) {
  // The card's padding, avatar and text scale with the UI font size, and the
  // card clips its overflow, so the carousel slot scales with them.
  const scale = useUiSpaceScale();
  const cardWidth = Math.round(MODEL_CARD_WIDTH_PX * scale);
  const cardHeight = Math.round(MODEL_CARD_HEIGHT_PX * scale);
  const showSkeleton = isLoading && rows.length === 0;
  if (!showSkeleton && rows.length === 0) {
    return null;
  }

  return (
    <section aria-label={title} className="group/carousel">
      <h2 className="mb-3">
        <button
          type="button"
          onClick={onOpenList}
          aria-label={`See all ${title}`}
          className="hub-section-title group/section -mx-1 inline-flex cursor-pointer items-center gap-1.5 rounded-md px-1 text-ui-18 font-semibold tracking-[-0.02em] text-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          {title}
          <ChevronRightIcon
            strokeWidth={2}
            className="hub-section-chevron size-4 text-muted-foreground"
          />
        </button>
      </h2>
      {showSkeleton ? (
        <HubSectionRowSkeleton width={cardWidth} height={cardHeight} />
      ) : (
        <CardCarousel
          items={rows}
          getKey={(row) => row.id}
          itemWidth={cardWidth}
          itemHeight={cardHeight}
          ariaLabel={title}
          renderItem={(row) => (
            <ModelCard
              row={row}
              deviceType={deviceType}
              isDataset={isDataset}
              onSelect={onSelect}
            />
          )}
        />
      )}
    </section>
  );
});
