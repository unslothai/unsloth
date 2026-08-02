// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Skeleton } from "@/components/ui/skeleton";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo } from "react";
import type { DiscoverRow } from "../types";
import { CardCarousel } from "./card-carousel";
import {
  MODEL_CARD_HEIGHT_PX,
  MODEL_CARD_WIDTH_PX,
  ModelCard,
} from "./model-card";

const SKELETON_KEYS = ["s0", "s1", "s2", "s3", "s4"] as const;

function HubSectionRowSkeleton() {
  return (
    <div className="flex gap-4 overflow-hidden pb-4 pt-2">
      {SKELETON_KEYS.map((key) => (
        <Skeleton
          key={key}
          className="shrink-0 rounded-[20px]"
          style={{ width: MODEL_CARD_WIDTH_PX, height: MODEL_CARD_HEIGHT_PX }}
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
          <HugeiconsIcon
            icon={ArrowRight01Icon}
            strokeWidth={2}
            className="hub-section-chevron size-4 text-muted-foreground"
          />
        </button>
      </h2>
      {showSkeleton ? (
        <HubSectionRowSkeleton />
      ) : (
        <CardCarousel
          items={rows}
          getKey={(row) => row.id}
          itemWidth={MODEL_CARD_WIDTH_PX}
          itemHeight={MODEL_CARD_HEIGHT_PX}
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
