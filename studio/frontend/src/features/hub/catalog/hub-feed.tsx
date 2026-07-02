// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { memo } from "react";
import { HUB_SECTION_TITLE, type HubSection } from "../lib/channels";
import type { DiscoverRow } from "../types";
import { HubSectionRow } from "./hub-section-row";

export interface HubFeedSectionData {
  rows: DiscoverRow[];
  isLoading: boolean;
}

export const HubFeed = memo(function HubFeed({
  trending,
  deviceType,
  isDataset,
  onSelect,
  onOpenChannel,
}: {
  trending: HubFeedSectionData;
  deviceType: string | null;
  isDataset: boolean;
  onSelect: (id: string) => void;
  onOpenChannel: (section: HubSection) => void;
}) {
  return (
    <div className="flex flex-col gap-10">
      <HubSectionRow
        title={HUB_SECTION_TITLE.trending}
        rows={trending.rows}
        isLoading={trending.isLoading}
        onSelect={onSelect}
        onOpenList={() => onOpenChannel("trending")}
        deviceType={deviceType}
        isDataset={isDataset}
      />
    </div>
  );
});
