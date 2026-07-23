// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { TrainIcon } from "@/features/hub/components/train-icon";
import { HUB_POST_DOWNLOAD_ACTIONS_VISIBLE } from "@/features/hub/lib/hub-feature-flags";
import { HugeiconsIcon } from "@hugeicons/react";
import type { LocalSource } from "../types";
import { DotTag } from "./dot-tag";
import { PathInfoButton } from "./path-info-button";

interface LocalDatasetCardProps {
  sourceLabel: string;
  source: LocalSource;
  path: string;
  onTrain?: () => void;
}

export function LocalDatasetCard({
  sourceLabel,
  source,
  path,
  onTrain,
}: LocalDatasetCardProps) {
  return (
    <div className="hub-download-card">
      <div className="group/dl flex items-center">
        <div className="relative flex h-9 min-w-0 flex-1 items-center pl-3 pr-2">
          <span className="flex min-w-0 items-center gap-1.5 text-[0.75rem] text-muted-foreground">
            <DotTag tone="success" label="On device" />
            {source !== "hf_cache" && (
              <span className="truncate text-muted-foreground/85">
                {sourceLabel}
              </span>
            )}
          </span>
          <div className="ml-auto flex items-center gap-0.5">
            <PathInfoButton path={path} />
          </div>
        </div>
        {HUB_POST_DOWNLOAD_ACTIONS_VISIBLE && (
          <div
            aria-hidden="true"
            className="ml-1 mr-0 h-5 w-px shrink-0 bg-foreground/[0.06] opacity-100 transition-opacity duration-150 group-hover/dl:opacity-0 dark:bg-white/[0.04]"
          />
        )}
        {/* Train CTA hidden until Hub→train picker ships. */}
        {onTrain && HUB_POST_DOWNLOAD_ACTIONS_VISIBLE && (
          <button
            type="button"
            onClick={() => onTrain()}
            className="hub-action-btn w-28"
          >
            <HugeiconsIcon icon={TrainIcon} strokeWidth={1.75} />
            Train
          </button>
        )}
      </div>
    </div>
  );
}
