// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LocalSource } from "../types";
import { DotTag } from "./dot-tag";
import { PathInfoButton } from "./path-info-button";

interface LocalDatasetCardProps {
  sourceLabel: string;
  source: LocalSource;
  path: string;
}

export function LocalDatasetCard({
  sourceLabel,
  source,
  path,
}: LocalDatasetCardProps) {
  return (
    <div className="hub-download-card">
      <div className="group/dl flex items-center">
        <div className="relative flex h-9 min-w-0 flex-1 items-center pl-3 pr-2">
          <span className="flex min-w-0 items-center gap-1.5 text-ui-12 text-muted-foreground">
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
      </div>
    </div>
  );
}
