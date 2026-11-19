// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDownStandardIcon, ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import type { AudioGalleryClip } from "../api";
import { audioModelLabel, formatClipDuration } from "../audio-workspace-utils";
import { variationLabel } from "../music/variation-groups";
import {
  CLIP_ROW,
  CLIP_ROW_ICON,
  CLIP_ROW_TILE,
  CLIPS_TITLE_INSET,
} from "./clips-card";

export function VariationGroupRow({
  clips,
  open,
  selected,
  onToggle,
}: {
  clips: readonly AudioGalleryClip[];
  open: boolean;
  selected: boolean;
  onToggle: () => void;
}) {
  const first = clips[0];
  if (!first) return null;
  return (
    <button
      type="button"
      aria-expanded={open}
      onClick={onToggle}
      className={cn(
        CLIP_ROW,
        "w-full min-w-0 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        selected && !open && "bg-wash-3",
      )}
    >
      <span className={CLIP_ROW_TILE}>
        <HugeiconsIcon
          icon={open ? ChevronDownStandardIcon : ChevronRightStandardIcon}
          strokeWidth={1.75}
          className={CLIP_ROW_ICON}
        />
      </span>
      <span className="flex min-w-0 flex-1 items-center gap-3 py-2.5 pe-4 ps-2">
        <span className="grid min-w-0 flex-1 gap-0.5">
          <span title={first.prompt} className="truncate text-ui-13">
            {first.prompt}
          </span>
          <span title={first.model} className="truncate text-ui-11p5 text-muted-foreground">
            {variationLabel(clips.length)} · {audioModelLabel(first.model)}
          </span>
        </span>
        {/* Plain, not CLIP_ROW_DURATION: this row has no menu to make room for. */}
        <span className="shrink-0 font-mono text-ui-11p5 tabular-nums text-muted-foreground">
          {formatClipDuration(first.duration_s)}
        </span>
      </span>
    </button>
  );
}

export function VariationChips({
  siblings,
  selectedId,
  onSelect,
}: {
  siblings: readonly AudioGalleryClip[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  if (siblings.length < 2) return null;
  return (
    <fieldset
      aria-label="Variations"
      className={cn("m-0 flex min-w-0 flex-wrap items-center gap-1 border-0 p-0", CLIPS_TITLE_INSET)}
    >
      <span className="me-1.5 text-ui-12 text-muted-foreground">
        {variationLabel(siblings.length)}
      </span>
      {siblings.map((clip, index) => {
        const current = clip.id === selectedId;
        return (
          <button
            key={clip.id}
            type="button"
            aria-pressed={current}
            title={`Variation ${index + 1}`}
            aria-label={`Variation ${index + 1} of ${siblings.length}`}
            onClick={() => onSelect(clip.id)}
            className={cn(
              "flex size-[calc(24px*var(--ui-space-scale,1))] items-center justify-center rounded-full font-mono text-ui-11p5 tabular-nums transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              current
                ? "bg-action text-background"
                : "bg-wash-8 text-muted-foreground hover:text-foreground",
            )}
          >
            {index + 1}
          </button>
        );
      })}
    </fieldset>
  );
}
