// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { ArrowDown01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { AudioGalleryClip } from "../api";
import { audioModelLabel, formatClipDuration } from "../audio-workspace-utils";
import { variationLabel } from "../music/variation-groups";

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
        "flex w-full min-w-0 items-center gap-2 rounded-row px-2 py-1.5 text-left text-ui-13 transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        selected && !open && "bg-muted",
      )}
    >
      <HugeiconsIcon
        icon={open ? ArrowDown01Icon : ArrowRight01Icon}
        className="size-3.5 shrink-0 text-muted-foreground"
      />
      <span className="min-w-0 flex-1 truncate">{first.prompt}</span>
      <span className="shrink-0 rounded-4xl bg-muted px-2 py-0.5 text-ui-11 text-muted-foreground">
        {variationLabel(clips.length)}
      </span>
      <span
        title={first.model}
        className="max-w-[30%] shrink-0 truncate text-ui-11p5 text-muted-foreground"
      >
        {audioModelLabel(first.model)}
      </span>
      <span className="shrink-0 text-ui-11p5 text-muted-foreground">
        {formatClipDuration(first.duration_s)}
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
      className="m-0 flex min-w-0 flex-wrap items-center gap-1 border-0 p-0"
    >
      <span className="mr-1 text-ui-11p5 text-muted-foreground">
        {variationLabel(siblings.length)}
      </span>
      {siblings.map((clip, index) => {
        const current = clip.id === selectedId;
        return (
          <button
            key={clip.id}
            type="button"
            aria-pressed={current}
            aria-label={`Variation ${index + 1} of ${siblings.length}`}
            onClick={() => onSelect(clip.id)}
            className={cn(
              "flex size-[calc(24px*var(--ui-space-scale,1))] items-center justify-center rounded-full font-mono text-ui-11p5 tabular-nums transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              current
                ? "bg-foreground text-background"
                : "bg-muted text-muted-foreground hover:text-foreground",
            )}
          >
            {index + 1}
          </button>
        );
      })}
    </fieldset>
  );
}
