// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Add01Icon, Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  type PointerEvent,
  useMemo,
  useRef,
  useState,
} from "react";
import { Waveform } from "./waveform";
import {
  type RangeEdge,
  type RangeLimits,
  type WaveRange,
  addRange,
  dragToRange,
  formatRange,
  formatRangeTime,
  moveRange,
  nextFreeRange,
  normalizeRanges,
  nudgeStep,
  pxToSeconds,
  rangeLabel,
  removeRange,
  secondsBeyondEnd,
  secondsToFraction,
  timelineEnd,
} from "./waveform-range";

// A press that moves less than this is a click (seek), not a drag.
const DRAG_THRESHOLD_PX = 4;

type Drag =
  | {
      kind: "create";
      pointerId: number;
      startX: number;
      anchorS: number;
      active: boolean;
    }
  | {
      kind: "edge";
      pointerId: number;
      index: number;
      edge: RangeEdge;
      grabS: number;
      base: WaveRange[];
    };

export function WaveformRangeSelect({
  peaks,
  durationS,
  src,
  label,
  ranges,
  onChange,
  maxRanges,
  beyondEndS = 0,
  disabled = false,
  className,
}: {
  peaks: readonly number[] | null;
  durationS: number | null;
  src: string | null;
  label: string;
  ranges: readonly WaveRange[];
  onChange: (next: WaveRange[]) => void;
  maxRanges: number;
  beyondEndS?: number;
  disabled?: boolean;
  className?: string;
}) {
  const duration = durationS && durationS > 0 ? durationS : 0;
  const limits: RangeLimits = useMemo(
    () => ({ durationS: duration, maxRanges, beyondEndS }),
    [duration, maxRanges, beyondEndS],
  );
  const timeline = timelineEnd(limits);
  const areaRef = useRef<HTMLDivElement | null>(null);
  const listRef = useRef<HTMLUListElement | null>(null);
  const drag = useRef<Drag | null>(null);
  const suppressClick = useRef(false);
  const [preview, setPreview] = useState<WaveRange[] | null>(null);
  const [focused, setFocused] = useState<number | null>(null);
  const shown = preview ?? ranges;
  const ready = duration > 0 && !disabled;
  const canAdd = ready && nextFreeRange(ranges, limits) !== null;

  const secondsAt = (clientX: number) => {
    const box = areaRef.current?.getBoundingClientRect();
    if (!box) return 0;
    return pxToSeconds(clientX - box.left, box.width, timeline);
  };

  const commit = (next: WaveRange[]) => onChange(normalizeRanges(next, limits));

  const focusChip = (index: number) => {
    const chips =
      listRef.current?.querySelectorAll<HTMLElement>("[data-range-chip]");
    if (!chips || chips.length === 0) return;
    chips[Math.max(0, Math.min(index, chips.length - 1))]?.focus();
  };

  const remove = (index: number) => {
    commit(removeRange(ranges, index));
    setFocused(null);
    requestAnimationFrame(() => focusChip(index));
  };

  const onRangeKey = (
    event: KeyboardEvent<HTMLElement>,
    index: number,
    edge: RangeEdge,
  ) => {
    if (disabled) return;
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      remove(index);
      return;
    }
    const step = nudgeStep(event);
    if (step === null) return;
    event.preventDefault();
    commit(moveRange(ranges, index, edge, step, limits));
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (!ready || event.button !== 0) return;
    const area = areaRef.current?.parentElement;
    if (!area?.contains(event.target as Node)) return;
    drag.current = {
      kind: "create",
      pointerId: event.pointerId,
      startX: event.clientX,
      anchorS: secondsAt(event.clientX),
      active: false,
    };
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    const at = secondsAt(event.clientX);
    if (current.kind === "create") {
      if (!current.active) {
        if (Math.abs(event.clientX - current.startX) < DRAG_THRESHOLD_PX)
          return;
        current.active = true;
        // Captured only once it is a drag, so a plain click still seeks.
        event.currentTarget.setPointerCapture(event.pointerId);
      }
      const range = dragToRange(current.anchorS, at, limits);
      setPreview(range ? addRange(ranges, range, limits) : [...ranges]);
      return;
    }
    setPreview(
      moveRange(
        current.base,
        current.index,
        current.edge,
        at - current.grabS,
        limits,
      ),
    );
  };

  const endDrag = (event: PointerEvent<HTMLDivElement>, cancelled: boolean) => {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    drag.current = null;
    const moved = current.kind === "edge" || current.active;
    if (moved) {
      // The click ending a drag must not seek; a touch drag may send none, so it lapses.
      suppressClick.current = true;
      window.setTimeout(() => {
        suppressClick.current = false;
      }, 0);
    }
    if (!cancelled && moved && preview) commit(preview);
    setPreview(null);
  };

  const startEdgeDrag = (
    event: PointerEvent<HTMLElement>,
    index: number,
    edge: RangeEdge,
  ) => {
    if (!ready || event.button !== 0) return;
    event.stopPropagation();
    event.preventDefault();
    const root = event.currentTarget.closest<HTMLElement>("[data-range-root]");
    root?.setPointerCapture(event.pointerId);
    drag.current = {
      kind: "edge",
      pointerId: event.pointerId,
      index,
      edge,
      grabS: secondsAt(event.clientX),
      base: [...ranges],
    };
    focusChip(index);
  };

  const overlay =
    duration > 0 ? (
      <div ref={areaRef} className="pointer-events-none absolute inset-0">
        {shown.map((range, index) => {
          const left = secondsToFraction(range.start_s, timeline) * 100;
          const width = secondsToFraction(range.end_s, timeline) * 100 - left;
          return (
            <div
              // biome-ignore lint/suspicious/noArrayIndexKey: ranges are kept sorted; the index is their name.
              key={index}
              className={cn(
                "absolute inset-y-0 rounded-md bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)] ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(45%*var(--contrast-edge-gain,1)),transparent)]",
                focused === index && "ring-2 ring-ring",
                ready && "pointer-events-auto cursor-grab",
              )}
              style={{ left: `${left}%`, width: `${width}%` }}
              onPointerDown={(event) => startEdgeDrag(event, index, "both")}
            >
              {(["start", "end"] as const).map((edge) => {
                const value = edge === "start" ? range.start_s : range.end_s;
                return (
                  <div
                    key={edge}
                    role="slider"
                    tabIndex={ready ? 0 : -1}
                    aria-label={`${edge === "start" ? "Start" : "End"} of range ${index + 1}`}
                    aria-valuemin={0}
                    aria-valuemax={Math.round(timeline * 10) / 10}
                    aria-valuenow={Math.round(value * 10) / 10}
                    aria-valuetext={`${formatRangeTime(value)}`}
                    aria-disabled={!ready}
                    className={cn(
                      "absolute inset-y-0 flex w-[calc(12px*var(--ui-space-scale,1))] cursor-ew-resize items-center justify-center outline-none focus-visible:ring-2 focus-visible:ring-ring",
                      edge === "start"
                        ? "left-0 -translate-x-1/2 rounded-l-md"
                        : "right-0 translate-x-1/2 rounded-r-md",
                    )}
                    onPointerDown={(event) => startEdgeDrag(event, index, edge)}
                    onKeyDown={(event) => onRangeKey(event, index, edge)}
                    onFocus={() => setFocused(index)}
                    onBlur={() => setFocused(null)}
                  >
                    <span className="h-1/2 w-[3px] rounded-full bg-foreground" />
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    ) : undefined;

  return (
    <div className={cn("grid gap-2", className)}>
      <div
        data-range-root=""
        className={cn(ready && "touch-none select-none")}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(event) => endDrag(event, false)}
        onPointerCancel={(event) => endDrag(event, true)}
        onClickCapture={(event) => {
          if (!suppressClick.current) return;
          suppressClick.current = false;
          event.stopPropagation();
          event.preventDefault();
        }}
      >
        <Waveform
          peaks={peaks}
          durationS={durationS}
          src={src}
          label={label}
          tailS={duration > 0 ? beyondEndS : 0}
          overlay={overlay}
        />
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <ul
          ref={listRef}
          aria-label={`Selected parts of ${label}`}
          className="contents"
        >
          {ranges.map((range, index) => {
            const beyond = secondsBeyondEnd(range, duration);
            return (
              <li
                // biome-ignore lint/suspicious/noArrayIndexKey: ranges are kept sorted; the index is their name.
                key={index}
                className="flex items-center rounded-4xl bg-muted"
              >
                <button
                  type="button"
                  data-range-chip=""
                  disabled={disabled}
                  aria-label={`${rangeLabel(index, range)}${beyond > 0 ? `, ${beyond} seconds past the end` : ""}`}
                  aria-keyshortcuts="Delete Backspace ArrowLeft ArrowRight"
                  title="Arrow keys move it, Shift+arrow by a second, Delete removes it"
                  className={cn(
                    "rounded-l-4xl py-0.5 pl-2.5 pr-1 font-mono text-ui-11p5 tabular-nums text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    focused === index && "underline underline-offset-2",
                  )}
                  onKeyDown={(event) => onRangeKey(event, index, "both")}
                  onFocus={() => setFocused(index)}
                  onBlur={() => setFocused(null)}
                >
                  {formatRange(range)}
                </button>
                <button
                  type="button"
                  disabled={disabled}
                  aria-label={`Remove range ${index + 1}`}
                  className="flex items-center rounded-r-4xl py-1 pl-0.5 pr-2 text-muted-foreground outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
                  onClick={() => remove(index)}
                >
                  <HugeiconsIcon icon={Cancel01Icon} className="size-3" />
                </button>
              </li>
            );
          })}
        </ul>
        {maxRanges > 0 && ranges.length < maxRanges ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-auto px-2 py-1 text-ui-11p5"
            disabled={!canAdd}
            onClick={() => {
              const next = nextFreeRange(ranges, limits);
              if (!next) return;
              const updated = addRange(ranges, next, limits);
              commit(updated);
              const at = updated.findIndex(
                (range) => range.start_s === next.start_s,
              );
              requestAnimationFrame(() => focusChip(at));
            }}
          >
            <HugeiconsIcon icon={Add01Icon} className="size-3" />
            {ranges.length === 0 ? "Select a part" : "Add a part"}
          </Button>
        ) : null}
      </div>
    </div>
  );
}
