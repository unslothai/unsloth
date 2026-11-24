// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import {
  type KeyboardEvent,
  type PointerEvent,
  type ReactNode,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { SliderLevel } from "./picker-model";

const H = 18;
const KNOB = 24;
const DOT = 4;

export function EffortSlider({
  levels,
  value,
  onChange,
  title,
  accessory,
  disabled = false,
}: {
  levels: readonly SliderLevel[];
  value: string;
  onChange: (value: string) => void;
  title: string;
  accessory?: ReactNode;
  disabled?: boolean;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [hover, setHover] = useState<number | null>(null);
  // The thumb steps level to level, so release has nothing to snap.
  const [live, setLive] = useState<number | null>(null);

  useLayoutEffect(() => {
    const el = trackRef.current;
    if (!el) {
      return;
    }
    setWidth(el.getBoundingClientRect().width);
    const observer = new ResizeObserver(([entry]) => {
      setWidth(entry.contentRect.width);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const count = levels.length;
  const selected = Math.max(
    0,
    levels.findIndex((l) => l.value === value),
  );
  const index = live ?? selected;
  const measured = width > KNOB;
  const frac = (i: number) => (count > 1 ? i / (count - 1) : 0);
  const spot = (f: number) => H / 2 + Math.round(f * (width - H));
  const nearest = (x: number) => {
    const span = width - H;
    const f = span > 0 ? (x - spot(0)) / span : 0;
    return Math.min(count - 1, Math.max(0, Math.round(f * (count - 1))));
  };
  const pos = spot(frac(index));

  const localX = (event: PointerEvent<HTMLDivElement>) =>
    event.clientX - (trackRef.current?.getBoundingClientRect().left ?? 0);
  const follow = (event: PointerEvent<HTMLDivElement>) => {
    setLive(nearest(localX(event)));
  };
  const commit = (i: number) => {
    const next = levels[i];
    if (next && next.value !== value) {
      onChange(next.value);
    }
  };
  const release = () => {
    if (live !== null) {
      commit(live);
    }
    setLive(null);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) {
      return;
    }
    let next: number | null = null;
    if (event.key === "ArrowLeft" || event.key === "ArrowDown") {
      next = Math.max(0, selected - 1);
    } else if (event.key === "ArrowRight" || event.key === "ArrowUp") {
      next = Math.min(count - 1, selected + 1);
    } else if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = count - 1;
    }
    if (next === null) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    commit(next);
  };

  const levelLabel = (
    <span
      key={levels[index]?.value}
      className="composer-effort-label text-ui-14 font-medium"
    >
      {levels[index]?.label}
    </span>
  );
  const titleLabel = (
    <span className="text-ui-10 font-semibold uppercase tracking-wider text-muted-foreground">
      {title}
    </span>
  );

  return (
    <div className="flex w-full flex-col gap-2">
      {accessory ? (
        <div className="flex items-center gap-2">
          <div className="flex min-w-0 flex-1 flex-col gap-0.5">
            {titleLabel}
            {levelLabel}
          </div>
          {accessory}
        </div>
      ) : (
        <div className="flex items-center justify-between gap-2">
          {titleLabel}
          {levelLabel}
        </div>
      )}
      <div
        ref={trackRef}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label={title}
        aria-valuemin={0}
        aria-valuemax={Math.max(0, count - 1)}
        aria-valuenow={index}
        aria-valuetext={levels[index]?.label}
        aria-disabled={disabled || undefined}
        className={cn(
          "composer-effort-slider relative w-full shrink-0 cursor-pointer touch-none rounded-full outline-none",
          disabled && "cursor-not-allowed opacity-50",
        )}
        style={{ height: KNOB }}
        onPointerDown={(event) => {
          if (disabled || event.button !== 0) {
            return;
          }
          event.currentTarget.setPointerCapture(event.pointerId);
          follow(event);
        }}
        onPointerMove={(event) => {
          if (disabled) {
            return;
          }
          setHover(nearest(localX(event)));
          if (live !== null) {
            follow(event);
          }
        }}
        onPointerUp={release}
        onPointerCancel={release}
        onPointerLeave={() => setHover(null)}
        onKeyDown={onKeyDown}
      >
        {measured ? (
          <div className="pointer-events-none absolute inset-0">
            <div
              className="composer-effort-track absolute inset-x-0 rounded-full"
              style={{ top: (KNOB - H) / 2, height: H }}
            />
            <div
              className="composer-effort-fill absolute left-0 rounded-full transition-[width] duration-150 ease-out"
              style={{ top: (KNOB - H) / 2, height: H, width: pos + H / 2 }}
            />
            {levels.map((level, i) => {
              const filled = spot(frac(i)) <= pos;
              const isHover = hover === i && i !== index;
              const size = isHover ? DOT + 3 : DOT;
              return (
                <div
                  key={level.value}
                  className={cn(
                    "absolute rounded-full transition-[width,height,top,left,opacity] duration-150",
                    filled
                      ? "composer-effort-dot-on"
                      : "composer-effort-dot-off",
                    i === index && "opacity-0",
                  )}
                  style={{
                    width: size,
                    height: size,
                    top: (KNOB - size) / 2,
                    left: spot(frac(i)) - size / 2,
                  }}
                />
              );
            })}
            <div
              className="composer-effort-thumb absolute rounded-full transition-transform duration-150 ease-out"
              style={{
                transform: `translateX(${pos - KNOB / 2}px)`,
                top: 0,
                width: KNOB,
                height: KNOB,
              }}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}
