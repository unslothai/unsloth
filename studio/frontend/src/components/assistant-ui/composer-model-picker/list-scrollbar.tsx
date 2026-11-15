// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A floating thumb over a list with its own scrollbar hidden, so rows keep full width.

import { cn } from "@/lib/utils";
import { type RefObject, useEffect, useRef, useState } from "react";

const INSET = 4;
const MIN_THUMB = 24;

export function ListScrollbar({
  target,
  edge = false,
  stopAt,
}: {
  target: RefObject<HTMLDivElement | null>;
  edge?: boolean;
  /** A pinned element at the list's foot; the track ends at its top rather than running under it. */
  stopAt?: string;
}) {
  const [thumb, setThumb] = useState<{
    top: number;
    height: number;
    track: number;
  } | null>(null);
  const grab = useRef<{ y: number; scrollTop: number } | null>(null);
  const [listHover, setListHover] = useState(false);
  const [barHover, setBarHover] = useState(false);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    const el = target.current;
    if (!el) {
      return;
    }
    const measure = () => {
      const max = el.scrollHeight - el.clientHeight;
      el.toggleAttribute("data-fade-top", el.scrollTop > 1);
      el.toggleAttribute("data-fade-bottom", el.scrollTop < max - 1);
      if (max <= 1) {
        setThumb(null);
        return;
      }
      // Pinned to the foot, so its own height is what it covers; layout size, not the open animation's scale.
      const foot = stopAt ? el.querySelector<HTMLElement>(stopAt) : null;
      const under = foot?.offsetHeight ?? 0;
      const track = el.clientHeight - under - INSET * 2;
      const height = Math.min(
        track,
        Math.max(MIN_THUMB, (track * el.clientHeight) / el.scrollHeight),
      );
      setThumb({
        top: el.offsetTop + INSET + ((track - height) * el.scrollTop) / max,
        height,
        track,
      });
    };
    measure();
    el.addEventListener("scroll", measure, { passive: true });
    const enter = () => setListHover(true);
    const leave = () => setListHover(false);
    el.addEventListener("pointerenter", enter);
    el.addEventListener("pointerleave", leave);
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    const mutations = new MutationObserver(measure);
    mutations.observe(el, { childList: true, subtree: true });
    return () => {
      el.removeEventListener("scroll", measure);
      el.removeEventListener("pointerenter", enter);
      el.removeEventListener("pointerleave", leave);
      observer.disconnect();
      mutations.disconnect();
    };
  }, [target, stopAt]);

  if (!thumb) {
    return null;
  }
  return (
    <div
      aria-hidden={true}
      className="group/bar absolute right-0 w-[calc(14px*var(--ui-space-scale,1))] cursor-default"
      style={{ top: thumb.top, height: thumb.height }}
      onPointerEnter={() => setBarHover(true)}
      onPointerLeave={() => setBarHover(false)}
      onPointerDown={(event) => {
        const el = target.current;
        if (!el || event.button !== 0) {
          return;
        }
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        grab.current = { y: event.clientY, scrollTop: el.scrollTop };
        setDragging(true);
      }}
      onPointerMove={(event) => {
        const el = target.current;
        const start = grab.current;
        if (!(el && start)) {
          return;
        }
        const track = thumb.track - thumb.height;
        const max = el.scrollHeight - el.clientHeight;
        if (track > 0) {
          el.scrollTop =
            start.scrollTop + ((event.clientY - start.y) * max) / track;
        }
      }}
      onPointerUp={() => {
        grab.current = null;
        setDragging(false);
      }}
      onPointerCancel={() => {
        grab.current = null;
        setDragging(false);
      }}
    >
      <div
        className={cn(
          "composer-list-thumb absolute inset-y-0 w-[calc(4px*var(--ui-space-scale,1))] rounded-full opacity-0 transition-[width,opacity] duration-150 group-hover/bar:w-[calc(6px*var(--ui-space-scale,1))]",
          edge ? "right-px" : "right-[calc(5px*var(--ui-space-scale,1))]",
          (listHover || barHover || dragging) && "opacity-100",
        )}
      />
    </div>
  );
}
