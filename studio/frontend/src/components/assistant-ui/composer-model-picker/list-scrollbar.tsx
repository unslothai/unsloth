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
    // Rows change with tab and search without resizing the capped list.
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

/** The tab strip's sideways twin: overlays the green line, so it never takes height from the tabs. */
export function StripScrollbar({ strip }: { strip: HTMLDivElement | null }) {
  const [thumb, setThumb] = useState<{
    left: number;
    width: number;
    top: number;
    track: number;
  } | null>(null);
  const grab = useRef<{ x: number; scrollLeft: number } | null>(null);
  const [stripHover, setStripHover] = useState(false);
  const [barHover, setBarHover] = useState(false);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!strip) {
      return;
    }
    const measure = () => {
      const max = strip.scrollWidth - strip.clientWidth;
      if (max <= 1) {
        setThumb(null);
        return;
      }
      const track = strip.clientWidth - INSET * 2;
      const width = Math.min(
        track,
        Math.max(MIN_THUMB, (track * strip.clientWidth) / strip.scrollWidth),
      );
      // Centred on the tabs' green line, which it overlays.
      const line = strip.querySelector<HTMLElement>(".composer-tab-line");
      const lineMid = line
        ? line.getBoundingClientRect().top +
          line.getBoundingClientRect().height / 2 -
          strip.getBoundingClientRect().top
        : strip.offsetHeight - 1;
      setThumb({
        left:
          strip.offsetLeft + INSET + ((track - width) * strip.scrollLeft) / max,
        width,
        top: strip.offsetTop + lineMid,
        track,
      });
    };
    measure();
    strip.addEventListener("scroll", measure, { passive: true });
    const enter = () => setStripHover(true);
    const leave = () => setStripHover(false);
    strip.addEventListener("pointerenter", enter);
    strip.addEventListener("pointerleave", leave);
    const observer = new ResizeObserver(measure);
    observer.observe(strip);
    const mutations = new MutationObserver(measure);
    mutations.observe(strip, { childList: true });
    return () => {
      strip.removeEventListener("scroll", measure);
      strip.removeEventListener("pointerenter", enter);
      strip.removeEventListener("pointerleave", leave);
      observer.disconnect();
      mutations.disconnect();
    };
  }, [strip]);

  if (!thumb) {
    return null;
  }
  return (
    <div
      aria-hidden={true}
      className="group/bar absolute h-[calc(10px*var(--ui-space-scale,1))] -translate-y-1/2 cursor-default"
      style={{ left: thumb.left, top: thumb.top, width: thumb.width }}
      onPointerEnter={() => setBarHover(true)}
      onPointerLeave={() => setBarHover(false)}
      onPointerDown={(event) => {
        if (!strip || event.button !== 0) {
          return;
        }
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        grab.current = { x: event.clientX, scrollLeft: strip.scrollLeft };
        setDragging(true);
      }}
      onPointerMove={(event) => {
        const start = grab.current;
        if (!(strip && start)) {
          return;
        }
        const track = thumb.track - thumb.width;
        const max = strip.scrollWidth - strip.clientWidth;
        if (track > 0) {
          strip.scrollLeft =
            start.scrollLeft + ((event.clientX - start.x) * max) / track;
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
          "composer-list-thumb absolute inset-x-0 top-1/2 h-[calc(4px*var(--ui-space-scale,1))] -translate-y-1/2 rounded-full opacity-0 transition-[height,opacity] duration-150 group-hover/bar:h-[calc(6px*var(--ui-space-scale,1))]",
          (stripHover || barHover || dragging) && "opacity-100",
        )}
      />
    </div>
  );
}
