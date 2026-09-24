// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import {
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";
import { RAISED_SURFACE } from "../surface";

// Between the controls and the tabs they cover.
const GAP_PX = 16;
// How far the tabs fade out where they are cut off.
const FADE_PX = 48;
// Movement before a press on the tabs counts as a drag, so a plain click still switches tab.
const DRAG_SLOP_PX = 4;

export interface HeaderTab {
  key: string;
  label: string;
}

function fadeMask(left: boolean, right: boolean): string | undefined {
  if (!left && !right) return undefined;
  const start = left ? `transparent 0, black ${FADE_PX}px` : "black 0";
  const end = right ? `black calc(100% - ${FADE_PX}px), transparent 100%` : "black 100%";
  return `linear-gradient(to right, ${start}, ${end})`;
}

/** The tabs in one line. Once they overflow they fade where cut off and scroll, or drag, sideways. */
function TabStrip({
  tabs,
  active,
  onChange,
  reserve,
}: {
  tabs: HeaderTab[];
  active: string;
  onChange: (key: string) => void;
  /** Room kept free on the right for the controls laid over the row. */
  reserve: number;
}) {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ left: false, right: false });
  const drag = useRef<{ id: number; x: number; left: number; moved: boolean } | null>(null);
  const dragged = useRef(false);

  useEffect(() => {
    const scroller = scrollerRef.current;
    const list = listRef.current;
    if (!scroller || !list) return;
    const measure = () =>
      setEdges({
        left: scroller.scrollLeft > 1,
        right: scroller.scrollLeft + scroller.clientWidth < scroller.scrollWidth - 1,
      });
    const observer = new ResizeObserver(measure);
    observer.observe(scroller);
    observer.observe(list);
    scroller.addEventListener("scroll", measure, { passive: true });
    return () => {
      observer.disconnect();
      scroller.removeEventListener("scroll", measure);
    };
  }, []);

  // A tab chosen, or left, out of view is brought back into it.
  useEffect(() => {
    const scroller = scrollerRef.current;
    const tab = scroller?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!scroller || !tab) return;
    const left = tab.offsetLeft - scroller.offsetLeft;
    const right = left + tab.offsetWidth;
    if (left < scroller.scrollLeft) scroller.scrollLeft = left - FADE_PX;
    else if (right > scroller.scrollLeft + scroller.clientWidth) {
      scroller.scrollLeft = right - scroller.clientWidth + FADE_PX;
    }
  }, [active, reserve]);

  function onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    dragged.current = false;
    if (event.pointerType !== "mouse" || event.button !== 0 || !(edges.left || edges.right)) {
      return;
    }
    drag.current = {
      id: event.pointerId,
      x: event.clientX,
      left: event.currentTarget.scrollLeft,
      moved: false,
    };
  }

  function onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const current = drag.current;
    if (!current || current.id !== event.pointerId) return;
    const dx = event.clientX - current.x;
    if (!current.moved) {
      if (Math.abs(dx) < DRAG_SLOP_PX) return;
      current.moved = true;
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    event.currentTarget.scrollLeft = current.left - dx;
  }

  function onPointerEnd(event: ReactPointerEvent<HTMLDivElement>) {
    if (drag.current?.id !== event.pointerId) return;
    dragged.current = drag.current.moved;
    drag.current = null;
  }

  // The click that ends a drag must not also switch tab.
  function onClickCapture(event: ReactMouseEvent) {
    if (!dragged.current) return;
    dragged.current = false;
    event.preventDefault();
    event.stopPropagation();
  }

  const mask = fadeMask(edges.left, edges.right);
  return (
    <div
      ref={scrollerRef}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onClickCapture={onClickCapture}
      // The padding keeps the active tab's shadow clear of the scroller's clipping.
      className="-m-1 overflow-x-auto p-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      style={{ marginRight: reserve || undefined, maskImage: mask, WebkitMaskImage: mask }}
    >
      <nav ref={listRef} className="flex w-max gap-1" aria-label="Library sections">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            type="button"
            aria-current={tab.key === active ? "page" : undefined}
            onClick={() => onChange(tab.key)}
            className={cn(
              "flex h-9 shrink-0 items-center rounded-full px-3.5 text-[15px] text-foreground/80 transition-colors hover:text-foreground",
              tab.key === active && cn(RAISED_SURFACE, "font-medium text-foreground"),
            )}
          >
            {tab.label}
          </button>
        ))}
      </nav>
    </div>
  );
}

/**
 * The page header, as ChatGPT's Library does it: the title scrolls away, the tab row sticks to the
 * top, and the controls ride up from the title row to settle on the tab row. Where they then cover
 * tabs, the tabs stop short of them, fade at the cut, and scroll sideways.
 */
export function LibraryHeader({
  title,
  controls,
  tabs,
  tabsClassName,
}: {
  title: ReactNode;
  controls: ReactNode;
  tabsClassName?: string;
  /** None inside a folder; the row still sticks, to hold the controls. */
  tabs: { items: HeaderTab[]; active: string; onChange: (key: string) => void } | null;
}) {
  const controlsRef = useRef<HTMLDivElement>(null);
  const rowRef = useRef<HTMLDivElement>(null);
  const [controlsWidth, setControlsWidth] = useState(0);
  const [covered, setCovered] = useState(false);

  useEffect(() => {
    const controlsNode = controlsRef.current;
    const row = rowRef.current;
    if (!controlsNode || !row) return;
    let frame = 0;
    const update = () => {
      frame = 0;
      setControlsWidth(controlsNode.getBoundingClientRect().width);
      setCovered(row.getBoundingClientRect().top < controlsNode.getBoundingClientRect().bottom);
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(controlsNode);
    // Capture: the page scrolls inside the app shell, not the window.
    document.addEventListener("scroll", schedule, { capture: true, passive: true });
    window.addEventListener("resize", schedule);
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      document.removeEventListener("scroll", schedule, { capture: true });
      window.removeEventListener("resize", schedule);
    };
  }, []);

  return (
    <>
      {/* No height of its own: the controls hang from it over the title row, then stick level with
          the tabs, whose row is as tall as they are plus its padding. */}
      <div className="pointer-events-none sticky top-4 z-30 flex h-0 justify-end">
        <div ref={controlsRef} className="pointer-events-auto">
          {controls}
        </div>
      </div>
      <div
        className="flex h-9 min-w-0 items-center"
        style={{ paddingRight: controlsWidth + GAP_PX }}
      >
        {title}
      </div>
      <div
        ref={rowRef}
        className="sticky top-0 z-20 -mx-6 mt-2 bg-background px-6 py-4 sm:-mx-10 sm:px-10"
      >
        {tabs ? (
          <div className={tabsClassName}>
            <TabStrip
              tabs={tabs.items}
              active={tabs.active}
              onChange={tabs.onChange}
              reserve={covered ? controlsWidth + GAP_PX : 0}
            />
          </div>
        ) : (
          <div className="h-9" />
        )}
        {/* Once stuck, what scrolls under the row fades out below it instead of cutting off: the
            chat page's fade under its header. */}
        <div
          aria-hidden
          className={cn(
            "pointer-events-none absolute inset-x-0 top-full h-6 bg-gradient-to-b from-background to-transparent transition-opacity",
            covered ? "opacity-100" : "opacity-0",
          )}
        />
      </div>
    </>
  );
}
