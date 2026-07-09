// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { ArrowLeft01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

export const CARD_GAP_PX = 16;
const CAROUSEL_TOP_PADDING_PX = 8;

export function CarouselArrow({
  side,
  visible,
  centerPx,
  onClick,
}: {
  side: "left" | "right";
  visible: boolean;
  centerPx: number;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-label={side === "left" ? "Scroll left" : "Scroll right"}
      aria-hidden={!visible}
      tabIndex={visible ? 0 : -1}
      onClick={onClick}
      style={{ top: centerPx }}
      className={cn(
        "hub-carousel-arrow absolute z-10 inline-flex size-9 -translate-y-1/2 items-center justify-center rounded-full",
        side === "left" ? "left-0 -translate-x-1/2" : "right-0 translate-x-1/2",
        visible
          ? "opacity-20 group-hover/carousel:opacity-100"
          : "pointer-events-none opacity-0",
      )}
    >
      <HugeiconsIcon
        icon={side === "left" ? ArrowLeft01Icon : ArrowRight01Icon}
        strokeWidth={2}
        className="size-4"
      />
    </button>
  );
}

export function CardCarousel<T>({
  items,
  getKey,
  renderItem,
  itemWidth,
  itemHeight,
  ariaLabel,
}: {
  items: T[];
  getKey: (item: T) => string;
  renderItem: (item: T) => ReactNode;
  itemWidth: number;
  itemHeight: number;
  ariaLabel: string;
}) {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const [canLeft, setCanLeft] = useState(false);
  const [canRight, setCanRight] = useState(false);
  const stepPx = itemWidth + CARD_GAP_PX;
  const arrowCenterPx = CAROUSEL_TOP_PADDING_PX + itemHeight / 2;

  const updateArrows = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    setCanLeft(el.scrollLeft > 1);
    setCanRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
  }, []);

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    updateArrows();
    const observer = new ResizeObserver(updateArrows);
    observer.observe(el);
    return () => observer.disconnect();
  }, [updateArrows]);

  useEffect(() => {
    updateArrows();
  }, [updateArrows, items]);

  const scrollByCards = useCallback(
    (direction: 1 | -1) => {
      scrollerRef.current?.scrollBy({
        left: direction * stepPx,
        behavior: "smooth",
      });
    },
    [stepPx],
  );

  // Click-and-drag panning (mouse only; touch/pen keep native scrolling).
  const drag = useRef<{ id: number; x: number; left: number; moved: boolean } | null>(
    null,
  );
  const suppressClick = useRef(false);

  const onPointerDown = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    suppressClick.current = false;
    const el = scrollerRef.current;
    if (!el || e.pointerType !== "mouse" || e.button !== 0) return;
    drag.current = { id: e.pointerId, x: e.clientX, left: el.scrollLeft, moved: false };
  }, []);

  const onPointerMove = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    const el = scrollerRef.current;
    if (!d || !el || e.pointerId !== d.id) return;
    // Primary button no longer held: the press ended off the scroller, so no
    // pointerup reached us. Drop the stale drag instead of scrolling on hover.
    if ((e.buttons & 1) === 0) {
      if (d.moved) el.style.scrollSnapType = "";
      drag.current = null;
      return;
    }
    const dx = e.clientX - d.x;
    // Ignore tiny moves so plain clicks still register.
    if (!d.moved && Math.abs(dx) < 5) return;
    if (!d.moved) {
      d.moved = true;
      // Snap fights the per-frame scrollLeft writes; disable it while dragging.
      el.style.scrollSnapType = "none";
      el.setPointerCapture(d.id);
    }
    el.scrollLeft = d.left - dx;
  }, []);

  const endDrag = useCallback((e: ReactPointerEvent<HTMLDivElement>) => {
    const d = drag.current;
    if (!d || e.pointerId !== d.id) return;
    if (d.moved) {
      // A drag just happened: swallow the click it would fire on a card.
      suppressClick.current = true;
      const el = scrollerRef.current;
      // Restore snap so the row settles on a card after the drag.
      if (el) el.style.scrollSnapType = "";
      el?.releasePointerCapture?.(d.id);
    }
    drag.current = null;
  }, []);

  const onClickCapture = useCallback((e: ReactMouseEvent<HTMLDivElement>) => {
    if (!suppressClick.current) return;
    suppressClick.current = false;
    e.preventDefault();
    e.stopPropagation();
  }, []);

  return (
    <div className="relative">
      <div
        ref={scrollerRef}
        onScroll={updateArrows}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onClickCapture={onClickCapture}
        // Stop the avatar image from starting a native drag during a pan.
        onDragStart={(e) => e.preventDefault()}
        aria-label={ariaLabel}
        // px-2 + -mx-2 give card shadows room so the edge cards aren't clipped;
        // scroll-px-2 keeps snap-start aligned with the heading.
        className="hub-carousel -mx-2 flex cursor-grab snap-x scroll-px-2 gap-4 overflow-x-auto px-2 pb-4 pt-2 select-none active:cursor-grabbing"
      >
        {items.map((item) => (
          <div
            key={getKey(item)}
            className="shrink-0 snap-start"
            style={{ width: itemWidth, height: itemHeight }}
          >
            {renderItem(item)}
          </div>
        ))}
      </div>
      <div
        aria-hidden="true"
        data-visible={canLeft || undefined}
        className="hub-carousel-fade hub-carousel-fade-left"
        style={{ top: CAROUSEL_TOP_PADDING_PX, height: itemHeight }}
      />
      <div
        aria-hidden="true"
        data-visible={canRight || undefined}
        className="hub-carousel-fade hub-carousel-fade-right"
        style={{ top: CAROUSEL_TOP_PADDING_PX, height: itemHeight }}
      />
      <CarouselArrow
        side="left"
        visible={canLeft}
        centerPx={arrowCenterPx}
        onClick={() => scrollByCards(-1)}
      />
      <CarouselArrow
        side="right"
        visible={canRight}
        centerPx={arrowCenterPx}
        onClick={() => scrollByCards(1)}
      />
    </div>
  );
}
