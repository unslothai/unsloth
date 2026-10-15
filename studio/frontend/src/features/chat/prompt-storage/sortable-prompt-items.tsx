// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { cn } from "@/lib/utils";
import { autoscrollDelta, clipSpan } from "./autoscroll";
import { flipShifts, insertionIndex, ownsDrag } from "./reorder";
import { GripVerticalIcon, XIcon } from "lucide-react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

export function AutoTextarea({
  value,
  onChange,
  placeholder,
  minRows = 1,
  maxHeight = 320,
  className,
}: {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  minRows?: number;
  maxHeight?: number;
  className?: string;
}): ReactElement {
  const ref = useRef<HTMLTextAreaElement>(null);

  const measure = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    const next = Math.min(el.scrollHeight, maxHeight);
    el.style.height = `${next}px`;
    el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [maxHeight]);

  // Layout effect: measuring after paint flashes the wrong height.
  useLayoutEffect(measure, [value, measure]);

  // Width changes rewrap without a `value` change, so re-measure.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    window.addEventListener("resize", measure);

    let observer: ResizeObserver | undefined;
    if (typeof ResizeObserver !== "undefined") {
      let lastWidth = el.clientWidth;
      observer = new ResizeObserver(() => {
        // Width only; it also fires for the height we just set, which would loop.
        const width = el.clientWidth;
        if (width === lastWidth) return;
        lastWidth = width;
        measure();
      });
      observer.observe(el);
    }

    return () => {
      window.removeEventListener("resize", measure);
      observer?.disconnect();
    };
  }, [measure]);

  return (
    <textarea
      ref={ref}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      rows={minRows}
      className={cn(
        "w-full resize-none rounded-lg border-0 bg-background/80 px-3 py-2 text-sm ring-1 ring-border/60 outline-none focus:ring-ring transition-shadow leading-relaxed",
        className,
      )}
    />
  );
}

function move<T>(arr: T[], from: number, to: number): T[] {
  const next = [...arr];
  const [item] = next.splice(from, 1);
  next.splice(to, 0, item);
  return next;
}

let uidSeq = 0;
function nextUid(): string {
  uidSeq += 1;
  return `i${uidSeq}`;
}

function findScrollParents(el: HTMLElement | null): HTMLElement[] {
  const out: HTMLElement[] = [];
  let node = el?.parentElement ?? null;
  while (node) {
    const overflowY = getComputedStyle(node).overflowY;
    if (overflowY === "auto" || overflowY === "scroll") out.push(node);
    node = node.parentElement;
  }
  return out;
}

// Keyed by synthetic uid, not index, or reorder swaps values under the caret.
export function SortablePromptItems({
  items,
  onChange,
  minItems = 0,
  preview = false,
}: {
  items: string[];
  onChange: (items: string[]) => void;
  minItems?: number;
  preview?: boolean;
}): ReactElement {
  const [uids, setUids] = useState<string[]>(() => items.map(() => nextUid()));
  const [draggingUid, setDraggingUid] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);
  const pointerYRef = useRef(0);
  const pointerIdRef = useRef<number | null>(null);
  const rowRefs = useRef(new Map<string, HTMLDivElement>());
  const prevOffsets = useRef(new Map<string, number>());
  const uidsRef = useRef(uids);
  // Only a reorder animates, or a growing textarea makes rows below wobble.
  const reorderTick = useRef(0);
  const animatedTick = useRef(0);

  // Resync during render, not an effect, to avoid painting a stale key.
  let rowUids = uids;
  if (uids.length !== items.length) {
    rowUids = items.map((_, i) => uids[i] ?? nextUid());
    setUids(rowUids);
  }

  // Layout offsets, not client rects: rects shift with autoscroll.
  const measureOffsets = useCallback(() => {
    const offsets = new Map<string, number>();
    rowRefs.current.forEach((el, uid) => offsets.set(uid, el.offsetTop));
    return offsets;
  }, []);

  useLayoutEffect(() => {
    if (reorderTick.current === animatedTick.current) return;
    animatedTick.current = reorderTick.current;
    flipShifts(prevOffsets.current, measureOffsets()).forEach((dy, uid) => {
      const el = rowRefs.current.get(uid);
      if (!el) return;
      el.style.transition = "none";
      el.style.transform = `translateY(${dy}px)`;
      requestAnimationFrame(() => {
        el.style.transition = "transform 180ms cubic-bezier(0.2, 0, 0, 1)";
        el.style.transform = "";
      });
    });
  }, [uids, measureOffsets]);

  const itemsRef = useRef(items);
  const onChangeRef = useRef(onChange);
  useLayoutEffect(() => {
    itemsRef.current = items;
    onChangeRef.current = onChange;
    uidsRef.current = rowUids;
  });

  const applyOrder = useCallback((from: number, to: number) => {
    if (from === to) return;
    // Read FLIP "first" now: row heights change independently of order.
    prevOffsets.current = measureOffsets();
    reorderTick.current += 1;
    const nextItems = move(itemsRef.current, from, to);
    const nextUids = move(uidsRef.current, from, to);
    // Advance mirrors now: a drag reorders faster than a commit.
    itemsRef.current = nextItems;
    uidsRef.current = nextUids;
    onChangeRef.current(nextItems);
    setUids(nextUids);
  }, [measureOffsets]);

  const handlePointerDown = useCallback(
    (uid: string, e: React.PointerEvent<HTMLButtonElement>) => {
      // Primary press only: other buttons are not caught by the zero-buttons release check.
      if (e.button !== 0 || !e.isPrimary) return;
      // isPrimary is per pointer type, so a mouse press can pass it mid-touch-drag.
      if (pointerIdRef.current !== null) return;
      pointerIdRef.current = e.pointerId;
      // No setPointerCapture: moving the row's DOM node releases capture.
      e.preventDefault();
      pointerYRef.current = e.clientY;
      setDraggingUid(uid);
    },
    [],
  );

  useEffect(() => {
    if (!draggingUid) return;
    const container = containerRef.current;
    if (!container) return;
    const scrollers = findScrollParents(container);
    let raf = 0;

    const evaluate = () => {
      const order = uidsRef.current;
      const from = order.indexOf(draggingUid);
      if (from < 0) return;

      // offsetTop/offsetHeight ignore transforms; client rects are wrong mid-FLIP.
      const localY = pointerYRef.current - container.getBoundingClientRect().top;
      const boxes = order.map((uid) => {
        const el = rowRefs.current.get(uid);
        return el ? { top: el.offsetTop, height: el.offsetHeight } : undefined;
      });
      const to = insertionIndex(boxes, from, localY);
      if (to !== from) applyOrder(from, to);
    };

    // pointerdown suppresses native drag scrolling, so scroll here.
    const tick = () => {
      for (let i = 0; i < scrollers.length; i++) {
        const scroller = scrollers[i];
        const limit = scroller.scrollHeight - scroller.clientHeight;
        if (limit <= 0) continue;
        const span = clipSpan(scroller.getBoundingClientRect(), [
          ...scrollers.slice(i + 1).map((el) => el.getBoundingClientRect()),
          { top: 0, bottom: window.innerHeight },
        ]);
        if (!span) continue;
        const delta = autoscrollDelta(pointerYRef.current, span.top, span.bottom);
        if (delta === 0) continue;
        const before = scroller.scrollTop;
        const next = Math.max(0, Math.min(limit, before + delta));
        if (next === before) continue;
        scroller.scrollTop = next;
        evaluate();
        break;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    const endDrag = () => {
      pointerIdRef.current = null;
      setDraggingUid(null);
    };

    // Ignore other pointers, or a second finger reorders or ends this drag.
    const isDragPointer = (e: PointerEvent) =>
      ownsDrag(pointerIdRef.current, e.pointerId);

    const onMove = (e: PointerEvent) => {
      if (!isDragPointer(e)) return;
      // Release outside the window sends no pointerup; treat a buttonless move as release.
      if (e.buttons === 0) {
        endDrag();
        return;
      }
      pointerYRef.current = e.clientY;
      evaluate();
    };

    const onPointerEnd = (e: PointerEvent) => {
      if (!isDragPointer(e)) return;
      endDrag();
    };

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onPointerEnd);
    window.addEventListener("pointercancel", onPointerEnd);
    window.addEventListener("blur", endDrag);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onPointerEnd);
      window.removeEventListener("pointercancel", onPointerEnd);
      window.removeEventListener("blur", endDrag);
    };
  }, [draggingUid, applyOrder]);

  const nudge = useCallback(
    (uid: string, delta: number) => {
      const from = uidsRef.current.indexOf(uid);
      const to = from + delta;
      if (from < 0 || to < 0 || to >= itemsRef.current.length) return;
      applyOrder(from, to);
    },
    [applyOrder],
  );

  return (
    // `relative` makes this the offsetParent the hit-test measures against.
    <div
      ref={containerRef}
      className={cn(
        "relative flex flex-col gap-2",
        draggingUid && "select-none",
      )}
    >
      {items.map((item, i) => {
        const uid = rowUids[i] ?? `fallback-${i}`;
        const isDragging = uid === draggingUid;
        return (
          <div
            key={uid}
            ref={(el) => {
              if (el) rowRefs.current.set(uid, el);
              else rowRefs.current.delete(uid);
            }}
            className={cn(
              "flex items-start gap-2 rounded-lg",
              isDragging && "relative z-10 bg-muted/60 ring-1 ring-border",
            )}
          >
            <button
              type="button"
              aria-label={`Reorder prompt ${i + 1}`}
              onPointerDown={(e) => handlePointerDown(uid, e)}
              onKeyDown={(e) => {
                if (e.key === "ArrowUp") {
                  e.preventDefault();
                  nudge(uid, -1);
                } else if (e.key === "ArrowDown") {
                  e.preventDefault();
                  nudge(uid, 1);
                }
              }}
              className={cn(
                "mt-1.5 flex h-7 w-5 shrink-0 touch-none items-center justify-center rounded text-muted-foreground/30 hover:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring outline-none transition-colors",
                isDragging ? "cursor-grabbing text-muted-foreground" : "cursor-grab",
              )}
            >
              <GripVerticalIcon className="size-4" />
            </button>
            <span className="mt-2.5 w-5 shrink-0 text-right text-xs font-medium tabular-nums text-muted-foreground/60">
              {i + 1}.
            </span>
            <div className="min-w-0 flex-1">
              {preview ? (
                <MarkdownPreview
                  markdown={item}
                  className="max-h-none w-full bg-background/40 p-3 text-sm"
                />
              ) : (
                <AutoTextarea
                  value={item}
                  onChange={(v) => onChange(items.map((x, idx) => (idx === i ? v : x)))}
                  placeholder={`Prompt ${i + 1}...`}
                />
              )}
            </div>
            {preview ? null : (
              <button
                type="button"
                onClick={() => {
                  onChange(items.filter((_, idx) => idx !== i));
                  setUids((prev) => prev.filter((_, idx) => idx !== i));
                }}
                disabled={items.length <= minItems}
                className="mt-1.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                title="Remove"
              >
                <XIcon className="size-3.5" />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
