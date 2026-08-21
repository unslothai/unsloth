// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { cn } from "@/lib/utils";
import { autoscrollDelta } from "./autoscroll";
import { GripVerticalIcon, XIcon } from "lucide-react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

// Textarea that grows with its content, capped at maxHeight.
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

  // Layout effect: measuring after paint flashes the wrong height while typing.
  useLayoutEffect(measure, [value, measure]);

  // A width change rewraps the text without touching `value`, and overflowY may
  // be "hidden", so the new lines would be clipped until the next keystroke.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    window.addEventListener("resize", measure);

    // Catches the column resizing without the window doing so.
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

// Nearest ancestor a drag can scroll.
function findScrollParent(el: HTMLElement | null): HTMLElement | null {
  let node = el?.parentElement ?? null;
  while (node) {
    const overflowY = getComputedStyle(node).overflowY;
    if (
      (overflowY === "auto" || overflowY === "scroll") &&
      node.scrollHeight > node.clientHeight
    ) {
      return node;
    }
    node = node.parentElement;
  }
  return null;
}

// Rows are keyed by a synthetic uid, not array index: index keys would swap the
// values under the caret on reorder and break the animation.
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
  const prevRects = useRef(new Map<string, DOMRect>());
  const uidsRef = useRef(uids);
  uidsRef.current = uids;
  // Rects are re-recorded on every commit but only a reorder animates, or an
  // auto-growing textarea would make the rows below it wobble as you type.
  const reorderTick = useRef(0);
  const animatedTick = useRef(0);

  // Resync when the count changes from outside (revert, switching lists, import).
  useEffect(() => {
    setUids((prev) =>
      prev.length === items.length
        ? prev
        : items.map((_, i) => prev[i] ?? nextUid()),
    );
  }, [items.length]);

  // FLIP: snap each moved row back to where it was, then release, so the browser
  // animates one transform per row rather than animating layout.
  useLayoutEffect(() => {
    const next = new Map<string, DOMRect>();
    rowRefs.current.forEach((el, uid) => next.set(uid, el.getBoundingClientRect()));

    if (reorderTick.current !== animatedTick.current) {
      animatedTick.current = reorderTick.current;
      next.forEach((rect, uid) => {
        const old = prevRects.current.get(uid);
        const el = rowRefs.current.get(uid);
        if (!old || !el) return;
        const dy = old.top - rect.top;
        if (Math.abs(dy) < 1) return;
        el.style.transition = "none";
        el.style.transform = `translateY(${dy}px)`;
        requestAnimationFrame(() => {
          el.style.transition = "transform 180ms cubic-bezier(0.2, 0, 0, 1)";
          el.style.transform = "";
        });
      });
    }

    prevRects.current = next;
  }, [uids, items]);

  // Refs so the drag listener does not resubscribe on every keystroke.
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const applyOrder = useCallback((from: number, to: number) => {
    if (from === to) return;
    reorderTick.current += 1;
    onChangeRef.current(move(itemsRef.current, from, to));
    setUids((prev) => move(prev, from, to));
  }, []);

  const handlePointerDown = useCallback(
    (uid: string, e: React.PointerEvent<HTMLButtonElement>) => {
      // Primary press only: a right/middle drag reports buttons 2 or 4, which
      // the zero-buttons release check below cannot end.
      if (e.button !== 0 || !e.isPrimary) return;
      // isPrimary is per pointer type, so a mouse press can pass it mid-touch-drag.
      if (pointerIdRef.current !== null) return;
      pointerIdRef.current = e.pointerId;
      // No setPointerCapture: reordering moves the row's DOM node, and detaching
      // releases capture, killing the drag one row in. The window listener below
      // outlives the move instead.
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
    const scroller = findScrollParent(container);
    let raf = 0;

    const evaluate = () => {
      const order = uidsRef.current;
      const from = order.indexOf(draggingUid);
      if (from < 0) return;

      // Layout offsets, not client rects: mid-FLIP a client rect still reports
      // the pre-animation position. offsetTop/offsetHeight ignore transforms.
      const localY = pointerYRef.current - container.getBoundingClientRect().top;
      let to = order.length - 1;
      for (let i = 0; i < order.length; i++) {
        const el = rowRefs.current.get(order[i]);
        if (!el) continue;
        if (localY <= el.offsetTop + el.offsetHeight) {
          to = i;
          break;
        }
      }
      if (to !== from) applyOrder(from, to);
    };

    // Holding near an edge scrolls and re-runs the hit-test; pointerdown
    // suppresses the browser's own gesture, so nothing else would scroll.
    const tick = () => {
      if (scroller) {
        const rect = scroller.getBoundingClientRect();
        const delta = autoscrollDelta(pointerYRef.current, rect.top, rect.bottom);
        if (delta !== 0) {
          const limit = scroller.scrollHeight - scroller.clientHeight;
          const before = scroller.scrollTop;
          scroller.scrollTop = Math.max(0, Math.min(limit, before + delta));
          if (scroller.scrollTop !== before) evaluate();
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    const endDrag = () => {
      pointerIdRef.current = null;
      setDraggingUid(null);
    };

    // Window listeners see every pointer; a second finger would otherwise
    // reorder with its own clientY and its release would end this drag.
    const isDragPointer = (e: PointerEvent) =>
      pointerIdRef.current === null || e.pointerId === pointerIdRef.current;

    const onMove = (e: PointerEvent) => {
      if (!isDragPointer(e)) return;
      // Releasing outside the window delivers no pointerup, so treat the first
      // move with no buttons held as the release we missed.
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
    // Release entirely outside the page, where no move follows to catch it.
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
        // The drag tracks the pointer window-wide; without this it paints a
        // text selection behind itself.
        draggingUid && "select-none",
      )}
    >
      {items.map((item, i) => {
        const uid = uids[i] ?? `fallback-${i}`;
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
