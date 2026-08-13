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

// Textarea that tracks its own content height. Prompts in a list run from one
// line to a paragraph, and a fixed `rows` either wastes space or hides text
// behind an inner scrollbar, so the element is measured on every change.
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

  // Layout effect, not a plain effect: resizing after paint shows a one-frame
  // flash of the wrong height while typing.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    const next = Math.min(el.scrollHeight, maxHeight);
    el.style.height = `${next}px`;
    el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [value, maxHeight]);

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

// Nearest ancestor that actually scrolls, so a drag can pull the list along with
// it. In the storage dialog this is the detail pane's overflow-y-auto wrapper.
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


// Reorderable list of prompt texts. Rows are keyed by a synthetic uid rather
// than by array index so React keeps each textarea (and its DOM node) attached
// to its text through a reorder. Index keys would swap the values under the
// caret and break the reorder animation.
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
  const rowRefs = useRef(new Map<string, HTMLDivElement>());
  const prevRects = useRef(new Map<string, DOMRect>());
  const uidsRef = useRef(uids);
  uidsRef.current = uids;
  // Bumped only by a reorder. Rows also move when a textarea auto-grows, and
  // animating those would turn ordinary typing into a wobbling list, so the
  // rects are always re-recorded but only a reorder actually animates.
  const reorderTick = useRef(0);
  const animatedTick = useRef(0);

  // Resync when the item count changes from outside this component (revert,
  // switching to another list, an import landing underneath us).
  useEffect(() => {
    setUids((prev) =>
      prev.length === items.length
        ? prev
        : items.map((_, i) => prev[i] ?? nextUid()),
    );
  }, [items.length]);

  // FLIP: rows are measured after every commit, and any row that moved is
  // snapped back to where it was and then released, so the browser animates
  // one transform per row instead of us animating layout.
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

  // Read through refs so the drag listener below never has to resubscribe just
  // because the text of an item changed.
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
      // Deliberately no setPointerCapture here. Reordering makes React move the
      // row's DOM node (insertBefore detaches and reattaches it), and detaching
      // a node implicitly releases its pointer capture -- so a captured grip
      // stops receiving pointermove after the very first swap and the drag dies
      // one row in. The window listener in the effect below outlives the move.
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

      // Hit-test against layout offsets rather than client rects: the FLIP
      // animation leaves a translateY on every row it moved, and a client rect
      // reports that transformed position, so mid-flight the rows would claim
      // to still be where they were. offsetTop/offsetHeight ignore transforms.
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

    // Holding near an edge scrolls the pane and re-runs the hit-test, so a list
    // taller than the pane can be reordered end to end. pointerdown suppresses
    // the browser's own drag gesture, so without this the reachable range is
    // whatever happens to be on screen -- which matters most on touch.
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

    const onMove = (e: PointerEvent) => {
      pointerYRef.current = e.clientY;
      evaluate();
    };
    const onUp = () => setDraggingUid(null);

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
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
    // `relative` makes this the offsetParent, so the row offsetTop values the
    // drag hit-test reads are measured against this container.
    <div
      ref={containerRef}
      className={cn(
        "relative flex flex-col gap-2",
        // The drag now tracks the pointer across the whole window, so kill text
        // selection for its duration or dragging paints a selection behind it.
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
