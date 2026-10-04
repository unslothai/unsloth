// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { DragDropVerticalIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ArrowUpIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import type { AnnotateEvent, AnnotateRect } from "./frame-message";
import { frameRect, onFrameAnnotate, sendFrameCommand } from "./page-frame";
import { useBrowserStore } from "./store";

// The blocks a click marks whole. Anything else marks the nearest element that holds text itself.
const BLOCK =
  "p, li, h1, h2, h3, h4, h5, h6, pre, blockquote, td, th, dt, dd, figcaption, caption, img";
// A list item marks its own line, not the lists nested under it.
const NESTED_BLOCK = "ul, ol, table, pre, blockquote, div";
const MAX_QUOTE_CHARS = 280;
const PAD = 5;
// A press that moves further than this draws an area instead of marking the block under it.
const DRAG_THRESHOLD = 4;
const BUBBLE = 28;

type Annotation = {
  id: number;
  ranges: Range[];
  quote: string;
  request: string;
};
type Pending = { id: number | null; ranges: Range[]; quote: string };
type Box = { left: number; top: number; width: number; height: number };

function hasOwnText(element: Element): boolean {
  return [...element.childNodes].some(
    (node) => node.nodeType === Node.TEXT_NODE && node.textContent?.trim(),
  );
}

function pdfLine(span: Element): Range[] {
  const layer = span.closest(".textLayer");
  if (!layer) return [];
  const target = span.getBoundingClientRect();
  const middle = target.top + target.height / 2;
  return [...layer.querySelectorAll("span")]
    .filter((candidate) => {
      if (candidate.querySelector("span") || !candidate.textContent?.trim())
        return false;
      const rect = candidate.getBoundingClientRect();
      return (
        Math.abs(rect.top + rect.height / 2 - middle) <
        Math.max(target.height, rect.height) * 0.6
      );
    })
    .map((candidate) => {
      const range = document.createRange();
      range.selectNodeContents(candidate);
      return range;
    });
}

/** A block's own content: an image whole, a list item up to the lists nested in it. */
function blockRange(block: Element): Range {
  const range = document.createRange();
  if (block.tagName === "IMG") {
    range.selectNode(block);
    return range;
  }
  range.selectNodeContents(block);
  if (block.tagName === "LI") {
    const nested = [...block.children].find((child) =>
      child.matches(NESTED_BLOCK),
    );
    if (nested) range.setEndBefore(nested);
  }
  return range;
}

/** What a click at `target` marks: its block, a list item's own line, or a PDF's text line. */
function blockAt(target: Element, root: Element): Range[] | null {
  if (target.closest(".textLayer")) {
    const line = pdfLine(target);
    return line.length > 0 ? line : null;
  }
  let block = target.closest(BLOCK);
  if (block && !root.contains(block)) block = null;
  if (!block) {
    let element: Element | null = target;
    while (element && element !== root && !hasOwnText(element))
      element = element.parentElement;
    if (!element || element === root) return null;
    block = element;
  }
  return [blockRange(block)];
}

const overlaps = (a: DOMRect, b: DOMRect) =>
  a.width > 0 &&
  a.height > 0 &&
  a.left < b.right &&
  a.right > b.left &&
  a.top < b.bottom &&
  a.bottom > b.top;

/** Everything a dragged area touches, block by block (a PDF by its text runs), in page order. */
function blocksIn(root: Element, area: DOMRect): Range[] {
  const candidates = [
    ...root.querySelectorAll(BLOCK),
    ...root.querySelectorAll(".textLayer span"),
  ].filter(
    (element) =>
      !element.closest("[data-annotate-ui]") &&
      (element.closest(".textLayer")
        ? !element.querySelector("span") && element.textContent?.trim()
        : true) &&
      // Cheap first: most of a long document is nowhere near the area.
      overlaps(element.getBoundingClientRect(), area),
  );
  const found = candidates.flatMap((element) => {
    const range = blockRange(element);
    return [...range.getClientRects()].some((rect) => overlaps(rect, area))
      ? [{ element, range }]
      : [];
  });
  // Drop blocks containing another hit (list items already stop before nested lists).
  return found
    .filter(
      ({ element, range }) =>
        !found.some(
          (other) =>
            other.element !== element &&
            element.contains(other.element) &&
            range.intersectsNode(other.element),
        ),
    )
    .map(({ range }) => range);
}

function quoteOf(ranges: Range[]): string {
  const text = ranges
    .map((range) => range.toString())
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
  return text.length > MAX_QUOTE_CHARS
    ? `${text.slice(0, MAX_QUOTE_CHARS - 1)}…`
    : text;
}

function boxOf(ranges: Range[], origin: DOMRect): Box | null {
  let left = Number.POSITIVE_INFINITY;
  let top = Number.POSITIVE_INFINITY;
  let right = Number.NEGATIVE_INFINITY;
  let bottom = Number.NEGATIVE_INFINITY;
  for (const range of ranges) {
    // A range whose nodes left the page (a virtualized page scrolled away) measures empty.
    for (const rect of range.getClientRects()) {
      if (rect.width === 0 && rect.height === 0) continue;
      left = Math.min(left, rect.left);
      top = Math.min(top, rect.top);
      right = Math.max(right, rect.right);
      bottom = Math.max(bottom, rect.bottom);
    }
  }
  if (!Number.isFinite(left)) return null;
  return {
    left: left - origin.left - PAD,
    top: top - origin.top - PAD,
    width: right - left + PAD * 2,
    height: bottom - top + PAD * 2,
  };
}

const sameRanges = (a: Range[] | null, b: Range[] | null) =>
  a !== null &&
  b !== null &&
  a.length === b.length &&
  a.every(
    (range, index) =>
      range.startContainer === b[index]?.startContainer &&
      range.startOffset === b[index]?.startOffset &&
      range.endContainer === b[index]?.endContainer &&
      range.endOffset === b[index]?.endOffset,
  );

/** Request edits on a file: click marks a block, drag marks an area, each takes a comment; Send
 *  posts all as one chat message. */
export function AnnotateLayer({
  page,
  fileName,
}: { page: HTMLElement; fileName: string }) {
  const t = useT();
  const layerRef = useRef<HTMLDivElement | null>(null);
  const cursorRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const lastHoverBox = useRef<Box | null>(null);
  const [area, setArea] = useState<Box | null>(null);
  const nextId = useRef(1);
  const [hover, setHover] = useState<Range[] | null>(null);
  const [items, setItems] = useState<Annotation[]>([]);
  const [pending, setPending] = useState<Pending | null>(null);
  const [draft, setDraft] = useState("");
  const [, setFrame] = useState(0);
  const { setAnnotating, sendAnnotations } = useBrowserStore.getState();

  const exit = () => setAnnotating(null);

  /** The annotations with the open comment applied; an emptied comment removes its mark. */
  const committed = (): Annotation[] => {
    if (!pending) return items;
    const request = draft.trim();
    if (pending.id === null) {
      if (!request) return items;
      return [...items, { id: nextId.current++, ranges: pending.ranges, quote: pending.quote, request }];
    }
    return request
      ? items.map((item) => (item.id === pending.id ? { ...item, request } : item))
      : items.filter((item) => item.id !== pending.id);
  };

  const save = () => {
    if (!pending) return;
    setItems(committed());
    setPending(null);
    setDraft("");
  };
  const saveRef = useRef(save);
  saveRef.current = save;

  const send = () => {
    const outgoing = committed();
    if (outgoing.length === 0) return;
    sendAnnotations?.({
      file: fileName,
      items: outgoing.map(({ quote, request }) => ({ quote, request })),
    });
    exit();
  };

  useEffect(() => {
    let frame = 0;
    let settle = 0;
    const redraw = () => {
      // Outlines glide between blocks, but track a scrolling page exactly.
      layerRef.current?.setAttribute("data-scrolling", "");
      window.clearTimeout(settle);
      settle = window.setTimeout(
        () => layerRef.current?.removeAttribute("data-scrolling"),
        150,
      );
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => setFrame((value) => value + 1));
    };
    page.addEventListener("scroll", redraw, { capture: true, passive: true });
    const resize = new ResizeObserver(redraw);
    resize.observe(page);
    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(settle);
      page.removeEventListener("scroll", redraw, { capture: true });
      resize.disconnect();
    };
  }, [page]);

  // Capture phase, so the page's own links and click handlers never see a marking press.
  useEffect(() => {
    const ownUi = (target: EventTarget | null) =>
      target instanceof Element && target.closest("[data-annotate-ui]");
    let press: { x: number; y: number; dragging: boolean } | null = null;
    // The bubble is the pointer: moved directly, so it never waits on a render.
    const moveBubble = (event: PointerEvent | null) => {
      const bubble = cursorRef.current;
      const layer = layerRef.current;
      if (!bubble || !layer) return;
      if (!event || ownUi(event.target)) {
        bubble.dataset.shown = "false";
        return;
      }
      const origin = layer.getBoundingClientRect();
      bubble.style.transform = `translate(${event.clientX - origin.left}px, ${event.clientY - origin.top - BUBBLE}px)`;
      bubble.dataset.shown = "true";
    };
    const areaFrom = (event: PointerEvent): DOMRect | null => {
      if (!press) return null;
      return new DOMRect(
        Math.min(press.x, event.clientX),
        Math.min(press.y, event.clientY),
        Math.abs(event.clientX - press.x),
        Math.abs(event.clientY - press.y),
      );
    };
    const mark = (ranges: Range[] | null) => {
      if (!ranges || ranges.length === 0) return;
      const image = ranges.some((range) =>
        range.cloneContents().querySelector("img"),
      );
      const quote =
        quoteOf(ranges) || (image ? t("browser.annotate.imageQuote") : "");
      if (!quote) return;
      setPending({ id: null, ranges, quote });
      setDraft("");
    };
    const onDown = (event: PointerEvent) => {
      if (event.button !== 0 || ownUi(event.target)) return;
      // No text selection or native drag: a press here marks.
      event.preventDefault();
      press = { x: event.clientX, y: event.clientY, dragging: false };
    };
    const handleMove = (event: PointerEvent) => {
      moveBubble(event);
      if (press) {
        if (
          !press.dragging &&
          Math.hypot(event.clientX - press.x, event.clientY - press.y) <
            DRAG_THRESHOLD
        )
          return;
        press.dragging = true;
        const rect = areaFrom(event);
        const origin = layerRef.current?.getBoundingClientRect();
        if (rect && origin) {
          setHover(null);
          setArea({
            left: rect.left - origin.left,
            top: rect.top - origin.top,
            width: rect.width,
            height: rect.height,
          });
        }
        return;
      }
      if (ownUi(event.target) || !(event.target instanceof Element)) return;
      const next = blockAt(event.target, page);
      setHover((current) => (sameRanges(current, next) ? current : next));
    };
    let moveFrame = 0;
    let lastMove: PointerEvent | null = null;
    const onMove = (event: PointerEvent) => {
      lastMove = event;
      if (moveFrame) return;
      moveFrame = requestAnimationFrame(() => {
        moveFrame = 0;
        const latest = lastMove;
        lastMove = null;
        if (latest) handleMove(latest);
      });
    };
    const cancelMove = () => {
      cancelAnimationFrame(moveFrame);
      moveFrame = 0;
      lastMove = null;
    };
    const onUp = (event: PointerEvent) => {
      cancelMove();
      if (!press) return;
      const { dragging } = press;
      const rect = areaFrom(event);
      press = null;
      setArea(null);
      // An unsaved comment is kept when moving on.
      saveRef.current();
      if (dragging && rect) mark(blocksIn(page, rect));
      else if (event.target instanceof Element && !ownUi(event.target))
        mark(blockAt(event.target, page));
    };
    const onLeave = () => {
      cancelMove();
      setHover(null);
      moveBubble(null);
    };
    const onClick = (event: MouseEvent) => {
      if (ownUi(event.target)) return;
      event.preventDefault();
      event.stopPropagation();
    };
    page.classList.add("browser-annotating");
    page.addEventListener("pointerdown", onDown, true);
    page.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
    page.addEventListener("pointerleave", onLeave);
    page.addEventListener("click", onClick, true);
    return () => {
      cancelMove();
      page.classList.remove("browser-annotating");
      page.removeEventListener("pointerdown", onDown, true);
      page.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      page.removeEventListener("pointerleave", onLeave);
      page.removeEventListener("click", onClick, true);
    };
  }, [page, t]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      if (pending) {
        setPending(null);
        setDraft("");
      } else exit();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  useLayoutEffect(() => {
    if (pending) inputRef.current?.focus();
  }, [pending]);

  const origin = layerRef.current?.getBoundingClientRect() ?? new DOMRect();
  const hoverBox = hover ? boxOf(hover, origin) : null;
  // Kept after the pointer leaves, so the outline fades where it was rather than jumping away.
  if (hoverBox) lastHoverBox.current = hoverBox;
  const shownHover = hoverBox ?? lastHoverBox.current;
  const pendingBox = pending ? boxOf(pending.ranges, origin) : null;
  const count = items.length;
  // A first comment still being typed can go too: Send commits it.
  const canSend = count > 0 || (pending?.id === null && draft.trim() !== "");

  return (
    <div
      ref={layerRef}
      className="group/annotate pointer-events-none absolute inset-0 z-10 overflow-hidden"
    >
      {shownHover ? (
        <div
          className={cn(
            "absolute rounded-md border-[1.5px] border-primary/70 transition-[left,top,width,height,opacity] duration-150 ease-out group-data-[scrolling]/annotate:transition-none motion-reduce:transition-none",
            hoverBox && !area ? "opacity-100" : "opacity-0",
          )}
          style={shownHover}
        />
      ) : null}
      {area ? (
        <div
          className="absolute rounded-md border-[1.5px] border-dashed border-primary/70 bg-primary/5"
          style={area}
        />
      ) : null}
      {items.map((item) => {
        const box = item.id === pending?.id ? null : boxOf(item.ranges, origin);
        return box ? (
          <Mark
            key={item.id}
            box={box}
            onOpen={() => {
              saveRef.current();
              setPending({
                id: item.id,
                ranges: item.ranges,
                quote: item.quote,
              });
              setDraft(item.request);
            }}
            label={item.request}
          />
        ) : null;
      })}
      {pending && pendingBox ? (
        <>
          <Mark box={pendingBox} label={draft} />
          <CommentForm
            inputRef={inputRef}
            box={pendingBox}
            layer={origin}
            draft={draft}
            onDraft={setDraft}
            onSave={save}
            canSave={Boolean(draft.trim()) || pending.id !== null}
          />
        </>
      ) : null}
      <AnnotateBar
        count={count}
        canSend={canSend}
        sendDisabled={!sendAnnotations}
        onSend={send}
        onExit={exit}
      />
      <div ref={cursorRef} aria-hidden={true} data-shown="false" className={CURSOR} />
    </div>
  );
}

function Mark({
  box,
  label,
  onOpen,
}: { box: Box; label: string; onOpen?: () => void }) {
  return (
    <>
      <div
        className="absolute animate-in rounded-md border-[1.5px] border-dashed border-primary/60 bg-primary/5 fade-in-0 duration-150"
        style={box}
      />
      <button
        type="button"
        data-annotate-ui=""
        aria-label={label}
        title={label}
        onClick={onOpen}
        disabled={!onOpen}
        className="pointer-events-auto absolute size-8 animate-in cursor-pointer rounded-full rounded-bl-[4px] bg-primary shadow-md ring-2 ring-background transition-transform fade-in-0 zoom-in-50 duration-150 hover:scale-110 disabled:cursor-default"
        style={{ left: box.left + box.width - 14, top: box.top - 18 }}
      />
    </>
  );
}

// The pointer while annotating: a comment bubble, its tail at the point.
const CURSOR =
  "absolute top-0 left-0 size-7 rounded-full rounded-bl-[4px] bg-primary shadow-md ring-2 ring-background transition-[opacity,scale] duration-100 data-[shown=false]:scale-50 data-[shown=false]:opacity-0 motion-reduce:transition-none";

/** The comment field for a mark: below it, or above when there is no room under it. */
function CommentForm({
  inputRef,
  box,
  layer,
  draft,
  onDraft,
  onSave,
  canSave,
  placeholder,
}: {
  placeholder?: string;
  inputRef: RefObject<HTMLInputElement | null>;
  box: Box;
  layer: { width: number; height: number };
  draft: string;
  onDraft: (value: string) => void;
  onSave: () => void;
  canSave: boolean;
}) {
  const t = useT();
  const width = Math.min(448, Math.max(240, layer.width - 24));
  const top =
    box.top + box.height + 8 + 48 > layer.height
      ? Math.max(8, box.top - 56)
      : box.top + box.height + 8;
  const left = Math.min(Math.max(12, box.left + box.width - width), layer.width - width - 12);
  return (
    <form
      data-annotate-ui=""
      onSubmit={(event) => {
        event.preventDefault();
        onSave();
      }}
      className="pointer-events-auto absolute flex h-12 animate-in items-center gap-2 rounded-full bg-neutral-800 pr-1.5 pl-5 text-white shadow-xl fade-in-0 zoom-in-95 duration-150"
      style={{ top, left, width }}
    >
      <input
        ref={inputRef}
        value={draft}
        onChange={(event) => onDraft(event.target.value)}
        placeholder={placeholder ?? t("browser.annotate.placeholder")}
        aria-label={placeholder ?? t("browser.annotate.placeholder")}
        className="min-w-0 flex-1 bg-transparent text-ui-15 outline-none placeholder:text-neutral-400"
      />
      <button
        type="submit"
        aria-label={t("browser.annotate.save")}
        disabled={!canSave}
        // The chat composer's own send button.
        className="flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-full bg-primary text-primary-foreground transition-[background-color,opacity] hover:bg-primary/80 disabled:cursor-default disabled:opacity-40 disabled:hover:bg-primary"
      >
        <ArrowUpIcon className="size-[calc(21px*var(--ui-space-scale,1))] stroke-2" />
      </button>
    </form>
  );
}

/** The bar along the bottom: how many marks, Cancel, Send. Drag its handle to move it. */
function AnnotateBar({
  count,
  canSend,
  sendDisabled,
  onSend,
  onExit,
  hint,
}: {
  hint?: string;
  count: number;
  canSend: boolean;
  sendDisabled: boolean;
  onSend: () => void;
  onExit: () => void;
}) {
  const t = useT();
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragBar = (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    const start = { x: event.clientX - offset.x, y: event.clientY - offset.y };
    const move = (moveEvent: PointerEvent) =>
      setOffset({
        x: moveEvent.clientX - start.x,
        y: moveEvent.clientY - start.y,
      });
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  return (
    <div
      data-annotate-ui=""
      className="pointer-events-auto absolute bottom-5 left-1/2 flex h-12 items-center gap-1 rounded-2xl bg-neutral-800 pr-1.5 pl-1 text-ui-15 text-white shadow-xl"
      style={{
        transform: `translate(calc(-50% + ${offset.x}px), ${offset.y}px)`,
      }}
    >
      <button
        type="button"
        aria-label={t("browser.annotate.move")}
        onPointerDown={dragBar}
        className="flex h-9 w-7 cursor-grab touch-none items-center justify-center text-neutral-400 active:cursor-grabbing"
      >
        <HugeiconsIcon icon={DragDropVerticalIcon} strokeWidth={2} className="size-4.5" />
      </button>
      <span className="whitespace-nowrap px-2">
        {count === 0
          ? (hint ?? t("browser.annotate.hint"))
          : t(count === 1 ? "browser.annotate.countOne" : "browser.annotate.countMany", { count })}
      </span>
      {canSend ? <span aria-hidden={true} className="mx-1 h-5 w-px bg-neutral-600" /> : null}
      <button
        type="button"
        onClick={onExit}
        className="h-9 cursor-pointer whitespace-nowrap rounded-xl px-3 transition-colors hover:bg-neutral-700"
      >
        {t("browser.annotate.cancel")}
      </button>
      {canSend ? (
        <button
          type="button"
          onClick={onSend}
          disabled={sendDisabled}
          className="h-9 cursor-pointer whitespace-nowrap rounded-xl bg-primary px-4 font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-default disabled:opacity-50"
        >
          {t("browser.annotate.send")}
        </button>
      ) : null}
    </div>
  );
}

type WebMark = { id: number; quote: string; request: string };
type WebPending = { id: number; quote: string; saved: boolean };

/**
 * Ask about a web page: the same marks and comments as a file's, drawn over the page's frame. The
 * page is another origin, so its own script finds what is under the pointer (see `annotation` in
 * routes/browser.py) and reports boxes in its viewport; this layer moves them onto the frame.
 */
export function WebAnnotateLayer({
  tabId,
  title,
  url,
}: { tabId: string; title: string; url: string }) {
  const t = useT();
  const layerRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const lastHover = useRef<AnnotateRect | null>(null);
  const settleRef = useRef(0);
  const [hover, setHover] = useState<AnnotateRect | null>(null);
  const [area, setArea] = useState<AnnotateRect | null>(null);
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(null);
  const [rects, setRects] = useState<ReadonlyMap<number, AnnotateRect | null>>(new Map());
  const [items, setItems] = useState<WebMark[]>([]);
  const [pending, setPending] = useState<WebPending | null>(null);
  const [draft, setDraft] = useState("");
  const [, setLayout] = useState(0);
  const { setAnnotating, sendAnnotations } = useBrowserStore.getState();

  const exit = () => setAnnotating(null);
  const forget = (id: number) => sendFrameCommand(tabId, { command: "annotateForget", id });

  /** The marks with the open comment applied; an emptied comment removes its mark. */
  const committed = (): WebMark[] => {
    if (!pending) return items;
    const request = draft.trim();
    if (!pending.saved) {
      if (!request) return items;
      return [...items, { id: pending.id, quote: pending.quote, request }];
    }
    return request
      ? items.map((item) => (item.id === pending.id ? { ...item, request } : item))
      : items.filter((item) => item.id !== pending.id);
  };

  const save = () => {
    if (!pending) return;
    const next = committed();
    if (!next.some((item) => item.id === pending.id)) forget(pending.id);
    setItems(next);
    setPending(null);
    setDraft("");
  };

  const discard = () => {
    if (pending && !pending.saved) forget(pending.id);
    setPending(null);
    setDraft("");
  };

  const send = () => {
    const outgoing = committed();
    if (outgoing.length === 0) return;
    sendAnnotations?.({
      file: title || url,
      url,
      items: outgoing.map(({ quote, request }) => ({ quote, request })),
    });
    exit();
  };

  // Read on each report, so one always sees this render's state.
  const handle = (event: AnnotateEvent) => {
    switch (event.kind) {
      case "ready":
        sendFrameCommand(tabId, { command: "annotate", on: true });
        break;
      case "up":
        save();
        break;
      case "escape":
        if (pending) discard();
        else exit();
        break;
      case "pointer":
        setPointer(event.point);
        break;
      case "hover":
        setHover(event.rect);
        break;
      case "area":
        setArea(event.rect);
        break;
      case "mark": {
        const quote =
          event.quote || (event.image ? event.alt || t("browser.annotate.imageQuote") : "");
        if (!quote) {
          forget(event.id);
          break;
        }
        setRects((current) => new Map(current).set(event.id, event.rect));
        setPending({ id: event.id, quote, saved: false });
        setDraft("");
        break;
      }
      case "rects": {
        // Outlines glide between blocks, but track a scrolling page exactly.
        const layer = layerRef.current;
        layer?.setAttribute("data-scrolling", "");
        window.clearTimeout(settleRef.current);
        settleRef.current = window.setTimeout(() => layer?.removeAttribute("data-scrolling"), 150);
        setRects(new Map(event.rects));
        setHover(event.hover);
        break;
      }
    }
  };
  const handleRef = useRef(handle);
  handleRef.current = handle;

  useEffect(() => {
    const stop = onFrameAnnotate(tabId, (event) => handleRef.current(event));
    // Now, for a page already loaded; a page still loading asks with "ready".
    sendFrameCommand(tabId, { command: "annotate", on: true });
    return () => {
      stop();
      sendFrameCommand(tabId, { command: "annotate", on: false });
    };
  }, [tabId]);

  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    const observer = new ResizeObserver(() => setLayout((value) => value + 1));
    observer.observe(layer);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      if (pending) discard();
      else exit();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  useLayoutEffect(() => {
    if (pending) inputRef.current?.focus();
  }, [pending]);

  const origin = layerRef.current?.getBoundingClientRect() ?? new DOMRect();
  const frame = frameRect(tabId) ?? origin;
  const dx = frame.left - origin.left;
  const dy = frame.top - origin.top;
  const onLayer = (rect: AnnotateRect | null | undefined): Box | null =>
    rect ? { left: rect.left + dx, top: rect.top + dy, width: rect.width, height: rect.height } : null;
  const hoverBox = onLayer(hover);
  if (hoverBox) lastHover.current = hover;
  const shownHover = hoverBox ?? onLayer(lastHover.current);
  const pendingBox = pending ? onLayer(rects.get(pending.id)) : null;
  const count = items.length;
  const canSend = count > 0 || (pending?.saved === false && draft.trim() !== "");

  return (
    <div
      ref={layerRef}
      className="group/annotate pointer-events-none absolute inset-0 z-10 overflow-hidden"
    >
      {shownHover ? (
        <div
          className={cn(
            "absolute rounded-md border-[1.5px] border-primary/70 transition-[left,top,width,height,opacity] duration-150 ease-out group-data-[scrolling]/annotate:transition-none motion-reduce:transition-none",
            hoverBox && !area ? "opacity-100" : "opacity-0",
          )}
          style={shownHover}
        />
      ) : null}
      {area ? (
        <div
          className="absolute rounded-md border-[1.5px] border-dashed border-primary/70 bg-primary/5"
          style={onLayer(area) ?? undefined}
        />
      ) : null}
      {items.map((item) => {
        const box = item.id === pending?.id ? null : onLayer(rects.get(item.id));
        return box ? (
          <Mark
            key={item.id}
            box={box}
            label={item.request}
            onOpen={() => {
              save();
              setPending({ id: item.id, quote: item.quote, saved: true });
              setDraft(item.request);
            }}
          />
        ) : null;
      })}
      {pending && pendingBox ? (
        <>
          <Mark box={pendingBox} label={draft} />
          <CommentForm
            inputRef={inputRef}
            box={pendingBox}
            layer={origin}
            draft={draft}
            onDraft={setDraft}
            onSave={save}
            canSave={Boolean(draft.trim()) || pending.saved}
            placeholder={t("browser.annotate.pagePlaceholder")}
          />
        </>
      ) : null}
      <AnnotateBar
        count={count}
        canSend={canSend}
        sendDisabled={!sendAnnotations}
        onSend={send}
        onExit={exit}
        hint={t("browser.annotate.pageHint")}
      />
      <div
        aria-hidden={true}
        data-shown={pointer ? "true" : "false"}
        className={CURSOR}
        style={
          pointer
            ? { transform: `translate(${pointer.x + dx}px, ${pointer.y + dy - BUBBLE}px)` }
            : undefined
        }
      />
    </div>
  );
}
