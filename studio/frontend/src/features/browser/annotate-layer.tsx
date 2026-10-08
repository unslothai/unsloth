// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  StudioDictationAdapter,
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
} from "@/features/chat";
import { useT } from "@/i18n";
import { DragDropVerticalIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { MicIcon } from "@/lib/mic-icon";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  type PointerEvent as ReactPointerEvent,
  type RefObject,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";
import { annotateScript } from "./api";
import { canScreenshot } from "./screenshot-support";
import { screenshotPage } from "./capture";
import { type AnnotateEvent, type AnnotateRect, MAX_MARKS } from "./frame-message";
import { startNativeAnnotate } from "./native-annotate";
import { focusPanel, nativeViewBounds } from "./native-view";
import { type FrameCommand, frameRect, onFrameAnnotate, sendFrameCommand } from "./page-frame";
import { useBrowserPrefsStore } from "./prefs-store";
import { useBrowserStore } from "./store";

// The blocks a click marks whole. Anything else marks the nearest element that holds text itself.
const BLOCK =
  "p, li, h1, h2, h3, h4, h5, h6, pre, blockquote, td, th, dt, dd, figcaption, caption, img";
const NESTED_BLOCK = "ul, ol, table, pre, blockquote, div";
const MAX_QUOTE_CHARS = 280;
const PAD = 5;
const DRAG_THRESHOLD = 4;
const BUBBLE = 28;

/** A screenshot of the marked page when Settings asks for one; failures just skip it. */
async function annotationScreenshot(page: HTMLElement | null): Promise<File[]> {
  if (useBrowserPrefsStore.getState().annotationScreenshots !== "always" || !page || !canScreenshot()) return [];
  const { tabs, annotateTabId } = useBrowserStore.getState();
  const tab = tabs.find((candidate) => candidate.id === annotateTabId);
  if (!tab) return [];
  const blob = await screenshotPage(tab, page).catch(() => null);
  return blob ? [new File([blob], "Annotated page.png", { type: "image/png" })] : [];
}

type Annotation = {
  id: number;
  ranges: Range[];
  quote: string;
  frame?: Frame;
  request: string;
};
/** A drag's box, offset from its content so it scrolls with it, at the zoom it was drawn at. */
type Frame = { left: number; top: number; width: number; height: number; zoom: number };
type Pending = { id: number | null; ranges: Range[]; quote: string; frame?: Frame };
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

/** Effective CSS zoom of an element. */
function cssZoomOf(element: Element | null): number {
  if (!element) return 1;
  // Typed as always present, but older engines lack it.
  const reported: unknown = element.currentCSSZoom;
  if (typeof reported === "number") return reported;
  // Older engines: multiply each ancestor's own zoom.
  let zoom = 1;
  for (let el: Element | null = element; el; el = el.parentElement) {
    zoom *= Number.parseFloat(getComputedStyle(el).zoom) || 1;
  }
  return zoom;
}

/** Effective CSS zoom at the content. */
function zoomAt(ranges: Range[]): number {
  const range = ranges[0];
  if (!range) return 1;
  const node = range.startContainer;
  // A selected node (an image) can carry its own zoom; its range starts at the parent.
  const selected = node.childNodes[range.startOffset];
  const element =
    selected instanceof Element && range.endContainer === node && range.endOffset === range.startOffset + 1
      ? selected
      : node instanceof Element
        ? node
        : node.parentElement;
  return cssZoomOf(element);
}

/** A drag's box if any, else the box around the content. */
function markBoxOf(ranges: Range[], frame: Frame | undefined, origin: DOMRect): Box | null {
  const content = boxOf(ranges, origin);
  if (!content || !frame) return content;
  // Scale by the zoom change since the drag, as the content did.
  const k = zoomAt(ranges) / frame.zoom;
  return {
    left: content.left + PAD + frame.left * k,
    top: content.top + PAD + frame.top * k,
    width: frame.width * k,
    height: frame.height * k,
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

/** Request edits on a file or a Studio page: click marks a block, drag marks an area; Send posts all as one message. */
export function AnnotateLayer({
  page,
  fileName,
  url,
  zoom = 1,
}: { page: HTMLElement; fileName: string; url?: string; zoom?: number }) {
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
  const [sending, setSending] = useState(false);
  const [, setFrame] = useState(0);
  const { setAnnotating, sendAnnotations } = useBrowserStore.getState();

  const exit = () => setAnnotating(null);

  const committed = (): Annotation[] => {
    if (!pending) return items;
    const request = draft.trim();
    if (pending.id === null) {
      if (!request) return items;
      return [
        ...items,
        { id: nextId.current++, ranges: pending.ranges, quote: pending.quote, frame: pending.frame, request },
      ];
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

  const send = async () => {
    const outgoing = committed();
    if (outgoing.length === 0 || !sendAnnotations || sending) return;
    // One at a time: a second click while staging would add the annotations twice.
    setSending(true);
    const files = await annotationScreenshot(page);
    const sent = await sendAnnotations(
      { file: fileName, url, items: outgoing.map(({ quote, request }) => ({ quote, request })) },
      files,
    ).finally(() => setSending(false));
    if (sent) exit();
  };

  useEffect(() => {
    let frame = 0;
    let settle = 0;
    const redraw = () => {
      layerRef.current?.setAttribute("data-scrolling", "");
      window.clearTimeout(settle);
      settle = window.setTimeout(
        () => layerRef.current?.removeAttribute("data-scrolling"),
        150,
      );
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => setFrame((value) => value + 1));
    };
    // Reduced motion gives zoom a 0.01ms transition, so content settles when it ends.
    const zoomed = (event: TransitionEvent) => event.propertyName === "zoom" && redraw();
    page.addEventListener("scroll", redraw, { capture: true, passive: true });
    page.addEventListener("transitionend", zoomed, true);
    const resize = new ResizeObserver(redraw);
    resize.observe(page);
    return () => {
      cancelAnimationFrame(frame);
      window.clearTimeout(settle);
      page.removeEventListener("scroll", redraw, { capture: true });
      page.removeEventListener("transitionend", zoomed, true);
      resize.disconnect();
    };
  }, [page]);

  // A zoom resizes the content after this render measured it, and fires no scroll or resize.
  useEffect(() => {
    const frame = requestAnimationFrame(() => setFrame((value) => value + 1));
    return () => cancelAnimationFrame(frame);
  }, [zoom]);

  // Capture phase, so the page's own links and click handlers never see a marking press.
  useEffect(() => {
    const ownUi = (target: EventTarget | null) =>
      target instanceof Element && target.closest("[data-annotate-ui]");
    let press: { x: number; y: number; dragging: boolean } | null = null;
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
    const mark = (ranges: Range[] | null, area?: DOMRect) => {
      if (!ranges || ranges.length === 0) return;
      const image = ranges.some((range) =>
        range.cloneContents().querySelector("img"),
      );
      const quote =
        quoteOf(ranges) || (image ? t("browser.annotate.imageQuote") : "");
      if (!quote) return;
      // Keep a drag's box as drawn, not shrunk to its text.
      const content = area ? boxOf(ranges, new DOMRect()) : null;
      const frame = area && content
        ? {
            left: area.left - content.left - PAD,
            top: area.top - content.top - PAD,
            width: area.width,
            height: area.height,
            zoom: zoomAt(ranges),
          }
        : undefined;
      setPending({ id: null, ranges, quote, frame });
      setDraft("");
    };
    const onDown = (event: PointerEvent) => {
      if (event.button !== 0 || ownUi(event.target)) return;
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
      saveRef.current();
      if (dragging && rect) mark(blocksIn(page, rect), rect);
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
  if (hoverBox) lastHoverBox.current = hoverBox;
  const shownHover = hoverBox ?? lastHoverBox.current;
  const pendingBox = pending ? markBoxOf(pending.ranges, pending.frame, origin) : null;
  const count = items.length;
  // A first comment still being typed can go too: Send commits it.
  const canSend = count > 0 || (pending?.id === null && draft.trim() !== "");
  const shown = count + (pending?.id === null && draft.trim() !== "" ? 1 : 0);

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
      {items.map((item, index) => {
        const box = item.id === pending?.id ? null : markBoxOf(item.ranges, item.frame, origin);
        return box ? (
          <Mark
            key={item.id}
            box={box}
            number={index + 1}
            onOpen={() => {
              saveRef.current();
              setPending({
                id: item.id,
                ranges: item.ranges,
                quote: item.quote,
                frame: item.frame,
              });
              setDraft(item.request);
            }}
            label={item.request}
          />
        ) : null;
      })}
      {pending && pendingBox ? (
        <>
          <Mark box={pendingBox} label={draft} number={markNumber(items, pending.id)} />
          <CommentForm
            inputRef={inputRef}
            box={pendingBox}
            layer={origin}
            draft={draft}
            onDraft={setDraft}
            onSave={save}
          />
        </>
      ) : null}
      {shown > 0 ? (
        <AnnotateBar
          count={shown}
          canSend={canSend}
          sendDisabled={!sendAnnotations || sending}
          onSend={send}
          onExit={exit}
        />
      ) : null}
      <div ref={cursorRef} aria-hidden={true} data-shown="false" className={CURSOR} />
    </div>
  );
}

function Mark({
  box,
  label,
  number,
  onOpen,
}: { box: Box; label: string; number: number; onOpen?: () => void }) {
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
        className="pointer-events-auto absolute flex size-8 animate-in cursor-pointer items-center justify-center rounded-full rounded-bl-[4px] bg-primary text-ui-13 font-semibold tabular-nums text-primary-foreground shadow-md ring-2 ring-background transition-transform fade-in-0 zoom-in-50 duration-150 hover:scale-110 disabled:cursor-default"
        style={{ left: box.left + box.width - 14, top: box.top - 18 }}
      >
        {number}
      </button>
    </>
  );
}

const CURSOR =
  "absolute top-0 left-0 size-7 rounded-full rounded-bl-[4px] bg-primary shadow-md ring-2 ring-background transition-[opacity,scale] duration-100 data-[shown=false]:scale-50 data-[shown=false]:opacity-0 motion-reduce:transition-none";

function markNumber<Item extends { id: number }>(items: Item[], id: number | null): number {
  const index = id === null ? -1 : items.findIndex((item) => item.id === id);
  return index === -1 ? items.length + 1 : index + 1;
}

const SURFACE =
  "border border-border bg-background text-foreground shadow-[0_8px_28px_-6px_rgba(0,0,0,0.18)] dark:border-transparent dark:bg-neutral-800 dark:text-white dark:shadow-xl";

/** Voice typing into a comment, as the composer's microphone does. */
function useCommentDictation(draft: string, onDraft: (value: string) => void) {
  const t = useT();
  const session = useRef<ReturnType<StudioDictationAdapter["listen"]> | null>(null);
  const [listening, setListening] = useState(false);
  useEffect(() => () => session.current?.cancel(), []);
  const start = () => {
    if (!isStudioDictationAvailable()) {
      notifyStudioDictationUnavailable();
      return;
    }
    let base = draft;
    const joined = (text: string) =>
      base && text && !base.endsWith(" ") ? `${base} ${text}` : base + text;
    try {
      const next = new StudioDictationAdapter({ chatId: null }).listen();
      session.current = next;
      setListening(true);
      next.onSpeech((result) => {
        if (session.current !== next) return;
        const text = joined(result.transcript);
        if (result.isFinal !== false) base = text;
        onDraft(text);
      });
      const end = () => {
        if (session.current !== next) return;
        session.current = null;
        setListening(false);
      };
      next.onSpeechEnd(end);
      next.onEnd?.(end);
    } catch (error) {
      session.current = null;
      setListening(false);
      toast.error(t("browser.annotate.dictateFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    }
  };
  const stop = () => void session.current?.stop();
  return { listening, toggle: () => (listening ? stop() : start()) };
}

function CommentForm({
  inputRef,
  box,
  layer,
  draft,
  onDraft,
  onSave,
  placeholder,
}: {
  placeholder?: string;
  inputRef: RefObject<HTMLInputElement | null>;
  box: Box;
  layer: { width: number; height: number };
  draft: string;
  onDraft: (value: string) => void;
  onSave: () => void;
}) {
  const t = useT();
  const dictation = useCommentDictation(draft, onDraft);
  const width = Math.min(448, Math.max(240, layer.width - 24));
  const top =
    box.top + box.height + 8 + 48 > layer.height
      ? Math.max(8, box.top - 56)
      : box.top + box.height + 8;
  const left = Math.min(Math.max(12, box.left + box.width - width), layer.width - width - 12);
  const micLabel = t(dictation.listening ? "browser.annotate.stopDictating" : "browser.annotate.dictate");
  // The tick replaces the microphone unless it is listening, so dictation can always be stopped.
  const canSave = Boolean(draft.trim()) && !dictation.listening;
  return (
    <form
      data-annotate-ui=""
      data-annotate-chrome=""
      data-native-cover=""
      onSubmit={(event) => {
        event.preventDefault();
        onSave();
      }}
      className={cn(
        "pointer-events-auto absolute flex h-12 animate-in items-center gap-2 rounded-full pr-1.5 pl-5 fade-in-0 zoom-in-95 duration-150",
        SURFACE,
      )}
      style={{ top, left, width }}
    >
      <input
        ref={inputRef}
        value={draft}
        onChange={(event) => onDraft(event.target.value)}
        placeholder={placeholder ?? t("browser.annotate.placeholder")}
        aria-label={placeholder ?? t("browser.annotate.placeholder")}
        className="min-w-0 flex-1 bg-transparent text-ui-15 outline-none placeholder:text-muted-foreground dark:placeholder:text-neutral-400"
      />
      {canSave ? (
        <button
          type="submit"
          aria-label={t("browser.annotate.save")}
          title={t("browser.annotate.save")}
          className="flex size-9 shrink-0 animate-in cursor-pointer items-center justify-center rounded-full bg-primary text-primary-foreground transition-opacity fade-in-0 zoom-in-75 duration-150 hover:opacity-90"
        >
          <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-4.5" />
        </button>
      ) : (
        <button
          type="button"
          aria-label={micLabel}
          title={micLabel}
          aria-pressed={dictation.listening}
          onClick={() => {
            dictation.toggle();
            inputRef.current?.focus();
          }}
          className={cn(
            "flex size-9 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground dark:text-neutral-300 dark:hover:bg-neutral-700 dark:hover:text-white",
            dictation.listening &&
              "animate-pulse bg-primary text-primary-foreground hover:bg-primary hover:text-primary-foreground dark:bg-primary dark:text-primary-foreground dark:hover:bg-primary dark:hover:text-primary-foreground",
          )}
        >
          <MicIcon className="size-4.5" />
        </button>
      )}
    </form>
  );
}

function AnnotateBar({
  count,
  canSend,
  sendDisabled,
  onSend,
  onExit,
  besidePage = false,
}: {
  count: number;
  canSend: boolean;
  sendDisabled: boolean;
  onSend: () => void;
  onExit: () => void;
  /** Over a native view: the page ends above the bar, so it only slides sideways. */
  besidePage?: boolean;
}) {
  const t = useT();
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const barRef = useRef<HTMLDivElement | null>(null);
  const drag = useRef<{ x: number; y: number; id: number } | null>(null);
  // Captured, so the drag keeps going over a web page's frame.
  const startDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { x: event.clientX - offset.x, y: event.clientY - offset.y, id: event.pointerId };
  };
  const moveDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    const start = drag.current;
    const bar = barRef.current;
    const layer = bar?.parentElement;
    if (!start || start.id !== event.pointerId || !bar || !layer) return;
    const bounds = layer.getBoundingClientRect();
    const width = bar.offsetWidth;
    const height = bar.offsetHeight;
    const maxX = Math.max(0, (bounds.width - width) / 2 - 8);
    const minY = -(bounds.height - height - 20 - 8);
    setOffset({
      x: Math.min(maxX, Math.max(-maxX, event.clientX - start.x)),
      y: besidePage ? 0 : Math.min(12, Math.max(minY, event.clientY - start.y)),
    });
  };
  const endDrag = (event: ReactPointerEvent<HTMLButtonElement>) => {
    if (drag.current?.id === event.pointerId) drag.current = null;
  };
  return (
    <div
      ref={barRef}
      data-annotate-ui=""
      data-annotate-chrome=""
      data-native-inset={besidePage ? "" : undefined}
      className={cn(
        "pointer-events-auto absolute bottom-5 left-1/2 flex h-12 items-center gap-1 rounded-2xl pr-1.5 pl-2.5 text-ui-15",
        SURFACE,
      )}
      style={{
        transform: `translate(calc(-50% + ${offset.x}px), ${offset.y}px)`,
      }}
    >
      <button
        type="button"
        aria-label={t("browser.annotate.move")}
        onPointerDown={startDrag}
        onPointerMove={moveDrag}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        className="flex h-9 w-7 cursor-grab touch-none items-center justify-center text-muted-foreground active:cursor-grabbing dark:text-neutral-400"
      >
        <HugeiconsIcon icon={DragDropVerticalIcon} strokeWidth={2} className="size-4.5" />
      </button>
      <span className="whitespace-nowrap px-2">
        {t(count === 1 ? "browser.annotate.countOne" : "browser.annotate.countMany", { count })}
      </span>
      {canSend ? <span aria-hidden={true} className="mx-1 h-5 w-px bg-border dark:bg-neutral-600" /> : null}
      <button
        type="button"
        onClick={onExit}
        className="h-9 cursor-pointer whitespace-nowrap rounded-xl px-3 transition-colors hover:bg-muted dark:hover:bg-neutral-700"
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

function accentColor(): string {
  const probe = document.createElement("span");
  probe.style.color = "var(--primary)";
  document.body.append(probe);
  const value = getComputedStyle(probe).color;
  probe.remove();
  // Computed colours can come back as oklch(); a canvas reads any of them out as RGB.
  const context = document.createElement("canvas").getContext("2d");
  if (!context) return value;
  context.fillStyle = value;
  context.fillRect(0, 0, 1, 1);
  const [r, g, b] = context.getImageData(0, 0, 1, 1).data;
  return `rgb(${r}, ${g}, ${b})`;
}

/** Ask about a web page: the page's own script tracks the pointer and draws marks (`annotation` in routes/browser.py); this layer holds the comments and the bar.
 *  Framed pages are driven through their shell, native views through `startNativeAnnotate`. */
export function WebAnnotateLayer({
  tabId,
  title,
  url,
  page,
  native = false,
}: { tabId: string; title: string; url: string; page: HTMLElement | null; native?: boolean }) {
  const t = useT();
  const layerRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  // The tab this layer annotates while mounted; a code fetch finishing after it left sends nothing.
  const liveTab = useRef<string | null>(null);
  const [rects, setRects] = useState<ReadonlyMap<number, AnnotateRect | null>>(new Map());
  const [items, setItems] = useState<WebMark[]>([]);
  const [pending, setPending] = useState<WebPending | null>(null);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [, setLayout] = useState(0);
  const zoom = useBrowserStore((state) => state.tabs.find((tab) => tab.id === tabId)?.zoom ?? 1);
  const { setAnnotating, sendAnnotations } = useBrowserStore.getState();
  const nativeSend = useRef<((command: FrameCommand) => void) | null>(null);

  const exit = () => setAnnotating(null);
  const command = (next: FrameCommand) =>
    native ? nativeSend.current?.(next) : sendFrameCommand(tabId, next);
  const forget = (id: number) => command({ command: "annotateForget", id });
  // The page gets the annotate code only now (each document once; repeats are ignored there).
  const start = () =>
    void annotateScript().then(
      (code) => {
        if (liveTab.current !== tabId) return;
        command({ command: "annotateInstall", code });
        command({ command: "annotate", on: true, color: accentColor() });
      },
      () => liveTab.current === tabId && exit(),
    );

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

  const send = async () => {
    const outgoing = committed();
    if (outgoing.length === 0 || !sendAnnotations || sending) return;
    // One at a time: a second click while staging would add the annotations twice.
    setSending(true);
    // Keep the open comment and close its form, so a native view shows again for the screenshot.
    setItems(outgoing);
    setPending(null);
    setDraft("");
    const files = await annotationScreenshot(page);
    const sent = await sendAnnotations(
      { file: title || url, url, items: outgoing.map(({ quote, request }) => ({ quote, request })) },
      files,
    ).finally(() => setSending(false));
    if (sent) exit();
  };

  // A press in a native view keeps key focus there: move it back to the comment.
  const takeKeys = () => {
    if (native) void focusPanel(tabId).then(() => inputRef.current?.focus());
  };

  // Read on each report, so one always sees this render's state.
  const handle = (event: AnnotateEvent) => {
    switch (event.kind) {
      case "ready":
        // New document (a native view navigated in place): drop the old marks.
        setItems([]);
        setPending(null);
        setDraft("");
        setRects(new Map());
        start();
        break;
      case "up":
        save();
        break;
      case "escape":
        if (pending) discard();
        else exit();
        break;
      case "open": {
        const item = items.find((entry) => entry.id === event.id);
        if (!item) break;
        setPending({ id: item.id, quote: item.quote, saved: true });
        setDraft(item.request);
        takeKeys();
        break;
      }
      case "mark": {
        const quote =
          event.quote ||
          (event.image
            ? event.alt || t("browser.annotate.imageQuote")
            : event.area
              ? t("browser.annotate.areaQuote")
              : "");
        // The page can post marks itself: cap both counts, as it can also empty its rects report.
        const full = !rects.has(event.id) && (rects.size >= MAX_MARKS || items.length >= MAX_MARKS);
        if (!quote || full) {
          forget(event.id);
          break;
        }
        setRects((current) => new Map(current).set(event.id, event.rect));
        setPending({ id: event.id, quote, saved: false });
        setDraft("");
        takeKeys();
        break;
      }
      case "rects":
        setRects(new Map(event.rects));
        break;
    }
  };
  const handleRef = useRef(handle);
  handleRef.current = handle;
  const startRef = useRef(start);
  startRef.current = start;

  useEffect(() => {
    const listener = (event: AnnotateEvent) => handleRef.current(event);
    let stop: () => void;
    if (native) {
      // One poll can carry several reports: render between them so each sees the last one's state.
      const channel = startNativeAnnotate(tabId, (event) => flushSync(() => listener(event)));
      nativeSend.current = channel.send;
      stop = () => {
        nativeSend.current = null;
        channel.stop();
      };
    } else {
      const unlisten = onFrameAnnotate(tabId, listener);
      stop = () => {
        unlisten();
        sendFrameCommand(tabId, { command: "annotate", on: false });
      };
    }
    liveTab.current = tabId;
    // Now, for a page already loaded; a page still loading asks with "ready".
    startRef.current();
    return () => {
      liveTab.current = null;
      stop();
    };
  }, [tabId, native]);

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

  useEffect(() => {
    const numbers: Array<[number, number]> = items.map((item, index) => [item.id, index + 1]);
    if (pending && !pending.saved) numbers.push([pending.id, items.length + 1]);
    command({ command: "annotateNumbers", numbers });
  }, [tabId, items, pending]);

  const origin = layerRef.current?.getBoundingClientRect() ?? new DOMRect();
  // Native views report CSS pixels; scale by their zoom.
  const bounds = native ? nativeViewBounds(tabId) : null;
  const frame = bounds ? { left: bounds.x, top: bounds.y } : (frameRect(tabId) ?? origin);
  const scale = bounds ? zoom : 1;
  const rect = pending ? rects.get(pending.id) : null;
  const pendingBox: Box | null = rect
    ? {
        left: rect.left * scale + frame.left - origin.left,
        top: rect.top * scale + frame.top - origin.top,
        width: rect.width * scale,
        height: rect.height * scale,
      }
    : null;
  const count = items.length;
  const canSend = count > 0 || (pending?.saved === false && draft.trim() !== "");
  const shown = count + (pending?.saved === false && draft.trim() !== "" ? 1 : 0);

  return (
    <div ref={layerRef} className="pointer-events-none absolute inset-0 z-10 overflow-hidden">
      {native && pending ? (
        // The page is a snapshot while commenting; a press on it saves, as on a live page.
        <div
          data-annotate-ui=""
          data-native-cover=""
          aria-hidden={true}
          className="pointer-events-auto absolute inset-0"
          onPointerDown={save}
        />
      ) : null}
      {pending && pendingBox ? (
        <CommentForm
          key={pending.id}
          inputRef={inputRef}
          box={pendingBox}
          layer={origin}
          draft={draft}
          onDraft={setDraft}
          onSave={save}
          placeholder={t("browser.annotate.pagePlaceholder")}
        />
      ) : null}
      {shown > 0 ? (
        <AnnotateBar
          count={shown}
          canSend={canSend}
          sendDisabled={!sendAnnotations || sending}
          onSend={send}
          onExit={exit}
          besidePage={native}
        />
      ) : null}
    </div>
  );
}
