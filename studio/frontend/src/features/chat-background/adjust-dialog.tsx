import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Slider } from "@/components/ui/slider";
import { useT } from "@/i18n";
// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { MinusSignIcon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type CSSProperties,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { WallpaperArtwork } from "./artwork";
import { dragPosition, imageBox, zoomPosition } from "./artwork-geometry";
import { type ChatBackgroundValue, parsePosition } from "./model";
import { patchBackground } from "./storage";

const MIN_ZOOM = 100;
const MAX_ZOOM = 300;
const ZOOM_STEP = 10;
const NUDGE = 8;

type Draft = { x: number; y: number; zoom: number };

function draftOf(value: ChatBackgroundValue): Draft {
  const [x, y] = parsePosition(value.position) ?? [50, 50];
  return { x: x / 100, y: y / 100, zoom: value.zoom };
}

/**
 * The page area the wallpaper paints, in CSS px. The preview renders this whole area and scales it
 * down, so every sizing (Original included) and any window shape frames what the page will show.
 */
function pageArea(fade: boolean): { w: number; h: number } {
  const w = Math.max(1, window.innerWidth);
  const h = fade
    ? Math.min(window.innerHeight * 0.72, 760)
    : window.innerHeight;
  return { w, h: Math.max(1, h) };
}

/**
 * Frames the wallpaper: drag to reposition, scroll or pinch to zoom
 * around the pointer, a Reset / zoom row, and Cancel / Apply. Edits stay a draft until Apply.
 */
export function WallpaperAdjustDialog({
  open,
  onOpenChange,
  value,
  fade,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  value: ChatBackgroundValue;
  fade: boolean;
}) {
  const t = useT();
  const [draft, setDraft] = useState<Draft>(() => draftOf(value));
  const [dragging, setDragging] = useState(false);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [box, setBox] = useState<{ w: number; h: number } | null>(null);
  // State, not a ref: the preview mounts in the dialog's portal after `open` flips, so effects keyed
  // on `open` would find no node and never measure it or attach the wheel and pinch listeners.
  const [preview, setPreview] = useState<HTMLDivElement | null>(null);
  const drag = useRef<{ id: number; x: number; y: number; from: Draft } | null>(
    null,
  );
  const pinch = useRef<number | null>(null);
  const url = useMemo(() => URL.createObjectURL(value.blob), [value.blob]);
  const area = useMemo(
    () => (open ? pageArea(fade) : { w: 1600, h: 900 }),
    [open, fade],
  );
  const aspect = area.w / area.h;

  useEffect(() => () => URL.revokeObjectURL(url), [url]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-seed only on open; the stored value changes under an open draft only by Apply.
  useEffect(() => {
    if (open) {
      setDraft(draftOf(value));
    }
  }, [open]);
  useEffect(() => {
    const image = new Image();
    image.onload = () =>
      setNatural({ w: image.naturalWidth, h: image.naturalHeight });
    image.src = url;
  }, [url]);
  useEffect(() => {
    const node = preview;
    if (!node) {
      return;
    }
    const read = () => setBox({ w: node.clientWidth, h: node.clientHeight });
    read();
    const observer = new ResizeObserver(read);
    observer.observe(node);
    return () => observer.disconnect();
  }, [preview]);

  // Preview px per page px; geometry runs in page px, where Apply's values live.
  const scale = box ? box.w / area.w : 1;
  // The drawn size before zoom, in page px: imageBox ignores position for size.
  const drawn = natural
    ? imageBox(area.w, area.h, natural.w, natural.h, value.sizing, "50% 50%")
    : null;
  const zoomTo = (next: number, at?: { x: number; y: number }) => {
    setDraft((d) => {
      const zoom = Math.round(Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next)));
      if (!drawn || zoom === d.zoom) {
        return { ...d, zoom };
      }
      const point = at
        ? { x: at.x / scale, y: at.y / scale }
        : { x: area.w / 2, y: area.h / 2 };
      return {
        zoom,
        x: zoomPosition(d.x, point.x, area.w, drawn.w, d.zoom / 100, zoom / 100),
        y: zoomPosition(d.y, point.y, area.h, drawn.h, d.zoom / 100, zoom / 100),
      };
    });
  };
  // dx, dy in preview px.
  const moveBy = (from: Draft, dx: number, dy: number): Draft =>
    drawn
      ? {
          ...from,
          x: dragPosition(from.x, dx / scale, area.w, drawn.w, from.zoom / 100),
          y: dragPosition(from.y, dy / scale, area.h, drawn.h, from.zoom / 100),
        }
      : from;
  const zoomRef = useRef(zoomTo);
  zoomRef.current = zoomTo;
  const draftRef = useRef(draft);
  draftRef.current = draft;

  // Native listeners: React's wheel is passive (no preventDefault), and WebKit pinches as gesture events.
  useEffect(() => {
    const node = preview;
    if (!node) {
      return;
    }
    const local = (event: { clientX: number; clientY: number }) => {
      const rect = node.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    };
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const lines = event.deltaMode === 1 ? 40 : 1;
      // A trackpad pinch arrives as a ctrl+wheel with small deltas: weight it more.
      const rate = event.ctrlKey ? 0.01 : 0.0025;
      const factor = Math.exp(
        Math.max(-2, Math.min(2, -event.deltaY * lines * rate)),
      );
      zoomRef.current(draftRef.current.zoom * factor, local(event));
    };
    type Gesture = Event & { scale: number; clientX: number; clientY: number };
    const onGestureStart = (event: Event) => {
      event.preventDefault();
      pinch.current = draftRef.current.zoom;
    };
    const onGestureChange = (event: Event) => {
      event.preventDefault();
      const gesture = event as Gesture;
      if (pinch.current !== null) {
        zoomRef.current(pinch.current * gesture.scale, local(gesture));
      }
    };
    const onGestureEnd = () => {
      pinch.current = null;
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    node.addEventListener("gesturestart", onGestureStart);
    node.addEventListener("gesturechange", onGestureChange);
    node.addEventListener("gestureend", onGestureEnd);
    return () => {
      node.removeEventListener("wheel", onWheel);
      node.removeEventListener("gesturestart", onGestureStart);
      node.removeEventListener("gesturechange", onGestureChange);
      node.removeEventListener("gestureend", onGestureEnd);
    };
  }, [preview]);

  const apply = () => {
    patchBackground({
      position: `${Math.round(draft.x * 100)}% ${Math.round(draft.y * 100)}%`,
      zoom: Math.round(draft.zoom),
    });
    onOpenChange(false);
  };
  const position = `${draft.x * 100}% ${draft.y * 100}%`;
  const ready = Boolean(natural && box);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-[calc(600px*var(--ui-space-scale,1))] gap-0 p-5"
        onKeyDown={(event) => {
          if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
            event.preventDefault();
            apply();
          }
        }}
      >
        <DialogHeader className="pe-8">
          <DialogTitle>{t("chatBackground.adjustTitle")}</DialogTitle>
          <DialogDescription>
            {t("chatBackground.adjustHint")}
          </DialogDescription>
        </DialogHeader>
        <div
          ref={setPreview}
          // biome-ignore lint/a11y/noNoninteractiveTabindex: arrow keys nudge the framing from here.
          tabIndex={0}
          role="img"
          aria-label={t("chatBackground.adjustTitle")}
          data-wallpaper-adjust-preview=""
          data-dragging={dragging || undefined}
          className="relative mx-auto mt-4 touch-none select-none overflow-hidden rounded-xl border border-border bg-background outline-none focus-visible:ring-2 focus-visible:ring-ring data-[dragging]:cursor-grabbing cursor-grab"
          style={
            {
              aspectRatio: String(aspect),
              // 300px tall at most, so a portrait window gives a narrow preview, not a square one.
              width: `min(100%, calc(300px * ${aspect} * var(--ui-space-scale, 1)))`,
            } as CSSProperties
          }
          onPointerDown={(event) => {
            if (event.button !== 0 || !ready) {
              return;
            }
            // A press would otherwise start a text selection that the drag sweeps across the dialog.
            // Focus by hand, as the default focus is cancelled too.
            event.preventDefault();
            window.getSelection()?.removeAllRanges();
            event.currentTarget.focus({ preventScroll: true });
            event.currentTarget.setPointerCapture(event.pointerId);
            drag.current = {
              id: event.pointerId,
              x: event.clientX,
              y: event.clientY,
              from: draft,
            };
            setDragging(true);
          }}
          // Selection starts from mousedown's default, which a cancelled pointerdown does not stop everywhere.
          onMouseDown={(event) => event.preventDefault()}
          onPointerMove={(event) => {
            const start = drag.current;
            if (!start || start.id !== event.pointerId) {
              return;
            }
            setDraft(
              moveBy(
                start.from,
                event.clientX - start.x,
                event.clientY - start.y,
              ),
            );
          }}
          onPointerUp={() => {
            drag.current = null;
            setDragging(false);
          }}
          onPointerCancel={() => {
            drag.current = null;
            setDragging(false);
          }}
          onKeyDown={(event) => {
            const step = event.shiftKey ? NUDGE * 3 : NUDGE;
            const delta = new Map([
              ["ArrowLeft", [step, 0]],
              ["ArrowRight", [-step, 0]],
              ["ArrowUp", [0, step]],
              ["ArrowDown", [0, -step]],
            ]).get(event.key);
            if (!delta) {
              return;
            }
            event.preventDefault();
            setDraft((d) => moveBy(d, delta[0], delta[1]));
          }}
        >
          <div
            aria-hidden="true"
            className="absolute top-0 left-0 origin-top-left"
            style={{
              width: area.w,
              height: area.h,
              transform: `scale(${scale})`,
            }}
          >
            <WallpaperArtwork
              url={url}
              sizing={value.sizing}
              position={position}
              zoom={draft.zoom}
              blur={0}
              bakeable={false}
            />
          </div>
          {dragging && (
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0"
            >
              {["left-1/3", "left-2/3"].map((at) => (
                <span
                  key={at}
                  className={`wallpaper-adjust-guide absolute inset-y-0 w-px ${at}`}
                />
              ))}
              {["top-1/3", "top-2/3"].map((at) => (
                <span
                  key={at}
                  className={`wallpaper-adjust-guide absolute inset-x-0 h-px ${at}`}
                />
              ))}
            </div>
          )}
        </div>
        <div className="mt-3.5 flex items-center justify-between gap-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setDraft({ x: 0.5, y: 0.5, zoom: MIN_ZOOM })}
          >
            {t("chatBackground.adjustReset")}
          </Button>
          <div className="flex items-center gap-1.5">
            <Button
              type="button"
              variant="outline"
              size="icon-sm"
              aria-label={t("chatBackground.zoomOut")}
              title={t("chatBackground.zoomOut")}
              disabled={draft.zoom <= MIN_ZOOM}
              onClick={() => zoomTo(draft.zoom - ZOOM_STEP)}
            >
              <HugeiconsIcon icon={MinusSignIcon} strokeWidth={2} />
            </Button>
            <Slider
              className="w-42"
              aria-label={t("chatBackground.zoom")}
              min={MIN_ZOOM}
              max={MAX_ZOOM}
              step={1}
              value={[draft.zoom]}
              onValueChange={([zoom]) => zoomTo(zoom)}
              thumbValueText={(zoom) => `${Math.round(zoom)}%`}
            />
            <span className="w-13 text-center text-xs tabular-nums text-muted-foreground">
              {Math.round(draft.zoom)}%
            </span>
            <Button
              type="button"
              variant="outline"
              size="icon-sm"
              aria-label={t("chatBackground.zoomIn")}
              title={t("chatBackground.zoomIn")}
              disabled={draft.zoom >= MAX_ZOOM}
              onClick={() => zoomTo(draft.zoom + ZOOM_STEP)}
            >
              <HugeiconsIcon icon={PlusSignIcon} strokeWidth={2} />
            </Button>
          </div>
        </div>
        <div className="mt-4.5 flex items-center justify-end gap-2 border-t border-border pt-3">
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            {t("common.cancel")}
          </Button>
          <Button
            type="button"
            variant="dark"
            disabled={!ready}
            onClick={apply}
          >
            {t("chatBackground.apply")}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
