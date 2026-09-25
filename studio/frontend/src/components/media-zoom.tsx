// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type SyntheticEvent,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import { cn } from "@/lib/utils";

export type MediaZoom = "fit" | number;

// A video's own controls sit along its bottom edge; a drag never starts there.
const VIDEO_CONTROLS_PX = 56;
const DRAG_SLOP_PX = 4;

type Size = { width: number; height: number };

function translate({ x, y }: { x: number; y: number }): string {
  return `translate3d(${x}px, ${y}px, 0)`;
}

function clamp(value: number, limit: number): number {
  return Math.max(-limit, Math.min(limit, value));
}

export function MediaZoomStage({
  zoom,
  onFitScale,
  children,
}: {
  zoom: MediaZoom;
  onFitScale: (scale: number | null) => void;
  children: ReactNode;
}) {
  const stageRef = useRef<HTMLDivElement>(null);
  const layerRef = useRef<HTMLDivElement>(null);
  const [stage, setStage] = useState<Size | null>(null);
  const [natural, setNatural] = useState<(Size & { src: string }) | null>(null);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const drag = useRef<{
    id: number;
    x: number;
    y: number;
    ox: number;
    oy: number;
    moved: boolean;
    next: { x: number; y: number };
    frame: number;
  } | null>(null);
  const dragged = useRef(false);
  const [grabbing, setGrabbing] = useState(false);

  useLayoutEffect(() => {
    const node = stageRef.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) =>
      setStage({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      }),
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const fit =
    stage && natural
      ? Math.min(stage.width / natural.width, stage.height / natural.height)
      : null;
  useEffect(() => onFitScale(fit), [fit, onFitScale]);

  const scale = zoom === "fit" ? fit : zoom;
  const box =
    natural && scale
      ? { width: natural.width * scale, height: natural.height * scale }
      : null;
  const slack =
    box && stage
      ? {
          x: Math.max(0, (box.width - stage.width) / 2),
          y: Math.max(0, (box.height - stage.height) / 2),
        }
      : { x: 0, y: 0 };
  const pan = { x: clamp(offset.x, slack.x), y: clamp(offset.y, slack.y) };
  const pannable = slack.x > 0 || slack.y > 0;

  function measure(event: SyntheticEvent) {
    const target = event.target;
    const size =
      target instanceof HTMLImageElement
        ? {
            width: target.naturalWidth,
            height: target.naturalHeight,
            src: target.currentSrc,
          }
        : target instanceof HTMLVideoElement
          ? {
              width: target.videoWidth,
              height: target.videoHeight,
              src: target.currentSrc,
            }
          : null;
    if (!size || !size.width || !size.height) return;
    if (size.src !== natural?.src) setOffset({ x: 0, y: 0 });
    setNatural(size);
  }

  function onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    dragged.current = false;
    if (!pannable || event.button !== 0) return;
    const target = event.target;
    if (
      target instanceof HTMLVideoElement &&
      event.clientY > target.getBoundingClientRect().bottom - VIDEO_CONTROLS_PX
    ) {
      return;
    }
    drag.current = {
      id: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      ox: pan.x,
      oy: pan.y,
      moved: false,
      next: pan,
      frame: 0,
    };
  }

  function onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    const current = drag.current;
    if (!current || current.id !== event.pointerId) return;
    if ((event.buttons & 1) === 0) {
      endDrag(current);
      return;
    }
    const dx = event.clientX - current.x;
    const dy = event.clientY - current.y;
    if (!current.moved) {
      if (Math.hypot(dx, dy) < DRAG_SLOP_PX) return;
      current.moved = true;
      setGrabbing(true);
      event.currentTarget.setPointerCapture(event.pointerId);
    }
    current.next = {
      x: clamp(current.ox + dx, slack.x),
      y: clamp(current.oy + dy, slack.y),
    };
    if (current.frame) return;
    current.frame = requestAnimationFrame(() => {
      current.frame = 0;
      const layer = layerRef.current;
      if (layer) layer.style.transform = translate(current.next);
    });
  }

  function onPointerEnd(event: ReactPointerEvent<HTMLDivElement>) {
    const current = drag.current;
    if (current?.id !== event.pointerId) return;
    endDrag(current);
  }

  function endDrag(current: NonNullable<typeof drag.current>) {
    cancelAnimationFrame(current.frame);
    dragged.current = current.moved;
    drag.current = null;
    setGrabbing(false);
    if (current.moved) setOffset(current.next);
  }

  return (
    <div
      ref={stageRef}
      onLoadCapture={measure}
      onLoadedMetadataCapture={measure}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerEnd}
      onPointerCancel={onPointerEnd}
      onLostPointerCapture={onPointerEnd}
      // The click that ends a drag must not also play or pause a video.
      onClickCapture={(event) => {
        if (!dragged.current) return;
        dragged.current = false;
        event.preventDefault();
        event.stopPropagation();
      }}
      onDragStart={(event) => pannable && event.preventDefault()}
      className={cn(
        "relative min-h-0 flex-1 select-none overflow-hidden",
        pannable && "touch-none",
        pannable && (grabbing ? "cursor-grabbing" : "cursor-grab"),
      )}
    >
      <div
        ref={layerRef}
        className="absolute flex will-change-transform"
        style={
          box && stage
            ? {
                width: box.width,
                height: box.height,
                left: (stage.width - box.width) / 2,
                top: (stage.height - box.height) / 2,
                transform: translate(pan),
              }
            : { inset: 0 }
        }
      >
        {children}
      </div>
    </div>
  );
}
