// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type PointerEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import type { LocalizedEditMode } from "./api";
import { ANNOTATION_COLORS, TRANSPARENCY_CHECKER } from "./edit-conditioning";

/** Draws the localized-edit layer over the source at its NATIVE resolution: an RGBA layer of
 *  coloured strokes for annotate, a white-on-black mask for paint and mask. */
export function LocalizedEditCanvas({
  image,
  mode,
  color,
  brushPct,
  resetKey,
  onLayerChange,
  onColorsChange,
}: {
  image: string;
  mode: LocalizedEditMode;
  color: string;
  brushPct: number;
  resetKey: number;
  onLayerChange: (dataUrl: string | null) => void;
  onColorsChange: (names: string[]) => void;
}) {
  const dispRef = useRef<HTMLCanvasElement | null>(null);
  const layerRef = useRef<HTMLCanvasElement | null>(null);
  const dims = useRef<{ w: number; h: number }>({ w: 0, h: 0 });
  const drawing = useRef(false);
  const dirty = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);
  const used = useRef<Set<string>>(new Set());
  // Drawing waits until the layer for THIS source, mode and clear has been sized.
  const layerKey = `${resetKey}|${mode}|${image}`;
  const [readyFor, setReadyFor] = useState<string | null>(null);
  const ready = readyFor === layerKey;

  // A new source, a new mode or Clear starts a blank layer at the source's natural size. The old
  // layer is dropped at once, not on load, so Generate never pairs it with the new source.
  useEffect(() => {
    let live = true;
    drawing.current = false;
    dirty.current = false;
    used.current = new Set();
    onLayerChange(null);
    onColorsChange([]);
    const img = new Image();
    img.onload = () => {
      if (!live) return;
      const w = img.naturalWidth;
      const h = img.naturalHeight;
      dims.current = { w, h };
      const disp = dispRef.current;
      const layer = layerRef.current ?? document.createElement("canvas");
      layerRef.current = layer;
      if (!disp) return;
      disp.width = w;
      disp.height = h;
      layer.width = w;
      layer.height = h;
      const lctx = layer.getContext("2d");
      const dctx = disp.getContext("2d");
      if (!lctx || !dctx) return;
      lctx.clearRect(0, 0, w, h);
      if (mode !== "annotate") {
        lctx.fillStyle = "#000";
        lctx.fillRect(0, 0, w, h);
      }
      dctx.clearRect(0, 0, w, h);
      setReadyFor(`${resetKey}|${mode}|${image}`);
    };
    img.src = image;
    return () => {
      live = false;
    };
  }, [image, mode, resetKey, onLayerChange, onColorsChange]);

  const radius = useCallback(() => {
    const base = Math.min(dims.current.w, dims.current.h) || 1024;
    // Annotations are outlines, so a fraction of the region brush.
    const pct = mode === "annotate" ? brushPct / 4 : brushPct;
    return Math.max(1.5, (pct / 100) * base);
  }, [brushPct, mode]);

  const toNatural = (e: PointerEvent<HTMLCanvasElement>) => {
    const disp = dispRef.current;
    if (!disp) return { x: 0, y: 0 };
    const r = disp.getBoundingClientRect();
    return {
      x: ((e.clientX - r.left) / r.width) * dims.current.w,
      y: ((e.clientY - r.top) / r.height) * dims.current.h,
    };
  };

  const stroke = (
    from: { x: number; y: number } | null,
    to: { x: number; y: number },
  ) => {
    const disp = dispRef.current;
    const layer = layerRef.current;
    if (!disp || !layer) return;
    const r = radius();
    const shown =
      mode === "annotate"
        ? color
        : mode === "paint"
          ? "#ffffff"
          : "rgba(244,114,114,0.55)";
    const stored = mode === "annotate" ? color : "#ffffff";
    const layers: Array<[CanvasRenderingContext2D | null, string]> = [
      [disp.getContext("2d"), shown],
      [layer.getContext("2d"), stored],
    ];
    for (const [ctx, style] of layers) {
      if (!ctx) continue;
      ctx.strokeStyle = style;
      ctx.fillStyle = style;
      ctx.lineWidth = r * 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.arc(to.x, to.y, r, 0, Math.PI * 2);
      ctx.fill();
      if (from) {
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      }
    }
    dirty.current = true;
    if (mode === "annotate") {
      const name = ANNOTATION_COLORS.find((c) => c.value === color)?.name;
      if (name && !used.current.has(name)) {
        used.current.add(name);
        onColorsChange([...used.current]);
      }
    }
  };

  const onDown = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!ready) return;
    drawing.current = true;
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      // setPointerCapture can throw for synthetic events; safe to ignore.
    }
    const p = toNatural(e);
    last.current = p;
    stroke(null, p);
  };
  const onMove = (e: PointerEvent<HTMLCanvasElement>) => {
    if (!drawing.current) return;
    const p = toNatural(e);
    stroke(last.current, p);
    last.current = p;
  };
  const onUp = () => {
    if (!drawing.current) return;
    drawing.current = false;
    last.current = null;
    const layer = layerRef.current;
    if (layer && dirty.current) onLayerChange(layer.toDataURL("image/png"));
  };

  return (
    <div
      className="relative overflow-hidden rounded-[10px] border border-border"
      style={TRANSPARENCY_CHECKER}
    >
      <img
        src={image}
        alt="Edit source"
        className="block w-full select-none"
        draggable={false}
      />
      <canvas
        ref={dispRef}
        data-testid="localized-edit-canvas"
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerLeave={onUp}
        className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
      />
    </div>
  );
}
