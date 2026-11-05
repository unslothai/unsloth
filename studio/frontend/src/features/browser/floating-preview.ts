// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  isDegenerateBox,
  nudgeDelta,
  placeAxis,
} from "@/hooks/floating-panel-position";
import { Z_LAYER } from "@/lib/z-layers";
import type {
  KeyboardEvent as ReactKeyboardEvent,
  PointerEvent as ReactPointerEvent,
} from "react";
import {
  FLOATING_ATTRIBUTE,
  FLOATING_MOVE_EVENT,
  FLOATING_SCALE_VAR,
} from "./floating-events";

const STORAGE_KEY = "unsloth_browser_floating_preview";
const EDGE_GAP = 12;
const MIN_WIDTH = 260;
const MIN_HEIGHT = 180;
// A drop this close to the right edge stays anchored there as the chat resizes.
const RIGHT_SNAP_PX = 16;
// The page keeps a layout at least this wide however small the preview gets.
const MIN_LAYOUT_WIDTH = 480;
const FRAME_PX = 2;

export type FloatingEdge =
  | "move"
  | "n"
  | "s"
  | "e"
  | "w"
  | "ne"
  | "nw"
  | "se"
  | "sw";
/** `left` null: anchored to the right edge. Lengths are CSS px within the room. */
type Geometry = {
  left: number | null;
  top: number;
  width: number;
  height: number;
};
type Box = { left: number; top: number; width: number; height: number };
type Room = Box & { composer: Box | null };

let geometry: Geometry | null = null;
// The page's side-pane layout width, shown scaled down.
let layoutWidth: number | null = null;

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function readGeometry(): Geometry | null {
  try {
    const parsed = JSON.parse(
      localStorage.getItem(STORAGE_KEY) ?? "null",
    ) as Record<string, unknown> | null;
    if (
      !parsed ||
      !finite(parsed.top) ||
      !finite(parsed.width) ||
      !finite(parsed.height)
    )
      return null;
    const left =
      parsed.left === null
        ? null
        : finite(parsed.left)
          ? parsed.left
          : undefined;
    if (left === undefined) return null;
    return {
      left,
      top: parsed.top,
      width: parsed.width,
      height: parsed.height,
    };
  } catch {
    return null;
  }
}

function writeGeometry(next: Geometry): void {
  geometry = next;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // storage unavailable
  }
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const toBox = (rect: DOMRect): Box => ({
  left: rect.left,
  top: rect.top,
  width: rect.width,
  height: rect.height,
});

function overlaps(a: Box, b: Box): boolean {
  return (
    a.left < b.left + b.width &&
    b.left < a.left + a.width &&
    a.top < b.top + b.height &&
    b.top < a.top + a.height
  );
}

function composerIn(split: HTMLElement): Box | null {
  for (const form of split.querySelectorAll?.<HTMLElement>(
    "form.aui-composer-root",
  ) ?? []) {
    if (
      form.offsetParent === null ||
      form.closest('[aria-hidden="true"], [inert]')
    )
      continue;
    const box = form.getBoundingClientRect();
    if (box.width > 0 && box.height > 0) return toBox(box);
  }
  return null;
}

function roomFor(surface: HTMLElement): Room | null {
  const split = surface.closest<HTMLElement>(".chat-artifact-split");
  if (!split) return null;
  const box = split.getBoundingClientRect();
  if (isDegenerateBox(box)) return null;
  const header = split
    .closest("[data-chat-content-root]")
    ?.querySelector<HTMLElement>(
      ":scope > [data-side-panel-inset]:not(.chat-header-fade)",
    );
  const top =
    Math.max(box.top, header?.getBoundingClientRect().bottom ?? box.top) +
    EDGE_GAP;
  const width = box.width - 2 * EDGE_GAP;
  const height = box.bottom - EDGE_GAP - top;
  return width > 0 && height > 0
    ? {
        left: box.left + EDGE_GAP,
        top,
        width,
        height,
        composer: composerIn(split),
      }
    : null;
}

function clearOfComposer(box: Box, room: Room): Box {
  const composer = room.composer;
  if (!composer || !overlaps(box, composer)) return box;
  const tops = [
    composer.top - EDGE_GAP - box.height,
    composer.top + composer.height + EDGE_GAP,
  ].filter(
    (top) => top >= room.top && top + box.height <= room.top + room.height,
  );
  if (tops.length === 0) return box;
  const top = tops.reduce((best, next) =>
    Math.abs(next - box.top) < Math.abs(best - box.top) ? next : best,
  );
  return { ...box, top };
}

function defaultGeometry(room: Box): Geometry {
  const width = Math.round(clamp(room.width * 0.38, 320, 480));
  return { left: null, top: 0, width, height: Math.round(width * 0.68) };
}

function boxIn(room: Room, intent: Geometry): Box {
  const width = clamp(
    intent.width,
    Math.min(MIN_WIDTH, room.width),
    room.width,
  );
  const height = clamp(
    intent.height,
    Math.min(MIN_HEIGHT, room.height),
    room.height,
  );
  return clearOfComposer(
    {
      left: room.left + placeAxis(intent.left, room.width - width),
      top: room.top + clamp(intent.top, 0, room.height - height),
      width,
      height,
    },
    room,
  );
}

function intentOf(room: Box, box: Box): Geometry {
  const left = box.left - room.left;
  const anchored = room.width - box.width - left <= RIGHT_SNAP_PX;
  return {
    left: anchored ? null : left,
    top: box.top - room.top,
    width: box.width,
    height: box.height,
  };
}

function currentBox(room: Room): Box {
  geometry ??= readGeometry();
  return boxIn(room, geometry ?? defaultGeometry(room));
}

function scaleFor(box: Box): number {
  if (!layoutWidth) return 1;
  const scale = Math.min(
    1,
    Math.max(0.05, (box.width - FRAME_PX) / layoutWidth),
  );
  // Rounded, so a native page is re-zoomed only on a real size change.
  return Math.round(scale * 1000) / 1000;
}

function paint(surface: HTMLElement, box: Box): void {
  Object.assign(surface.style, {
    position: "fixed",
    left: `${box.left}px`,
    top: `${box.top}px`,
    width: `${box.width}px`,
    height: `${box.height}px`,
  });
  surface.style.setProperty(FLOATING_SCALE_VAR, String(scaleFor(box)));
}

// Not transform: React owns it on the surface, and a drag's own is cleared when it ends.
const PLACED_PROPERTIES = [
  "position",
  "left",
  "top",
  "width",
  "height",
  FLOATING_SCALE_VAR,
] as const;
// Below this a press is a click (a double-click docks), not a drag.
const DRAG_SLOP_PX = 3;

export function floatSurface(surface: HTMLElement): () => void {
  const saved = PLACED_PROPERTIES.map((name) =>
    surface.style.getPropertyValue(name),
  );
  const split = surface.closest<HTMLElement>(".chat-artifact-split");
  // Measured before it floats: still the side pane's width, or about none when it opens floating.
  const docked = surface.getBoundingClientRect().width;
  const splitWidth = split?.getBoundingClientRect().width ?? 0;
  layoutWidth = Math.round(
    Math.max(
      MIN_LAYOUT_WIDTH,
      docked >= MIN_LAYOUT_WIDTH ? docked : (layoutWidth ?? splitWidth / 2),
    ),
  );
  const place = () => {
    if (surface.hasAttribute("data-floating-gesture")) return;
    const room = roomFor(surface);
    if (room) paint(surface, currentBox(room));
  };
  place();
  const observer = new ResizeObserver(place);
  if (split) observer.observe(split);
  const observeComposer = () => {
    const composer = split?.querySelector?.<HTMLElement>(
      "form.aui-composer-root",
    );
    if (composer) observer.observe(composer);
  };
  observeComposer();
  window.addEventListener("resize", place);
  return () => {
    observer.disconnect();
    window.removeEventListener("resize", place);
    PLACED_PROPERTIES.forEach((name, index) => {
      if (saved[index]) surface.style.setProperty(name, saved[index]);
      else surface.style.removeProperty(name);
    });
  };
}

function resized(
  start: Box,
  edge: FloatingEdge,
  dx: number,
  dy: number,
  room: Box,
): Box {
  const minWidth = Math.min(MIN_WIDTH, room.width);
  const minHeight = Math.min(MIN_HEIGHT, room.height);
  const roomRight = room.left + room.width;
  const roomBottom = room.top + room.height;
  if (edge === "move") {
    return {
      ...start,
      left: clamp(start.left + dx, room.left, roomRight - start.width),
      top: clamp(start.top + dy, room.top, roomBottom - start.height),
    };
  }
  let left = start.left;
  let right = start.left + start.width;
  let top = start.top;
  let bottom = start.top + start.height;
  if (edge.includes("w")) left = clamp(left + dx, room.left, right - minWidth);
  if (edge.includes("e")) right = clamp(right + dx, left + minWidth, roomRight);
  if (edge.includes("n")) top = clamp(top + dy, room.top, bottom - minHeight);
  if (edge.includes("s"))
    bottom = clamp(bottom + dy, top + minHeight, roomBottom);
  return { left, top, width: right - left, height: bottom - top };
}

function settled(
  start: Box,
  previous: Box,
  next: Box,
  edge: FloatingEdge,
  room: Room,
): Box {
  if (edge === "move") return clearOfComposer(next, room);
  const composer = room.composer;
  if (!composer || !overlaps(next, composer) || overlaps(start, composer))
    return next;
  let { top, height } = next;
  const bottom = top + height;
  if (edge.includes("s") && start.top + start.height <= composer.top) {
    height = composer.top - EDGE_GAP - top;
  } else if (
    edge.includes("n") &&
    start.top >= composer.top + composer.height
  ) {
    top = composer.top + composer.height + EDGE_GAP;
    height = bottom - top;
  }
  const stopped = { ...next, top, height };
  return height >= Math.min(MIN_HEIGHT, room.height) &&
    !overlaps(stopped, composer)
    ? stopped
    : previous;
}

const CURSORS: Record<FloatingEdge, string> = {
  move: "grabbing",
  n: "ns-resize",
  s: "ns-resize",
  e: "ew-resize",
  w: "ew-resize",
  ne: "nesw-resize",
  sw: "nesw-resize",
  nw: "nwse-resize",
  se: "nwse-resize",
};

const announceMove = () => window.dispatchEvent(new Event(FLOATING_MOVE_EVENT));

/** A move rides a transform and a resize only rescales, once per frame; a sheet keeps the page's frame off the pointer. */
export function startFloatingGesture(
  event: ReactPointerEvent<HTMLElement>,
  edge: FloatingEdge,
): void {
  if (event.button !== 0) return;
  if (
    edge === "move" &&
    (event.target as Element).closest(
      "button:not([data-floating-handle]), a, input",
    )
  )
    return;
  const surface = event.currentTarget.closest<HTMLElement>(
    `[${FLOATING_ATTRIBUTE}]`,
  );
  const room = surface ? roomFor(surface) : null;
  if (!surface || !room) return;
  event.preventDefault();
  const start = currentBox(room);
  const startX = event.clientX;
  const startY = event.clientY;
  let latest = start;
  let frame = 0;
  // Only on the first real move: from the press it would take the double-click.
  // `[data-slot$="-overlay"]`: native-view.ts treats it as covering the page, so input stays here.
  let sheet: HTMLElement | null = null;
  const begin = () => {
    sheet = document.createElement("div");
    sheet.setAttribute("data-slot", "browser-floating-drag-overlay");
    sheet.setAttribute("aria-hidden", "true");
    Object.assign(sheet.style, {
      position: "fixed",
      inset: "0",
      zIndex: String(Z_LAYER.DRAG_CURSOR_OVERLAY),
      cursor: CURSORS[edge],
      userSelect: "none",
      touchAction: "none",
    });
    document.body.appendChild(sheet);
    surface.setAttribute(
      "data-floating-gesture",
      edge === "move" ? "move" : "resize",
    );
  };
  // Docked or closed mid-drag: the surface is no longer the window's.
  const floating = () => surface.hasAttribute(FLOATING_ATTRIBUTE);
  const draw = () => {
    frame = 0;
    if (!floating()) return;
    if (edge === "move") {
      surface.style.transform = `translate3d(${latest.left - start.left}px, ${latest.top - start.top}px, 0)`;
    } else {
      paint(surface, latest);
    }
    announceMove();
  };
  const move = (next: PointerEvent) => {
    if (next.buttons === 0) {
      finish();
      return;
    }
    const dx = next.clientX - startX;
    const dy = next.clientY - startY;
    if (!sheet) {
      if (Math.hypot(dx, dy) < DRAG_SLOP_PX) return;
      begin();
    }
    latest = settled(
      start,
      latest,
      resized(start, edge, dx, dy, room),
      edge,
      room,
    );
    if (!frame) frame = requestAnimationFrame(draw);
  };
  const finish = () => {
    window.removeEventListener("pointermove", move);
    window.removeEventListener("pointerup", finish);
    window.removeEventListener("pointercancel", finish);
    cancelAnimationFrame(frame);
    if (!sheet) return;
    sheet.remove();
    surface.removeAttribute("data-floating-gesture");
    if (edge === "move") surface.style.transform = "none";
    if (!floating()) return;
    paint(surface, latest);
    writeGeometry(intentOf(room, latest));
    announceMove();
  };
  window.addEventListener("pointermove", move);
  window.addEventListener("pointerup", finish);
  window.addEventListener("pointercancel", finish);
}

export function nudgeFloating(event: ReactKeyboardEvent<HTMLElement>): void {
  const delta = nudgeDelta(event.key, false);
  const surface = event.currentTarget.closest<HTMLElement>(
    `[${FLOATING_ATTRIBUTE}]`,
  );
  const room = surface ? roomFor(surface) : null;
  if (!delta || !surface || !room) return;
  event.preventDefault();
  const step = 2;
  const edge = event.shiftKey ? "se" : "move";
  const start = currentBox(room);
  const next = settled(
    start,
    start,
    resized(start, edge, delta.dx * step, delta.dy * step, room),
    edge,
    room,
  );
  paint(surface, next);
  writeGeometry(intentOf(room, next));
  announceMove();
}
