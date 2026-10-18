// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { prefersReducedMotion } from "@/features/settings";

// Slides a segmented control's selected pill from the old segment to the new one
// without running script per frame or forcing layout during the click:
// - geometry is read once per switch, in the first ResizeObserver callback for
//   the new pill's segment, after the browser has laid out the frame anyway.
//   Both the old and the new segment are measured then, so nothing is cached
//   that could go stale;
// - the slide is a transform-only Web Animation on both axes (tracks can wrap),
//   which the compositor runs off the main thread. When the two segments differ
//   in width, a fill inside the pill changes width through a round clip-path
//   inset, so the ends stay round (a scaleX would squash them). Chromium runs
//   that on the compositor too; other engines repaint it on the main thread,
//   still with no script.
// At rest the pill paints itself and the fill and its shadow are hidden.

/** Spring the design system gives segmented controls (design.md, 2.5 Motion). */
const SPRING = { stiffness: 500, damping: 35, mass: 0.5 };

/** The spring's progress toward the new segment, sampled every millisecond until it settles. */
const SPRING_CURVE: number[] = (() => {
  const curve: number[] = [];
  let offset = 1;
  let velocity = 0;
  while (curve.length < 2000) {
    curve.push(1 - offset);
    if (Math.abs(offset) < 0.002 && Math.abs(velocity) < 0.05) break;
    velocity +=
      ((-SPRING.stiffness * offset - SPRING.damping * velocity) / SPRING.mass) *
      0.001;
    offset += velocity * 0.001;
  }
  curve[curve.length - 1] = 1;
  return curve;
})();

const SLIDE_MS = SPRING_CURVE.length - 1;

/** Where the CSS curve takes its points: dense early, where the spring moves fastest. */
const STOPS = [
  0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.36, 0.43, 0.5, 0.6,
  0.7, 0.85,
];

/** design.md's reveal curve, for engines without linear(). */
const FALLBACK = [0.22, 1, 0.36, 1] as const;

/** cubic-bezier(FALLBACK) sampled every millisecond of the slide. */
function bezierCurve(): number[] {
  const [x1, y1, x2, y2] = FALLBACK;
  const bezier = (a: number, b: number, t: number) =>
    3 * (1 - t) ** 2 * t * a + 3 * (1 - t) * t * t * b + t ** 3;
  const curve: number[] = [];
  for (let ms = 0; ms <= SLIDE_MS; ms++) {
    // x(t) only grows, so bisect for the t where it reaches this point in time.
    const x = ms / SLIDE_MS;
    let lo = 0;
    let hi = 1;
    for (let i = 0; i < 30; i++) {
      const t = (lo + hi) / 2;
      if (bezier(x1, x2, t) < x) lo = t;
      else hi = t;
    }
    curve.push(bezier(y1, y2, (lo + hi) / 2));
  }
  curve[SLIDE_MS] = 1;
  return curve;
}

const HAS_LINEAR =
  typeof CSS !== "undefined" &&
  !!CSS.supports?.("transition-timing-function", "linear(0, 1)");

const EASING = HAS_LINEAR
  ? `linear(0, ${STOPS.map(
      (at) =>
        `${SPRING_CURVE[Math.round(at * SLIDE_MS)].toFixed(3)} ${Math.round(at * 100)}%`,
    ).join(", ")}, 1)`
  : `cubic-bezier(${FALLBACK.join(", ")})`;

/** Progress of the curve the slide actually runs, so an interrupted slide
 *  restarts from where its pill is drawn. */
const CURVE = HAS_LINEAR ? SPRING_CURVE : bezierCurve();

function progressAt(ms: number): number {
  if (ms >= SLIDE_MS) return 1;
  return CURVE[Math.max(0, Math.round(ms))];
}

/** Animation timeline time, which the slide's startTime is set from. */
function now(): number {
  const time = document.timeline.currentTime;
  return typeof time === "number" ? time : performance.now();
}

interface Box {
  x: number;
  y: number;
  w: number;
}

interface PillState {
  track: HTMLElement;
  segment: HTMLElement;
  /** Offset of the drawn pill from its segment's box when the slide began. */
  slide?: { dx: number; dy: number; dw: number; start: number };
}

/** The last pill of each control, keyed by the control's group id. */
const pills = new Map<string, PillState>();

/** The track isolates (SegmentedPill sits at -z-10 in it); the segment does not. */
function findTrack(segment: HTMLElement): HTMLElement {
  for (let node = segment.parentElement; node; node = node.parentElement) {
    if (getComputedStyle(node).isolation === "isolate") return node;
  }
  return segment.parentElement ?? segment;
}

/** The pill fills its segment's padding box. Layout pixels, transforms
 *  ignored, relative to the page: only differences between two segments of
 *  one track are used. */
function measure(segment: HTMLElement): Box {
  let x = segment.clientLeft;
  let y = segment.clientTop;
  for (
    let node: HTMLElement | null = segment;
    node;
    node = node.offsetParent as HTMLElement | null
  ) {
    x += node.offsetLeft;
    y += node.offsetTop;
  }
  return { x, y, w: segment.clientWidth };
}

function slide(
  root: HTMLElement,
  w: number,
  dx: number,
  dy: number,
  dw: number,
) {
  const timing = { duration: SLIDE_MS, easing: EASING };
  const animations = [
    root.animate(
      [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }],
      timing,
    ),
  ];
  const [shadow, fill] = Array.from(root.children) as HTMLElement[];
  const resize = fill && Math.abs(dw) >= 0.5;
  if (resize) {
    // The pill spans the wider of the two widths; the fill's clip shows the
    // current one. A clip also cuts the fill's shadow, so a shadow-only copy
    // of the pill underneath stretches with it. Its scale bends the shadow's
    // ends a little, which the blur hides; a filter would follow the clip
    // instead, but Chromium draws it from a stale clip and leaves a ghost.
    const span = Math.max(w, w + dw);
    const inset = (width: number) =>
      `inset(0 ${span - width}px 0 0 round 9999px)`;
    root.style.right = "auto";
    root.style.width = `${span}px`;
    root.style.setProperty("background-color", "transparent", "important");
    root.style.setProperty("box-shadow", "none", "important");
    shadow.style.width = `${w}px`;
    shadow.style.transformOrigin = "left";
    shadow.style.setProperty("background-color", "transparent", "important");
    shadow.style.display = "block";
    fill.style.setProperty("box-shadow", "none", "important");
    fill.style.display = "block";
    animations.push(
      shadow.animate(
        [{ transform: `scaleX(${(w + dw) / w})` }, { transform: "none" }],
        timing,
      ),
      fill.animate(
        [{ clipPath: inset(w + dw) }, { clipPath: inset(w) }],
        timing,
      ),
    );
  }
  // One start time, so the move and the resize stay in step.
  const start = document.timeline.currentTime;
  for (const animation of animations) animation.startTime = start;
  if (!resize) return;
  Promise.all(animations.map((animation) => animation.finished)).then(
    () => {
      for (const property of [
        "right",
        "width",
        "background-color",
        "box-shadow",
      ]) {
        root.style.removeProperty(property);
      }
      shadow.style.display = "none";
      fill.style.display = "none";
    },
    () => {},
  );
}

function place(group: string, root: HTMLElement, segment: HTMLElement) {
  const track = findTrack(segment);
  const state: PillState = { track, segment };
  const prev = pills.get(group);
  for (const [key, other] of pills) {
    if (!other.track.isConnected) pills.delete(key);
  }
  pills.set(group, state);
  if (
    !prev ||
    prev.track !== track ||
    !prev.segment.isConnected ||
    prefersReducedMotion()
  ) {
    return;
  }
  const from = measure(prev.segment);
  const to = measure(segment);
  if (!from.w || !to.w) return;
  // Start from where the old pill is drawn now, mid-slide included.
  const time = now();
  const remaining = prev.slide ? 1 - progressAt(time - prev.slide.start) : 0;
  const dx = from.x + (prev.slide?.dx ?? 0) * remaining - to.x;
  const dy = from.y + (prev.slide?.dy ?? 0) * remaining - to.y;
  const dw = from.w + (prev.slide?.dw ?? 0) * remaining - to.w;
  if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5 && Math.abs(dw) < 0.5) return;
  state.slide = { dx, dy, dw, start: time };
  slide(root, to.w, dx, dy, dw);
}

/** Track a mounted pill: once its segment has been laid out, slide it in from
 *  the control's previous pill. Returns the cleanup. */
export function watchSegmentedPill(group: string, root: HTMLElement) {
  const segment = root.parentElement ?? root;
  const observer = new ResizeObserver(() => {
    observer.disconnect();
    place(group, root, segment);
  });
  observer.observe(segment);
  return () => observer.disconnect();
}
