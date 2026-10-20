// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { prefersReducedMotion } from "@/features/settings";

// Slides a segmented control's selected pill from the old segment to the new one
// without running script per frame or forcing layout during the click:
// - geometry is read once per switch, in the first ResizeObserver callback for
//   the new pill's segment, after the browser has laid out the frame anyway.
//   Both the old and the new segment are measured then, so nothing is cached
//   that could go stale;
// - the slide is transform-only Web Animations on both axes (tracks can wrap),
//   which every engine's compositor runs off the main thread. When the two
//   segments differ in width, the pill is drawn as two round caps and a middle
//   strip: the end cap moves and the strip stretches with scaleX, so the ends
//   stay round (a scaleX of the whole pill would squash them). The move and
//   the resize are then both transforms on one clock, so a busy main thread
//   (a list re-rendering after the click) cannot hold the width back while the
//   pill moves. A clip-path resize could: Chromium repaints it on the main thread.
// At rest the pill paints itself and the caps, strip and shadow are hidden.

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
  h: number;
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
  return { x, y, w: segment.clientWidth, h: segment.clientHeight };
}

function slide(
  root: HTMLElement,
  w: number,
  h: number,
  dx: number,
  dy: number,
  dw: number,
) {
  const [shadow, startCap, strip, endCap] = Array.from(
    root.children,
  ) as HTMLElement[];
  // The caps are half circles. The strip runs between their flat sides and
  // tucks 1px under each, so no seam shows; any deeper and its square corners
  // would poke out past the curve. Every width involved must leave it room.
  const r = h / 2;
  const inner = w - 2 * r + 2;
  const resize =
    endCap && Math.abs(dw) >= 0.5 && Math.min(w, w + dw) > 2 * r + 1;
  // Held until the cleanup below, so the pieces never show their rest state
  // between the end of the slide and the cleanup.
  const timing = {
    duration: SLIDE_MS,
    easing: EASING,
    fill: resize ? ("forwards" as const) : ("none" as const),
  };
  const animations = [
    root.animate(
      [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: "none" }],
      timing,
    ),
  ];
  if (resize) {
    // The pill keeps its left end on the root's; the end cap moves by the
    // width change and the strip stretches to meet it. Only the strip scales,
    // and it has no round ends to squash. A shadow-only copy of the whole pill
    // stretches underneath, so the pieces need no shadow of their own; its
    // scale bends the shadow's ends a little, which the blur hides.
    root.style.setProperty("background-color", "transparent", "important");
    root.style.setProperty("box-shadow", "none", "important");
    shadow.style.width = `${w}px`;
    shadow.style.setProperty("background-color", "transparent", "important");
    shadow.style.display = "block";
    startCap.style.width = `${r}px`;
    strip.style.left = `${r - 1}px`;
    strip.style.width = `${inner}px`;
    endCap.style.width = `${r}px`;
    for (const piece of [startCap, strip, endCap]) {
      piece.style.setProperty("box-shadow", "none", "important");
      piece.style.display = "block";
    }
    animations.push(
      shadow.animate(
        [{ transform: `scaleX(${(w + dw) / w})` }, { transform: "none" }],
        timing,
      ),
      strip.animate(
        [
          { transform: `scaleX(${(inner + dw) / inner})` },
          { transform: "none" },
        ],
        timing,
      ),
      endCap.animate(
        [{ transform: `translateX(${dw}px)` }, { transform: "none" }],
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
      root.style.removeProperty("background-color");
      root.style.removeProperty("box-shadow");
      for (const piece of [shadow, startCap, strip, endCap]) {
        piece.style.display = "none";
      }
      for (const animation of animations) animation.cancel();
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
  slide(root, to.w, to.h, dx, dy, dw);
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
