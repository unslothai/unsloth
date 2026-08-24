// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_overlay_rail.py, shaped like smoke-settings.html: a
// vite entry with no backend and no auth, driving the real useStackGeometry.
//
// Everything the rail's geometry rests on -- the computed padding, both scrollHeight probes,
// the clip, the hit box -- needs a DOM, and the node suite has none. The full app has one but
// only behind a backend, a router and four polling cards.
//
// The rail below is a copy of the one in src/app/provider.tsx, className and style keys
// included; playwright_overlay_rail.py pins the two against each other so it cannot drift.

/* eslint-disable no-restricted-imports -- a harness entry point, not app code. */
import {
  STACK_SHADOW_GUTTER_BOTTOM,
  STACK_SHADOW_GUTTER_TOP,
  type MonitorFrame,
  railBottomOffset,
  railMaxHeight,
  useMonitorFrameStore,
  useStackGeometry,
} from "@/features/settings/stores/monitor-frame-store";
/* eslint-enable no-restricted-imports */
import { Z_LAYER } from "@/lib/z-layers";
import { cn } from "@/lib/utils";
import { StrictMode, useSyncExternalStore } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

/** One card in the rail, as the driver describes it. */
type CardSpec = {
  /** Natural height, the card's own `min-h-[...]` floor equivalent. */
  height: number;
  /** How far it may be squeezed. Equal to `height` models a card that cannot. */
  floor?: number;
  /** Transient banners carry this; the download and loaded-model cards do not. */
  dismissible?: boolean;
  /** Dragged out of the rail's flow, the way the loaded-models card can be. */
  dragged?: boolean;
};

declare global {
  interface Window {
    // Optional: the app typechecks this entry, but only the harness page installs it.
    __railSmoke?: {
      setCards: (cards: CardSpec[]) => void;
      publish: (who: string, frame: MonitorFrame) => void;
      clear: (who: string) => void;
      /** What the hook last handed the rail, before either offset is applied. */
      geometry: () => { bottom: number; maxHeight: number; overflowing: boolean };
      /** The reserved room, so the driver asserts against the source, not a copy. */
      gutter: () => { top: number; bottom: number };
      state: () => { cards: CardSpec[]; errors: string[] };
      errors: () => string[];
    };
  }
}

const seenErrors: string[] = [];
window.addEventListener("error", (e) => {
  seenErrors.push(String(e.message));
});
window.addEventListener("unhandledrejection", (e) => {
  seenErrors.push(String(e.reason));
});

// Publisher tokens are compared by identity and the store folds frames one at a time, so
// each obstacle needs its own stable token; their union under one token is a different
// input, and a wrong one.
const publishers = new Map<string, object>();
function publisherFor(who: string): object {
  let token = publishers.get(who);
  if (!token) {
    token = {};
    publishers.set(who, token);
  }
  return token;
}

// A store of one value, so setCards re-renders without pulling in any app state.
let cards: CardSpec[] = [];
const listeners = new Set<() => void>();
function setCards(next: CardSpec[]): void {
  cards = next;
  for (const listener of listeners) listener();
}
function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function Card({ spec, index }: { spec: CardSpec; index: number }) {
  const floor = spec.floor ?? spec.height;
  return (
    <div
      className={cn(
        "pointer-events-auto flex w-[calc(100vw-2rem)] max-w-[448px] flex-col",
        spec.dragged && "fixed top-2 left-2 z-[9999] w-fit",
      )}
      data-overlay-dismissible={spec.dismissible ? "true" : undefined}
      data-card-index={index}
      style={{ minHeight: floor, flex: floor === spec.height ? "none" : "0 1 auto" }}
    >
      {/* The shadow the gutter is sized against: the same pair every rail surface carries. */}
      <div
        className="menu-soft-surface menu-soft-edgeless relative flex min-h-0 flex-col overflow-hidden rounded-[24px] bg-card"
        data-card-surface={index}
        style={{ height: spec.height, minHeight: floor }}
      >
        {/* Its own scroller, so the squeeze the rail probes for is a real one. */}
        <div className="min-h-0 flex-1 overflow-y-auto p-4 text-sm">
          {`card ${index}`}
          <div style={{ height: spec.height * 2 }} />
        </div>
      </div>
    </div>
  );
}

// What the hook last returned, so the driver can compare the band the cards were given
// against the box that was laid out.
let lastGeometry = { bottom: 0, maxHeight: 0, overflowing: false };

function Rail() {
  const live = useSyncExternalStore(subscribe, () => cards);
  const stack = useStackGeometry();
  lastGeometry = {
    bottom: stack.bottom,
    maxHeight: stack.maxHeight,
    overflowing: stack.overflowing,
  };
  return (
    <div
      ref={stack.ref}
      data-testid="overlay-rail"
      // Copied from src/app/provider.tsx. Keep both in step; the driver asserts it.
      className={cn(
        "fixed right-4 -mx-3 flex flex-col items-end gap-2 overflow-y-auto overflow-x-hidden overscroll-contain px-3",
        stack.overflowing ? "pointer-events-auto" : "pointer-events-none",
      )}
      style={{
        bottom: railBottomOffset(stack.bottom),
        maxHeight: railMaxHeight(stack.maxHeight),
        paddingTop: STACK_SHADOW_GUTTER_TOP,
        paddingBottom: STACK_SHADOW_GUTTER_BOTTOM,
        zIndex: Z_LAYER.OVERLAY_STACK,
      }}
    >
      {live.map((spec, index) => (
        <Card key={index} spec={spec} index={index} />
      ))}
    </div>
  );
}

/**
 * Stand-ins for the two surfaces that publish a frame: the Live monitor, which is a floating
 * panel and so paints over the rail, and the chat composer, which does not.
 */
function Obstacles() {
  const frames = useMonitorFrameStore((s) => s.frames);
  const boxes: Array<[string, MonitorFrame]> = [];
  for (const [token, frame] of frames) {
    for (const [who, mine] of publishers) {
      if (mine === token) boxes.push([who, frame]);
    }
  }
  return (
    <>
      {boxes.map(([who, frame]) => (
        <div
          key={who}
          data-testid={`obstacle-${who}`}
          className="fixed rounded-xl border border-border/70 bg-muted"
          style={{
            left: frame.left,
            top: frame.top,
            width: Math.max(0, frame.right - frame.left),
            height: Math.max(0, frame.bottom - frame.top),
            // The Live monitor is a floating panel, which outranks the rail; the composer is
            // in-page chrome and sits below it.
            zIndex: who === "monitor" ? Z_LAYER.FLOATING_PANEL : 100,
            resize: who === "monitor" ? "both" : "none",
            overflow: who === "monitor" ? "auto" : "hidden",
          }}
        />
      ))}
    </>
  );
}

/**
 * All eight of the window's resize targets, copied from window-titlebar.tsx, so the driver
 * can ask which ones the rail's box reaches rather than being told. A card is
 * `w-[calc(100vw-2rem)]` up to its max, so a narrow window brings the left-hand ones in too.
 */
// Each carries the layer the app gives it, so the driver can separate which targets the
// box reaches from which it takes: the difference is what a named layer is already covering.
const RESIZE_TARGETS = [
  ["resize-north", "fixed inset-x-2 top-0 h-1 cursor-n-resize"],
  ["resize-south", "fixed inset-x-2 bottom-0 h-1 cursor-s-resize"],
  ["resize-west", "fixed inset-y-2 left-0 w-1 cursor-w-resize"],
  ["resize-east", "fixed inset-y-2 right-0 w-1 cursor-e-resize"],
  ["resize-northwest", "fixed top-0 left-0 size-3 cursor-nw-resize"],
  ["resize-northeast", "fixed top-0 right-0 size-3 cursor-ne-resize"],
  ["resize-southwest", "fixed bottom-0 left-0 size-3 cursor-sw-resize"],
  ["resize-southeast", "fixed right-0 bottom-0 size-3 cursor-se-resize"],
] as const;

// Overridable per load, so one run measures the shape this PR inherited as well as the one
// it ships: `?gripz=70&headerz=70` is the titlebar before it was touched.
const params = new URLSearchParams(window.location.search);
function layerParam(name: string, fallback: number): number {
  const raw = params.get(name);
  if (raw === null) return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}
const GRIP_Z = layerParam("gripz", Z_LAYER.WINDOW_RESIZE_EDGE);
// 70 is the app's own header, on Tailwind's scale with the rest of the in-page chrome.
const HEADER_Z = layerParam("headerz", 70);
// None by default: a z-index here cannot escape the header above it. `?toolbarz=` puts one
// back so the driver can show it changes nothing.
const TOOLBAR_Z = params.has("toolbarz")
  ? layerParam("toolbarz", 0)
  : undefined;

function WindowResizeTargets() {
  return (
    <>
      {RESIZE_TARGETS.map(([id, shape]) => (
        <div
          key={id}
          aria-hidden="true"
          data-testid={id}
          className={cn("pointer-events-auto", shape)}
          style={{ zIndex: GRIP_Z }}
        />
      ))}
    </>
  );
}

/**
 * The window controls, whose corner the north-east grip lands on. Shaped like the toolbar in
 * window-titlebar.tsx: `right-1` plus `px-1` leaves Close's right edge 8px in, and a 30px
 * button centred in the band puts its top a couple of px down, so the grip's corner and the
 * button's overlap.
 *
 * The `<header>` is not decoration. In the app the toolbar sits inside a positioned, numbered
 * header, which by CSS is a stacking context and confines every z-index inside it. Without it
 * the buttons compare against the grips directly, which is a more forgiving question than the
 * one the app asks: it is the ancestor that has to out-rank a grip.
 */
function WindowControls() {
  return (
    <header
      data-testid="titlebar-header"
      className="pointer-events-none absolute inset-x-0 top-0 h-[34px] select-none"
      style={{ zIndex: HEADER_Z }}
    >
      <div
        className="pointer-events-auto absolute right-1 top-0 flex h-full items-center gap-0.5 px-1"
        role="toolbar"
        aria-label="Window controls"
        style={TOOLBAR_Z === undefined ? undefined : { zIndex: TOOLBAR_Z }}
      >
        {["minimize", "maximize", "close"].map((name) => (
          <button
            key={name}
            type="button"
            data-testid={`control-${name}`}
            aria-label={name}
            className="relative z-[80] inline-flex size-[30px] items-center justify-center rounded-[10px]"
          />
        ))}
      </div>
    </header>
  );
}

function Harness() {
  return (
    <>
      <Obstacles />
      {/* Header before grips, as window-titlebar.tsx renders them. Not cosmetic: equal
          z-indexes are resolved by document order, so swapping these two hands the corner
          to whichever comes last and the pre-PR control run measures the wrong titlebar. */}
      <WindowControls />
      <WindowResizeTargets />
      <Rail />
    </>
  );
}

window.__railSmoke = {
  setCards,
  publish: (who, frame) =>
    useMonitorFrameStore.getState().setFrame(publisherFor(who), frame),
  clear: (who) => {
    useMonitorFrameStore.getState().clearFrame(publisherFor(who));
    publishers.delete(who);
  },
  geometry: () => ({ ...lastGeometry }),
  gutter: () => ({
    top: STACK_SHADOW_GUTTER_TOP,
    bottom: STACK_SHADOW_GUTTER_BOTTOM,
  }),
  state: () => ({ cards, errors: [...seenErrors] }),
  errors: () => [...seenErrors],
};

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Root element not found");
const root = createRoot(rootElement);
// StrictMode attaches the ref twice and its cleanup tears the observers down, so the driver
// takes ?nostrict and never reads mid-remount.
const strict = !new URLSearchParams(window.location.search).has("nostrict");
const tree = <Harness />;
root.render(strict ? <StrictMode>{tree}</StrictMode> : tree);
