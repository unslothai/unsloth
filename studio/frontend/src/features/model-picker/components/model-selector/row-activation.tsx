// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-row activation for the On Device list.
//
// The list renders every cached repo, and each row carried three Radix tooltips and a Radix
// dropdown whether or not the pointer was anywhere near it. None of those four are VISIBLE until
// the row is hovered or focused -- a tooltip has nothing on screen while it is closed, and the row
// action buttons sit under `opacity-0` until `group-hover` / `focus-within` -- so mounting them for
// a thousand rows buys nothing and costs a Radix Root, a Presence state machine and a portal each.
//
// A row wrapped in `ModelRowShell` therefore starts INACTIVE: it renders the same markup with the
// tooltip wrappers left off and the action buttons replaced by identical, focusable placeholders.
// The first pointer or focus that reaches the row switches it to ACTIVE, which mounts exactly what
// the merge base always had. Everything outside such a shell reads the context default, `true`, so
// no other list in the app changes at all.
//
// Two things have to survive the swap, and both are replayed in a layout effect so nothing paints
// in between:
//   1. FOCUS. Activating replaces the focused element (a placeholder, or the row button itself once
//      its tooltip wrapper appears), which would otherwise drop focus to <body> mid-Tab.
//   2. HOVER. Radix's tooltip trigger opens on `pointermove`, not on `pointerenter`. A pointer that
//      enters a row and then holds still fires no further move, so the freshly mounted trigger
//      would never learn the pointer is on it and the tooltip would never open. One synthetic
//      `pointermove` at the recorded position, only if that position is still inside this row,
//      hands it the event it missed.
//
// Coarse-pointer devices opt out entirely (`active` starts true). They show the row actions at all
// times (`[@media(hover:none)]:opacity-100`), a tap is a pointerdown and a click on the SAME node,
// and Radix's tap-to-pin tooltip behaviour has no hover to fall back on -- so the safe answer there
// is to change nothing.

import { createContext, useCallback, useContext, useLayoutEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, ReactNode } from "react";

/** True = mount everything, which is what every row outside a shell does. */
const RowActiveContext = createContext(true);

/** Whether this row has been reached by a pointer or by focus yet. */
export function useRowActive(): boolean {
  return useContext(RowActiveContext);
}

/** Hover is what the deferral trades on, so a device without it keeps the merge base's tree. */
function pointerIsCoarse(): boolean {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
  try {
    return (
      window.matchMedia("(hover: none)").matches ||
      window.matchMedia("(pointer: coarse)").matches
    );
  } catch {
    // A matchMedia that throws on an unsupported query (older WebViews) must not take the list
    // with it; erring towards "not coarse" only costs the deferral, never correctness.
    return false;
  }
}

/** Child index chain from `root` down to `node`, so the same slot can be found again after the
 *  subtree is rebuilt. The two trees are structurally identical -- `asChild` triggers add no
 *  elements, and the placeholders are the same tags in the same order -- so the path still lands
 *  on the element that had focus. */
function childIndexPath(root: Element, node: Element): number[] | null {
  const path: number[] = [];
  let current: Element | null = node;
  while (current && current !== root) {
    const parent: Element | null = current.parentElement;
    if (!parent) return null;
    path.push(Array.prototype.indexOf.call(parent.children, current));
    current = parent;
  }
  return current === root ? path.reverse() : null;
}

function elementAtPath(root: Element, path: number[]): HTMLElement | null {
  let current: Element | undefined = root;
  for (const index of path) {
    current = current?.children[index];
    if (!current) return null;
  }
  return current instanceof HTMLElement ? current : null;
}

type PendingReplay = {
  focusPath: number[] | null;
  pointer: { x: number; y: number } | null;
};

export function ModelRowShell({
  className,
  children,
}: {
  className?: string;
  children: ReactNode;
}) {
  const shellRef = useRef<HTMLDivElement | null>(null);
  // Read once per row and kept in state: a media query evaluated in the render body would be a new
  // answer on every render, and the point of this flag is that it never changes under the row.
  const [active, setActive] = useState(pointerIsCoarse);
  // The state flip is async from the handler's point of view, so `active` is not readable there.
  const activeRef = useRef(active);
  const pending = useRef<PendingReplay | null>(null);
  // A press is in progress in this row, so the swap has to wait: see `applyPending`.
  const pressed = useRef(false);
  const frame = useRef(0);

  const applyPending = useCallback(() => {
    if (activeRef.current || !pending.current || pressed.current) return;
    activeRef.current = true;
    setActive(true);
  }, []);

  /**
   * Record what has to be replayed, then hand the swap to the next frame.
   *
   * NOT immediately, and this is the whole reason this function is shaped like this. A click is
   * one gesture -- move, down, up -- and the browser fires the click on the nearest common
   * ancestor of the down and up targets. Activating replaces the row's button, so a swap that
   * lands between the down and the up leaves those on two different elements and NO click is
   * fired at all: the row silently fails to select. Measured on the smoke harness with a single
   * `mouse.click`, which is exactly that gesture with nothing in between: the merge base selected
   * the model and closed the panel, an eagerly swapping build did neither.
   *
   * `flushSync` looks like the fix and is not: `pointerenter` can be dispatched while React is
   * already committing (the DOM moving under a still pointer is enough), and React then refuses
   * the flush, logs, and schedules the update anyway -- so the race is still there, now with a
   * console error. A frame is late enough to be outside the current event and early enough that
   * nothing can be seen; if a press arrives first, the swap waits for its click.
   */
  const armActivation = useCallback(
    (pointer: { x: number; y: number } | null) => {
      if (activeRef.current || pending.current) return;
      const shell = shellRef.current;
      const focused = document.activeElement;
      pending.current = {
        focusPath:
          shell && focused instanceof HTMLElement && focused !== shell && shell.contains(focused)
            ? childIndexPath(shell, focused)
            : null,
        pointer,
      };
      if (pointer === null) {
        // Focus is not a two-part gesture: a keyboard user who tabs in has already arrived, and
        // waiting a frame would let a Tab straight through the row leave it inert.
        applyPending();
        return;
      }
      cancelAnimationFrame(frame.current);
      frame.current = requestAnimationFrame(applyPending);
    },
    [applyPending],
  );

  const onPointerEnter = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      // A touch "enter" is the first half of a tap: swapping the subtree under the finger would
      // move the click's target out from under it. Those devices are opted out above anyway; this
      // is the belt for a hybrid whose primary pointer is a mouse.
      if (event.pointerType === "touch") return;
      armActivation({ x: event.clientX, y: event.clientY });
    },
    [armActivation],
  );

  // Capture, because the row's own action buttons stop propagation: a bubbling listener would
  // never learn that the dots button had been pressed.
  const onPointerDownCapture = useCallback(() => {
    pressed.current = true;
  }, []);

  // On the NEXT frame, not from inside this handler. React collects a click's listener path once,
  // at the start of the dispatch, and skips the ones whose instance has been unmounted by the time
  // it gets to them -- so a swap applied in the capture phase silently eats the row button's own
  // onClick in the bubble phase. Measured: the click landed on the row (capture saw it, the DOM
  // swap was recorded after it) and the model was still not selected. Deferring to a frame puts
  // the swap outside the whole dispatch.
  const onClickCapture = useCallback(() => {
    pressed.current = false;
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(applyPending);
  }, [applyPending]);

  // A press that leaves the row will not produce a click here, so release the hold: the click the
  // browser does fire lands on a common ancestor outside this row either way.
  const onPointerLeave = useCallback(() => {
    pressed.current = false;
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(applyPending);
  }, [applyPending]);

  const onFocusCapture = useCallback(() => {
    armActivation(null);
  }, [armActivation]);

  useLayoutEffect(() => () => cancelAnimationFrame(frame.current), []);

  useLayoutEffect(() => {
    const replay = pending.current;
    pending.current = null;
    const shell = shellRef.current;
    if (!active || !replay || !shell) return;
    if (replay.focusPath) {
      const target = elementAtPath(shell, replay.focusPath);
      // Only when focus actually fell out of the row: a browser that kept it (or a user who moved
      // it on already) must not be yanked back.
      if (target && !shell.contains(document.activeElement)) {
        target.focus({ preventScroll: true });
      }
    }
    // No constructible PointerEvent (an old WebView) only costs the replay: the row is mounted
    // either way and the next real move opens the tooltip.
    if (replay.pointer && typeof PointerEvent === "function") {
      const under = document.elementFromPoint(replay.pointer.x, replay.pointer.y);
      // Stale coordinates are worse than no replay: if the pointer has already left this row, the
      // event would announce a hover that is not happening.
      if (under && shell.contains(under)) {
        under.dispatchEvent(
          new PointerEvent("pointermove", {
            bubbles: true,
            cancelable: true,
            composed: true,
            clientX: replay.pointer.x,
            clientY: replay.pointer.y,
            pointerType: "mouse",
          }),
        );
      }
    }
  }, [active]);

  return (
    <div
      ref={shellRef}
      className={className}
      onPointerEnter={onPointerEnter}
      onPointerDownCapture={onPointerDownCapture}
      onClickCapture={onClickCapture}
      onPointerLeave={onPointerLeave}
      onFocusCapture={onFocusCapture}
    >
      <RowActiveContext.Provider value={active}>{children}</RowActiveContext.Provider>
    </div>
  );
}
