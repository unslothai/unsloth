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

  const activate = useCallback((pointer: { x: number; y: number } | null) => {
    if (activeRef.current) return;
    activeRef.current = true;
    const shell = shellRef.current;
    const focused = document.activeElement;
    pending.current = {
      focusPath:
        shell && focused instanceof HTMLElement && focused !== shell && shell.contains(focused)
          ? childIndexPath(shell, focused)
          : null,
      pointer,
    };
    setActive(true);
  }, []);

  const onPointerEnter = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      // A touch "enter" is the first half of a tap: swapping the subtree under the finger would
      // move the click's target out from under it. Those devices are opted out above anyway; this
      // is the belt for a hybrid whose primary pointer is a mouse.
      if (event.pointerType === "touch") return;
      activate({ x: event.clientX, y: event.clientY });
    },
    [activate],
  );

  const onFocusCapture = useCallback(() => {
    activate(null);
  }, [activate]);

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
    if (replay.pointer) {
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
      onFocusCapture={onFocusCapture}
    >
      <RowActiveContext.Provider value={active}>{children}</RowActiveContext.Provider>
    </div>
  );
}
