// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

/**
 * The React half of the code-block layout hold. See ./code-block-layout.ts for what it decides
 * and why; this file only wires that decision to the thread root's DOM attribute.
 *
 * Separate file so the decision can be unit-tested without pulling assistant-ui and React into
 * the test runner, which strips types rather than bundling.
 *
 * A MutationObserver rather than an effect per path, and the timing that allows it
 * -------------------------------------------------------------------------------
 * The hold has to be taken back whenever code blocks are re-created on a thread that has
 * already been released, because a fresh element has no last remembered size and lays out at
 * streamdown's 200px `contain-intrinsic-size` fallback for one frame. Three paths do that with
 * `thread.isRunning` false throughout, and none of them is reported by any state this component
 * can subscribe to; a fourth is always possible. So the trigger is the thing itself: a node
 * carrying, or containing, streamdown's code-block marker being added under the thread root.
 *
 * The reason that is allowed to replace a `useLayoutEffect` is a timing property, and it was
 * measured rather than argued. The attribute has to be back on the root before the browser
 * paints the commit that created those elements, because that paint IS the flicker frame. A
 * MutationObserver callback is a microtask, and the microtask checkpoint at the end of a task
 * runs before the rendering update in which paint happens. Measured in Chromium 151 against the
 * same 2,169px block, sampling `offsetHeight` on each of the first frames after a fresh block
 * replaces a rendered one:
 *
 *   released, nothing takes the hold back   frame 0: 200px   frame 1+: 2,169px
 *   hold restored synchronously             frame 0: 2,169px               (the layout-effect shape)
 *   hold restored from a MutationObserver   frame 0: 2,169px
 *
 * and the observer result is the same whether the mutation is made from a plain task, from
 * inside a `requestAnimationFrame` callback, or from a microtask. So the observer lands on the
 * right side of paint in every context React can commit from, and the per-path effect it
 * replaces bought nothing the observer does not.
 */

import {
  CODE_BLOCK_LAYOUT_ATTRIBUTE,
  CODE_BLOCK_SELECTOR,
  type CodeBlockLayoutController,
  type CodeBlockRemountWatcher,
  createCodeBlockLayoutController,
  createCodeBlockRemountWatcher,
} from "@/components/assistant-ui/code-block-layout";
import { useAuiState } from "@assistant-ui/react";
import { type RefObject, useEffect, useRef } from "react";

/**
 * What the observer records.
 *
 * `childList` only, and deliberately no `attributes`: the attribute this component writes lives
 * on the very node being observed, so recording attributes would make the watcher's own effect
 * visible to itself. It is disconnected before that write happens either way; this is the belt
 * to that's braces.
 */
const OBSERVED: MutationObserverInit = { childList: true, subtree: true };

/**
 * Is this added node a code block, or does it contain one?
 *
 * The containment half is the whole point. React mounts a reply's rendered parts as one added
 * node with the blocks somewhere beneath it, so a check that only asked whether the added node
 * itself carries the marker would answer no on every path this exists to catch.
 */
function addsACodeBlock(node: Node): boolean {
  if (node.nodeType !== Node.ELEMENT_NODE) return false;
  const element = node as Element;
  return (
    element.matches(CODE_BLOCK_SELECTOR) ||
    element.querySelector(CODE_BLOCK_SELECTOR) !== null
  );
}

/**
 * Writes `data-code-block-layout` onto the thread root.
 *
 * Renders nothing. It is mounted inside ThreadPrimitive.Root and handed a ref to it, so the only
 * component that re-renders when the thread starts or stops running is this one. Putting the
 * attribute on the root as a rendered prop instead would subscribe the component that owns
 * ThreadPrimitive.Root to the run state, and that component is memoised precisely so a render of
 * it does not reconcile the whole message list.
 */
export function CodeBlockLayoutSignal({
  rootRef,
  settleMs,
}: {
  rootRef: RefObject<HTMLElement | null>;
  settleMs?: number;
}): null {
  const isRunning = useAuiState(({ thread }) => thread.isRunning);
  const controllerRef = useRef<CodeBlockLayoutController | null>(null);

  useEffect(() => {
    // Captured once. React clears refs before running an unmounting element's cleanup, so
    // reading rootRef.current in the cleanup below would find null and leave the attribute
    // saying "settled" on a root that no longer has anything driving it.
    const root = rootRef.current;

    // The general half of the hold: a run is not the only thing that mounts fresh code blocks,
    // and the other paths are not enumerable. See createCodeBlockRemountWatcher for the three
    // known ones and for why this watches the DOM instead of hooking each of them.
    //
    // `let` rather than an ordering that avoids it: the observer callback needs the watcher and
    // the watcher's connect needs the observer, and this is the cycle broken at the cheaper end.
    let watcher: CodeBlockRemountWatcher | null = null;
    const observer =
      root && typeof MutationObserver !== "undefined"
        ? new MutationObserver((records) => {
            watcher?.sawMutations(records, addsACodeBlock);
          })
        : null;

    const controller = createCodeBlockLayoutController({
      settleMs,
      onChange: (layout) => {
        root?.setAttribute(CODE_BLOCK_LAYOUT_ATTRIBUTE, layout);
        watcher?.layoutChanged(layout);
      },
    });

    watcher = createCodeBlockRemountWatcher({
      connect: () => {
        if (root) observer?.observe(root, OBSERVED);
      },
      disconnect: () => {
        observer?.disconnect();
      },
      onRemount: () => {
        controller.remeasure();
      },
    });

    controllerRef.current = controller;
    // Written before anything is armed, so the DOM never says something the controller does not.
    // Its absence already means held, so this is about legibility rather than correctness.
    root?.setAttribute(CODE_BLOCK_LAYOUT_ATTRIBUTE, controller.layout());
    watcher.layoutChanged(controller.layout());
    return () => {
      watcher?.dispose();
      watcher = null;
      controller.dispose();
      controllerRef.current = null;
      root?.setAttribute(CODE_BLOCK_LAYOUT_ATTRIBUTE, "building");
    };
  }, [rootRef, settleMs]);

  useEffect(() => {
    controllerRef.current?.setRunning(isRunning);
  }, [isRunning]);

  return null;
}
