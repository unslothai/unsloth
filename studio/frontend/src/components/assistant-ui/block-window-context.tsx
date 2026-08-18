// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

/* eslint-disable react-refresh/only-export-components */

/**
 * The DOM and React half of the streaming reasoning pane's block window. The arithmetic lives in
 * block-window.ts, which imports neither.
 *
 * There are two contexts, and the split is not decorative. A reasoning GROUP can hold more than
 * one reasoning part, and each part renders its own <MarkdownText/>, i.e. its own Streamdown
 * document with its own block indices starting at 0. So:
 *
 *   BlockWindowPaneContext   supplied by ReasoningText, carries the scroll container. Present
 *                            only while the pane is streaming, which is also the only time the
 *                            pane is height-capped, so the window and the 256px cap are exactly
 *                            co-extensive.
 *   BlockWindowContext       supplied per Streamdown document by BlockWindowDocument, carries
 *                            that document's controller. Block components read this one.
 *
 * Without a pane context nothing is provided, `useBlockWindowMounted` says true, and the tree
 * is what it is today. The main answer's MarkdownText never sees a pane context.
 */

import {
  type ReactNode,
  type RefObject,
  createContext,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useState,
  useSyncExternalStore,
} from "react";

import { BlockWindowController } from "./block-window-controller";

const NO_UNSUBSCRIBE = (): void => {};

// ── contexts ────────────────────────────────────────────────────────

export type BlockWindowPane = {
  paneRef: RefObject<HTMLDivElement | null>;
};

const BlockWindowPaneContext = createContext<BlockWindowPane | null>(null);
const BlockWindowContext = createContext<BlockWindowController | null>(null);

/**
 * Marks the subtree as a windowable pane. Rendered by ReasoningText while the group streams, and
 * by nothing else, which is what keeps the answer text and settled panes out of scope.
 */
export function BlockWindowPaneProvider({
  paneRef,
  enabled,
  children,
}: {
  paneRef: RefObject<HTMLDivElement | null>;
  enabled: boolean;
  children: ReactNode;
}): ReactNode {
  const value = useMemo(
    () => (enabled ? { paneRef } : null),
    [enabled, paneRef],
  );
  return (
    <BlockWindowPaneContext.Provider value={value}>
      {children}
    </BlockWindowPaneContext.Provider>
  );
}

/**
 * Whether this subtree is inside a pane that windows its blocks. Read by MarkdownText to pick a
 * block component; a document that is not windowed renders exactly the tree it renders today.
 */
export function useBlockWindowPaneActive(): boolean {
  return useContext(BlockWindowPaneContext) !== null;
}

/**
 * One Streamdown document's window: owns the controller, renders the spacer, and is where the
 * scroll compensation runs, in a layout effect of the very commit that dropped the blocks.
 */
export function BlockWindowDocument({
  children,
}: {
  children: ReactNode;
}): ReactNode {
  const pane = useContext(BlockWindowPaneContext);
  // Lazy state rather than a ref: the controller has to survive re-renders and it is read during
  // render (the context value), which a ref may not be.
  const [ownController] = useState(() => new BlockWindowController());
  const controller = pane ? ownController : null;

  useEffect(() => {
    if (!(controller && pane)) {
      return;
    }
    return controller.attach(pane.paneRef.current);
  }, [controller, pane]);

  const subscribe = useMemo(
    () => controller?.subscribeDocument ?? (() => NO_UNSUBSCRIBE),
    [controller],
  );
  const getSpacerHeight = useCallback(
    () => controller?.spacerHeight() ?? 0,
    [controller],
  );
  const spacerHeight = useSyncExternalStore(
    subscribe,
    getSpacerHeight,
    getSpacerHeight,
  );

  // No dependency array: this has to run on the commit that moved the window, and that commit is
  // driven by the store subscription above rather than by a prop.
  useLayoutEffect(() => {
    controller?.settleAfterCommit();
  });

  if (!controller) {
    return children;
  }
  return (
    <BlockWindowContext.Provider value={controller}>
      <div
        aria-hidden="true"
        data-aui-block-window-spacer=""
        ref={controller.spacerRef}
        style={{ height: spacerHeight }}
      />
      {children}
    </BlockWindowContext.Provider>
  );
}

/**
 * Whether a Streamdown block belongs in the tree, and the block's report of its own content.
 *
 * Every block calls this, mounted or not: a withheld block still renders (it returns null), so the
 * whole document is observed even though only a suffix of it is in the DOM, which is what lets a
 * retroactive re-parse be noticed at all.
 *
 * Outside a windowed pane there is no controller and the answer is always true.
 */
export function useBlockWindowMounted(index: number, content: string): boolean {
  const controller = useContext(BlockWindowContext);

  const subscribe = useCallback(
    (onChange: () => void) =>
      controller ? controller.subscribeBlock(index, onChange) : NO_UNSUBSCRIBE,
    [controller, index],
  );
  const getMounted = useCallback(
    () => controller?.isMounted(index) ?? true,
    [controller, index],
  );
  const mounted = useSyncExternalStore(subscribe, getMounted, getMounted);

  useEffect(() => {
    controller?.reportContent(index, content);
  }, [controller, index, content]);

  return mounted;
}

/**
 * The callback ref for a mounted block's slot element.
 *
 * Returned rather than read: `react-hooks/refs` rejects a component that INSPECTS a ref during
 * render, so the caller hands this straight to `ref=` and never looks at it.
 */
export function useBlockWindowMarker(
  index: number,
): ((element: HTMLElement | null) => void) | null {
  const controller = useContext(BlockWindowContext);
  return controller ? controller.markerRef(index) : null;
}
