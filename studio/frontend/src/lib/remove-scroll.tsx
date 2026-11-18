// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Stands in for `react-remove-scroll` (aliased in vite.config.ts), the scroll lock Radix dialogs, modal
 * menus, selects and modal popovers mount while open.
 *
 * WebKit folds window and document wheel / touch listeners into the style of every element (event
 * listener regions), so the library's non-passive document listeners restyled the whole page on each
 * open and again on close: about 2.5 s per step on a 100-turn chat. Removing its injected <style> on
 * close rebuilt every style once more. Chromium pays neither.
 *
 * What the lock did here is kept: the `data-scroll-locked` counter on <body>, with the styles the
 * library injected now static in index.css. Its wheel and touch cancelling is dropped: under a modal
 * layer nothing outside it can be wheel-scrolled anyway (body is `pointer-events: none`, the thread
 * sits under a cover, the page itself never scrolls), and lists inside dialogs already scroll on
 * their own (`useWheelScrollRef`, popovers portaled into the dialog).
 */

import { forwardRef, type ReactElement, useEffect } from "react";
import type { IRemoveScrollProps } from "react-remove-scroll/dist/es2015/types";
import { RemoveScroll as RemoveScrollUI } from "react-remove-scroll/UI";

const LOCK_ATTRIBUTE = "data-scroll-locked";

/** Same counter as react-remove-scroll-bar, so nested and mixed locks agree on when it ends. */
function lockCount(): number {
  const count = Number.parseInt(document.body.getAttribute(LOCK_ATTRIBUTE) || "0", 10);
  return Number.isFinite(count) ? count : 0;
}

function ScrollLockSideCar({ removeScrollBar }: { removeScrollBar?: boolean }): ReactElement | null {
  useEffect(() => {
    if (removeScrollBar === false) return;
    document.body.setAttribute(LOCK_ATTRIBUTE, String(lockCount() + 1));
    return () => {
      const next = lockCount() - 1;
      if (next <= 0) document.body.removeAttribute(LOCK_ATTRIBUTE);
      else document.body.setAttribute(LOCK_ATTRIBUTE, String(next));
    };
  }, [removeScrollBar]);
  return null;
}

// The library's defaults, spelled out: JSX does not apply its `defaultProps` (React 19), and Radix
// passes none of these, so without them the lock would never engage.
const RemoveScrollLock = forwardRef<HTMLElement, IRemoveScrollProps>(function RemoveScroll(
  { enabled = true, removeScrollBar = true, inert = false, ...props },
  ref,
) {
  return (
    <RemoveScrollUI
      {...props}
      enabled={enabled}
      removeScrollBar={removeScrollBar}
      inert={inert}
      ref={ref}
      sideCar={ScrollLockSideCar}
    />
  );
});

export const RemoveScroll = Object.assign(RemoveScrollLock, {
  classNames: RemoveScrollUI.classNames,
});

export default RemoveScroll;
