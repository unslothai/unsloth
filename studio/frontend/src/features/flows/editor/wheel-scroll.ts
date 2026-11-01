// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type Overflow = Pick<CSSStyleDeclaration, "overflowX" | "overflowY">;
type Box = Pick<
  HTMLElement,
  "scrollTop" | "scrollLeft" | "scrollHeight" | "scrollWidth" | "clientHeight" | "clientWidth"
> & { parentElement: Box | null };

const SCROLLS = /auto|scroll|overlay/;

/** Whether a wheel over `target` should scroll a box inside the canvas rather than zoom it: only
 *  when some box between it and `canvas` can still move that way. A pinch (ctrl) always zooms. */
export function wheelScrollsInside(
  target: Box | null,
  canvas: Box,
  deltaX: number,
  deltaY: number,
  ctrlKey: boolean,
  styleOf: (el: Box) => Overflow = (el) => getComputedStyle(el as unknown as Element),
): boolean {
  if (ctrlKey) return false;
  for (let el = target; el && el !== canvas; el = el.parentElement) {
    const style = styleOf(el);
    if (deltaY !== 0 && SCROLLS.test(style.overflowY) && el.scrollHeight > el.clientHeight + 1) {
      if (deltaY < 0 ? el.scrollTop > 0 : el.scrollTop + el.clientHeight < el.scrollHeight - 1)
        return true;
    }
    if (deltaX !== 0 && SCROLLS.test(style.overflowX) && el.scrollWidth > el.clientWidth + 1) {
      if (deltaX < 0 ? el.scrollLeft > 0 : el.scrollLeft + el.clientWidth < el.scrollWidth - 1)
        return true;
    }
  }
  return false;
}
