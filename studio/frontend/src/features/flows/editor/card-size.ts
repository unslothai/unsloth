// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeSpec } from "../types";

export const CARD_MIN_WIDTH = 220;

/** Smallest card height that still shows the header and every port row. */
export function cardMinHeight(spec: Pick<NodeSpec, "inputs" | "outputs" | "flow_out">): number {
  const rows = Math.max(
    spec.inputs.length,
    spec.outputs.length + spec.flow_out.length,
  );
  return 56 + rows * 24 + 16;
}

/** Room the output preview keeps when a card is pulled down to its minimum. */
const GROW_MIN = 24;

type StyleOf = (el: Element) => Pick<
  CSSStyleDeclaration,
  "position" | "rowGap" | "paddingTop" | "paddingBottom"
>;

/** Height a card needs to show its header, ports, fields and errors. Only the output preview
 *  (`data-card-grow`) gives way; it clips instead. Absolutely placed children (resize lines,
 *  progress bar) take no room. */
export function cardContentMinHeight(
  card: HTMLElement,
  styleOf: StyleOf = (el) => getComputedStyle(el),
): number {
  const px = (v: string) => Number.parseFloat(v) || 0;
  let total = card.offsetHeight - card.clientHeight;
  for (const el of Array.from(card.children) as HTMLElement[]) {
    const style = styleOf(el);
    if (style.position === "absolute") continue;
    if (!el.hasAttribute("data-card-body")) {
      total += el.offsetHeight;
      continue;
    }
    const kids = Array.from(el.children) as HTMLElement[];
    total +=
      px(style.paddingTop) +
      px(style.paddingBottom) +
      px(style.rowGap) * Math.max(0, kids.length - 1);
    for (const kid of kids)
      total += kid.hasAttribute("data-card-grow") ? GROW_MIN : kid.offsetHeight;
  }
  return Math.ceil(total);
}
