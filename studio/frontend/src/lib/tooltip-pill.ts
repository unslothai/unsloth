// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A one-line tooltip is a full pill; a wrapped one keeps the squarer corners so it does not read as
 *  a capsule. Measured, since the same surface holds one word or a paragraph. */
export function roundIfSingleLine(el: HTMLElement | null): void {
  if (!el) return;
  const cs = getComputedStyle(el);
  const lineHeight = Number.parseFloat(cs.lineHeight) || 16;
  const inner =
    el.clientHeight - Number.parseFloat(cs.paddingTop) - Number.parseFloat(cs.paddingBottom);
  el.classList.toggle("rounded-full!", inner < lineHeight * 1.5);
}
