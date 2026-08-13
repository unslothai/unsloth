// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether a tooltip trigger sits below the active modal rather than inside it.
 *
 * Radix's DismissableLayer writes `pointer-events` inline: `none` on the body
 * while a modal is up, `auto` on the active layer, `none` on the layers under
 * it. Walking the ancestors for the nearest of those answers which layer owns
 * the trigger.
 *
 * The trigger's own style is deliberately skipped. A trigger can be authored
 * `pointer-events: none` and still belong to the modal, which is exactly what
 * the hint anchors inside the MCP dropdown rows do.
 *
 * Read the trigger, never the tooltip content: layers rank by mount order, so
 * content opened after the modal reads `auto` wherever its trigger sits.
 */
export function isBlockedByActiveModal(element: HTMLElement): boolean {
  for (let node = element.parentElement; node; node = node.parentElement) {
    const pointerEvents = node.style?.pointerEvents;
    if (pointerEvents === "auto") return false;
    if (pointerEvents === "none") return true;
  }
  return false;
}

// Whether a modal layer is up at all, shared by every tooltip. Radix sets body pointer-events to
// none while one is, which is also when a hovered trigger stops receiving pointerleave, so a
// tooltip already on screen hangs over the dialog with nothing able to close it.
//
// Two observers, because the two questions cost very different amounts. Whether a modal is up is
// one attribute on one node. Which layer owns a trigger needs the whole subtree, and only has an
// answer worth asking for while a modal is up.
let modalLayerUp = false;
const modalLayerListeners = new Set<() => void>();
let bodyLayerObserver: MutationObserver | null = null;
let stackedLayerObserver: MutationObserver | null = null;

function notifyModalLayer(): void {
  for (const listener of modalLayerListeners) listener();
}

function readPointerEvents(style: string | null): string {
  return (
    /(?:^|;)\s*pointer-events\s*:\s*([^;]+)/.exec(style ?? "")?.[1]?.trim() ??
    ""
  );
}

function readStackedLayerMutations(records: MutationRecord[]): void {
  for (const record of records) {
    const previous = record.oldValue;
    // The live property, not getAttribute: that serialises the whole declaration, and the
    // records that land here while a modal is up are mostly inline styles being animated.
    const current = (record.target as HTMLElement).style?.pointerEvents ?? "";
    // A style that did not name pointer-events and still does not cannot have changed it. That
    // is every frame of a motion/react animation, every popper reposition and every resize drag,
    // and this test is the whole reason none of them reach the regex.
    if (
      current === "" &&
      (previous === null || !previous.includes("pointer-events"))
    ) {
      continue;
    }
    if (readPointerEvents(previous) !== current) {
      notifyModalLayer();
      return;
    }
  }
}

// Stacking only matters while a modal is up; otherwise this fires on every animated inline style.
function syncStackedLayerObserver(): void {
  if (!modalLayerUp) {
    stackedLayerObserver?.disconnect();
    stackedLayerObserver = null;
    return;
  }
  if (stackedLayerObserver) return;
  stackedLayerObserver = new MutationObserver(readStackedLayerMutations);
  stackedLayerObserver.observe(document.body, {
    attributes: true,
    attributeFilter: ["style"],
    attributeOldValue: true,
    subtree: true,
  });
}

function readModalLayer(): void {
  const next = document.body.style.pointerEvents === "none";
  if (next === modalLayerUp) return;
  modalLayerUp = next;
  syncStackedLayerObserver();
  notifyModalLayer();
}

export function subscribeModalLayer(listener: () => void): () => void {
  // A fresh identity per subscription: the Set would otherwise collapse two subscribers that
  // share a callback, and the first cleanup would tear the observers down under the second.
  const subscription = () => listener();
  modalLayerListeners.add(subscription);
  // Both conditions: the set dedupes, so the same listener added twice would otherwise build a
  // second body observer and orphan the first, which is the leak the teardown below exists for.
  if (
    modalLayerListeners.size === 1 &&
    !bodyLayerObserver &&
    typeof MutationObserver !== "undefined"
  ) {
    bodyLayerObserver = new MutationObserver(readModalLayer);
    bodyLayerObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ["style"],
    });
    // Nothing watched the body while there were no listeners, so the flag is only as good as
    // this read.
    readModalLayer();
  }
  return () => {
    modalLayerListeners.delete(subscription);
    if (modalLayerListeners.size > 0) return;
    // No reader left. A modal that is up when the last tooltip unmounts would otherwise leave
    // the subtree observer running regexes with nobody to notify until the modal closed.
    // Clearing the flag before the sync is what drops that observer, and it is also what keeps
    // `readModalLayer`'s early return honest for the next subscriber: flag and observers are
    // only ever out of step between these two lines.
    bodyLayerObserver?.disconnect();
    bodyLayerObserver = null;
    modalLayerUp = false;
    syncStackedLayerObserver();
  };
}

export function getModalLayer(): boolean {
  return modalLayerUp;
}
