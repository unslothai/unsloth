// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether a tooltip trigger sits below the active modal. Radix writes inline pointer-events on
 * layers; the nearest ancestor with one owns the trigger. The trigger's own style is skipped.
 */
export function isBlockedByActiveModal(element: HTMLElement): boolean {
  for (let node = element.parentElement; node; node = node.parentElement) {
    const pointerEvents = node.style?.pointerEvents;
    if (pointerEvents === "auto") return false;
    if (pointerEvents === "none") return true;
  }
  return false;
}

// While a modal is up a hovered trigger never gets pointerleave, so tooltips must close. Two
// observers: one cheap body attribute check, and a subtree one only while a modal is up.
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
    // Live property, not getAttribute, which serialises the whole (often animated) style.
    const current = (record.target as HTMLElement).style?.pointerEvents ?? "";
    // Styles that never named pointer-events are skipped (every animation frame and resize drag).
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
  // Fresh identity per subscription so shared callbacks do not collapse in the Set.
  const subscription = () => listener();
  modalLayerListeners.add(subscription);
  // Both checks: a duplicate listener would otherwise orphan a body observer and leak.
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
    // Nothing watched the body while there were no listeners, so the flag needs this read.
    readModalLayer();
  }
  return () => {
    modalLayerListeners.delete(subscription);
    if (modalLayerListeners.size > 0) return;
    // No reader left: disconnect, or the subtree observer runs with nobody to notify.
    bodyLayerObserver?.disconnect();
    bodyLayerObserver = null;
    modalLayerUp = false;
    syncStackedLayerObserver();
  };
}

export function getModalLayer(): boolean {
  return modalLayerUp;
}
