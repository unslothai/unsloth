


/**
 * Whether a tooltip trigger sits below the active modal rather than inside it.
 *
 * Radix's DismissableLayer writes `pointer-events` inline: `none` on the body while a modal is up,
 * `auto` on the active layer, `none` on the layers under it. The nearest ancestor with one of
 * those owns the trigger.
 *
 * The trigger's own style is skipped on purpose: it can be authored `pointer-events: none` and
 * still belong to the modal, as the hint anchors in the MCP dropdown rows do. Read the trigger,
 * never the content: layers rank by mount order, so content opened after the modal reads `auto`.
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
// none while one is, which is also when a hovered trigger stops receiving pointerleave, so an
// open tooltip hangs over the dialog with nothing able to close it.
//
// Two observers, because the questions cost differently: "is a modal up" is one attribute on one
// node; "which layer owns a trigger" needs the whole subtree, and only matters while one is up.
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
    // The live property, not getAttribute: that serialises the whole declaration, and most
    // records landing here while a modal is up are inline styles being animated.
    const current = (record.target as HTMLElement).style?.pointerEvents ?? "";
    // A style that never named pointer-events cannot have changed it. That is every animation
    // frame, popper reposition and resize drag, and this test is why none reach the regex.
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
  // A fresh identity per subscription: the Set would otherwise collapse two subscribers sharing a
  // callback, and the first cleanup would tear the observers down under the second.
  const subscription = () => listener();
  modalLayerListeners.add(subscription);
  // Both conditions: without the size check a duplicate listener builds a second body observer
  // and orphans the first, which is the leak the teardown below exists for.
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
    // No reader left. A modal still up when the last tooltip unmounts would otherwise leave the
    // subtree observer running with nobody to notify. Clearing the flag before the sync drops
    // that observer and keeps `readModalLayer`'s early return honest for the next subscriber.
    bodyLayerObserver?.disconnect();
    bodyLayerObserver = null;
    modalLayerUp = false;
    syncStackedLayerObserver();
  };
}

export function getModalLayer(): boolean {
  return modalLayerUp;
}
