// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Copy handler that writes text/plain itself, skipping the costly styled text/html flavour.
 * `Selection.toString()` differs from the clipboard's text/plain, so the known deltas (img alt,
 * U+00A0, text-transform) are patched into the live DOM for one synchronous turn, the engine's own
 * `toString()` is read, and everything is restored. Only engines whose `toString()` is proven to
 * match (Chromium) take this path; form controls are refused, since a password could leak.
 */

/** Why a copy was left to the browser. Named so a test can assert the reason, not just the miss. */
export type NativeCopyReason =
  | "already-handled"
  | "no-clipboard-data"
  | "editable-origin"
  | "empty-selection"
  | "selection-leaves-thread"
  | "form-control"
  | "unmapped-engine";

export type ThreadCopyDecision =
  | { readonly kind: "fast" }
  | { readonly kind: "native"; readonly reason: NativeCopyReason };

/** Structural copy event so the gate can be unit tested without a DOM. */
export type CopyEventLike = {
  readonly defaultPrevented: boolean;
  readonly target: unknown;
  readonly clipboardData: {
    setData(format: string, data: string): void;
  } | null;
};

export type SelectionLike = {
  readonly isCollapsed: boolean;
  readonly rangeCount: number;
  getRangeAt(index: number): { readonly commonAncestorContainer: unknown };
  toString(): string;
};

export type ThreadViewportLike = {
  contains(node: unknown): boolean;
  querySelector(selectors: string): unknown;
};

/** Chromium wraps a control's value in control-dependent block breaks, so the copy is refused. */
const FORM_CONTROL = "input, textarea, select";

const EDITABLE_ORIGIN =
  'input, textarea, select, [contenteditable=""], [contenteditable="true"], [contenteditable="plaintext-only"]';

const TRANSFORMED = new Set(["uppercase", "lowercase", "capitalize"]);

function matchesAncestor(target: unknown, selectors: string): boolean {
  const closest = (
    target as { closest?: (selectors: string) => unknown } | null
  )?.closest;
  if (typeof closest !== "function") return false;
  return closest.call(target, selectors) != null;
}

/**
 * The range's common ancestor, not the viewport, so one open textarea elsewhere does not refuse
 * every copy. A superset of the selection, so it can only over-refuse.
 */
function scopeOf(
  selection: SelectionLike,
  viewport: ThreadViewportLike,
): Pick<ThreadViewportLike, "querySelector"> {
  // Several disjoint ranges have no single ancestor short of the viewport.
  if (selection.rangeCount !== 1) return viewport;

  const container = selection.getRangeAt(0).commonAncestorContainer as {
    querySelector?: unknown;
    parentElement?: { querySelector?: unknown } | null;
  } | null;

  if (typeof container?.querySelector === "function") {
    return container as Pick<ThreadViewportLike, "querySelector">;
  }
  const parent = container?.parentElement;
  if (typeof parent?.querySelector === "function") {
    return parent as Pick<ThreadViewportLike, "querySelector">;
  }
  return viewport;
}

/**
 * Behavioural probe: does `toString()` append trailing block breaks the clipboard lacks (WebKit)?
 * Cached per document. Restores via `restoreSelection` to keep an upward drag's direction.
 */
export function engineClipboardIsMapped(
  view: Window & typeof globalThis,
): boolean {
  const cache = view as { __sbFastCopyMapped?: boolean };
  if (typeof cache.__sbFastCopyMapped === "boolean") {
    return cache.__sbFastCopyMapped;
  }
  let mapped = false;
  try {
    const ua = view.navigator?.userAgent ?? "";
    if (
      /Chrome\/|Chromium\/|Edg\//.test(ua) &&
      !/\bAppleWebKit\b(?!.*Chrome)/.test(ua)
    ) {
      const doc = view.document;
      const probe = doc.createElement("div");
      probe.setAttribute("aria-hidden", "true");
      probe.style.cssText =
        "position:fixed;left:-9999px;top:0;width:1px;height:1px;overflow:hidden";
      probe.innerHTML = "<p>a</p>";
      doc.body.appendChild(probe);
      const selection = view.getSelection();
      const saved: Range[] = [];
      if (selection) {
        for (let index = 0; index < selection.rangeCount; index += 1) {
          saved.push(selection.getRangeAt(index).cloneRange());
        }
        // Before the probe replaces it, because that is the only moment the direction exists.
        const direction = captureDirection(selection);
        selection.selectAllChildren(probe);
        mapped = selection.toString() === "a";
        restoreSelection(selection, saved, direction);
      }
      probe.remove();
    }
  } catch {
    mapped = false;
  }
  cache.__sbFastCopyMapped = mapped;
  return mapped;
}

/** Hidden, invisible or unselectable images contribute no alt text natively. */
function nativeWouldEmitAlt(image: HTMLImageElement): boolean {
  const computed = getComputedStyle(image);
  if (computed.display === "none") return false;
  if (computed.visibility !== "visible") return false;
  const selectable = computed.userSelect ?? computed.webkitUserSelect;
  if (selectable === "none") return false;
  return true;
}

/** `style.removeProperty` does not restore the serialised attribute. */
function restoreStyleAttribute(element: Element, had: string | null): void {
  if (had !== null) {
    element.setAttribute("style", had);
    return;
  }
  // `removeAttribute("style")` leaves `style=""` once the declaration was touched (Chromium);
  // removing the attribute NODE works.
  try {
    element.attributes.removeNamedItem("style");
  } catch {
    element.removeAttribute("style");
  }
}

/** Returns the undo list so the caller can guarantee the restore runs even if `toString()` throws. */
function patchClipboardDeltas(root: Element): Array<() => void> {
  const undo: Array<() => void> = [];

  // The clipboard carries SOURCE text. Include the root itself: querySelectorAll("*") excludes it,
  // and text-transform inherits, so getComputedStyle(root) covers ancestors.
  const scoped: HTMLElement[] = [];
  if (root instanceof HTMLElement) scoped.push(root);
  scoped.push(...Array.from(root.querySelectorAll<HTMLElement>("*")));
  for (const element of scoped) {
    const transform = getComputedStyle(element).textTransform;
    if (!TRANSFORMED.has(transform)) continue;
    // Restore the raw attribute: removeProperty leaves `style=""` or rewrites the serialisation.
    const had = element.getAttribute("style");
    element.style.setProperty("text-transform", "none", "important");
    undo.push(() => restoreStyleAttribute(element, had));
  }

  // The alt holder must not have a box: next to a block image it gets an anonymous block and a
  // stray newline. Hiding the image removes it, and the image has no text of its own.
  for (const image of Array.from(
    root.querySelectorAll<HTMLImageElement>("img[alt]"),
  )) {
    const alt = image.getAttribute("alt");
    if (!alt) continue;
    // Only images the native iterator would emit, e.g. not ImagePreview's `invisible` loading state.
    if (!nativeWouldEmitAlt(image)) continue;
    const had = image.getAttribute("style");
    image.style.setProperty("display", "none", "important");
    const holder = image.ownerDocument.createElement("span");
    holder.textContent = alt;
    image.parentNode?.insertBefore(holder, image);
    undo.push(() => {
      holder.remove();
      restoreStyleAttribute(image, had);
    });
  }

  return undo;
}

/**
 * Only the direction: holders are inserted before images, so element-offset endpoints would
 * shift, while cloned live Ranges already track the DOM.
 */
type SelectionDirection = { readonly backward: boolean };

function captureDirection(selection: Selection): SelectionDirection {
  const { anchorNode, anchorOffset, focusNode, focusOffset } = selection;
  if (!anchorNode || !focusNode) return { backward: false };
  try {
    const probe = anchorNode.ownerDocument?.createRange();
    if (!probe) return { backward: false };
    probe.setStart(anchorNode, anchorOffset);
    probe.setEnd(anchorNode, anchorOffset);
    // -1 means the focus lies before the anchor, which is a selection dragged upwards.
    return { backward: probe.comparePoint(focusNode, focusOffset) < 0 };
  } catch {
    // Different trees, or a detached node. Forward is the safe assumption.
    return { backward: false };
  }
}

/**
 * Restore the selection including its direction: `addRange` always yields a forward selection.
 * Multi-range selections (Firefox only) still use `addRange`; this path does not run there.
 */
function restoreSelection(
  selection: Selection,
  saved: readonly Range[],
  direction: SelectionDirection,
): void {
  selection.removeAllRanges();
  if (saved.length === 1) {
    // The live range, already adjusted by holder insertions, ordered by drag direction.
    const range = saved[0];
    try {
      if (direction.backward) {
        selection.setBaseAndExtent(
          range.endContainer,
          range.endOffset,
          range.startContainer,
          range.startOffset,
        );
      } else {
        selection.setBaseAndExtent(
          range.startContainer,
          range.startOffset,
          range.endContainer,
          range.endOffset,
        );
      }
      return;
    } catch {
      // A detached node is better answered with the range than with nothing at all.
    }
  }
  for (const range of saved) selection.addRange(range);
}

/** The string the browser would have copied. The selection is restored whatever happens. */
export function faithfulSelectionText(
  selection: Selection,
  root: Element,
): string {
  const saved: Range[] = [];
  for (let index = 0; index < selection.rangeCount; index += 1) {
    saved.push(selection.getRangeAt(index).cloneRange());
  }
  const direction = captureDirection(selection);
  const undo = patchClipboardDeltas(root);
  let raw: string;
  try {
    if (undo.length > 0) restoreSelection(selection, saved, direction);
    raw = selection.toString();
  } finally {
    for (let index = undo.length - 1; index >= 0; index -= 1) undo[index]();
    if (undo.length > 0) restoreSelection(selection, saved, direction);
  }
  // Both engines' clipboards fold a no-break space to a plain one; neither `toString()` does.
  return raw.replace(/\u00a0/g, " ");
}

/** Pure; every rejection carries its reason so tests can tell the failure modes apart. */
export function decideThreadCopy(
  event: CopyEventLike,
  selection: SelectionLike | null,
  viewport: ThreadViewportLike,
  engineIsMapped = true,
): ThreadCopyDecision {
  // Somebody upstream already produced this clipboard payload. Do not overwrite it.
  if (event.defaultPrevented) {
    return { kind: "native", reason: "already-handled" };
  }

  // No transfer to write into, so preventing the default would copy nothing at all.
  if (!event.clipboardData) {
    return { kind: "native", reason: "no-clipboard-data" };
  }

  // A copy out of a text control is the control's own selection; window.getSelection() is stale.
  if (matchesAncestor(event.target, EDITABLE_ORIGIN)) {
    return { kind: "native", reason: "editable-origin" };
  }

  // A caret rather than a selection. The browser copies nothing; so should we.
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return { kind: "native", reason: "empty-selection" };
  }

  // A selection leaving the thread has an ancestor above the viewport; its text is unproven.
  for (let index = 0; index < selection.rangeCount; index += 1) {
    if (
      !viewport.contains(selection.getRangeAt(index).commonAncestorContainer)
    ) {
      return { kind: "native", reason: "selection-leaves-thread" };
    }
  }

  if (scopeOf(selection, viewport).querySelector(FORM_CONTROL) != null) {
    return { kind: "native", reason: "form-control" };
  }

  // Last: the only branch that touches the document.
  if (!engineIsMapped) {
    return { kind: "native", reason: "unmapped-engine" };
  }

  return { kind: "fast" };
}

/** Only `copy`: a cut must mutate the document, and the thread is not editable. */
export function attachThreadFastCopy(viewport: HTMLElement): () => void {
  const onCopy = (event: ClipboardEvent) => {
    const view = viewport.ownerDocument.defaultView;
    if (!view) return;
    const selection = view.getSelection();
    const decision = decideThreadCopy(
      event,
      selection,
      viewport,
      engineClipboardIsMapped(view as Window & typeof globalThis),
    );
    if (decision.kind !== "fast") return;

    let text: string;
    try {
      text = faithfulSelectionText(
        selection as Selection,
        scopeElement(selection as Selection, viewport),
      );
    } catch {
      // The patch failed; let the browser copy (slow and right beats fast and wrong).
      return;
    }
    // A non-collapsed selection can serialise to ""; writing it would clear the clipboard.
    if (text === "") return;

    event.preventDefault();
    event.clipboardData?.setData("text/plain", text);
  };

  viewport.addEventListener("copy", onCopy);
  return () => viewport.removeEventListener("copy", onCopy);
}

function scopeElement(selection: Selection, viewport: HTMLElement): Element {
  if (selection.rangeCount !== 1) return viewport;
  const container = selection.getRangeAt(0).commonAncestorContainer;
  if (container.nodeType === 1) return container as Element;
  return container.parentElement ?? viewport;
}
