// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Copying a selection out of the thread, without paying for the styled clipboard flavour.
 *
 * Almost all of the cost of a copy on a long thread is the browser building the annotated
 * `text/html` flavour, not extracting text: on smoke-heavy-thread.html at the 100K rung a
 * 40,626-character selection costs 347.0ms to copy and 11.9ms to produce ourselves. So take the
 * event, write the plain text ourselves, and the styled flavour is never built.
 *
 * `Selection.toString()` IS NOT THE SAME STRING AS THE CLIPBOARD'S `text/plain`, and this file
 * exists because of that: Blink builds them with different `TextIteratorBehavior` flags, and
 * `WebViewImpl::ApplyWebPreferences` turns `SetSelectionIncludesAltImageText(true)` on for every
 * web view. Copying the thirty-one constructs in tests/studio/_thread_fast_copy_constructs.py one
 * at a time on Chromium and WebKit found exactly four divergences: img[alt] emits alt text (both,
 * REPRODUCED), U+00A0 becomes a space (both, REPRODUCED), text-transform is ignored on chromium
 * and applied on webkit (REPRODUCED for chromium), and a form control's value is emitted as its
 * own block on chromium and omitted on webkit (REFUSED). Everything else is identical because it
 * is the same iterator, so this does NOT reimplement the iterator: it patches the enumerated
 * deltas into the live DOM inside one synchronous turn of the copy event, asks the engine for its
 * own `toString()`, and puts everything back. Every semantic not on that list stays right by
 * construction, which a hand-written walker could not promise.
 *
 * ONE ENGINE FAMILY ONLY. WebKit's `toString()` appends trailing block breaks its clipboard does
 * not carry, and the count depends on what the selection ends with (+2 after a paragraph or
 * heading, +1 after a div, list, `<pre>` or blockquote, +0 after a table or an inline). Chromium's
 * delta is +0 in all eleven endings measured. A silently wrong clipboard is worse than a slow one,
 * so an unproven engine gets the browser's own copy. The check is a BEHAVIOURAL probe of
 * `toString()`, not a version test; the user agent narrows it further and never decides bytes.
 *
 * A FORM CONTROL IS REFUSED rather than reproduced, because chromium's emission carries a
 * control-dependent block break ("value" then a break for a text input, breaks on both sides for a
 * select). The 100K-rung viewport contains ZERO form controls, so the refusal costs nothing that
 * exists, and a password field would copy as its mask, where guessing the glyph wrong would put a
 * real password on the clipboard.
 *
 * NOT COVERED: a document-wide Ctrl+A, which targets the first sidebar row rather than the
 * viewport (measured in both engines) and would be refused anyway for spanning the composer's
 * textarea. Moving the listener to the document changes neither.
 *
 * NOT SERIALISED FROM THE MESSAGE STORE, which would be faster still but puts role headings,
 * thinking markers and tool lines on the clipboard that the DOM selection never contained. That
 * trade is worth making on a virtualized list, where the DOM cannot select an unmounted message;
 * on the shipped fully mounted list it only hands the user a different document than they
 * highlighted.
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

/**
 * A copy event, structurally. The GATE needs no DOM, so its branches can be unit tested against
 * plain objects. The SERIALISER below must be proven in a browser against a real clipboard, and is,
 * by tests/studio/playwright_thread_fast_copy.py: per construct, Chromium must match byte for byte
 * and WebKit must refuse with a divergence the same run measures.
 */
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

/**
 * A form control anywhere in the selected subtree. Chromium's clipboard reads its value and
 * wraps it in block breaks whose shape depends on the control, so the copy is handed back.
 */
const FORM_CONTROL = "input, textarea, select";

/** Where a copy must be left alone because the selection is not the document's. */
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
 * The smallest subtree the checks have to be right about: the range's common ancestor, not the
 * viewport. That difference is what makes this worth shipping, since one open `<textarea>`
 * anywhere would otherwise cost the fast path for the whole thread. The common ancestor is a
 * superset of the selected content, so scanning it can only refuse a copy the fast path could have
 * taken, never accept one it could not. A range ending in a text node has no `querySelector`, hence
 * the parent; anything unrecognisable falls back to the viewport, the conservative direction.
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
 * Does this engine's clipboard agree with its own `toString()` about trailing block breaks?
 * A hidden paragraph is selected and `toString()` asked whether it appends one: Chromium says "a",
 * WebKit says "a" plus two breaks. Cached, since the answer cannot change within a document.
 *
 * The restore goes through the serialiser's `restoreSelection` for the same reason it does there:
 * rebuilding from ranges alone flips a selection the user dragged upwards, so the text would be
 * right while their highlight silently reversed and the next Shift+Arrow moved the far end.
 * Reached on a plain right-to-left drag, and on the FIRST copy of a document.
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

/**
 * The two deltas that are reproduced, patched into the live DOM and taken straight back out.
 * Returns the undo list rather than cleaning up itself, so the caller can guarantee the restore
 * runs even if `toString()` throws. Nothing here paints: the whole sequence is one synchronous
 * turn of the copy event.
 */
/**
 * Put an element's `style` attribute back exactly as it was found, including not being there.
 * `style.removeProperty` is not the inverse of `style.setProperty` at the serialisation level,
 * which is the level a DOM comparison works at.
 */
/**
 * Would the clipboard's own iterator emit this image's alt text? `SkipsUnselectableContent` and
 * ordinary rendering rules mean an image that is not displayed, not visible or not selectable
 * contributes nothing. `visibility` and `user-select` inherit, so the computed style already
 * answers for an ancestor that set them.
 */
function nativeWouldEmitAlt(image: HTMLImageElement): boolean {
  const computed = getComputedStyle(image);
  if (computed.display === "none") return false;
  if (computed.visibility !== "visible") return false;
  const selectable = computed.userSelect ?? computed.webkitUserSelect;
  if (selectable === "none") return false;
  return true;
}

function restoreStyleAttribute(element: Element, had: string | null): void {
  if (had !== null) {
    element.setAttribute("style", had);
    return;
  }
  // `removeAttribute("style")` DOES NOT REMOVE IT once the inline declaration has been touched.
  // Measured in Chromium: setProperty then removeAttribute leaves `style=""` on the element and in
  // its serialisation, twice does not help, nor does clearing `cssText` first. Removing the
  // attribute NODE does work. The residue is invisible to a user, visible to every DOM comparison,
  // and permanent, so a thread accumulates one per element the fast path ever touched.
  try {
    element.attributes.removeNamedItem("style");
  } catch {
    element.removeAttribute("style");
  }
}

function patchClipboardDeltas(root: Element): Array<() => void> {
  const undo: Array<() => void> = [];

  // IgnoresCssTextTransforms: the clipboard carries the SOURCE text. ROOT ITSELF, not only its
  // descendants. A selection lying entirely inside `<span class="uppercase">text</span>` has that
  // span as its common ancestor and `querySelectorAll("*")` excludes the element it is called on,
  // so the TRANSFORMED text was written where the clipboard carries the source (the INSIDE_SCOPE
  // cases in tests/studio/_thread_fast_copy_constructs.py all failed). `text-transform` inherits,
  // so `getComputedStyle(root)` already reports a transform set on any ancestor above the scope.
  const scoped: HTMLElement[] = [];
  if (root instanceof HTMLElement) scoped.push(root);
  scoped.push(...Array.from(root.querySelectorAll<HTMLElement>("*")));
  for (const element of scoped) {
    const transform = getComputedStyle(element).textTransform;
    if (!TRANSFORMED.has(transform)) continue;
    // THE RAW ATTRIBUTE, not the property. `style.removeProperty` restores the computed value and NOT
    // the serialisation: an element that had no `style` attribute is left carrying `style=""`, and one
    // that had `style="text-transform:uppercase"` gets it rewritten with a space and a semicolon.
    // Invisible to a user, visible to any DOM comparison, so every copy silently rewrote the document.
    const had = element.getAttribute("style");
    element.style.setProperty("text-transform", "none", "important");
    undo.push(() => restoreStyleAttribute(element, had));
  }

  // EmitsImageAltText: the clipboard carries the alt text, which is not in the DOM as text. THE
  // HOLDER MUST NOT HAVE A BOX. Unsloth's message images are display:block, so an inline holder
  // beside one sits between two blocks, the engine wraps it in an anonymous block, and the alt text
  // arrives with a leading newline the real clipboard does not have (measured on the real thread as
  // 40,650 characters against 40,648, two images each contributing one extra break). Taking the
  // image out of the flow removes the box the break came from, and an image contributes no text of
  // its own, so hiding it changes nothing else.
  for (const image of Array.from(
    root.querySelectorAll<HTMLImageElement>("img[alt]"),
  )) {
    const alt = image.getAttribute("alt");
    if (!alt) continue;
    // ONLY AN IMAGE THE NATIVE ITERATOR WOULD EMIT. Chromium skips an image that is not rendered or
    // not selectable, so inserting alt text unconditionally ADDS text the clipboard never carried:
    // `display: none`, `visibility: hidden`, `user-select: none` and Unsloth's own `ImagePreview`,
    // which carries an `invisible` class until the image loads, all diverge.
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
 * The string the browser would have put on the clipboard, produced without building the styled
 * flavour. Proven byte-for-byte against Chromium's real clipboard over 31 selections (the 23
 * constructs it answers, two element-offset endpoints, three scoped to a transformed element and
 * three partials) and on the real thread at 40,648 characters. The selection is restored whatever
 * happens, because a copy that quietly moved the user's highlight would be the worse bug.
 */
/**
 * A selection's DIRECTION, and only its direction.
 *
 * Deliberately not the endpoints: the alt holders are inserted BEFORE their images, so an endpoint
 * expressed as an element/child offset in the same parent points at a different child afterwards.
 * Measured with two images and a trailing text node selected by parent offsets 0..3, the tail was
 * silently dropped, and because the result is non-empty the listener took the copy anyway. A
 * cloned `Range` is live and its boundaries move with the DOM, so the ranges already hold the
 * correct positions; only which end the user dragged from is missing from them.
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
 * Put the selection back the way it was, INCLUDING ITS DIRECTION.
 *
 * A `Range` carries ordered boundaries and nothing else, so `addRange` always produces a forward
 * selection: a selection dragged upwards came back with anchor and focus swapped, so the user's
 * next Shift+Arrow moved the opposite edge. Only the patched path rebuilds the selection, which is
 * why this never showed on ordinary prose. `setBaseAndExtent` preserves direction but takes one
 * anchor/focus pair, so a multi-range selection still falls back to `addRange`; only Firefox
 * produces those, and this fast path does not run there.
 */
function restoreSelection(
  selection: Selection,
  saved: readonly Range[],
  direction: SelectionDirection,
): void {
  selection.removeAllRanges();
  if (saved.length === 1) {
    // The LIVE range's boundaries, which the holder insertions have already adjusted, ordered
    // by the direction the user dragged in.
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

/**
 * Should this copy be answered by us instead of the browser, and if not, why not? Pure, and every
 * rejection carries its reason, because each branch is a distinct way of getting somebody's copy
 * wrong and "the fast path did not run" cannot tell them apart.
 */
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

  // A copy out of a text control is that control's own selection, and `window.getSelection()` is not
  // it: the document selection is usually collapsed or stale while a textarea has focus, so
  // substituting it replaces the copied field with unrelated text. The thread viewport contains these
  // (a message being edited mounts a textarea), so this cannot be left to the listener's placement.
  if (matchesAncestor(event.target, EDITABLE_ORIGIN)) {
    return { kind: "native", reason: "editable-origin" };
  }

  // A caret rather than a selection. The browser copies nothing; so should we.
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return { kind: "native", reason: "empty-selection" };
  }

  // A selection that starts inside the thread and runs out of it has a common ancestor above the
  // viewport. Its text is no longer something this file has proven.
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

  // Last, because it is the only branch that touches the document, and every cheaper refusal
  // above should have run first.
  if (!engineIsMapped) {
    return { kind: "native", reason: "unmapped-engine" };
  }

  return { kind: "fast" };
}

/**
 * Wire the decision to a live viewport. Returns the detach function. Only `copy` is listened for:
 * a `cut` must also mutate the document it cut from, the thread is not editable, and one inside a
 * message being edited belongs to that textarea.
 */
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
      // The patch could not be applied or undone cleanly. Let the browser copy: slow and right
      // beats fast and silently different.
      return;
    }
    // A selection can be non-collapsed and still serialise to nothing (an image with an empty
    // alt on its own). Writing "" would clear a clipboard the browser would have left alone.
    if (text === "") return;

    event.preventDefault();
    event.clipboardData?.setData("text/plain", text);
  };

  viewport.addEventListener("copy", onCopy);
  return () => viewport.removeEventListener("copy", onCopy);
}

/** The element form of `scopeOf`, for the patching path, which needs a real `Element`. */
function scopeElement(selection: Selection, viewport: HTMLElement): Element {
  if (selection.rangeCount !== 1) return viewport;
  const container = selection.getRangeAt(0).commonAncestorContainer;
  if (container.nodeType === 1) return container as Element;
  return container.parentElement ?? viewport;
}
