// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Copying a selection out of the thread, without paying for the styled clipboard flavour.
 *
 * WHERE THE TIME GOES. Almost all of the cost of a copy on a long thread is the browser building
 * the annotated `text/html` flavour, not extracting text. Measured on smoke-heavy-thread.html at
 * the 100K rung -- the real Thread, 14,137 elements in the viewport, Chromium, median of 5 --
 * a copy of the whole viewport takes 678ms, and writing `Selection.toString()` into the event
 * instead takes 1.3ms. So: take the event, write the plain text ourselves, and the styled flavour
 * is never built. The paste target that wanted styling is the one that pays for this, which is why
 * the substitution is only made when it is provably a no-op for plain text (below).
 *
 * `Selection.toString()` IS NOT THE SAME STRING AS THE CLIPBOARD'S `text/plain`. It is close, and
 * on ordinary prose it is identical, but the two are built by different code with different
 * flags, and the difference is not a rounding error -- it is whole runs of text appearing or not
 * appearing. In Blink:
 *
 *   DOMSelection::toString()                    ForSelectionToString, SkipsUnselectableContent
 *   FrameSelection::SelectedTextForClipboard()  SkipsUnselectableContent, EntersTextControls,
 *                                               IgnoresCssTextTransforms, EmitsImageAltText
 *
 * and `WebViewImpl::ApplyWebPreferences` turns `SetSelectionIncludesAltImageText(true)` on for
 * every web view, so the alt-text one is not conditional in practice. Measured on Playwright's
 * Chromium and WebKit by copying a fixture and pasting it back, `toString()` against the real
 * clipboard:
 *
 *   text-transform: uppercase   toString "TRANSFORMED HEADING", clipboard "transformed heading"
 *                               (Chromium only; WebKit uppercases both)
 *   <img alt="...">             clipboard emits the alt text, toString omits it (both engines)
 *   <input>, <textarea>         clipboard emits the value, toString omits it (Chromium only)
 *   user-select: none           no difference on either engine
 *
 * Every one of those constructs occurs in a Studio thread, and it is not hypothetical: on the
 * heavy fixture, an ungated fast path over 40,626 characters of conversation dropped the two
 * "SVG preview" alt strings the browser's own copy put on the clipboard, 40,648 characters
 * against 40,626. Images carry alt text all over the thread (attachment previews, generated
 * images, tool result images, SVG previews) and a message being edited puts a real `<textarea>`
 * inside the viewport. So the fast path is taken only when none of them is in range, and
 * otherwise the browser's own copy runs untouched. That is the whole reason
 * `CLIPBOARD_ONLY_CONTENT` exists: it is not a stylistic preference, it is the list of things
 * that would silently change what a user copied. The check is scoped to the selection's own
 * common ancestor rather than to the whole viewport, so one image in one message does not turn
 * the fast path off for the rest of the conversation. Even at the worst scope -- 42,000
 * elements, nothing matching, so the walk does not stop early -- the `querySelector` costs 2.6ms
 * in Chromium and 1ms in WebKit, against the copy it is trying to avoid.
 *
 * WHAT THIS DOES NOT COVER. The listener is on the thread viewport, so it only sees a copy whose
 * selection lies inside the thread. The browser's own document-wide Ctrl+A does NOT route
 * through it: the copy event targets the node the selection starts in, which for a whole-document
 * selection is the first row of the sidebar, and the event bubbles from there and never touches
 * the viewport at all (measured in both engines). What is fixed here is the case that actually
 * dominates a user's session -- dragging a selection across a long stretch of the conversation.
 *
 * WHY NOT SERIALISE THE MESSAGE STORE. Building the text from the store instead is faster still,
 * because nothing is read out of the DOM at all. It also changes what lands on the clipboard:
 * role headings, thinking markers and tool lines that the DOM selection never contained. That is
 * a trade worth making on a virtualized list, where the DOM physically cannot select an unmounted
 * message and the alternative is losing text outright. On the shipped, fully mounted list it buys
 * only speed, at the cost of handing the user a different document than the one they highlighted,
 * and it would be flatly wrong for a partial selection -- which is the case this is for.
 */

/** Why a copy was left to the browser. Named so a test can assert the reason, not just the miss. */
export type NativeCopyReason =
  | "already-handled"
  | "no-clipboard-data"
  | "editable-origin"
  | "empty-selection"
  | "selection-leaves-thread"
  | "clipboard-only-content";

export type ThreadCopyDecision =
  | { readonly kind: "fast"; readonly text: string }
  | { readonly kind: "native"; readonly reason: NativeCopyReason };

/**
 * A copy event, structurally. Nothing here needs a DOM, so the decision can be unit tested
 * against plain objects rather than against a headless browser that would only be timing itself.
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
 * Anything the clipboard's `text/plain` serialiser treats differently from
 * `Selection.toString()`. One of these anywhere in the selected subtree and the copy is handed
 * back to the browser, because the substitution would no longer be invisible.
 *
 * `alt=""` is excluded on purpose: an empty alt emits nothing either way, and it is what every
 * decorative image in the thread already carries.
 */
const CLIPBOARD_ONLY_CONTENT = [
  // EmitsImageAltText: the clipboard carries the alt text, toString() does not.
  'img[alt]:not([alt=""])',
  // EntersTextControls: the clipboard carries the control's value, toString() does not. A message
  // being edited, and a queued prompt being renamed, both mount one of these in the viewport.
  "input",
  "textarea",
  "select",
  // IgnoresCssTextTransforms: the clipboard carries the source text, toString() carries the
  // transformed text. Studio styles with Tailwind utilities, so the utility classes are the
  // reachable form of this; the attribute selector covers a hand-written inline style.
  ".uppercase",
  ".lowercase",
  ".capitalize",
  '[style*="text-transform"]',
].join(", ");

/** Where a copy must be left alone because the selection is not the document's. */
const EDITABLE_ORIGIN =
  'input, textarea, select, [contenteditable=""], [contenteditable="true"], [contenteditable="plaintext-only"]';

function matchesAncestor(target: unknown, selectors: string): boolean {
  const closest = (
    target as { closest?: (selectors: string) => unknown } | null
  )?.closest;
  if (typeof closest !== "function") return false;
  return closest.call(target, selectors) != null;
}

/**
 * The smallest subtree the content check has to be right about.
 *
 * The range's common ancestor, not the viewport, and the difference is what makes this worth
 * shipping: a thread that showed one image anywhere would otherwise never take the fast path
 * again, however far from the image the user selected. The common ancestor is by definition a
 * superset of the selected content, so scanning it is still an over-approximation -- it can only
 * ever refuse a copy the fast path could have taken, never accept one it could not.
 *
 * A range that ends in a text node has no `querySelector` of its own, hence the parent. Anything
 * unrecognisable falls back to the viewport, which is the conservative direction.
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
 * Should this copy be answered with `Selection.toString()` instead of the browser's own
 * serialisation, and if not, why not?
 *
 * Pure, and every rejection carries its reason, because each branch is a distinct way of getting
 * somebody's copy wrong and "the fast path did not run" is not enough to tell them apart.
 */
export function decideThreadCopy(
  event: CopyEventLike,
  selection: SelectionLike | null,
  viewport: ThreadViewportLike,
): ThreadCopyDecision {
  // Somebody upstream already produced this clipboard payload. Do not overwrite it.
  if (event.defaultPrevented) {
    return { kind: "native", reason: "already-handled" };
  }

  // No transfer to write into, so preventing the default would copy nothing at all.
  if (!event.clipboardData) {
    return { kind: "native", reason: "no-clipboard-data" };
  }

  // A copy out of a text control is that control's own selection, and `window.getSelection()` is
  // not it: the document selection is usually collapsed or stale while a textarea has focus, so
  // substituting it would replace the copied field with unrelated text. The thread viewport
  // contains these -- a message being edited mounts a textarea -- so this cannot be left to the
  // listener's placement.
  if (matchesAncestor(event.target, EDITABLE_ORIGIN)) {
    return { kind: "native", reason: "editable-origin" };
  }

  // A caret rather than a selection. The browser copies nothing; so should we.
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
    return { kind: "native", reason: "empty-selection" };
  }

  // A selection that starts inside the thread and runs out of it -- into the composer, the
  // sidebar, the header -- has a common ancestor above the viewport. Its text is no longer
  // something this file has checked, so it is not something this file should be rewriting.
  for (let index = 0; index < selection.rangeCount; index += 1) {
    if (
      !viewport.contains(selection.getRangeAt(index).commonAncestorContainer)
    ) {
      return { kind: "native", reason: "selection-leaves-thread" };
    }
  }

  if (
    scopeOf(selection, viewport).querySelector(CLIPBOARD_ONLY_CONTENT) != null
  ) {
    return { kind: "native", reason: "clipboard-only-content" };
  }

  const text = selection.toString();
  // A selection can be non-collapsed and still serialise to nothing (an image on its own, a
  // collapsed reasoning block). Writing "" would clear a clipboard the browser would have left
  // alone.
  if (text === "") {
    return { kind: "native", reason: "empty-selection" };
  }

  return { kind: "fast", text };
}

/**
 * Wire the decision to a live viewport. Returns the detach function.
 *
 * Only `copy` is listened for. A `cut` also has to mutate the document it cut from, and the
 * thread is not editable, so a cut inside it is already a no-op and one inside a message being
 * edited belongs to that textarea.
 */
export function attachThreadFastCopy(viewport: HTMLElement): () => void {
  const onCopy = (event: ClipboardEvent) => {
    const selection =
      viewport.ownerDocument.defaultView?.getSelection() ?? null;
    const decision = decideThreadCopy(event, selection, viewport);
    if (decision.kind !== "fast") return;
    event.preventDefault();
    event.clipboardData?.setData("text/plain", decision.text);
  };

  viewport.addEventListener("copy", onCopy);
  return () => viewport.removeEventListener("copy", onCopy);
}
