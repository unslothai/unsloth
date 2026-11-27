// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Radix writes `pointer-events: none` on <body> while a modal layer is open. Rules keyed on that inline
 * style (`body[style*=...] X`) made WebKit re-match every element under <body> on each flip, about 45 ms
 * per open and close on a 150-turn chat. This mirrors the lock onto attributes that never sit above the
 * thread, so a toggle only reaches the elements the rules style:
 *   - `data-modal-lock-thread` on each thread root, `data-modal-lock-path` on its ancestors;
 *   - `data-modal-lock-region` on every other child of those ancestors (the rest of the page);
 *   - `data-modal-lock-lower` on a dialog, menu or listbox a stacked modal layer pushed below it.
 * Idempotent: one watcher per document.
 */

export const MODAL_LOCK_THREAD_ATTR = "data-modal-lock-thread";
export const MODAL_LOCK_PATH_ATTR = "data-modal-lock-path";
export const MODAL_LOCK_REGION_ATTR = "data-modal-lock-region";
export const MODAL_LOCK_LOWER_ATTR = "data-modal-lock-lower";

const THREAD_ROOT_CLASS = "aui-thread-root";
const LAYER_ROLES = new Set(["dialog", "alertdialog", "menu", "listbox"]);
const LAYER_SELECTOR = '[role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"]';

type Attr =
  | typeof MODAL_LOCK_THREAD_ATTR
  | typeof MODAL_LOCK_PATH_ATTR
  | typeof MODAL_LOCK_REGION_ATTR;

interface LockWatcher {
  locked: boolean;
  marks: Map<Element, Attr>;
  lower: Set<Element>;
  structure: MutationObserver;
  layers: MutationObserver;
}

const watchers = new WeakMap<Document, LockWatcher>();

function isBodyLocked(doc: Document): boolean {
  return doc.body?.style.pointerEvents === "none";
}

function isLowerLayer(element: Element): boolean {
  const role = element.getAttribute("role");
  return (
    role !== null &&
    LAYER_ROLES.has(role) &&
    (element as HTMLElement).style?.pointerEvents === "none"
  );
}

/** Thread roots, their ancestors under <body>, and every other child of those ancestors. */
function desiredMarks(doc: Document): { marks: Map<Element, Attr>; path: Element[] } {
  const body = doc.body;
  const marks = new Map<Element, Attr>();
  const onPath = new Set<Element>([body]);
  const roots = Array.from(body.getElementsByClassName(THREAD_ROOT_CLASS));
  for (const root of roots) {
    // A thread root nested in another one stays inside that one (it is never split).
    if (roots.some((other) => other !== root && other.contains(root))) continue;
    marks.set(root, MODAL_LOCK_THREAD_ATTR);
    for (let node = root.parentElement; node && node !== body; node = node.parentElement) {
      if (onPath.has(node)) break;
      onPath.add(node);
    }
  }
  for (const node of onPath) {
    if (node !== body) marks.set(node, MODAL_LOCK_PATH_ATTR);
    for (const child of node.children) {
      if (!onPath.has(child) && !marks.has(child)) marks.set(child, MODAL_LOCK_REGION_ATTR);
    }
  }
  return { marks, path: [...onPath] };
}

function applyMarks(watcher: LockWatcher, next: Map<Element, Attr>): void {
  for (const [element, attr] of watcher.marks) {
    if (next.get(element) !== attr) element.removeAttribute(attr);
  }
  for (const [element, attr] of next) {
    if (watcher.marks.get(element) !== attr) element.setAttribute(attr, "");
  }
  watcher.marks = next;
}

function markLower(watcher: LockWatcher, element: Element): void {
  if (isLowerLayer(element)) {
    if (!watcher.lower.has(element)) {
      element.setAttribute(MODAL_LOCK_LOWER_ATTR, "");
      watcher.lower.add(element);
    }
  } else if (watcher.lower.delete(element)) {
    element.removeAttribute(MODAL_LOCK_LOWER_ATTR);
  }
}

function syncStructure(doc: Document, watcher: LockWatcher): void {
  const { marks, path } = desiredMarks(doc);
  applyMarks(watcher, marks);
  watcher.structure.disconnect();
  for (const node of path) watcher.structure.observe(node, { childList: true });
}

function lock(doc: Document, watcher: LockWatcher): void {
  watcher.locked = true;
  syncStructure(doc, watcher);
  // A layer that was already open (a non-modal menu under the new dialog) turned lower in the same
  // commit. Layers are never inside a thread, so only the regions are searched.
  for (const [element, attr] of watcher.marks) {
    if (attr !== MODAL_LOCK_REGION_ATTR) continue;
    markLower(watcher, element);
    for (const layer of element.querySelectorAll(LAYER_SELECTOR)) markLower(watcher, layer);
  }
  watcher.layers.observe(doc.body, {
    attributes: true,
    attributeFilter: ["style"],
    subtree: true,
  });
}

function unlock(watcher: LockWatcher): void {
  watcher.locked = false;
  watcher.structure.disconnect();
  watcher.layers.disconnect();
  applyMarks(watcher, new Map());
  for (const element of watcher.lower) element.removeAttribute(MODAL_LOCK_LOWER_ATTR);
  watcher.lower.clear();
}

/** Mirror Radix's body pointer-events lock onto the attributes index.css keys its modal rules on. */
export function watchModalPointerLock(doc: Document = document): () => void {
  const existing = watchers.get(doc);
  if (existing || typeof MutationObserver === "undefined" || !doc.body) {
    return () => undefined;
  }
  const watcher: LockWatcher = {
    locked: false,
    marks: new Map(),
    lower: new Set(),
    structure: new MutationObserver(() => {
      if (watcher.locked) syncStructure(doc, watcher);
    }),
    layers: new MutationObserver((records) => {
      for (const record of records) {
        if (record.target !== doc.body) markLower(watcher, record.target as Element);
      }
    }),
  };
  watchers.set(doc, watcher);
  const read = () => {
    const next = isBodyLocked(doc);
    if (next === watcher.locked) return;
    if (next) lock(doc, watcher);
    else unlock(watcher);
  };
  const body = new MutationObserver(read);
  body.observe(doc.body, { attributes: true, attributeFilter: ["style"] });
  read();
  return () => {
    body.disconnect();
    unlock(watcher);
    watchers.delete(doc);
  };
}
