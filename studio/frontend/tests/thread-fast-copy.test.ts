// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A thread copy writes its own text/plain so the browser skips the slow styled flavour.
// Tests pin the gate and serialiser against a hand-rolled DOM (no jsdom in this project).

import assert from "node:assert/strict";
import test from "node:test";

import {
  type CopyEventLike,
  type SelectionLike,
  type ThreadViewportLike,
  attachThreadFastCopy,
  decideThreadCopy,
  engineClipboardIsMapped,
  faithfulSelectionText,
} from "../src/components/assistant-ui/thread-fast-copy.ts";

function plainViewport(
  options: {
    contains?: boolean;
    matches?: string[];
  } = {},
): ThreadViewportLike & { queried: string[] } {
  const { contains = true, matches = [] } = options;
  const queried: string[] = [];
  return {
    queried,
    contains: () => contains,
    querySelector(selectors: string) {
      queried.push(selectors);
      const hit = matches.find((selector) => selectors.includes(selector));
      return hit === undefined ? null : { selector: hit };
    },
  };
}

function selectionOf(text: string): SelectionLike {
  return {
    isCollapsed: false,
    rangeCount: 1,
    getRangeAt: () => ({ commonAncestorContainer: { node: "message" } }),
    toString: () => text,
  };
}

function copyEvent(overrides: Partial<CopyEventLike> = {}): CopyEventLike {
  return {
    defaultPrevented: false,
    target: { closest: () => null },
    clipboardData: { setData: () => undefined },
    ...overrides,
  };
}

test("an ordinary selection inside the thread is answered by the fast path", () => {
  const decision = decideThreadCopy(
    copyEvent(),
    selectionOf("first message\n\nsecond message"),
    plainViewport(),
  );

  assert.deepEqual(decision, { kind: "fast" });
});

test("a copy another handler already answered is left alone", () => {
  const decision = decideThreadCopy(
    copyEvent({ defaultPrevented: true }),
    selectionOf("anything"),
    plainViewport(),
  );

  assert.deepEqual(decision, { kind: "native", reason: "already-handled" });
});

test("a copy with no clipboardData is left alone rather than prevented into nothing", () => {
  // preventDefault() with nowhere to write the replacement copies nothing.
  const decision = decideThreadCopy(
    copyEvent({ clipboardData: null }),
    selectionOf("anything"),
    plainViewport(),
  );

  assert.deepEqual(decision, { kind: "native", reason: "no-clipboard-data" });
});

test("a copy out of a text control keeps the browser's own copy", () => {
  // Inside a textarea, window.getSelection() is the document's selection, not the field's.
  for (const tag of ["input", "textarea", "select"]) {
    const decision = decideThreadCopy(
      copyEvent({
        target: {
          closest: (selectors: string) =>
            selectors.includes(tag) ? { tag } : null,
        },
      }),
      selectionOf("some text elsewhere in the document"),
      plainViewport(),
    );

    assert.deepEqual(
      decision,
      { kind: "native", reason: "editable-origin" },
      `copy originating in <${tag}>`,
    );
  }
});

test("a copy out of a contenteditable keeps the browser's own copy", () => {
  const decision = decideThreadCopy(
    copyEvent({
      target: {
        closest: (selectors: string) =>
          selectors.includes("contenteditable") ? { tag: "div" } : null,
      },
    }),
    selectionOf("text"),
    plainViewport(),
  );

  assert.deepEqual(decision, { kind: "native", reason: "editable-origin" });
});

test("a caret rather than a selection is left alone", () => {
  const collapsed: SelectionLike = { ...selectionOf("x"), isCollapsed: true };
  assert.deepEqual(decideThreadCopy(copyEvent(), collapsed, plainViewport()), {
    kind: "native",
    reason: "empty-selection",
  });

  const empty: SelectionLike = { ...selectionOf("x"), rangeCount: 0 };
  assert.deepEqual(decideThreadCopy(copyEvent(), empty, plainViewport()), {
    kind: "native",
    reason: "empty-selection",
  });

  assert.deepEqual(decideThreadCopy(copyEvent(), null, plainViewport()), {
    kind: "native",
    reason: "empty-selection",
  });
});

test("a selection running out of the thread is left alone", () => {
  // Text past the viewport boundary is unchecked, so it is not rewritten.
  const decision = decideThreadCopy(
    copyEvent(),
    selectionOf("thread text plus composer draft"),
    plainViewport({ contains: false }),
  );

  assert.deepEqual(decision, {
    kind: "native",
    reason: "selection-leaves-thread",
  });
});

test("every range of a multi-range selection has to be inside the thread", () => {
  // Firefox builds multi-range selections from ctrl-click.
  let range = 0;
  const selection: SelectionLike = {
    isCollapsed: false,
    rangeCount: 2,
    getRangeAt: () => ({ commonAncestorContainer: { index: range++ } }),
    toString: () => "two disjoint runs",
  };
  const viewport: ThreadViewportLike = {
    contains: (node) => (node as { index: number }).index === 0,
    querySelector: () => null,
  };

  assert.deepEqual(decideThreadCopy(copyEvent(), selection, viewport), {
    kind: "native",
    reason: "selection-leaves-thread",
  });
});

test("a form control in the selected subtree hands the copy back", () => {
  // Form controls are refused: Chromium emits their value as its own block, varying by control.
  for (const control of ["input", "textarea", "select"]) {
    const decision = decideThreadCopy(
      copyEvent(),
      selectionOf("some prose and then the field"),
      plainViewport({ matches: [control] }),
    );

    assert.deepEqual(
      decision,
      { kind: "native", reason: "form-control" },
      `<${control}> inside the selection`,
    );
  }
});

test("an image with alt text no longer refuses the fast path", () => {
  const viewport = plainViewport();
  const decision = decideThreadCopy(
    copyEvent(),
    selectionOf("prose next to an image of a cat"),
    viewport,
  );

  assert.deepEqual(decision, { kind: "fast" });
  assert.deepEqual(viewport.queried, ["input, textarea, select"]);
});

test("text under a css text-transform no longer refuses the fast path", () => {
  // Chromium's clipboard ignores text-transform; the serialiser neutralises it for the copy.
  const viewport = plainViewport({ matches: [".uppercase"] });

  assert.deepEqual(
    decideThreadCopy(copyEvent(), selectionOf("STDOUT"), viewport),
    { kind: "fast" },
  );
});

test("an engine whose clipboard mapping is unproven hands the copy back", () => {
  // WebKit's toString() appends trailing block breaks its clipboard does not carry.
  const decision = decideThreadCopy(
    copyEvent(),
    selectionOf("a paragraph"),
    plainViewport(),
    false,
  );

  assert.deepEqual(decision, { kind: "native", reason: "unmapped-engine" });
});

test("the engine check runs after every cheaper refusal", () => {
  const cheaper: ReadonlyArray<readonly [string, () => unknown]> = [
    [
      "already-handled",
      () =>
        decideThreadCopy(
          copyEvent({ defaultPrevented: true }),
          selectionOf("text"),
          plainViewport(),
          false,
        ),
    ],
    [
      "no-clipboard-data",
      () =>
        decideThreadCopy(
          copyEvent({ clipboardData: null }),
          selectionOf("text"),
          plainViewport(),
          false,
        ),
    ],
    [
      "editable-origin",
      () =>
        decideThreadCopy(
          copyEvent({ target: { closest: () => ({}) } }),
          selectionOf("text"),
          plainViewport(),
          false,
        ),
    ],
    [
      "empty-selection",
      () => decideThreadCopy(copyEvent(), null, plainViewport(), false),
    ],
    [
      "selection-leaves-thread",
      () =>
        decideThreadCopy(
          copyEvent(),
          selectionOf("text"),
          plainViewport({ contains: false }),
          false,
        ),
    ],
    [
      "form-control",
      () =>
        decideThreadCopy(
          copyEvent(),
          selectionOf("text"),
          plainViewport({ matches: ["textarea"] }),
          false,
        ),
    ],
  ];

  for (const [reason, run] of cheaper) {
    assert.deepEqual(run(), { kind: "native", reason }, reason);
  }
});

test("the checks run cheapest-first, so a rejected copy never walks the thread", () => {
  // The form control check is the only DOM query, so other rejections must come first.
  for (const event of [
    copyEvent({ defaultPrevented: true }),
    copyEvent({ clipboardData: null }),
    copyEvent({ target: { closest: () => ({}) } }),
  ]) {
    const viewport = plainViewport();
    const decision = decideThreadCopy(event, selectionOf("text"), viewport);

    assert.equal(decision.kind, "native");
    assert.deepEqual(viewport.queried, []);
  }
});

test("the button copy path is untouched", () => {
  // The copy fallback textarea is outside the viewport and also hits the editable-origin guard.
  const decision = decideThreadCopy(
    copyEvent({
      target: {
        closest: (selectors: string) =>
          selectors.includes("textarea") ? { tag: "textarea" } : null,
      },
    }),
    selectionOf("whatever the thread happens to have selected"),
    plainViewport(),
  );

  assert.deepEqual(decision, { kind: "native", reason: "editable-origin" });
});

// Scoping the check to the whole viewport would disable the fast path after any textarea.

function selectionInside(
  element: { querySelector(selectors: string): unknown },
  text = "selected prose",
): SelectionLike {
  return {
    isCollapsed: false,
    rangeCount: 1,
    getRangeAt: () => ({ commonAncestorContainer: element }),
    toString: () => text,
  };
}

test("the form control check looks at the selection's ancestor, not at the whole thread", () => {
  const message = {
    querySelector: () => null,
  };
  const viewport: ThreadViewportLike = {
    contains: () => true,
    querySelector: () => ({ tag: "textarea" }),
  };

  const decision = decideThreadCopy(
    copyEvent(),
    selectionInside(message),
    viewport,
  );

  assert.deepEqual(decision, { kind: "fast" });
});

test("a form control inside the selected subtree still refuses the fast path", () => {
  const message = { querySelector: () => ({ tag: "textarea" }) };
  const viewport: ThreadViewportLike = {
    contains: () => true,
    querySelector: () => null,
  };

  assert.deepEqual(
    decideThreadCopy(copyEvent(), selectionInside(message), viewport),
    { kind: "native", reason: "form-control" },
  );
});

test("a range ending in a text node is checked against that node's element", () => {
  // Range.commonAncestorContainer is often a text node, which has no querySelector.
  const paragraph = { querySelector: () => ({ tag: "textarea" }) };
  const textNode = { parentElement: paragraph };
  const selection: SelectionLike = {
    isCollapsed: false,
    rangeCount: 1,
    getRangeAt: () => ({ commonAncestorContainer: textNode }),
    toString: () => "half a sentence",
  };
  const viewport: ThreadViewportLike = {
    contains: () => true,
    querySelector: () => null,
  };

  assert.deepEqual(decideThreadCopy(copyEvent(), selection, viewport), {
    kind: "native",
    reason: "form-control",
  });
});

test("a multi-range selection is checked against the whole viewport", () => {
  // Disjoint ranges have no common ancestor short of the viewport.
  const viewport: ThreadViewportLike = {
    contains: () => true,
    querySelector: () => ({ tag: "input" }),
  };
  const selection: SelectionLike = {
    isCollapsed: false,
    rangeCount: 2,
    getRangeAt: () => ({
      commonAncestorContainer: { querySelector: () => null },
    }),
    toString: () => "two disjoint runs",
  };

  assert.deepEqual(decideThreadCopy(copyEvent(), selection, viewport), {
    kind: "native",
    reason: "form-control",
  });
});

// The fake throws on unimplemented selectors and computes toString() live from the tree.
// It cannot prove toString() equals the real clipboard; that was measured in a browser.

type StyleEntry = { value: string; priority: string };

type FakeText = {
  readonly nodeType: 3;
  readonly data: string;
  parentNode: FakeElement | null;
  readonly ownerDocument: typeof fakeDocument;
};

type FakeNode = FakeText | FakeElement;

type FakeElement = {
  readonly nodeType: 1;
  readonly tagName: string;
  readonly attrs: Map<string, string>;
  readonly childNodes: FakeNode[];
  parentNode: FakeElement | null;
  readonly parentElement: FakeElement | null;
  readonly ruleTextTransform: string | null;
  readonly ruleVisibility: string | null;
  readonly styleEntries: Map<string, StyleEntry>;
  readonly style: {
    getPropertyValue(name: string): string;
    getPropertyPriority(name: string): string;
    setProperty(name: string, value: string, priority?: string): void;
    removeProperty(name: string): void;
  };
  readonly ownerDocument: typeof fakeDocument;
  textContent: string;
  getAttribute(name: string): string | null;
  setAttribute(name: string, value: string): void;
  removeAttribute(name: string): void;
  readonly attributes: { removeNamedItem(name: string): void };
  querySelector(selectors: string): FakeElement | null;
  querySelectorAll(selectors: string): FakeElement[];
  insertBefore(node: FakeNode, ref: FakeNode | null): FakeNode;
  remove(): void;
};

/** patchClipboardDeltas checks `instanceof HTMLElement`; only the prototype matters. */
class FakeHtmlElement {}

Object.defineProperty(globalThis, "HTMLElement", {
  configurable: true,
  writable: true,
  value: FakeHtmlElement,
});

function text(data: string): FakeText {
  return { nodeType: 3, data, parentNode: null, ownerDocument: fakeDocument };
}

/** Only the selectors this module uses; anything else is a hole in the fake, not a pass. */
function matchesSelector(node: FakeElement, selectors: string): boolean {
  return selectors.split(",").some((part) => {
    const one = part.trim();
    if (one === "*") return true;
    if (one === "img[alt]") {
      return node.tagName === "img" && node.attrs.has("alt");
    }
    if (one === "input" || one === "textarea" || one === "select") {
      return node.tagName === one;
    }
    throw new Error(`the fake DOM does not implement the selector ${one}`);
  });
}

/**
 * Mutating inside a live range collapses it in Chromium, hence the double restore.
 * Listeners outlive their test, so each checks the mutation is inside its own root.
 */
const mutationListeners: Array<(parent: FakeElement) => void> = [];

function notifyMutation(parent: FakeElement): void {
  for (const listener of mutationListeners) listener(parent);
}

function isWithin(node: FakeElement | null, root: FakeElement): boolean {
  for (let at = node; at !== null; at = at.parentNode) {
    if (at === root) return true;
  }
  return false;
}

function descendants(node: FakeElement): FakeElement[] {
  const found: FakeElement[] = [];
  for (const child of node.childNodes) {
    if (child.nodeType !== 1) continue;
    found.push(child, ...descendants(child));
  }
  return found;
}

function el(
  tagName: string,
  options: {
    alt?: string;
    ruleVisibility?: string;
    inline?: ReadonlyArray<readonly [string, string, string?]>;
    rule?: string;
    children?: ReadonlyArray<FakeNode | string>;
  } = {},
): FakeElement {
  const attrs = new Map<string, string>();
  if (options.alt !== undefined) attrs.set("alt", options.alt);
  const styleEntries = new Map<string, StyleEntry>();
  for (const [name, value, priority] of options.inline ?? []) {
    styleEntries.set(name, { value, priority: priority ?? "" });
  }
  const childNodes: FakeNode[] = [];
  // Inline declaration and `style` attribute are one thing on a real element; emptying it
  // leaves `style=""`. Separate maps hid residue the restore left.
  const syncStyleAttribute = () => {
    attrs.set(
      "style",
      [...styleEntries]
        .map(
          ([name, entry]) =>
            `${name}: ${entry.value}${entry.priority ? ` !${entry.priority}` : ""};`,
        )
        .join(" "),
    );
  };
  const readStyleAttribute = (value: string) => {
    styleEntries.clear();
    for (const part of value.split(";")) {
      const [rawName, ...rest] = part.split(":");
      if (rest.length === 0) continue;
      const name = rawName.trim();
      if (!name) continue;
      const raw = rest.join(":").trim();
      const important = raw.endsWith("!important");
      styleEntries.set(name, {
        value: important ? raw.slice(0, -"!important".length).trim() : raw,
        priority: important ? "important" : "",
      });
    }
  };

  // Elements built with inline styles must carry the matching style attribute, as parsed markup does.
  if (styleEntries.size > 0) syncStyleAttribute();

  const node: FakeElement = {
    nodeType: 1,
    tagName,
    attrs,
    childNodes,
    parentNode: null,
    get parentElement() {
      return node.parentNode;
    },
    ruleTextTransform: options.rule ?? null,
    ruleVisibility: options.ruleVisibility ?? null,
    styleEntries,
    style: {
      getPropertyValue: (name) => styleEntries.get(name)?.value ?? "",
      getPropertyPriority: (name) => styleEntries.get(name)?.priority ?? "",
      setProperty: (name, value, priority = "") => {
        styleEntries.set(name, { value, priority });
        syncStyleAttribute();
      },
      removeProperty: (name) => {
        styleEntries.delete(name);
        syncStyleAttribute();
      },
    },
    ownerDocument: fakeDocument,
    get textContent() {
      return childNodes.map(renderSource).join("");
    },
    set textContent(value: string) {
      childNodes.length = 0;
      node.insertBefore(text(value), null);
    },
    getAttribute: (name) => attrs.get(name) ?? null,
    setAttribute: (name, value) => {
      attrs.set(name, value);
      if (name === "style") readStyleAttribute(value);
    },
    // removeAttribute and attribute node removal differ on a real element; both are modelled.
    removeAttribute: (name) => {
      attrs.delete(name);
      if (name === "style") styleEntries.clear();
    },
    attributes: {
      removeNamedItem: (name: string) => {
        if (!attrs.has(name)) {
          // The real DOM throws NotFoundError, which the module relies on catching.
          throw new Error(`NotFoundError: no attribute named ${name}`);
        }
        attrs.delete(name);
        if (name === "style") styleEntries.clear();
      },
    },
    querySelector: (selectors) =>
      descendants(node).find((child) => matchesSelector(child, selectors)) ??
      null,
    querySelectorAll: (selectors) =>
      descendants(node).filter((child) => matchesSelector(child, selectors)),
    insertBefore(child, ref) {
      const at = ref === null ? -1 : childNodes.indexOf(ref);
      childNodes.splice(at < 0 ? childNodes.length : at, 0, child);
      child.parentNode = node;
      notifyMutation(node);
      return child;
    },
    remove() {
      const parent = node.parentNode;
      if (!parent) return;
      parent.childNodes.splice(parent.childNodes.indexOf(node), 1);
      node.parentNode = null;
      notifyMutation(parent);
    },
  };
  Object.setPrototypeOf(node, FakeHtmlElement.prototype);
  for (const child of options.children ?? []) {
    node.insertBefore(typeof child === "string" ? text(child) : child, null);
  }
  return node;
}

function renderSource(node: FakeNode): string {
  return node.nodeType === 3
    ? node.data
    : node.childNodes.map(renderSource).join("");
}

function computedTextTransform(node: FakeElement | null): string {
  for (let at = node; at !== null; at = at.parentNode) {
    const inline = at.styleEntries.get("text-transform");
    if (inline) return inline.value;
    if (at.ruleTextTransform) return at.ruleTextTransform;
  }
  return "none";
}

function isDisplayNone(node: FakeElement | null): boolean {
  for (let at = node; at !== null; at = at.parentNode) {
    if (at.styleEntries.get("display")?.value === "none") return true;
  }
  return false;
}

function applyTransform(value: string, transform: string): string {
  if (transform === "uppercase") return value.toUpperCase();
  if (transform === "lowercase") return value.toLowerCase();
  if (transform === "capitalize") {
    return value.replace(/\b\w/g, (letter) => letter.toUpperCase());
  }
  return value;
}

/** Selection.toString(): text as rendered (transforms apply); images emit nothing. */
function renderSelected(node: FakeNode): string {
  if (node.nodeType === 3) {
    if (isDisplayNone(node.parentNode)) return "";
    return applyTransform(node.data, computedTextTransform(node.parentNode));
  }
  if (isDisplayNone(node)) return "";
  if (node.tagName === "img") return "";
  return node.childNodes.map(renderSelected).join("");
}

/** Structural snapshot; inline style compared as a set since engines reorder declarations. */
function snapshot(node: FakeNode): unknown {
  if (node.nodeType === 3) return { text: node.data };
  return {
    tag: node.tagName,
    attrs: [...node.attrs].sort(),
    style: [...node.styleEntries]
      .map(
        ([name, entry]) =>
          `${name}: ${entry.value}${entry.priority ? ` !${entry.priority}` : ""}`,
      )
      .sort(),
    children: node.childNodes.map(snapshot),
  };
}

type FakeRange = {
  readonly id: number;
  readonly commonAncestorContainer: FakeElement | null;
  readonly startContainer: FakeNode | null;
  readonly startOffset: number;
  readonly endContainer: FakeNode | null;
  readonly endOffset: number;
  cloneRange(): FakeRange;
};

function fakeRange(
  id: number,
  container: FakeElement | null = null,
  bounds: {
    startContainer?: FakeNode | null;
    startOffset?: number;
    endContainer?: FakeNode | null;
    endOffset?: number;
  } = {},
): FakeRange {
  // A real Range is live, so the restore reads its boundaries, not pre-patch offsets.
  return {
    id,
    commonAncestorContainer: container,
    startContainer: bounds.startContainer ?? container,
    startOffset: bounds.startOffset ?? 0,
    endContainer: bounds.endContainer ?? container,
    endOffset: bounds.endOffset ?? 0,
    cloneRange: () => fakeRange(id, container, bounds),
  };
}

function fakeSelection(
  root: FakeElement,
  options: {
    ids?: number[];
    probeText?: string;
    anchorNode?: FakeNode;
    anchorOffset?: number;
    focusNode?: FakeNode;
    focusOffset?: number;
    bounds?: {
      startContainer?: FakeNode | null;
      startOffset?: number;
      endContainer?: FakeNode | null;
      endOffset?: number;
    };
  } = {},
) {
  const { ids = [1], probeText = "a" } = options;
  const bounds = options.bounds ?? {};
  // Direction lives only in anchor/focus; rebuilding from cloned ranges flips backward selections.
  const calls: {
    setBaseAndExtent: Array<{
      anchorNode: FakeNode | null;
      anchorOffset: number;
      focusNode: FakeNode | null;
      focusOffset: number;
    }>;
  } = { setBaseAndExtent: [] };
  let anchorNode: FakeNode | null = options.anchorNode ?? null;
  let anchorOffset = options.anchorOffset ?? 0;
  let focusNode: FakeNode | null = options.focusNode ?? null;
  let focusOffset = options.focusOffset ?? 0;
  let ranges = ids.map((id) => fakeRange(id, root, bounds));
  let probing = false;
  let collapsedByMutation = false;
  mutationListeners.push((parent) => {
    if (isWithin(parent, root)) collapsedByMutation = true;
  });
  const sync = () => {
    if (!collapsedByMutation) return;
    collapsedByMutation = false;
    ranges = [];
  };
  const selection = {
    get rangeCount() {
      sync();
      return ranges.length;
    },
    getRangeAt: (index: number) => {
      sync();
      return ranges[index];
    },
    // Anchor and focus move with the ranges, or a direction-flipping restore would pass.
    removeAllRanges() {
      sync();
      ranges = [];
      probing = false;
      anchorNode = null;
      anchorOffset = 0;
      focusNode = null;
      focusOffset = 0;
    },
    addRange(range: FakeRange) {
      sync();
      ranges.push(range);
      anchorNode = range.startContainer;
      anchorOffset = range.startOffset;
      focusNode = range.endContainer;
      focusOffset = range.endOffset;
    },
    selectAllChildren() {
      sync();
      ranges = [];
      probing = true;
      anchorNode = null;
      anchorOffset = 0;
      focusNode = null;
      focusOffset = 0;
    },
    toString: () => {
      sync();
      if (probing) return probeText;
      return ranges.length === 0 ? "" : renderSelected(root);
    },
    get anchorNode() {
      return anchorNode;
    },
    get anchorOffset() {
      return anchorOffset;
    },
    get focusNode() {
      return focusNode;
    },
    get focusOffset() {
      return focusOffset;
    },
    setBaseAndExtent(
      newAnchor: FakeNode,
      newAnchorOffset: number,
      newFocus: FakeNode,
      newFocusOffset: number,
    ) {
      sync();
      anchorNode = newAnchor;
      anchorOffset = newAnchorOffset;
      focusNode = newFocus;
      focusOffset = newFocusOffset;
      ranges = ids.map((id) => fakeRange(id, root, bounds));
      calls.setBaseAndExtent.push({
        anchorNode: newAnchor,
        anchorOffset: newAnchorOffset,
        focusNode: newFocus,
        focusOffset: newFocusOffset,
      });
    },
  };
  return {
    selection,
    calls,
    currentIds: () => {
      sync();
      return ranges.map((range) => range.id);
    },
  };
}

/** captureDirection needs ownerDocument.createRange(); order comes from parent chains. */
function chain(node: FakeNode): FakeNode[] {
  const path: FakeNode[] = [];
  for (let at: FakeNode | null = node; at !== null; at = at.parentNode)
    path.unshift(at);
  return path;
}

function documentOrder(
  a: FakeNode,
  aOffset: number,
  b: FakeNode,
  bOffset: number,
): number {
  const pa = chain(a);
  const pb = chain(b);
  let depth = 0;
  while (depth < pa.length && depth < pb.length && pa[depth] === pb[depth]) {
    depth += 1;
  }
  if (depth === pa.length && depth === pb.length) {
    return bOffset === aOffset ? 0 : bOffset < aOffset ? -1 : 1;
  }
  const parent = pa[depth - 1];
  if (!parent || parent.nodeType !== 1) return 0;
  const kids = parent.childNodes;
  const ia = depth < pa.length ? kids.indexOf(pa[depth]) : aOffset;
  const ib = depth < pb.length ? kids.indexOf(pb[depth]) : bOffset;
  if (ib === ia) return 0;
  return ib < ia ? -1 : 1;
}

const fakeDocument = {
  createElement: (tag: string) => el(tag),
  createRange: () => {
    let anchor: FakeNode | null = null;
    let anchorAt = 0;
    return {
      setStart(node: FakeNode, offset: number) {
        anchor = node;
        anchorAt = offset;
      },
      setEnd() {},
      comparePoint: (node: FakeNode, offset: number) =>
        anchor === null ? 0 : documentOrder(anchor, anchorAt, node, offset),
    };
  },
};

/** The module reads the bare global, not `view.getComputedStyle`. */
Object.defineProperty(globalThis, "getComputedStyle", {
  configurable: true,
  writable: true,
  value: (node: FakeElement) => ({
    textTransform: computedTextTransform(node),
    // display is not inherited; visibility and user-select are. Skipped images get no alt holder.
    display: node.styleEntries.get("display")?.value ?? "inline",
    visibility: inheritedStyle(node, "visibility", "visible"),
    userSelect: inheritedStyle(node, "user-select", "auto"),
    webkitUserSelect: inheritedStyle(node, "-webkit-user-select", "auto"),
  }),
});

function inheritedStyle(
  node: FakeElement | null,
  name: string,
  fallback: string,
): string {
  for (let at = node; at !== null; at = at.parentNode) {
    const inline = at.styleEntries.get(name);
    if (inline) return inline.value;
    if (name === "visibility" && at.ruleVisibility) return at.ruleVisibility;
  }
  return fallback;
}

const CHROMIUM_UA =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const WEBKIT_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15";

function fakeView(options: {
  userAgent: string;
  probeText?: string;
  root?: FakeElement;
  selectionIds?: number[];
  anchorNode?: FakeNode;
  anchorOffset?: number;
  focusNode?: FakeNode;
  focusOffset?: number;
  bounds?: {
    startContainer?: FakeNode | null;
    startOffset?: number;
    endContainer?: FakeNode | null;
    endOffset?: number;
  };
}) {
  const root = options.root ?? el("div", { children: ["thread text"] });
  const { selection, currentIds, calls } = fakeSelection(root, {
    ids: options.selectionIds ?? [1],
    probeText: options.probeText ?? "a",
    anchorNode: options.anchorNode,
    anchorOffset: options.anchorOffset,
    focusNode: options.focusNode,
    focusOffset: options.focusOffset,
    bounds: options.bounds,
  });
  const body = el("body");
  // A counter, not a throwing getter: engineClipboardIsMapped swallows throws as false.
  let documentReads = 0;
  const document = {
    createElement: (tag: string) => el(tag),
    body: {
      appendChild: (child: FakeElement) => body.insertBefore(child, null),
    },
  };
  const view = {
    navigator: { userAgent: options.userAgent },
    get document() {
      documentReads += 1;
      return document;
    },
    getSelection: () => selection,
  };
  return {
    view,
    root,
    selection,
    currentIds,
    calls,
    body,
    documentReads: () => documentReads,
  };
}

function asWindow(view: unknown): Window & typeof globalThis {
  return view as Window & typeof globalThis;
}

test("a non-chromium user agent is unmapped without the document being touched", () => {
  const world = fakeView({ userAgent: WEBKIT_UA, probeText: "a" });

  assert.equal(engineClipboardIsMapped(asWindow(world.view)), false);
  assert.equal(world.documentReads(), 0);
  assert.equal(world.body.childNodes.length, 0);
});

test("a chromium engine whose toString appends a block break is unmapped", () => {
  // The user agent never decides the bytes; the probe does.
  const world = fakeView({ userAgent: CHROMIUM_UA, probeText: "a\n\n" });

  assert.equal(engineClipboardIsMapped(asWindow(world.view)), false);
  assert.equal(world.documentReads(), 1);
});

test("a chromium engine whose toString matches its clipboard is mapped", () => {
  const world = fakeView({ userAgent: CHROMIUM_UA, probeText: "a" });

  assert.equal(engineClipboardIsMapped(asWindow(world.view)), true);
  assert.equal(world.body.childNodes.length, 0);
});

test("the probe's answer is cached on the view", () => {
  const world = fakeView({ userAgent: CHROMIUM_UA, probeText: "a" });

  assert.equal(engineClipboardIsMapped(asWindow(world.view)), true);
  assert.equal(world.documentReads(), 1);
  assert.equal(engineClipboardIsMapped(asWindow(world.view)), true);
  assert.equal(world.documentReads(), 1);
  assert.equal(
    (world.view as { __sbFastCopyMapped?: boolean }).__sbFastCopyMapped,
    true,
  );
});

test("the probe puts a backward selection back backward", () => {
  // Restoring from saved ranges alone always yields a forward selection; assert the anchor/focus pair.
  const first = text("first paragraph");
  const second = text("second paragraph");
  const root = el("div", {
    children: [
      el("p", { children: [first] }),
      el("p", { children: [second] }),
    ],
  });
  const world = fakeView({
    userAgent: CHROMIUM_UA,
    probeText: "a",
    root,
    anchorNode: second,
    anchorOffset: 16,
    focusNode: first,
    focusOffset: 0,
    bounds: {
      startContainer: first,
      startOffset: 0,
      endContainer: second,
      endOffset: 16,
    },
  });

  engineClipboardIsMapped(asWindow(world.view));

  assert.equal(world.selection.anchorNode, second);
  assert.equal(world.selection.anchorOffset, 16);
  assert.equal(world.selection.focusNode, first);
  assert.equal(world.selection.focusOffset, 0);
});

test("the probe leaves a forward selection forward", () => {
  const first = text("first paragraph");
  const second = text("second paragraph");
  const root = el("div", {
    children: [
      el("p", { children: [first] }),
      el("p", { children: [second] }),
    ],
  });
  const world = fakeView({
    userAgent: CHROMIUM_UA,
    probeText: "a",
    root,
    anchorNode: first,
    anchorOffset: 0,
    focusNode: second,
    focusOffset: 16,
    bounds: {
      startContainer: first,
      startOffset: 0,
      endContainer: second,
      endOffset: 16,
    },
  });

  engineClipboardIsMapped(asWindow(world.view));

  assert.equal(world.selection.anchorNode, first);
  assert.equal(world.selection.anchorOffset, 0);
  assert.equal(world.selection.focusNode, second);
  assert.equal(world.selection.focusOffset, 16);
});

test("the probe puts the user's selection back", () => {
  const world = fakeView({
    userAgent: CHROMIUM_UA,
    probeText: "a",
    selectionIds: [7, 8],
  });

  engineClipboardIsMapped(asWindow(world.view));

  assert.deepEqual(world.currentIds(), [7, 8]);
  // addRange, not setBaseAndExtent, so multiple ranges are not dropped.
  assert.deepEqual(world.calls.setBaseAndExtent, []);
});

function serialise(root: FakeElement, ids: number[] = [1]) {
  const { selection, currentIds } = fakeSelection(root, { ids });
  const before = snapshot(root);
  const output = faithfulSelectionText(
    selection as unknown as Selection,
    root as unknown as Element,
  );
  return { output, before, after: snapshot(root), currentIds };
}

test("the alt text of an image reaches the output", () => {
  // Chromium's clipboard emits image alt text but toString() drops it, so insert a real node.
  const root = el("p", {
    children: ["before ", el("img", { alt: "a cat" }), " after"],
  });

  assert.equal(serialise(root).output, "before a cat after");
});

test("an image with an empty alt contributes nothing", () => {
  // alt="" must insert no node: each insertion collapses the user's live selection.
  const root = el("p", {
    children: ["before ", el("img", { alt: "" }), "after"],
  });
  const { selection } = fakeSelection(root);
  let childrenWhileReading = 0;
  const watched = {
    ...selection,
    toString: () => {
      childrenWhileReading = root.childNodes.length;
      return renderSelected(root);
    },
  };

  const output = faithfulSelectionText(
    watched as unknown as Selection,
    root as unknown as Element,
  );

  assert.equal(output, "before after");
  assert.equal(childrenWhileReading, 3);
});

test("the alt text holder has no box of its own", () => {
  // A block image makes an inline holder gain a leading newline; hiding the image avoids it.
  const image = el("img", { alt: "a cat", inline: [["display", "block"]] });
  const root = el("p", { children: ["before ", image, " after"] });
  let displayWhileReading = "";
  const { selection } = fakeSelection(root);
  const watched = {
    ...selection,
    toString: () => {
      displayWhileReading = image.style.getPropertyValue("display");
      return renderSelected(root);
    },
  };

  const output = faithfulSelectionText(
    watched as unknown as Selection,
    root as unknown as Element,
  );

  assert.equal(output, "before a cat after");
  assert.equal(displayWhileReading, "none");
});

test("a no-break space is folded to a plain space", () => {
  // Both engines' clipboards fold U+00A0 to a space; neither toString() does.
  const root = el("p", { children: ["one\u00a0two\u00a0three"] });

  const { output } = serialise(root);

  assert.equal(output, "one two three");
  assert.equal(output.includes("\u00a0"), false);
});

test("the source text under a text-transform is what is serialised", () => {
  const root = el("div", {
    children: [el("span", { rule: "uppercase", children: ["stdout"] })],
  });

  assert.equal(serialise(root).output, "stdout");
});

test("a transform inherited from an ancestor is neutralised too", () => {
  // text-transform inherits, so the patch must reach children without the class.
  const root = el("div", {
    rule: "uppercase",
    children: [el("span", { children: ["stdout"] })],
  });

  assert.equal(serialise(root).output, "stdout");
});

test("the dom is left exactly as it was found", () => {
  // The stub only checks bookkeeping; real serialisation is checked by
  // tests/studio/playwright_thread_fast_copy.py (e.g. Chromium keeps `style=""`).
  const root = el("div", {
    children: [
      el("span", { rule: "uppercase", children: ["stdout"] }),
      " ",
      el("img", { alt: "a cat" }),
      el("p", { children: ["tail"] }),
    ],
  });

  const { before, after } = serialise(root);

  assert.deepEqual(after, before);
  assert.deepEqual(
    root.querySelectorAll("*").map((node) => node.tagName),
    ["span", "img", "p"],
  );
});

test("an element's own inline text-transform keeps its value and priority", () => {
  // Restoring the value without `!important` would change rendering from then on.
  const span = el("span", {
    inline: [
      ["color", "red"],
      ["text-transform", "capitalize", "important"],
    ],
    children: ["stdout"],
  });
  const root = el("div", { children: [span] });

  const { before, after, output } = serialise(root);

  assert.equal(output, "stdout");
  assert.equal(span.style.getPropertyValue("text-transform"), "capitalize");
  assert.equal(span.style.getPropertyPriority("text-transform"), "important");
  assert.deepEqual(after, before);
});

test("an image's own inline display keeps its value and priority", () => {
  const image = el("img", {
    alt: "a cat",
    inline: [["display", "inline-block", "important"]],
  });
  const root = el("p", { children: [image] });

  const { before, after, output } = serialise(root);

  assert.equal(output, "a cat");
  assert.equal(image.style.getPropertyValue("display"), "inline-block");
  assert.equal(image.style.getPropertyPriority("display"), "important");
  assert.deepEqual(after, before);
});

test("an element with no inline style of its own keeps none", () => {
  const span = el("span", { rule: "uppercase", children: ["stdout"] });
  const image = el("img", { alt: "a cat" });
  const root = el("div", { children: [span, image] });

  serialise(root);

  assert.deepEqual([...span.styleEntries.keys()], []);
  assert.deepEqual([...image.styleEntries.keys()], []);
});

test("the user's selection ranges are put back", () => {
  // Holder insert and removal each collapse the live range, so ranges are restored twice.
  const root = el("div", {
    children: [
      el("span", { rule: "uppercase", children: ["stdout"] }),
      el("img", { alt: "a cat" }),
    ],
  });

  const { currentIds, output } = serialise(root, [4, 5]);

  assert.equal(output, "stdouta cat");
  assert.deepEqual(currentIds(), [4, 5]);
});

test("nothing untouched is patched, so an unremarkable selection restores trivially", () => {
  const root = el("div", { children: ["plain prose"] });
  const { selection, currentIds } = fakeSelection(root, { ids: [9] });
  const seen: number[] = [];
  const watched = {
    ...selection,
    removeAllRanges() {
      seen.push(-1);
      selection.removeAllRanges();
    },
  };

  const output = faithfulSelectionText(
    watched as unknown as Selection,
    root as unknown as Element,
  );

  assert.equal(output, "plain prose");
  assert.deepEqual(seen, []);
  assert.deepEqual(currentIds(), [9]);
});

test("a selection wholly inside a text-transformed element serialises the source text", () => {
  // The scope element may itself be transformed; querySelectorAll("*") excludes the root.
  const root = el("span", { rule: "uppercase", children: ["stdout"] });

  const { output, before, after } = serialise(root);

  assert.equal(output, "stdout");
  assert.deepEqual(after, before);
});

test("a transform patched onto the root itself is undone with the rest", () => {
  const root = el("span", {
    inline: [["text-transform", "uppercase", "important"]],
    children: ["stdout"],
  });

  const { output } = serialise(root);

  assert.equal(output, "stdout");
  assert.equal(root.style.getPropertyValue("text-transform"), "uppercase");
  assert.equal(root.style.getPropertyPriority("text-transform"), "important");
});

type FakeListener = (event: unknown) => void;

function fakeViewport(
  options: { root?: FakeElement; userAgent?: string; probeText?: string } = {},
) {
  const world = fakeView({
    userAgent: options.userAgent ?? CHROMIUM_UA,
    probeText: options.probeText ?? "a",
    root: options.root,
  });
  const listeners = new Map<string, Set<FakeListener>>();
  const viewport = {
    contains: () => true,
    querySelector: () => null,
    ownerDocument: { defaultView: world.view },
    addEventListener(type: string, listener: FakeListener) {
      const set = listeners.get(type) ?? new Set<FakeListener>();
      set.add(listener);
      listeners.set(type, set);
    },
    removeEventListener(type: string, listener: FakeListener) {
      listeners.get(type)?.delete(listener);
    },
  };
  const dispatch = (type: string, event: unknown) => {
    for (const listener of listeners.get(type) ?? []) listener(event);
  };
  return { viewport, listeners, dispatch, world };
}

function fakeCopyEvent(overrides: Partial<CopyEventLike> = {}) {
  const written: [string, string][] = [];
  let prevented = 0;
  return {
    written,
    prevented: () => prevented,
    event: {
      ...copyEvent(overrides),
      clipboardData:
        overrides.clipboardData === null
          ? null
          : {
              setData(format: string, data: string) {
                written.push([format, data]);
              },
            },
      preventDefault() {
        prevented += 1;
      },
    },
  };
}

test("the listener writes text/plain and takes the event away from the browser", () => {
  // The fake does not model block boundaries; the module delegates that to the engine.
  const root = el("div", { children: ["first message\n\nsecond message"] });
  const { viewport, dispatch } = fakeViewport({ root });
  attachThreadFastCopy(viewport as unknown as HTMLElement);
  const copy = fakeCopyEvent();

  dispatch("copy", copy.event);

  assert.equal(copy.prevented(), 1);
  assert.deepEqual(copy.written, [
    ["text/plain", "first message\n\nsecond message"],
  ]);
});

test("a copy that serialises to nothing is neither prevented nor written to", () => {
  // A non-collapsed selection can serialise to ""; writing it would clear the clipboard.
  const root = el("div", { children: [el("img", { alt: "" })] });
  const { viewport, dispatch } = fakeViewport({ root });
  attachThreadFastCopy(viewport as unknown as HTMLElement);
  const copy = fakeCopyEvent();

  dispatch("copy", copy.event);

  assert.equal(copy.prevented(), 0);
  assert.deepEqual(copy.written, []);
});

test("a serialiser that throws leaves the copy to the browser", () => {
  // Slow and right beats fast and wrong; preventDefault has not been called yet here.
  const root = el("div", { children: ["text"] });
  const exploding = {
    ...root,
    querySelectorAll: () => {
      throw new Error("style recalc failed");
    },
  };
  const { viewport, dispatch, world } = fakeViewport({
    root: exploding as unknown as FakeElement,
  });
  world.selection.getRangeAt = () =>
    fakeRange(1, exploding as unknown as FakeElement);
  attachThreadFastCopy(viewport as unknown as HTMLElement);
  const copy = fakeCopyEvent();

  dispatch("copy", copy.event);

  assert.equal(copy.prevented(), 0);
  assert.deepEqual(copy.written, []);
});

test("a refused copy is neither prevented nor written to", () => {
  const { viewport, dispatch } = fakeViewport();
  attachThreadFastCopy(viewport as unknown as HTMLElement);
  const copy = fakeCopyEvent({ defaultPrevented: true });

  dispatch("copy", copy.event);

  assert.equal(copy.prevented(), 0);
  assert.deepEqual(copy.written, []);
});

test("an unmapped engine leaves the listener's copy to the browser", () => {
  const { viewport, dispatch } = fakeViewport({ userAgent: WEBKIT_UA });
  attachThreadFastCopy(viewport as unknown as HTMLElement);
  const copy = fakeCopyEvent();

  dispatch("copy", copy.event);

  assert.equal(copy.prevented(), 0);
  assert.deepEqual(copy.written, []);
});

test("only copy is listened for, so a cut still cuts", () => {
  // The thread is not editable, so a cut there is a no-op or belongs to the textarea.
  const { viewport, listeners } = fakeViewport();
  attachThreadFastCopy(viewport as unknown as HTMLElement);

  assert.deepEqual([...listeners.keys()], ["copy"]);
});

test("detaching removes the listener", () => {
  // The viewport remounts per thread, so a leaked handler accumulates.
  const { viewport, dispatch, listeners } = fakeViewport();
  const detach = attachThreadFastCopy(viewport as unknown as HTMLElement);

  detach();

  assert.equal(listeners.get("copy")?.size, 0);
  const copy = fakeCopyEvent();
  dispatch("copy", copy.event);
  assert.equal(copy.prevented(), 0);
});

test("an image the native iterator skips contributes no alt text", () => {
  // Chromium skips unrendered or unselectable images, so their alt text must not be added.
  for (const inline of [
    [["display", "none"]],
    [["visibility", "hidden"]],
    [["user-select", "none"]],
  ] as const) {
    const root = el("p", {
      children: [
        "before ",
        el("img", { alt: "SVG preview", inline: [...inline] }),
        " after",
      ],
    });
    assert.equal(serialise(root).output, "before  after");
  }
});

test("studio's own invisible class suppresses the alt text", () => {
  // ImagePreview is `invisible` until load, so this case is reachable.
  const root = el("p", {
    children: [
      "before ",
      el("img", { alt: "SVG preview", ruleVisibility: "hidden" }),
      " after",
    ],
  });

  assert.equal(serialise(root).output, "before  after");
});

test("a visible image still contributes its alt text", () => {
  const root = el("p", {
    children: ["before ", el("img", { alt: "SVG preview" }), " after"],
  });

  assert.equal(serialise(root).output, "before SVG preview after");
});

test("a backward selection comes back backward", () => {
  // The restore must read live boundaries (holders shift child offsets) and keep direction.
  const first = text("first paragraph");
  const second = text("second paragraph");
  const root = el("div", {
    children: [
      el("p", { children: [first] }),
      el("p", { children: [second, el("img", { alt: "a cat" })] }),
    ],
  });
  const { selection, calls } = fakeSelection(root, {
    anchorNode: second,
    anchorOffset: 16,
    focusNode: first,
    focusOffset: 0,
    bounds: {
      startContainer: first,
      startOffset: 0,
      endContainer: second,
      endOffset: 16,
    },
  });

  faithfulSelectionText(
    selection as unknown as Selection,
    root as unknown as Element,
  );

  assert.deepEqual(calls.setBaseAndExtent.at(-1), {
    anchorNode: second,
    anchorOffset: 16,
    focusNode: first,
    focusOffset: 0,
  });
});

test("a forward selection is restored from the live boundaries in order", () => {
  const first = text("first paragraph");
  const second = text("second paragraph");
  const root = el("div", {
    children: [
      el("p", { children: [first] }),
      el("p", { children: [second, el("img", { alt: "a cat" })] }),
    ],
  });
  const { selection, calls } = fakeSelection(root, {
    anchorNode: first,
    anchorOffset: 0,
    focusNode: second,
    focusOffset: 16,
    bounds: {
      startContainer: first,
      startOffset: 0,
      endContainer: second,
      endOffset: 16,
    },
  });

  faithfulSelectionText(
    selection as unknown as Selection,
    root as unknown as Element,
  );

  assert.deepEqual(calls.setBaseAndExtent.at(-1), {
    anchorNode: first,
    anchorOffset: 0,
    focusNode: second,
    focusOffset: 16,
  });
});
