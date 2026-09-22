// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A copy inside the thread writes its own `text/plain` so the browser never builds the styled
// clipboard flavour. Measured on smoke-heavy-thread.html at the 100K rung, a 40,626-character
// selection costs 347.0ms to copy and 11.9ms to produce ourselves, so essentially all of it.
//
// What is tested here is the DECISION and the SERIALISER, not the timing. A unit test cannot time
// a clipboard, and one that tried would be measuring the test runner. What it can do is pin the
// two things that decide whether a user gets the right bytes:
//
//   * the gate, whose every branch is a distinct way of handing somebody the wrong text -- a copy
//     that started in a textarea, where `window.getSelection()` is not the selection being copied
//     at all; a selection that runs out of the thread, whose text has not been checked; a form
//     control, whose value the clipboard emits as its own block; an engine whose `toString()` has
//     not been proven to agree with its clipboard
//   * the serialiser, which reproduces the enumerated deltas by PATCHING THE LIVE DOM and taking
//     the engine's own `toString()` back out. What has to be true of it is that the patch is
//     complete and that the document and the user's selection are exactly as they were found.
//
// There is no DOM library in this project -- no jsdom, no happy-dom, no vitest environment; the
// runner is `node --test` and every sibling test that needs a document hand-rolls one (see
// tests/overlay-scrollbar-gutter.test.ts). So the gate runs against plain objects, which is what
// its structural types are for, and the serialiser runs against the small DOM below. That DOM is
// honest about its limits: its `querySelectorAll` throws on any selector it does not really
// implement, and its `toString()` is computed from the tree at the moment it is asked, so an alt
// text only appears in the output if the module actually put a holder in the document.

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

// --- the gate ----------------------------------------------------------------------------------

/** A viewport holding no form control, containing everything. */
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
      // Stand in for a real matcher: report a hit if any of the named constructs is in the list
      // the module asked about.
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
  // The decision no longer carries the text: the string is produced separately, from the live
  // DOM, because reproducing the clipboard's deltas needs a document and the gate does not.
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
  // preventDefault() with nowhere to write the replacement copies NOTHING, which is strictly
  // worse than a slow copy.
  const decision = decideThreadCopy(
    copyEvent({ clipboardData: null }),
    selectionOf("anything"),
    plainViewport(),
  );

  assert.deepEqual(decision, { kind: "native", reason: "no-clipboard-data" });
});

test("a copy out of a text control keeps the browser's own copy", () => {
  // The edit composer and the queued-prompt editor both mount a textarea INSIDE the viewport, so
  // this is reachable, and `window.getSelection()` there is the document's selection rather than
  // the field's: substituting it would replace the copied field with unrelated text.
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
  // Dragging from the last message down into the composer, or up into the sidebar, puts the
  // common ancestor above the viewport. The text past the boundary is not text this file has
  // checked, so it is not text this file rewrites.
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
  // Firefox builds a multi-range selection from ctrl-click. One range inside the thread does not
  // make the rest of them so.
  let range = 0;
  const selection: SelectionLike = {
    isCollapsed: false,
    rangeCount: 2,
    getRangeAt: () => ({ commonAncestorContainer: { index: range++ } }),
    toString: () => "two disjoint runs",
  };
  const viewport: ThreadViewportLike = {
    // First range inside, second outside.
    contains: (node) => (node as { index: number }).index === 0,
    querySelector: () => null,
  };

  assert.deepEqual(decideThreadCopy(copyEvent(), selection, viewport), {
    kind: "native",
    reason: "selection-leaves-thread",
  });
});

test("a form control in the selected subtree hands the copy back", () => {
  // The one construct that is REFUSED rather than reproduced. Chromium's clipboard emits the
  // control's value AND treats it as its own block, and the break depends on the control: a text
  // input lands as "value\n", a select as "\nvalue\n". Guessing that wrong is the same
  // block-boundary problem the whole file exists to avoid -- and a password field would copy as
  // its mask, where guessing the glyph wrong puts a real password on the clipboard.
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
  // It used to. The clipboard emits an image's alt text and `toString()` does not, but that delta
  // is now REPRODUCED by faithfulSelectionText rather than being a reason to give up: the images
  // in a thread are message attachments, and refusing on them turned the fast path off for
  // exactly the messages most worth copying.
  const viewport = plainViewport();
  const decision = decideThreadCopy(
    copyEvent(),
    selectionOf("prose next to an image of a cat"),
    viewport,
  );

  assert.deepEqual(decision, { kind: "fast" });
  // And the gate no longer even asks about images.
  assert.deepEqual(viewport.queried, ["input, textarea, select"]);
});

test("text under a css text-transform no longer refuses the fast path", () => {
  // Also used to. Chromium's clipboard ignores text-transform and carries the source text while
  // `toString()` carries the rendered text; faithfulSelectionText neutralises the transform for
  // the length of the copy instead of handing the copy back.
  const viewport = plainViewport({ matches: [".uppercase"] });

  assert.deepEqual(
    decideThreadCopy(copyEvent(), selectionOf("STDOUT"), viewport),
    { kind: "fast" },
  );
});

test("an engine whose clipboard mapping is unproven hands the copy back", () => {
  // WebKit's `toString()` appends trailing block breaks its clipboard does not carry, and the
  // count depends on what the selection ends with: +2 after a paragraph or heading, +1 after a
  // div, list, <pre> or blockquote, +0 after a table or an inline. A clipboard that is silently
  // wrong is worse than one that is slow.
  const decision = decideThreadCopy(
    copyEvent(),
    selectionOf("a paragraph"),
    plainViewport(),
    false,
  );

  assert.deepEqual(decision, { kind: "native", reason: "unmapped-engine" });
});

test("the engine check runs after every cheaper refusal", () => {
  // It is the branch the module documents as last, and the one whose answer costs a probe of the
  // document. Anything that refuses for a cheaper reason has to say so instead.
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
  // The form control check is the only branch of the gate that queries the DOM. Anything that
  // rejects for another reason has to reject before paying for it.
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
  // lib/copy-to-clipboard.ts falls back to a hidden <textarea> appended to document.body. It is
  // outside the viewport, so the listener never sees it; and if the tree ever moves, the
  // editable-origin guard rejects it anyway. Both, because either one alone is a silent
  // dependency on the other.
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

// --- how far the form control check looks ------------------------------------------------------
// Scoping it to the whole viewport would be correct and nearly useless: one textarea anywhere in
// the conversation would turn the fast path off for every later copy, however far from it the user
// selected.

/** A range whose common ancestor is an element that answers querySelector itself. */
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
    // This message holds no form control.
    querySelector: () => null,
  };
  // The viewport does, elsewhere in the conversation. It must not be consulted.
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
  // Range.commonAncestorContainer is very often a text node, which has no querySelector.
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
  // Disjoint ranges have no common ancestor short of the viewport, so the wide check is the only
  // sound one.
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

// --- a document, hand-rolled ------------------------------------------------------------------
// Enough of one for the serialiser and the probe, and no more. Three properties make it worth
// trusting: `querySelectorAll` throws on any selector it does not really implement, so the module
// cannot quietly ask for something the fake glosses over; `toString()` is computed from the tree
// at the instant it is asked, so alt text reaches the output only if a holder was really inserted;
// and inline style is a property map with priorities, so a restore that loses `!important` shows.
//
// What it does NOT prove is the thing only a browser can: that `toString()` under these patches
// equals the clipboard's `text/plain`. That was measured, over 27 constructs, and is what the doc
// comment records.

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
  /** What a class rule gives this element. Inherited by descendants, the way CSS does it. */
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

/**
 * `patchClipboardDeltas` narrows the root with `root instanceof HTMLElement`, so a fake element
 * has to be one. The constructor is never called: elements are built as object literals and given
 * this prototype, which is all `instanceof` looks at.
 */
class FakeHtmlElement {}

Object.defineProperty(globalThis, "HTMLElement", {
  configurable: true,
  writable: true,
  value: FakeHtmlElement,
});

function text(data: string): FakeText {
  // Text nodes carry an ownerDocument too, because a selection endpoint is usually one of
  // these and that is where `captureDirection` reaches for `createRange`.
  return { nodeType: 3, data, parentNode: null, ownerDocument: fakeDocument };
}

/** Only the selectors this module actually uses. Anything else is a hole in the fake, not a pass. */
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
 * Removing or inserting a node inside a live range COLLAPSES that range in Chromium, which is the
 * whole reason faithfulSelectionText re-establishes the selection twice: once after patching and
 * once after undoing. Without modelling it, the second restore looks like dead code.
 *
 * One listener per fake selection, and they outlive their test, which is why each of them checks
 * that the mutation was inside its own root before reacting.
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
    /** What a class rule gives this element's visibility, e.g. Unsloth's `invisible`. */
    ruleVisibility?: string;
    /** [property, value, priority] triples, as an inline style attribute. */
    inline?: ReadonlyArray<readonly [string, string, string?]>;
    /** A stylesheet rule on this element, e.g. Tailwind's `uppercase`. */
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
  // THE COUPLING THE STUB WAS MISSING. On a real element the inline declaration and the `style`
  // attribute are two views of one thing: touching a property writes the attribute, and emptying
  // the declaration leaves `style=""` rather than removing it. Keeping them in separate maps is
  // why "the dom is left exactly as it was found" stayed green while every copy was leaving
  // residue in the document.
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

  // An element built WITH inline styles must carry the matching `style` attribute from the
  // start, exactly as parsed markup would. Seeding the declaration alone left `getAttribute`
  // returning null, so a restore saw "there was no style attribute" and removed the element's
  // own styling.
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
    // `removeAttribute` and the attribute NODE removal are different operations on a real
    // element, and the difference is the bug this stub failed to catch. Both are modelled.
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

/** Untransformed text, which is what `textContent` reports and what the clipboard carries. */
function renderSource(node: FakeNode): string {
  return node.nodeType === 3
    ? node.data
    : node.childNodes.map(renderSource).join("");
}

/** The cascade, as far as this module can see it: own inline, own rule, then inherited. */
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

/**
 * What `Selection.toString()` would give for everything under `node`.
 *
 * The two behaviours that matter are the ones the module patches around: text is emitted as it is
 * RENDERED, so a text-transform shows up; and an image emits nothing of its own, which is why the
 * alt text has to arrive as a real inserted node or not at all.
 */
function renderSelected(node: FakeNode): string {
  if (node.nodeType === 3) {
    if (isDisplayNone(node.parentNode)) return "";
    return applyTransform(node.data, computedTextTransform(node.parentNode));
  }
  if (isDisplayNone(node)) return "";
  if (node.tagName === "img") return "";
  return node.childNodes.map(renderSelected).join("");
}

/**
 * A structural snapshot, for asserting the document was put back.
 *
 * Inline style is compared as a SET of (property, value, priority): declaration order inside a
 * style attribute is not observable to anything the app reads, and a real CSSStyleDeclaration
 * reorders on removeProperty/setProperty anyway, so ordering here would fail a correct restore.
 */
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
  // A real Range is LIVE: its boundaries move as the DOM around them changes, which is why the
  // restore reads them instead of the raw offsets captured before the patch ran.
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

/**
 * A selection over `root`, plus the probe behaviour engineClipboardIsMapped needs.
 *
 * `selectAllChildren` switches what `toString()` answers, and `removeAllRanges` switches it back,
 * which is exactly the sequence the probe performs.
 */
function fakeSelection(
  root: FakeElement,
  options: {
    ids?: number[];
    probeText?: string;
    anchorNode?: FakeNode;
    anchorOffset?: number;
    focusNode?: FakeNode;
    focusOffset?: number;
    /** The LIVE range boundaries, which the restore reads instead of the raw offsets. */
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
  // A selection's DIRECTION lives in its anchor/focus pair and nowhere else. A cloned Range
  // carries ordered boundaries only, so a restore that rebuilds from ranges silently flips a
  // backward selection forward. Recording the calls is how a test can tell which API was used.
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
  /** Take the collapse the last mutation inside the selection would really have caused. */
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
    // ANCHOR AND FOCUS MOVE WITH THE RANGES, as they do in a real engine. A stub that left
    // them where the test set them would report the ORIGINAL direction no matter what the code
    // did to the selection, so a restore that flips a backward selection forward would pass.
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
      // Per the spec: the anchor is the range's START and the focus its END, which is exactly
      // why rebuilding this way can only ever produce a forward selection.
      anchorNode = range.startContainer;
      anchorOffset = range.startOffset;
      focusNode = range.endContainer;
      focusOffset = range.endOffset;
    },
    selectAllChildren() {
      sync();
      ranges = [];
      probing = true;
      // Whatever the engine puts here, it is the probe's node and not the user's selection.
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
      // The boundaries are restored too, so the selection is usable afterwards.
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

/**
 * `captureDirection` asks a throwaway Range whether the focus lies before the anchor, so the
 * stub needs `ownerDocument.createRange()` on every node. Document order is computed from the
 * parent chains, which is what `comparePoint` reports and needs no knowledge of a root.
 */
function chain(node: FakeNode): FakeNode[] {
  const path: FakeNode[] = [];
  for (let at: FakeNode | null = node; at !== null; at = at.parentNode)
    path.unshift(at);
  return path;
}

/** -1 if `b` precedes `a` in document order, 1 if it follows, 0 if they are the same point. */
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
    // `display` is not inherited; `visibility` and `user-select` are, which is why the module
    // can read the computed value and not walk ancestors itself. An image the native iterator
    // skips must not get an alt-text holder, so these three decide whether it does.
    display: node.styleEntries.get("display")?.value ?? "inline",
    visibility: inheritedStyle(node, "visibility", "visible"),
    userSelect: inheritedStyle(node, "user-select", "auto"),
    webkitUserSelect: inheritedStyle(node, "-webkit-user-select", "auto"),
  }),
});

/** An inherited property, resolved the way the cascade does: own inline, own rule, ancestors. */
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

// --- the behavioural probe ---------------------------------------------------------------------

const CHROMIUM_UA =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36";
const WEBKIT_UA =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15";

function fakeView(options: {
  userAgent: string;
  probeText?: string;
  root?: FakeElement;
  selectionIds?: number[];
  /** The user's own selection, when a test cares which way round it is. */
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
  // A counter rather than a throwing getter: engineClipboardIsMapped swallows everything it
  // catches, so a throw would be indistinguishable from an honest `false`.
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

/** The probe assigns `style.cssText` and `innerHTML`, which the fake element does not model. */
function asWindow(view: unknown): Window & typeof globalThis {
  return view as Window & typeof globalThis;
}

test("a non-chromium user agent is unmapped without the document being touched", () => {
  // WebKit's `toString()` appends block breaks its clipboard does not carry, so the answer is
  // known before any probe runs, and running one would cost a style recalc for nothing.
  const world = fakeView({ userAgent: WEBKIT_UA, probeText: "a" });

  assert.equal(engineClipboardIsMapped(asWindow(world.view)), false);
  assert.equal(world.documentReads(), 0);
  assert.equal(world.body.childNodes.length, 0);
});

test("a chromium engine whose toString appends a block break is unmapped", () => {
  // The probe is a hidden <p>a</p>. An engine that answers "a\n\n" to that is the WebKit
  // behaviour under a Chromium-looking user agent, and the user agent is never allowed to decide
  // what bytes are produced.
  const world = fakeView({ userAgent: CHROMIUM_UA, probeText: "a\n\n" });

  assert.equal(engineClipboardIsMapped(asWindow(world.view)), false);
  assert.equal(world.documentReads(), 1);
});

test("a chromium engine whose toString matches its clipboard is mapped", () => {
  const world = fakeView({ userAgent: CHROMIUM_UA, probeText: "a" });

  assert.equal(engineClipboardIsMapped(asWindow(world.view)), true);
  // And the probe took its own node back out of the body.
  assert.equal(world.body.childNodes.length, 0);
});

test("the probe's answer is cached on the view", () => {
  // It cannot change within a document, and it costs a node insertion, a style recalc and a round
  // trip through the user's selection.
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
  // The probe takes the selection away and rebuilds it, so it has the serialiser's problem and
  // needs the serialiser's answer. Rebuilding from the saved ranges alone always produces a
  // forward selection, so a user who dragged right to left had their highlight silently reversed
  // by the first copy in the document, and their next Shift+Arrow moved the opposite end.
  //
  // Asserted on the anchor/focus PAIR, not on the copied text, which is identical either way.
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
  // The other half of the rule, so the fix cannot be "always swap".
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
  // And by `addRange`, not `setBaseAndExtent`, which takes ONE anchor/focus pair and so cannot
  // express two ranges. Firefox is the only engine that produces them, and this fast path does
  // not run there, but dropping a range would be worse than losing its direction.
  assert.deepEqual(world.calls.setBaseAndExtent, []);
});

// --- the serialiser ----------------------------------------------------------------------------

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
  // Blink's EmitsImageAltText, which ApplyWebPreferences turns on for every web view, so this is
  // not conditional in practice. `toString()` drops it, so it has to arrive as a real node.
  const root = el("p", {
    children: ["before ", el("img", { alt: "a cat" }), " after"],
  });

  assert.equal(serialise(root).output, "before a cat after");
});

test("an image with an empty alt contributes nothing", () => {
  // alt="" is what every decorative image in the thread carries, and it emits nothing on either
  // side. An empty holder would serialise to nothing too, so the output alone does not show the
  // difference: what is checked is that no node is inserted at all. Inserting one costs a
  // document mutation inside the user's live selection, which collapses it, per image, per copy.
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
  // Unsloth's message images are display:block, so an inline holder placed beside one sits between
  // two blocks, the engine wraps it in an anonymous block, and the alt text arrives with a leading
  // newline the clipboard does not have. Measured on the real thread as 40,650 characters against
  // the clipboard's 40,648, two images each contributing one extra break. Taking the image out of
  // the flow removes the box the break came from; an image contributes no text of its own, so
  // hiding it changes nothing else.
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
  // Both engines' clipboards fold U+00A0 to a plain space; neither `toString()` does.
  const root = el("p", { children: ["one\u00a0two\u00a0three"] });

  const { output } = serialise(root);

  assert.equal(output, "one two three");
  assert.equal(output.includes("\u00a0"), false);
});

test("the source text under a text-transform is what is serialised", () => {
  // Blink's IgnoresCssTextTransforms: the clipboard carries the source text, `toString()` carries
  // the rendered text. Chromium is the engine this runs on, so the source text is the target.
  const root = el("div", {
    children: [el("span", { rule: "uppercase", children: ["stdout"] })],
  });

  assert.equal(serialise(root).output, "stdout");
});

test("a transform inherited from an ancestor is neutralised too", () => {
  // text-transform inherits, so a child with no rule of its own still renders transformed, and
  // the patch has to reach it rather than only the element carrying the class.
  const root = el("div", {
    rule: "uppercase",
    children: [el("span", { children: ["stdout"] })],
  });

  assert.equal(serialise(root).output, "stdout");
});

test("the dom is left exactly as it was found", () => {
  // A copy that left a stray <span> in the message, or an inline `display: none` on an image,
  // would be a far worse bug than the one being fixed -- and it would persist, because nothing
  // ever re-renders that subtree on its own.
  //
  // WHAT THIS TEST CANNOT PROVE, stated because it once passed while the code was wrong. These
  // run against a hand-rolled stub, so they check that the patch's own bookkeeping balances --
  // no leftover node, no leftover property. They CANNOT see how a real engine serialises what is
  // left: Chromium keeps a `style=""` attribute on an element whose inline declaration has been
  // touched, even after `removeAttribute("style")`, so every copy was permanently rewriting the
  // document while this test was green. The structural parity digest found it, six actions
  // differing against a null that matched fifteen of sixteen. The authoritative check is
  // `tests/studio/playwright_thread_fast_copy.py`, which compares `outerHTML` before and after
  // in a real browser, per construct.
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
  // The patch writes `text-transform: none !important` over whatever was there. Restoring the
  // value but dropping `!important` would silently change how the message renders from then on.
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
  // removeProperty on a property that was never there must not leave an empty declaration behind.
  const span = el("span", { rule: "uppercase", children: ["stdout"] });
  const image = el("img", { alt: "a cat" });
  const root = el("div", { children: [span, image] });

  serialise(root);

  assert.deepEqual([...span.styleEntries.keys()], []);
  assert.deepEqual([...image.styleEntries.keys()], []);
});

test("the user's selection ranges are put back", () => {
  // An alt text holder is INSERTED before the read and REMOVED after it, and each of those is a
  // node mutation inside the user's live range, which collapses it. So the ranges have to be
  // re-established twice, and the second time is after the undo. A copy that quietly dropped the
  // user's highlight would be its own bug, and a visible one.
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
  // With no transform and no alt text there is nothing to undo, and the selection is never
  // disturbed at all.
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
  // The root, not only its descendants. `scopeElement` hands the serialiser the range's common
  // ancestor, and for a selection inside one leaf -- `<span class="uppercase">stdout</span>`,
  // which the code execution card renders five times over -- that ancestor IS the transformed
  // span. `querySelectorAll("*")` does not include the element it is called on, so this wrote
  // "STDOUT" where the clipboard carries "stdout" until the root was added to the scan.
  const root = el("span", { rule: "uppercase", children: ["stdout"] });

  const { output, before, after } = serialise(root);

  assert.equal(output, "stdout");
  assert.deepEqual(after, before);
});

test("a transform patched onto the root itself is undone with the rest", () => {
  // The root is patched by a different line from its descendants, so its undo is worth its own
  // check: a leftover `text-transform: none !important` would flatten the message's own styling
  // from the first copy onwards.
  const root = el("span", {
    inline: [["text-transform", "uppercase", "important"]],
    children: ["stdout"],
  });

  const { output } = serialise(root);

  assert.equal(output, "stdout");
  assert.equal(root.style.getPropertyValue("text-transform"), "uppercase");
  assert.equal(root.style.getPropertyPriority("text-transform"), "important");
});

// --- the listener ------------------------------------------------------------------------------
// The decision and the serialiser are only worth anything wired to a real event, writing the
// flavour they claim to.

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
  // One text node, not two paragraphs: the fake DOM deliberately does not model block boundary
  // emission, because the module does not either -- delegating to the engine's own iterator
  // instead of writing a walker is the whole point of the file.
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
  // A selection can be non-collapsed and still serialise to nothing -- an image with an empty alt
  // on its own. Writing "" would clear a clipboard the browser would have left alone.
  const root = el("div", { children: [el("img", { alt: "" })] });
  const { viewport, dispatch } = fakeViewport({ root });
  attachThreadFastCopy(viewport as unknown as HTMLElement);
  const copy = fakeCopyEvent();

  dispatch("copy", copy.event);

  assert.equal(copy.prevented(), 0);
  assert.deepEqual(copy.written, []);
});

test("a serialiser that throws leaves the copy to the browser", () => {
  // The patch could not be applied or undone cleanly. Slow and right beats fast and silently
  // different, and preventDefault has not been called yet when it happens.
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
  // The range has to point at the exploding element for the serialiser to reach it.
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
  // A cut has to mutate the document it cut from. The thread is not editable, so a cut inside it
  // is already a no-op, and one inside a message being edited belongs to that textarea.
  const { viewport, listeners } = fakeViewport();
  attachThreadFastCopy(viewport as unknown as HTMLElement);

  assert.deepEqual([...listeners.keys()], ["copy"]);
});

test("detaching removes the listener", () => {
  // The viewport is remounted on every thread switch, so a handler that outlived its element
  // would accumulate one per thread the user opened.
  const { viewport, dispatch, listeners } = fakeViewport();
  const detach = attachThreadFastCopy(viewport as unknown as HTMLElement);

  detach();

  assert.equal(listeners.get("copy")?.size, 0);
  const copy = fakeCopyEvent();
  dispatch("copy", copy.event);
  assert.equal(copy.prevented(), 0);
});

// --- what the native iterator skips ------------------------------------------------------------

test("an image the native iterator skips contributes no alt text", () => {
  // Raised in review and confirmed against the real clipboard. Chromium's iterator skips an
  // image that is not rendered or not selectable, so inserting its alt text unconditionally
  // ADDED text the clipboard never carried: 'before SVG preview after' against 'before after'.
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
  // `ImagePreview` carries `invisible` until the image loads, so a copy across a message whose
  // image had not finished loading gained an alt string. This is the reachable case, not a
  // hypothetical one.
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
  // The guard must not swallow the case the feature exists for.
  const root = el("p", {
    children: ["before ", el("img", { alt: "SVG preview" }), " after"],
  });

  assert.equal(serialise(root).output, "before SVG preview after");
});

test("a backward selection comes back backward", () => {
  // A cloned Range carries ordered boundaries and no direction, so rebuilding with `addRange`
  // turned a selection dragged upwards into a forward one and the user's next Shift+Arrow moved
  // the opposite edge. Only the patched path rebuilds, hence the image.
  //
  // The restore reads the range's LIVE boundaries rather than the offsets captured before the
  // patch, because the alt holders are inserted before their images and an element/child offset
  // would then point at the wrong child. So what is asserted here is that the live boundaries
  // came back SWAPPED, which is direction and nothing else.
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
  // The other half of the same rule: forward stays forward, and both ends come from the range
  // rather than from the pre-patch offsets.
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
