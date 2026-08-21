// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A copy inside the thread writes its own `text/plain` so the browser never builds the styled
// clipboard flavour, which is where over 99% of a long thread's copy time goes.
//
// What is tested here is the DECISION, not the timing. A unit test cannot time a clipboard, and
// one that tried would be measuring the test runner. What it can do is pin every branch that
// decides whether the substitution happens, and each of those branches is a distinct way of
// handing a user the wrong text:
//
//   * taking a copy that started in a textarea, where `window.getSelection()` is not the
//     selection being copied at all
//   * taking a copy whose selection runs out of the thread, whose text has not been checked
//   * taking a copy over content the clipboard serialises differently from `Selection.toString()`
//     -- image alt text, form control values, CSS text transforms -- which would silently drop or
//     alter runs of text
//   * writing an empty string over a clipboard the browser would have left alone
//
// The types the module exports are structural on purpose, so all of this runs against plain
// objects with no DOM.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type CopyEventLike,
  type SelectionLike,
  type ThreadViewportLike,
  attachThreadFastCopy,
  decideThreadCopy,
} from "../src/components/assistant-ui/thread-fast-copy.ts";

/** A viewport holding nothing the clipboard would serialise differently, containing everything. */
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
    clipboardData: { setData: () => {} },
    ...overrides,
  };
}

test("an ordinary selection inside the thread is answered from Selection.toString()", () => {
  const decision = decideThreadCopy(
    copyEvent(),
    selectionOf("first message\n\nsecond message"),
    plainViewport(),
  );

  assert.deepEqual(decision, {
    kind: "fast",
    text: "first message\n\nsecond message",
  });
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

test("a selection that serialises to nothing does not clear the clipboard", () => {
  const decision = decideThreadCopy(
    copyEvent(),
    selectionOf(""),
    plainViewport(),
  );

  assert.deepEqual(decision, { kind: "native", reason: "empty-selection" });
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

test("content the clipboard serialises differently is left to the browser", () => {
  // Each of these was measured against a real clipboard: `Selection.toString()` and the
  // clipboard's own `text/plain` disagree, so substituting one for the other would change what
  // the user copied.
  const cases: ReadonlyArray<readonly [string, string]> = [
    // Blink's EmitsImageAltText: the clipboard carries the alt text, toString() drops it.
    ["image with alt text", 'img[alt]:not([alt=""])'],
    // Blink's EntersTextControls: the clipboard carries the value, toString() drops it.
    ["message being edited", "textarea"],
    ["form input", "input"],
    // Blink's IgnoresCssTextTransforms: the clipboard carries the source text, toString()
    // carries the transformed text.
    ["uppercased label", ".uppercase"],
  ];

  for (const [what, selector] of cases) {
    const decision = decideThreadCopy(
      copyEvent(),
      selectionOf("some prose and then the awkward thing"),
      plainViewport({ matches: [selector] }),
    );

    assert.deepEqual(
      decision,
      { kind: "native", reason: "clipboard-only-content" },
      what,
    );
  }
});

test("a decorative image does not block the fast path", () => {
  // alt="" emits nothing on either side, and it is what every decorative image in the thread
  // already carries. Excluding it would turn the fast path off for most threads for no reason.
  const viewport = plainViewport();
  const decision = decideThreadCopy(
    copyEvent(),
    selectionOf("prose next to a decorative image"),
    viewport,
  );

  assert.equal(decision.kind, "fast");
  assert.equal(viewport.queried.length, 1);
  assert.match(viewport.queried[0], /img\[alt\]:not\(\[alt=""\]\)/);
});

test("the checks run cheapest-first, so a rejected copy never walks the thread", () => {
  // The content check is the only branch that touches the DOM at scale. Anything that rejects
  // for another reason has to reject before paying for it.
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

// --- the listener itself ----------------------------------------------------------------------
// The decision above is only worth anything if it is wired to a real event and writes the flavour
// it claims to. These use a hand-rolled element rather than a DOM library: what is being pinned
// is the four calls the handler makes, and a DOM would only hide them.

type FakeListener = (event: unknown) => void;

function fakeViewport(selectionText: string) {
  const listeners = new Map<string, Set<FakeListener>>();
  const viewport = {
    contains: () => true,
    querySelector: () => null,
    ownerDocument: {
      defaultView: {
        getSelection: (): SelectionLike => selectionOf(selectionText),
      },
    },
    addEventListener(type: string, listener: FakeListener) {
      const set = listeners.get(type) ?? new Set();
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
  return { viewport, listeners, dispatch };
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
  const { viewport, dispatch } = fakeViewport(
    "first message\n\nsecond message",
  );
  attachThreadFastCopy(viewport as unknown as HTMLElement);
  const copy = fakeCopyEvent();

  dispatch("copy", copy.event);

  assert.equal(copy.prevented(), 1);
  assert.deepEqual(copy.written, [
    ["text/plain", "first message\n\nsecond message"],
  ]);
});

test("a rejected copy is neither prevented nor written to", () => {
  const { viewport, dispatch } = fakeViewport("");
  attachThreadFastCopy(viewport as unknown as HTMLElement);
  const copy = fakeCopyEvent();

  dispatch("copy", copy.event);

  assert.equal(copy.prevented(), 0);
  assert.deepEqual(copy.written, []);
});

test("only copy is listened for, so a cut still cuts", () => {
  // A cut has to mutate the document it cut from. The thread is not editable, so a cut inside it
  // is already a no-op, and one inside a message being edited belongs to that textarea.
  const { viewport, listeners } = fakeViewport("text");
  attachThreadFastCopy(viewport as unknown as HTMLElement);

  assert.deepEqual([...listeners.keys()], ["copy"]);
});

test("detaching removes the listener", () => {
  // The viewport is remounted on every thread switch, so a handler that outlived its element
  // would accumulate one per thread the user opened.
  const { viewport, dispatch, listeners } = fakeViewport("text");
  const detach = attachThreadFastCopy(viewport as unknown as HTMLElement);

  detach();

  assert.equal(listeners.get("copy")?.size, 0);
  const copy = fakeCopyEvent();
  dispatch("copy", copy.event);
  assert.equal(copy.prevented(), 0);
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

// --- how far the content check looks ----------------------------------------------------------
// Scoping it to the whole viewport would be correct and nearly useless: one image anywhere in the
// conversation would turn the fast path off for every later copy, however far from the image the
// user selected.

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

test("the content check looks at the selection's ancestor, not at the whole thread", () => {
  const message = {
    // This message holds nothing awkward.
    querySelector: () => null,
  };
  // The viewport does, elsewhere in the conversation. It must not be consulted.
  const viewport: ThreadViewportLike = {
    contains: () => true,
    querySelector: () => ({ tag: "img" }),
  };

  const decision = decideThreadCopy(
    copyEvent(),
    selectionInside(message),
    viewport,
  );

  assert.deepEqual(decision, { kind: "fast", text: "selected prose" });
});

test("an awkward element inside the selected subtree still refuses the fast path", () => {
  const message = { querySelector: () => ({ tag: "img" }) };
  const viewport: ThreadViewportLike = {
    contains: () => true,
    querySelector: () => null,
  };

  assert.deepEqual(
    decideThreadCopy(copyEvent(), selectionInside(message), viewport),
    {
      kind: "native",
      reason: "clipboard-only-content",
    },
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
    reason: "clipboard-only-content",
  });
});

test("a multi-range selection is checked against the whole viewport", () => {
  // Disjoint ranges have no common ancestor short of the viewport, so the wide check is the only
  // sound one.
  const viewport: ThreadViewportLike = {
    contains: () => true,
    querySelector: () => ({ tag: "img" }),
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
    reason: "clipboard-only-content",
  });
});
