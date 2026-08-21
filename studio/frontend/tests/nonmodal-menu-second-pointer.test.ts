// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Two pointers can be down at once -- a finger and a mouse on a touchscreen laptop -- and only
// one of them owes the click `swallowClick` exists to eat. `installDismissingClickGuard`'s
// `pointerdown` handler already filters a second pointer's PRESS. Its release was not filtered,
// and the release is the half that retires the guard: `startGrace` took no event, so a second
// pointer's `pointerup` marked the FIRST pointer's gesture as released and started its 500 ms
// bound. Measured on chromium with real CDP touch, hold the unconfirmed "Delete message" button
// with a finger while the action-bar menu is open, press and release inside the menu with the
// mouse, wait past the bound and lift the finger: the message was deleted.
//
// The state machine is driven here against a fake document rather than through a browser,
// because the ordering that produces the bug is a handful of events and a clock, and a test that
// can advance the clock by hand pins the bound itself rather than a sleep that happens to be
// longer than it. tests/studio/probe_dismiss_guard.py carries the same case against real
// engines with real input.

import assert from "node:assert/strict";
import test from "node:test";

// ---------------------------------------------------------------------------
// The smallest document that answers every question menu-dismiss.ts asks of one.
// ---------------------------------------------------------------------------

type Listener = (event: FakeEvent) => void;

interface FakeEvent {
  type: string;
  target?: unknown;
  detail?: number;
  button?: number;
  pointerId?: number;
  pointerType?: string;
  key?: string;
  stopped?: boolean;
  prevented?: boolean;
  stopPropagation?: () => void;
  preventDefault?: () => void;
}

class FakeNode {}

/** `closest` answers the one selector the guard uses: is this inside a menu surface. */
class FakeElement extends FakeNode {
  inMenu: boolean;
  readonly parent?: FakeElement;
  readonly textEntry: boolean;
  constructor(inMenu: boolean, parent?: FakeElement, textEntry = false) {
    super();
    this.inMenu = inMenu;
    this.parent = parent;
    this.textEntry = textEntry;
  }
  closest(): FakeElement | null {
    if (this.inMenu) return this;
    for (let node = this.parent; node; node = node.parent) {
      if (node.inMenu) return node;
    }
    return null;
  }
  contains(candidate: unknown): boolean {
    for (
      let node = candidate instanceof FakeElement ? candidate : undefined;
      node;
      node = node.parent
    ) {
      if (node === this) return true;
    }
    return false;
  }
  matches(): boolean {
    return this.textEntry;
  }
}

class FakeHTMLElement extends FakeElement {
  isContentEditable = false;
  blurCount = 0;
  blur(): void {
    this.blurCount += 1;
    if (fakeDocument.activeElement === this) fakeDocument.activeElement = null;
  }
}

class FakeMouseEvent implements FakeEvent {
  detail = 0;
  type: string;
  constructor(type: string) {
    this.type = type;
  }
}

class FakeDocument {
  private readonly capturing = new Map<string, Listener[]>();
  private readonly bubbling = new Map<string, Listener[]>();
  activeElement: unknown = null;

  addEventListener(type: string, fn: Listener, capture?: boolean): void {
    const bucket = capture ? this.capturing : this.bubbling;
    const list = bucket.get(type) ?? [];
    if (!list.includes(fn)) list.push(fn);
    bucket.set(type, list);
  }

  removeEventListener(type: string, fn: Listener, capture?: boolean): void {
    const bucket = capture ? this.capturing : this.bubbling;
    const list = bucket.get(type);
    if (!list) return;
    const at = list.indexOf(fn);
    if (at >= 0) list.splice(at, 1);
  }

  /** Capture first, then bubble, and `stopPropagation` in capture ends the dispatch. */
  dispatchEvent(event: FakeEvent): void {
    event.stopped = false;
    event.prevented = false;
    event.stopPropagation = () => {
      event.stopped = true;
    };
    event.preventDefault = () => {
      event.prevented = true;
    };
    for (const phase of [this.capturing, this.bubbling]) {
      for (const fn of [...(phase.get(event.type) ?? [])]) {
        if (event.stopped) return;
        fn(event);
      }
      if (event.stopped) return;
    }
  }
}

/** A clock the test advances by hand, so the 500 ms bound is pinned rather than slept through. */
class FakeWindow {
  private next = 1;
  private readonly timers = new Map<number, { at: number; fn: () => void }>();
  private now = 0;
  private readonly windowListeners = new Map<string, Listener[]>();

  setTimeout(fn: () => void, ms: number): number {
    const id = this.next++;
    this.timers.set(id, { at: this.now + ms, fn });
    return id;
  }
  clearTimeout(id: number | undefined): void {
    if (id !== undefined) this.timers.delete(id);
  }
  addEventListener(type: string, fn: Listener): void {
    const list = this.windowListeners.get(type) ?? [];
    if (!list.includes(fn)) list.push(fn);
    this.windowListeners.set(type, list);
  }
  removeEventListener(type: string, fn: Listener): void {
    const list = this.windowListeners.get(type);
    if (!list) return;
    const at = list.indexOf(fn);
    if (at >= 0) list.splice(at, 1);
  }
  dispatchEvent(event: FakeEvent): void {
    for (const fn of [...(this.windowListeners.get(event.type) ?? [])]) fn(event);
  }
  advance(ms: number): void {
    this.now += ms;
    for (const [id, timer] of [...this.timers]) {
      if (timer.at <= this.now) {
        this.timers.delete(id);
        timer.fn();
      }
    }
  }
}

const fakeDocument = new FakeDocument();
const fakeWindow = new FakeWindow();

const globals = globalThis as unknown as Record<string, unknown>;
globals.document = fakeDocument;
globals.window = fakeWindow;
globals.Node = FakeNode;
globals.Element = FakeElement;
globals.HTMLElement = FakeHTMLElement;
globals.MouseEvent = FakeMouseEvent;

const { installDismissingClickGuard } = await import(
  "../src/lib/menu-dismiss.ts"
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const OUTSIDE = new FakeHTMLElement(false);
const INSIDE_MENU = new FakeHTMLElement(true);

const down = (
  pointerId: number,
  pointerType: string,
  target: unknown,
): void => {
  fakeDocument.dispatchEvent({
    type: "pointerdown",
    pointerId,
    pointerType,
    button: 0,
    target,
  });
};
const up = (pointerId: number, pointerType: string): void => {
  fakeDocument.dispatchEvent({ type: "pointerup", pointerId, pointerType });
};

/**
 * Dispatch the compatibility click a pointer's release synthesises, and answer whether it
 * survived to the bubble phase. The guard's whole job is that it does not.
 *
 * Matched on TARGET, not on "a click arrived": when the swallowed gesture was a touch the guard
 * re-raises a bare `click` at `document` to release Radix's deferred touch dismissal, and
 * counting that one would report every touch case as delivered on a tree that swallowed
 * correctly.
 */
function clickReachedTheControl(target: unknown): boolean {
  let delivered = false;
  const watch = (event: FakeEvent): void => {
    if (event.target === target) delivered = true;
  };
  fakeDocument.addEventListener("click", watch, false);
  try {
    fakeDocument.dispatchEvent({ type: "click", detail: 1, target });
  } finally {
    fakeDocument.removeEventListener("click", watch, false);
  }
  return delivered;
}

/** A guard for exactly one test, always removed, so no test inherits another's arm state. */
function withOpenMenu(body: () => void): void {
  const remove = installDismissingClickGuard();
  try {
    body();
  } finally {
    remove();
    // Whatever state the body left, end the gesture: a new press supersedes any arm, and the
    // release plus the bound retires it, so the next test starts from a disarmed module.
    down(999, "mouse", INSIDE_MENU);
    up(999, "mouse");
    fakeWindow.advance(5000);
  }
}

// ---------------------------------------------------------------------------
// The bug
// ---------------------------------------------------------------------------

test("a second pointer's release does not retire the gesture that armed the guard", () => {
  withOpenMenu(() => {
    // A finger presses and HOLDS a control outside the menu. Radix defers a touch dismissal to
    // the resulting click, so the menu is still open underneath.
    down(11, "touch", OUTSIDE);
    // A mouse presses INSIDE the still-open menu and releases there. Its press is already
    // filtered; its release is the one that was not.
    down(22, "mouse", INSIDE_MENU);
    up(22, "mouse");
    // Past CLICK_GRACE_MS. If that release started the bound on the FINGER's behalf, the guard
    // is gone by now and the finger is still down.
    fakeWindow.advance(900);
    assert.equal(
      clickReachedTheControl(OUTSIDE),
      false,
      "the held finger's compatibility click reached the control underneath: a second " +
        "pointer's pointerup retired a gesture that was never released",
    );
  });
});

test("a second pointer's cancel does not retire it either", () => {
  withOpenMenu(() => {
    down(11, "touch", OUTSIDE);
    down(22, "mouse", INSIDE_MENU);
    fakeDocument.dispatchEvent({
      type: "pointercancel",
      pointerId: 22,
      pointerType: "mouse",
    });
    fakeWindow.advance(900);
    assert.equal(
      clickReachedTheControl(OUTSIDE),
      false,
      "a second pointer being cancelled disarmed a guard whose own pointer was still down",
    );
  });
});

// ---------------------------------------------------------------------------
// and the other direction, which is just as broken and looks identical from outside
// ---------------------------------------------------------------------------

test("the armed pointer's OWN release still starts the bound", () => {
  withOpenMenu(() => {
    down(11, "mouse", OUTSIDE);
    up(11, "mouse");
    fakeWindow.advance(900);
    assert.equal(
      clickReachedTheControl(OUTSIDE),
      true,
      "a gesture that produced no click must not leave the swallower armed for an unrelated " +
        "one; filtering by pointer id must not cost the bound its own release",
    );
  });
});

test("the armed pointer's own click is still swallowed, once", () => {
  withOpenMenu(() => {
    down(11, "mouse", OUTSIDE);
    up(11, "mouse");
    assert.equal(
      clickReachedTheControl(OUTSIDE),
      false,
      "the dismissing click is the one click this guard exists to eat",
    );
    assert.equal(
      clickReachedTheControl(OUTSIDE),
      true,
      "and exactly one: the click after it is the user's",
    );
  });
});

test("a new gesture supersedes an armed no-click gesture after menu cleanup", () => {
  const remove = installDismissingClickGuard();
  try {
    down(11, "mouse", OUTSIDE);
    up(11, "mouse");
    // Radix unmounts the menu content after the outside press. The first gesture pressed a
    // disabled native button, so it produced no click, but the swallower remains armed in grace.
    remove();

    down(12, "mouse", OUTSIDE);
    up(12, "mouse");
    assert.equal(
      clickReachedTheControl(OUTSIDE),
      true,
      "the first click of a new gesture was swallowed after the menu's watcher unmounted",
    );
  } finally {
    remove();
    fakeWindow.advance(5000);
  }
});

test("the armed pointer's own cancel still disarms", () => {
  withOpenMenu(() => {
    down(11, "touch", OUTSIDE);
    fakeDocument.dispatchEvent({
      type: "pointercancel",
      pointerId: 11,
      pointerType: "touch",
    });
    assert.equal(
      clickReachedTheControl(OUTSIDE),
      true,
      "a cancelled gesture produces no click of its own, so the guard must not survive it",
    );
  });
});

// ---------------------------------------------------------------------------
// Focus acquired by a guarded press
// ---------------------------------------------------------------------------

test("a drag-retargeted click releases the control focused by the guarded press", () => {
  withOpenMenu(() => {
    const clickAncestor = new FakeHTMLElement(false);
    const button = new FakeHTMLElement(false, clickAncestor);
    const icon = new FakeElement(false, button);
    fakeDocument.activeElement = INSIDE_MENU;

    down(11, "mouse", icon);
    // Pointerdown's default action runs after the capture listener and focuses the button.
    fakeDocument.activeElement = button;
    up(11, "mouse");

    assert.equal(
      clickReachedTheControl(clickAncestor),
      false,
      "the drag-retargeted click must still be swallowed",
    );
    assert.equal(
      button.blurCount,
      1,
      "focus cleanup must use the original press target, not the retargeted click ancestor",
    );
  });
});

test("focus moved elsewhere during the gesture is preserved", () => {
  withOpenMenu(() => {
    const clickAncestor = new FakeHTMLElement(false);
    const pressedButton = new FakeHTMLElement(false, clickAncestor);
    const icon = new FakeElement(false, pressedButton);
    const appFocusedControl = new FakeHTMLElement(false);
    fakeDocument.activeElement = INSIDE_MENU;

    down(11, "mouse", icon);
    fakeDocument.activeElement = appFocusedControl;
    up(11, "mouse");
    clickReachedTheControl(clickAncestor);

    assert.equal(
      appFocusedControl.blurCount,
      0,
      "the guard must not blur focus the app moved away from the pressed control",
    );
  });
});

test("a control focused before the guarded press is preserved", () => {
  withOpenMenu(() => {
    const button = new FakeHTMLElement(false);
    fakeDocument.activeElement = button;

    down(11, "mouse", button);
    up(11, "mouse");
    assert.equal(clickReachedTheControl(button), false);

    assert.equal(button.blurCount, 0, "the press did not acquire this existing focus");
    assert.equal(fakeDocument.activeElement, button);
  });
});

test("window blur retires the guard and releases press-acquired focus", () => {
  withOpenMenu(() => {
    const button = new FakeHTMLElement(false);
    fakeDocument.activeElement = INSIDE_MENU;

    down(11, "mouse", button);
    fakeDocument.activeElement = button;
    fakeWindow.dispatchEvent({ type: "blur" });

    assert.equal(button.blurCount, 1);
    assert.equal(
      clickReachedTheControl(button),
      true,
      "a guard retired on window blur must not swallow a later click",
    );
  });
});

test("a no-click grace timeout also releases press-acquired focus", () => {
  withOpenMenu(() => {
    const button = new FakeHTMLElement(false);
    fakeDocument.activeElement = INSIDE_MENU;

    down(11, "mouse", button);
    fakeDocument.activeElement = button;
    up(11, "mouse");
    fakeWindow.advance(900);

    assert.equal(button.blurCount, 1);
    assert.equal(clickReachedTheControl(button), true);
  });
});

test("text-entry focus acquired by the guarded press is preserved", () => {
  withOpenMenu(() => {
    const input = new FakeHTMLElement(false, undefined, true);
    fakeDocument.activeElement = INSIDE_MENU;

    down(11, "mouse", input);
    fakeDocument.activeElement = input;
    up(11, "mouse");
    assert.equal(clickReachedTheControl(input), false);

    assert.equal(input.blurCount, 0, "typing focus must keep its caret");
    assert.equal(fakeDocument.activeElement, input);
  });
});
