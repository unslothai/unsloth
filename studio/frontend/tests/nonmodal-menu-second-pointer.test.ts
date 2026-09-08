// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pin the multi-pointer ordering and release grace period with a deterministic fake DOM.

import assert from "node:assert/strict";
import test from "node:test";

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

/** Minimal `closest` implementation for the guard's menu check. */
class FakeElement extends FakeNode {
  inMenu: boolean;
  inert = false;
  readonly parent?: FakeElement;
  readonly textEntry: boolean;
  constructor(inMenu: boolean, parent?: FakeElement, textEntry = false) {
    super();
    this.inMenu = inMenu;
    this.parent = parent;
    this.textEntry = textEntry;
  }
  closest(selector?: string): FakeElement | null {
    if (selector === "[inert]") {
      if (this.inert) return this;
      for (let node = this.parent; node; node = node.parent) {
        if (node.inert) return node;
      }
      return null;
    }
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
  isConnected = true;
  disabled = false;
  acceptsFocus = true;
  blurCount = 0;
  focusCount = 0;
  focusOptions: FocusOptions | undefined;
  blur(): void {
    this.blurCount += 1;
    if (fakeDocument.activeElement === this) {
      fakeDocument.activeElement = fakeDocument.body;
    }
  }
  focus(options?: FocusOptions): void {
    this.focusCount += 1;
    this.focusOptions = options;
    if (this.isConnected && !this.disabled && this.acceptsFocus) {
      fakeDocument.activeElement = this;
    }
  }
  matches(selector?: string): boolean {
    if (selector === ":disabled") return this.disabled;
    return super.matches();
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
  readonly body = new FakeHTMLElement(false);
  activeElement: unknown = this.body;

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

  /** Dispatch capture then bubble listeners. */
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

/** Deterministic timer clock. */
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
    for (const fn of [...(this.windowListeners.get(event.type) ?? [])])
      fn(event);
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

/** Dispatch a target click and report whether it reached bubble listeners. */
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

/** Install a guard for one isolated test. */
function withOpenMenu(
  body: () => void,
  triggerRef?: { current: FakeHTMLElement | null },
): void {
  const install = installDismissingClickGuard as (ref?: {
    current: FakeHTMLElement | null;
  }) => () => void;
  const remove = install(triggerRef);
  try {
    body();
  } finally {
    remove();
    down(999, "mouse", INSIDE_MENU);
    up(999, "mouse");
    fakeWindow.advance(5000);
  }
}

test("a second pointer's release does not retire the gesture that armed the guard", () => {
  withOpenMenu(() => {
    // A finger holds a control outside the menu; touch dismissal is deferred to its click.
    down(11, "touch", OUTSIDE);
    // A mouse presses and releases inside the still-open menu.
    down(22, "mouse", INSIDE_MENU);
    up(22, "mouse");
    // Advance beyond CLICK_GRACE_MS while the finger remains down.
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

test("a swallowed outside press restores focus to its captured trigger", () => {
  const trigger = new FakeHTMLElement(false);
  const triggerRef = { current: trigger as FakeHTMLElement | null };
  const outsideButton = new FakeHTMLElement(false);
  let outsideActivations = 0;
  const activateOutside = (event: FakeEvent): void => {
    if (event.target === outsideButton) outsideActivations += 1;
  };
  fakeDocument.addEventListener("click", activateOutside, false);

  try {
    withOpenMenu(() => {
      fakeDocument.activeElement = INSIDE_MENU;
      down(11, "mouse", outsideButton);
      // The menu can unmount and clear its ref before the compatibility click arrives.
      triggerRef.current = null;
      // Pointerdown's default action focuses the button after capture listeners run.
      fakeDocument.activeElement = outsideButton;
      up(11, "mouse");
      fakeDocument.dispatchEvent({
        type: "click",
        detail: 1,
        target: outsideButton,
      });

      assert.equal(
        outsideActivations,
        0,
        "the dismissing click reached the outside button",
      );
      assert.equal(
        fakeDocument.activeElement,
        trigger,
        "focus fell back to BODY instead of returning to the menu trigger",
      );
      assert.deepEqual(trigger.focusOptions, { preventScroll: true });

      // Space only activates a button that retained focus from the swallowed press.
      if (fakeDocument.activeElement === outsideButton) {
        fakeDocument.dispatchEvent({
          type: "click",
          detail: 0,
          target: outsideButton,
        });
      }
      assert.equal(
        outsideActivations,
        0,
        "Space activated the outside button after its dismissing click was swallowed",
      );
    }, triggerRef);
  } finally {
    fakeDocument.removeEventListener("click", activateOutside, false);
  }
});

test("an unavailable trigger falls back to blurring the pressed control", () => {
  const scenarios: [string, FakeHTMLElement | null][] = [
    ["absent", null],
    [
      "detached",
      Object.assign(new FakeHTMLElement(false), { isConnected: false }),
    ],
    ["disabled", Object.assign(new FakeHTMLElement(false), { disabled: true })],
    ["inert", Object.assign(new FakeHTMLElement(false), { inert: true })],
    [
      "unfocusable",
      Object.assign(new FakeHTMLElement(false), { acceptsFocus: false }),
    ],
  ];

  for (const [label, trigger] of scenarios) {
    withOpenMenu(
      () => {
        const outsideButton = new FakeHTMLElement(false);
        fakeDocument.activeElement = INSIDE_MENU;
        down(11, "mouse", outsideButton);
        fakeDocument.activeElement = outsideButton;
        up(11, "mouse");
        assert.equal(clickReachedTheControl(outsideButton), false, label);

        assert.equal(outsideButton.blurCount, 1, label);
        assert.equal(fakeDocument.activeElement, fakeDocument.body, label);
      },
      { current: trigger },
    );
  }
});

test("a new gesture supersedes an armed no-click gesture after menu cleanup", () => {
  const remove = installDismissingClickGuard();
  try {
    down(11, "mouse", OUTSIDE);
    up(11, "mouse");
    // The first gesture produced no click, but the swallower remains armed during grace.
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
  const trigger = new FakeHTMLElement(false);
  withOpenMenu(
    () => {
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
      assert.equal(trigger.focusCount, 0);
    },
    { current: trigger },
  );
});

test("a control focused before the guarded press is preserved", () => {
  withOpenMenu(() => {
    const button = new FakeHTMLElement(false);
    fakeDocument.activeElement = button;

    down(11, "mouse", button);
    up(11, "mouse");
    assert.equal(clickReachedTheControl(button), false);

    assert.equal(
      button.blurCount,
      0,
      "the press did not acquire this existing focus",
    );
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
  const trigger = new FakeHTMLElement(false);
  withOpenMenu(
    () => {
      const input = new FakeHTMLElement(false, undefined, true);
      fakeDocument.activeElement = INSIDE_MENU;

      down(11, "mouse", input);
      fakeDocument.activeElement = input;
      up(11, "mouse");
      assert.equal(clickReachedTheControl(input), false);

      assert.equal(input.blurCount, 0, "typing focus must keep its caret");
      assert.equal(fakeDocument.activeElement, input);
      assert.equal(trigger.focusCount, 0);
    },
    { current: trigger },
  );
});
