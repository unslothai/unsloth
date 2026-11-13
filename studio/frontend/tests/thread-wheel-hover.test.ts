// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { mock, test } from "node:test";

import { attachWheelHoverSuppression } from "../src/components/assistant-ui/thread-wheel-hover.ts";
import { readSrc } from "./helpers/kit.ts";

globalThis.MouseEvent ??=
  class extends Event {} as unknown as typeof MouseEvent;

function message(id: string) {
  const got: string[] = [];
  const el = {
    id,
    got,
    hover: false,
    matches: (selector: string) =>
      selector === "[data-message-id]" ||
      (selector === "[data-message-id]:hover" && el.hover),
    dispatchEvent(event: Event) {
      thread.boundary(event, el);
      return true;
    },
  };
  return el;
}

type Message = ReturnType<typeof message>;

const thread = {
  listeners: new Map<string, (event: Event) => void>(),
  messages: [] as Message[],
  boundary(event: Event, target: Message) {
    let stopped = false;
    Object.defineProperty(event, "target", { value: target });
    event.stopPropagation = () => {
      stopped = true;
    };
    this.listeners.get(`${event.type}:capture`)?.(event);
    if (!stopped) target.got.push(event.type);
  },
  fire(type: string, extra: Record<string, unknown> = {}) {
    this.listeners.get(type)?.({
      type,
      timeStamp: performance.now(),
      ...extra,
    } as unknown as Event);
  },
};

function attach(...ids: string[]) {
  thread.listeners.clear();
  thread.messages = ids.map(message);
  const viewport = {
    addEventListener(
      type: string,
      fn: (event: Event) => void,
      options?: boolean | AddEventListenerOptions,
    ) {
      thread.listeners.set(options === true ? `${type}:capture` : type, fn);
    },
    removeEventListener(type: string, _fn: unknown, options?: boolean) {
      thread.listeners.delete(options === true ? `${type}:capture` : type);
    },
    querySelector: (selector: string) =>
      thread.messages.find((m) => m.matches(selector)) ?? null,
  };
  return {
    detach: attachWheelHoverSuppression(viewport as unknown as HTMLElement),
    messages: thread.messages,
  };
}

// the browser updates `:hover` even when the event does not reach assistant-ui.
const enter = (m: Message, buttons = 0) => {
  m.hover = true;
  const event = new MouseEvent("mouseenter");
  Object.defineProperty(event, "buttons", { value: buttons });
  thread.boundary(event, m);
};
const leave = (m: Message, buttons = 0) => {
  m.hover = false;
  const event = new MouseEvent("mouseleave");
  Object.defineProperty(event, "buttons", { value: buttons });
  thread.boundary(event, m);
};

test("a wheel scroll holds message hover events, then moves hover once it settles", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const {
      detach,
      messages: [a, b, c],
    } = attach("a", "b", "c");
    enter(a);
    thread.fire("wheel", { buttons: 0 });
    thread.fire("scroll");
    leave(a);
    enter(b);
    leave(b);
    enter(c);
    mock.timers.tick(100);
    thread.fire("scroll");
    mock.timers.tick(100);
    assert.deepEqual([a.got, b.got, c.got], [["mouseenter"], [], []]);
    mock.timers.tick(100);
    assert.deepEqual(
      [a.got, b.got, c.got],
      [["mouseenter", "mouseleave"], [], ["mouseenter"]],
    );
    leave(c);
    assert.deepEqual(c.got, ["mouseenter", "mouseleave"]);
    detach();
  } finally {
    mock.timers.reset();
  }
});

test("hover that has not moved by the time the scroll settles is left alone", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const {
      detach,
      messages: [a],
    } = attach("a", "b");
    enter(a);
    thread.fire("wheel", { buttons: 0 });
    thread.fire("scroll");
    mock.timers.tick(200);
    assert.deepEqual(a.got, ["mouseenter"]);
    detach();
  } finally {
    mock.timers.reset();
  }
});

test("a message hovered from mount, with no mouseenter, still loses hover after the scroll", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const {
      detach,
      messages: [a, b],
    } = attach("a", "b");
    a.hover = true;
    thread.fire("wheel", { buttons: 0 });
    thread.fire("scroll");
    leave(a);
    enter(b);
    mock.timers.tick(200);
    assert.deepEqual([a.got, b.got], [["mouseleave"], ["mouseenter"]]);
    detach();
  } finally {
    mock.timers.reset();
  }
});

test("programmatic scrolls (follow-to-bottom) and button-held wheels keep hover events", () => {
  const {
    detach,
    messages: [a, b],
  } = attach("a", "b");
  thread.fire("scroll");
  enter(a);
  thread.fire("wheel", { buttons: 1 });
  thread.fire("scroll");
  leave(a);
  enter(b);
  assert.deepEqual(
    [a.got, b.got],
    [["mouseenter", "mouseleave"], ["mouseenter"]],
  );
  detach();
});

test("a selection drag passes through during an active wheel tail", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const {
      detach,
      messages: [a, b],
    } = attach("a", "b");
    enter(a);
    thread.fire("wheel", { buttons: 0 });
    thread.fire("scroll");
    leave(a, 1);
    enter(b, 1);
    assert.deepEqual(
      [a.got, b.got],
      [["mouseenter", "mouseleave"], ["mouseenter"]],
    );
    mock.timers.tick(200);
    assert.deepEqual(
      [a.got, b.got],
      [["mouseenter", "mouseleave"], ["mouseenter"]],
    );
    detach();
  } finally {
    mock.timers.reset();
  }
});

test("detaching mid-scroll settles hover and stops listening", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const {
      detach,
      messages: [a, b],
    } = attach("a", "b");
    enter(a);
    thread.fire("wheel", { buttons: 0 });
    thread.fire("scroll");
    leave(a);
    enter(b);
    detach();
    assert.deepEqual(
      [a.got, b.got],
      [["mouseenter", "mouseleave"], ["mouseenter"]],
    );
    assert.equal(thread.listeners.size, 0);
  } finally {
    mock.timers.reset();
  }
});

test("the thread attaches it to its viewport", () => {
  assert.match(
    readSrc("components/assistant-ui/thread.tsx"),
    /return attachWheelHoverSuppression\(viewportEl\);/,
  );
});
