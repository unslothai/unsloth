// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The highlight controller keeps a Range alive for four seconds. These tests
// pin its mutation behavior: unrelated DOM churn must not clear a still-exact
// range, while a changed passage must.

import assert from "node:assert/strict";
import test from "node:test";

import {
  clearQuoteHighlight,
  paintQuoteHighlight,
} from "../src/components/assistant-ui/message-quote-highlight.ts";

class FakeElement {
  readonly attributes = new Map<string, string>();
  readonly isConnected = true;
  content: FakeElement | null = null;
  readonly tagName: string;

  constructor(tagName: string) {
    this.tagName = tagName;
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
  }

  removeAttribute(name: string): void {
    this.attributes.delete(name);
  }

  hasAttribute(name: string): boolean {
    return this.attributes.has(name);
  }

  querySelector(selector: string): FakeElement | null {
    return selector === ".aui-assistant-message-content"
      ? this.content
      : null;
  }

  addEventListener(): void {}
  removeEventListener(): void {}
}

class FakeHighlight {
  ranges: Range[];
  priority = 0;

  constructor(...ranges: Range[]) {
    this.ranges = ranges;
  }

  clear(): void {
    this.ranges = [];
  }
}

class FakeMutationObserver {
  private active = true;
  private readonly callback: () => void;

  constructor(callback: () => void) {
    this.callback = callback;
    observers.push(this);
  }

  observe(): void {}
  disconnect(): void {
    this.active = false;
  }

  fire(): void {
    if (this.active) {
      this.callback();
    }
  }
}

const observers: FakeMutationObserver[] = [];
const frames = new Map<number, () => void>();
let nextFrame = 1;

function createWorld() {
  const registry = new Map<string, FakeHighlight>();
  const previous = {
    document: (globalThis as { document?: unknown }).document,
    window: (globalThis as { window?: unknown }).window,
    CSS: (globalThis as { CSS?: unknown }).CSS,
    Highlight: (globalThis as { Highlight?: unknown }).Highlight,
    MutationObserver: (globalThis as { MutationObserver?: unknown }).MutationObserver,
    requestAnimationFrame: (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame,
    cancelAnimationFrame: (globalThis as { cancelAnimationFrame?: unknown }).cancelAnimationFrame,
  };

  const document = {
    createElement: (tagName: string) => new FakeElement(tagName.toUpperCase()),
  };
  const window = {
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  };
  const CSS = { highlights: registry };
  const requestAnimationFrame = (callback: () => void) => {
    const id = nextFrame++;
    frames.set(id, callback);
    return id;
  };
  const cancelAnimationFrame = (id: number) => {
    frames.delete(id);
  };

  (globalThis as { document?: unknown }).document = document;
  (globalThis as { window?: unknown }).window = window;
  (globalThis as { CSS?: unknown }).CSS = CSS;
  (globalThis as { Highlight?: unknown }).Highlight = FakeHighlight;
  (globalThis as { MutationObserver?: unknown }).MutationObserver =
    FakeMutationObserver;
  (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame =
    requestAnimationFrame;
  (globalThis as { cancelAnimationFrame?: unknown }).cancelAnimationFrame =
    cancelAnimationFrame;

  const row = new FakeElement("DIV");
  const content = new FakeElement("DIV");
  const viewport = new FakeElement("DIV");
  row.content = content;

  const range = {
    startContainer: { isConnected: true },
    endContainer: { isConnected: true },
    collapsed: false,
    text: "alpha onebeta two",
    toString() {
      return this.text;
    },
  };

  return {
    registry,
    row,
    content,
    viewport,
    range,
    mutate: () => {
      for (const observer of [...observers]) {
        observer.fire();
      }
    },
    flush: () => {
      while (frames.size > 0) {
        const pending = [...frames.values()];
        frames.clear();
        for (const callback of pending) {
          callback();
        }
      }
    },
    restore: () => {
      clearQuoteHighlight();
      observers.length = 0;
      frames.clear();
      (globalThis as { document?: unknown }).document = previous.document;
      (globalThis as { window?: unknown }).window = previous.window;
      (globalThis as { CSS?: unknown }).CSS = previous.CSS;
      (globalThis as { Highlight?: unknown }).Highlight = previous.Highlight;
      (globalThis as { MutationObserver?: unknown }).MutationObserver =
        previous.MutationObserver;
      (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame =
        previous.requestAnimationFrame;
      (globalThis as { cancelAnimationFrame?: unknown }).cancelAnimationFrame =
        previous.cancelAnimationFrame;
    },
  };
}

test("unrelated DOM mutations preserve an exact highlight", () => {
  const world = createWorld();
  try {
    paintQuoteHighlight(world.range as unknown as Range, {
      row: world.row as unknown as HTMLElement,
      viewport: world.viewport as unknown as HTMLElement,
      expectedText: "alpha onebeta two",
      durationMs: 60_000,
    });
    assert.ok(world.registry.get("unsloth-quote"));
    assert.equal(world.row.hasAttribute("data-quote-source-highlight"), true);

    world.mutate();
    world.flush();

    assert.ok(
      world.registry.get("unsloth-quote"),
      "an unrelated mutation must keep the highlight",
    );
    assert.equal(world.row.hasAttribute("data-quote-source-highlight"), true);
  } finally {
    world.restore();
  }
});

test("mutating the quoted passage clears the highlight", () => {
  const world = createWorld();
  try {
    paintQuoteHighlight(world.range as unknown as Range, {
      row: world.row as unknown as HTMLElement,
      viewport: world.viewport as unknown as HTMLElement,
      expectedText: "alpha onebeta two",
      durationMs: 60_000,
    });
    assert.ok(world.registry.get("unsloth-quote"));

    world.range.text = "alpha one changed beta two";
    world.mutate();
    world.flush();

    assert.equal(world.registry.get("unsloth-quote"), undefined);
    assert.equal(world.row.hasAttribute("data-quote-source-highlight"), false);
  } finally {
    world.restore();
  }
});
