// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { browserScopeContains } from "../../src/features/browser/browser-scope.ts";

class TestElement {
  parentElement: TestElement | null = null;
  dataset: { browserScopeTab?: string } = {};
  blocked = false;
  children: TestElement[] = [];
  contains(node: TestElement): boolean {
    return node === this || this.children.some((child) => child.contains(node));
  }
  closest(selector: string): TestElement | null {
    for (let node: TestElement | null = this; node; node = node.parentElement) {
      if (selector === "[data-browser-scope-tab]" && node.dataset.browserScopeTab) return node;
      if (selector === "[inert], [hidden]" && node.blocked) return node;
    }
    return null;
  }
}
Object.assign(globalThis, { Element: TestElement });
const asNode = (node: TestElement) => node as unknown as Node;

test("zoom and find scopes include only the active tab's visible frame and annotation portal", () => {
  const section = new TestElement();
  const toolbar = new TestElement();
  section.children.push(toolbar);
  const frameHost = new TestElement();
  frameHost.dataset.browserScopeTab = "a";
  const frame = new TestElement();
  frame.parentElement = frameHost;
  const noteHost = new TestElement();
  noteHost.dataset.browserScopeTab = "a";
  const note = new TestElement();
  note.parentElement = noteHost;
  assert.equal(browserScopeContains(section as unknown as HTMLElement, "a", asNode(toolbar)), true);
  assert.equal(browserScopeContains(section as unknown as HTMLElement, "a", asNode(frame)), true);
  assert.equal(browserScopeContains(section as unknown as HTMLElement, "a", asNode(note)), true);
  assert.equal(browserScopeContains(section as unknown as HTMLElement, "b", asNode(frame)), false);
  frameHost.blocked = true;
  noteHost.blocked = true;
  assert.equal(browserScopeContains(section as unknown as HTMLElement, "a", asNode(frame)), false);
  assert.equal(browserScopeContains(section as unknown as HTMLElement, "a", asNode(note)), false);
});
