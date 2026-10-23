// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { outsideThread } from "../src/features/browser/outside-thread.ts";
import { readSrc } from "./helpers/kit.ts";

class Node {
  readonly children: Node[] = [];
  parentElement: Node | null = null;
  readonly name: string;
  readonly marks: string[];
  constructor(name: string, marks: string[] = []) {
    this.name = name;
    this.marks = marks;
  }
  add(...nodes: Node[]): this {
    for (const node of nodes) {
      node.parentElement = this;
      this.children.push(node);
    }
    return this;
  }
  matches(selector: string): boolean {
    return this.marks.includes(selector);
  }
  descendants(): Node[] {
    return this.children.flatMap((child) => [child, ...child.descendants()]);
  }
  querySelectorAll(selector: string): Node[] {
    return this.descendants().filter((node) => node.matches(selector));
  }
  querySelector(selector: string): Node | null {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

function page(withThread: boolean) {
  const inside = new Node("inside", ["overlay"]);
  const viewport = new Node("viewport", withThread ? [".aui-thread-viewport"] : []).add(
    new Node("message").add(inside),
  );
  const dock = new Node("dock", ["dock"]).add(viewport, new Node("composer"));
  const html = new Node("html").add(
    new Node("head"),
    new Node("body").add(
      new Node("portal", ["overlay"]).add(new Node("menu", ["overlay"])),
      new Node("root").add(
        new Node("sidebar").add(new Node("inline", ["overlay"])),
        new Node("main").add(
          dock,
          new Node("panel").add(new Node("cover", ["overlay"]), new Node("inset", ["dock"])),
        ),
      ),
    ),
  );
  return { html, document: { querySelector: (s: string) => html.querySelector(s), querySelectorAll: (s: string) => html.querySelectorAll(s) } };
}

const names = (nodes: unknown[]) => (nodes as Node[]).map((node) => node.name).sort();

test("the native page's overlay scan covers everything but the thread's messages", () => {
  const { document } = page(true);
  assert.deepEqual(names(outsideThread("overlay", document as never)), ["cover", "inline", "menu", "portal"]);
  assert.deepEqual(
    names(outsideThread("dock", document as never)),
    ["dock", "inset"],
    "the full-view dock encloses the thread and is still found",
  );
});

test("without a thread on the page the scan is the whole document", () => {
  const { document } = page(false);
  assert.deepEqual(names(outsideThread("overlay", document as never)), ["cover", "inline", "inside", "menu", "portal"]);
});

test("nothing that can cover a native page renders inside the thread", () => {
  const native = readSrc("features/browser/native-view.ts");
  assert.match(native, /for \(const element of outsideThread(?:<HTMLElement>)?\(selector\)\)/);
  assert.match(native, /for \(const dock of outsideThread\(\s*"\.chat-full-view-dock, \.chat-full-view-dock-minimized, \[data-native-inset\]",\s*\)\)/);
  assert.doesNotMatch(native, /document\.querySelectorAll/, "a document-wide scan walked every message of a long thread on each 300 ms recheck, about 10 ms a time beside a native page");
  for (const wrapper of ["dialog", "sheet", "alert-dialog", "popover", "dropdown-menu", "select", "tooltip", "hover-card", "context-menu", "combobox", "menubar"]) {
    assert.match(readSrc(`components/ui/${wrapper}.tsx`), /Portal/, `${wrapper} renders into a portal`);
  }
  assert.match(readSrc("components/assistant-ui/image.tsx"), /createPortal\(\s*<button[\s\S]*?data-slot="image-zoom-overlay"[\s\S]*?document\.body,?\s*\)/, "the image zoom overlay leaves the thread");
  assert.match(readSrc("app/provider.tsx"), /<Toaster\b/, "toasts render from the app shell");
  for (const attribute of ["data-native-cover", "data-native-inset"]) {
    assert.match(readSrc("features/browser/annotate-layer.tsx"), new RegExp(attribute));
  }
});
