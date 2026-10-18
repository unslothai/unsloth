// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

// The page shell's link clicks (routes/browser.py), run as shipped.
const shell = readFileSync(new URL("../../backend/routes/browser.py", import.meta.url), "utf8");
const source = shell.slice(shell.indexOf("const linkFrom = "), shell.indexOf("const submitForm = "));

const nativeDispatch = EventTarget.prototype.dispatchEvent;

function setup() {
  class PageEventTarget extends EventTarget {}
  const document = new PageEventTarget();
  class Element extends PageEventTarget {
    isConnected = false;
    attributes: Map<string, string>;
    constructor(attributes: Record<string, string>) {
      super();
      this.attributes = new Map(Object.entries(attributes));
    }
    getAttribute(name: string) {
      return this.attributes.get(name) ?? null;
    }
    hasAttribute(name: string) {
      return this.attributes.has(name);
    }
    matches(selector: string) {
      return selector === "a[href], area[href]" && this.hasAttribute("href");
    }
  }
  const fire = (target: Element, event: Event) => {
    if (!target.isConnected) return nativeDispatch.call(target, event);
    Object.defineProperty(event, "composedPath", { value: () => [target, document] });
    return nativeDispatch.call(document, event);
  };
  class HTMLAnchorElement extends Element {
    click() {
      fire(this, new Event("click", { cancelable: true }));
    }
  }
  const posted: Array<Record<string, unknown>> = [];
  const navigated: string[] = [];
  runInNewContext(source, {
    document,
    Element,
    HTMLAnchorElement,
    EventTarget: PageEventTarget,
    Event,
    setTimeout,
    nativeFetch: fetch,
    post: (message: Record<string, unknown>) => posted.push(message),
    navigate: (url: string) => navigated.push(url),
    followHash: () => {},
  });
  const link = (attributes: Record<string, string>, connected = true) => {
    const anchor = new HTMLAnchorElement(attributes);
    anchor.isConnected = connected;
    return anchor;
  };
  const userClick = (anchor: Element) => fire(anchor, new Event("click", { cancelable: true }));
  return { document, link, userClick, posted, navigated };
}

async function settled(done: () => boolean) {
  for (let i = 0; i < 100 && !done(); i++) await new Promise((resolve) => setTimeout(resolve, 10));
}

async function savedFile(posted: Array<Record<string, unknown>>) {
  await settled(() => posted.length > 0);
  assert.equal(posted.length, 1);
  const [{ type, name, blob }] = posted;
  assert.equal(type, "download");
  assert.ok(blob instanceof Blob);
  return { name, type: blob.type, text: await blob.text() };
}

test("a download link to a data: URL hands its file to the panel", async () => {
  const page = setup();
  page.userClick(page.link({ href: "data:text/csv,name%2Cage", download: "people.csv" }));
  assert.deepEqual(await savedFile(page.posted), { name: "people.csv", type: "text/csv", text: "name,age" });
  assert.deepEqual(page.navigated, []);
});

test("a blob saved through a link outside the page is handed over, though the page revokes it at once", async () => {
  for (const click of ["dispatchEvent", "click"] as const) {
    const page = setup();
    const href = URL.createObjectURL(new Blob(['{"a":1}'], { type: "application/json" }));
    const anchor = page.link({ href, download: "convertcsv.json" }, false);
    if (click === "dispatchEvent") anchor.dispatchEvent(new Event("click"));
    else anchor.click();
    URL.revokeObjectURL(href);
    assert.deepEqual(await savedFile(page.posted), {
      name: "convertcsv.json",
      type: "application/json",
      text: '{"a":1}',
    });
  }
});

test("a blob saved through a link in the page is handed over", async () => {
  const page = setup();
  const href = URL.createObjectURL(new Blob(["hello blob"], { type: "text/plain" }));
  page.link({ href, download: "" }).click();
  URL.revokeObjectURL(href);
  assert.deepEqual(await savedFile(page.posted), { name: "download", type: "text/plain", text: "hello blob" });
});

test("ordinary links still navigate, and a download the page cancels saves nothing", async () => {
  const page = setup();
  page.userClick(page.link({ href: "https://example.com/next" }));
  await settled(() => page.navigated.length > 0);
  assert.deepEqual(page.navigated, ["https://example.com/next"]);
  const routed = setup();
  routed.document.addEventListener("click", (event) => event.preventDefault());
  routed.userClick(routed.link({ href: "data:text/plain,x", download: "x.txt" }));
  await new Promise((resolve) => setTimeout(resolve, 100));
  assert.deepEqual([routed.posted, routed.navigated], [[], []]);
  assert.deepEqual(page.posted, []);
});
