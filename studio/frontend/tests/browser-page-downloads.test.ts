// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

const shell = readFileSync(new URL("../../backend/routes/browser.py", import.meta.url), "utf8");
const source = shell.slice(shell.indexOf("const linkFrom = "), shell.indexOf("const submitForm = "));

const nativeDispatch = EventTarget.prototype.dispatchEvent;

function setup() {
  class PageEventTarget extends EventTarget {}
  const capturing = new PageEventTarget();
  const bubbling = new PageEventTarget();
  class Document extends PageEventTarget {
    addEventListener(type: string, listener: EventListener, capture?: boolean) {
      (capture ? capturing : bubbling).addEventListener(type, listener);
    }
  }
  const document = new Document();
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
    for (const node of [capturing, target, bubbling]) nativeDispatch.call(node, event);
    return !event.defaultPrevented;
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
    URL,
    pageUrl: "https://a.example/docs/page",
    resolve: (href: string) => new URL(href, "https://a.example/docs/page").href,
    post: (message: Record<string, unknown>) => message.type !== "linkClick" && posted.push(message),
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

test("a download link the page fills in its own click handler saves the new file", async () => {
  const saved: unknown[] = [];
  const links: Array<Record<string, string>> = [{ href: "#" }, { href: URL.createObjectURL(new Blob(["old"])), download: "old.txt" }];
  for (const before of links) {
    const page = setup();
    const anchor = page.link(before);
    anchor.addEventListener("click", () => {
      anchor.attributes.set("href", URL.createObjectURL(new Blob(["new"], { type: "text/csv" })));
      anchor.attributes.set("download", "export.csv");
      URL.revokeObjectURL(before.href);
    });
    page.userClick(anchor);
    await settled(() => page.posted.length + page.navigated.length > 0);
    const [file] = page.posted;
    saved.push(file?.blob instanceof Blob ? [file.name, await file.blob.text()] : page.navigated);
  }
  assert.deepEqual(saved, [["export.csv", "new"], ["export.csv", "new"]]);
});

test("a link renamed in its own click handler keeps the bytes, though the caller revokes them at once", async () => {
  const page = setup();
  const href = URL.createObjectURL(new Blob(["report"], { type: "text/plain" }));
  const anchor = page.link({ href, download: "draft.txt" });
  anchor.addEventListener("click", () => anchor.attributes.set("download", "final.txt"));
  anchor.click();
  URL.revokeObjectURL(href);
  assert.deepEqual(await savedFile(page.posted), { name: "final.txt", type: "text/plain", text: "report" });
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

test("a download link to the page's own site is saved through the panel under the link's name, and the page stays", async () => {
  const saves: unknown[] = [];
  for (const download of ["data.txt", ""]) {
    const page = setup();
    page.userClick(page.link({ href: "/downloads/documents/txt/data.txt", download }));
    await settled(() => page.posted.length + page.navigated.length > 0);
    saves.push([page.posted.map((message) => ({ ...message })), page.navigated]);
  }
  const url = "https://a.example/downloads/documents/txt/data.txt";
  assert.deepEqual(saves, [
    [[{ type: "download", url, name: "data.txt" }], []],
    [[{ type: "download", url, name: "" }], []],
  ]);
});

test("a download link to another site opens it, as in a browser", async () => {
  const page = setup();
  page.userClick(page.link({ href: "https://b.example/data.txt", download: "data.txt" }));
  await settled(() => page.posted.length + page.navigated.length > 0);
  assert.deepEqual([page.posted, page.navigated], [[], ["https://b.example/data.txt"]]);
});

function shellNavigation(active = false) {
  const part = (from: string, to: string) => {
    const at = shell.indexOf(from);
    return shell.slice(at, shell.indexOf(to, at));
  };
  const code = [
    part("const navigate = ", "/* PAGE_STORAGE_JS */"),
    part("// A blank popup", "const linkFrom = "),
    part("const fileAt = ", "// Read at the click"),
    "({ navigate });",
  ].join("\n");
  const posted: Array<Record<string, unknown>> = [];
  const window: { open?: (url?: string, target?: string) => unknown } = {};
  const base = "https://a.example/page";
  const { navigate } = runInNewContext(code, {
    window,
    navigator: { userActivation: { isActive: active } },
    cfg: { url: base },
    shellOrigin: null,
    URL,
    nativeFetch: fetch,
    resolve: (href: string) => new URL(href, base).href,
    samePage: () => false,
    followHash: () => {},
    post: (message: Record<string, unknown>) => posted.push({ ...message }),
  }) as { navigate: (url: string, newTab?: boolean) => void };
  return { open: window.open!, navigate, posted };
}

test("a file the page sends a window to, or opens, is saved; it would otherwise be dropped as an outside link", async () => {
  const opened = shellNavigation();
  assert.equal(opened.open("data:text/plain,hi"), null);
  assert.deepEqual(await savedFile(opened.posted), { name: "download", type: "text/plain", text: "hi" });
  const sent = shellNavigation();
  const href = URL.createObjectURL(new Blob(["a,b"], { type: "text/csv" }));
  sent.navigate(href);
  assert.deepEqual(await savedFile(sent.posted), { name: "download", type: "text/csv", text: "a,b" });
  URL.revokeObjectURL(href);
  const blocked = shellNavigation();
  blocked.open("https://b.example/");
  await new Promise((resolve) => setTimeout(resolve, 50));
  assert.deepEqual(blocked.posted, []);
});

test("FileSaver.js's blank popup, pointed at the file, saves it", async () => {
  for (const send of ["href", "location"] as const) {
    const page = shellNavigation();
    const popup = page.open("", "_blank") as { document: { title: string; body: { innerText?: string } }; location: unknown };
    popup.document.title = popup.document.body.innerText = "downloading...";
    const url = "data:attachment/file;base64,aGk=";
    if (send === "href") (popup.location as { href: string }).href = url;
    else popup.location = url;
    assert.deepEqual(await savedFile(page.posted), { name: "download", type: "attachment/file", text: "hi" });
  }
  const page = shellNavigation(true);
  (page.open() as { location: string }).location = "https://b.example/next";
  assert.deepEqual(page.posted, [{ type: "navigate", url: "https://b.example/next", newTab: true, background: false }]);
  const late = shellNavigation(false);
  (late.open() as { location: string }).location = "https://b.example/next";
  assert.deepEqual(late.posted, [{ type: "navigate", url: "https://b.example/next", newTab: false, background: false }]);
});
