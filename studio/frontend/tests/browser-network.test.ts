// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

// The page shell's fetch and XHR (routes/browser_network.py), run as shipped.
const source = readFileSync(new URL("../../backend/routes/browser_network.py", import.meta.url), "utf8")
  .split('r"""')[1]
  .split('"""')[0];

/** A stand-in for the browser's own XHR: a request ends when the test fires its loadend. */
class NativeXHR extends EventTarget {
  calls: unknown[][] = [];
  state = 0;
  get readyState() {
    return this.state;
  }
  get status() {
    return 0;
  }
  open(...args: unknown[]) {
    this.calls.push(["open", ...args]);
    this.state = 1;
  }
  setRequestHeader() {}
  send(...args: unknown[]) {
    this.calls.push(["send", ...args]);
  }
  abort() {
    this.calls.push(["abort"]);
  }
}

class ProgressEvent extends Event {}

/** As a browser's: send() on a request already in flight throws InvalidStateError. */
class StrictXHR extends NativeXHR {
  inFlight = false;
  constructor() {
    super();
    this.addEventListener("loadend", () => (this.inFlight = false));
  }
  open() {
    this.inFlight = false;
  }
  send() {
    if (this.inFlight) throw new Error("InvalidStateError");
    this.inFlight = true;
  }
}

/** Stand-ins for the page's Document and XMLSerializer, enough for an XHR's Document body. */
class PageDocument {
  contentType: string;
  markup: string;
  constructor(contentType: string, markup: string) {
    this.contentType = contentType;
    this.markup = markup;
  }
  get doctype() {
    return this.contentType === "text/html" ? { name: "html" } : null;
  }
  get documentElement() {
    return { outerHTML: this.markup };
  }
}
class PageXMLSerializer {
  serializeToString(document: PageDocument) {
    return `<?xml version="1.0"?>${document.markup}`;
  }
}

/** `pageUrl`: the page's address; with none (a local file) nothing goes through the panel.
 *  `browserReads`: own-site addresses the sandbox can read itself (the site answers its origin);
 *  the browser's fetch of any other own-site address fails, as CORS makes it. */
function setup(
  XHR: typeof NativeXHR = NativeXHR,
  pageUrl: string | null = null,
  anyOrigin = false,
  browserReads: (url: string) => boolean = () => false,
) {
  let now = 0;
  // The script patches its Response's body readers: a class of its own keeps that out of other tests.
  class PageResponse extends Response {}
  const answers: Array<(response: Response) => void> = [];
  const refused: string[] = [];
  const window = Object.assign(new EventTarget(), {
    fetch: (input: unknown) =>
      new Promise<Response>((resolve, reject) => {
        const url = input instanceof Request ? input.url : String(input);
        const own = pageUrl !== null && new URL(url, pageUrl).origin === new URL(pageUrl).origin;
        if (own && !browserReads(url)) {
          refused.push(url);
          reject(new TypeError("Failed to fetch"));
        } else answers.push(resolve);
      }),
    XMLHttpRequest: XHR,
  });
  // What the page sent its parent, the shell.
  const posted: Array<Record<string, unknown>> = [];
  const parent = {};
  const base = pageUrl ?? "about:srcdoc";
  const resolve = (href: string) => {
    try {
      return new URL(href, base).href;
    } catch {
      return null;
    }
  };
  runInNewContext(source, {
    window, Date: { now: () => now }, Response: PageResponse, Symbol, Object, Set,
    cfg: { url: pageUrl, anyOrigin }, post: (message: Record<string, unknown>) => posted.push(message), resolve, parent,
    document: { baseURI: base }, URL, Request, Headers, Blob, Event, ProgressEvent, DOMException, AbortController,
    setTimeout, clearTimeout, Document: PageDocument, XMLSerializer: PageXMLSerializer, URLSearchParams,
  });
  /** The panel's answer to request `id`, as the shell relays it. */
  const reply = (id: unknown, answer: Record<string, unknown>, source: unknown = parent) =>
    window.dispatchEvent(Object.assign(new Event("message"), {
      source,
      data: { type: "unsloth:browser-command", command: "response", id, ...answer },
    }));
  const pending = (since: number) => (window as unknown as Record<symbol, (since: number) => number>)[Symbol.for("unsloth.browser.pending")](since);
  const page = window as unknown as { fetch: (url: string, init?: RequestInit) => Promise<Response>; XMLHttpRequest: new () => NativeXHR };
  return {
    window: page, pending, advance: (ms: number) => (now += ms),
    answer: (i: number, body = "{}") => answers[i](new PageResponse(body)),
    Response: PageResponse,
    native: answers,
    refused,
    posted,
    reply,
  };
}

test("only requests started since the action count, however long they stay open", async () => {
  const t = setup();
  const poll = t.window.fetch("/stream");
  // A long poll the page already had open isn't the action's.
  t.advance(1000);
  assert.equal(t.pending(1000), 0);
  // Cross-origin requests count too: a list fetched from another API is still content on its way.
  const api = t.window.fetch("https://api.github.com/x");
  t.advance(60_000);
  assert.equal(t.pending(1000), 1);
  assert.equal(t.pending(0), 2);
  t.answer(0);
  t.answer(1);
  await Promise.all([poll, api]);
  assert.equal(t.pending(0), 0);
});

test("reading a response's body still counts, since the page draws only once it has it", async () => {
  const t = setup();
  const request = t.window.fetch("https://api.github.com/repos/x/git/trees/y");
  t.answer(0, "tree");
  const response = await request;
  assert.equal(t.pending(0), 0);
  const body = response.text();
  assert.equal(t.pending(0), 1);
  assert.equal(await body, "tree");
  assert.equal(t.pending(0), 0);
});

test("an XHR counts from send until loadend", () => {
  const t = setup();
  const xhr = new t.window.XMLHttpRequest();
  xhr.open();
  xhr.send();
  assert.equal(t.pending(0), 1);
  xhr.dispatchEvent(new Event("loadend"));
  assert.equal(t.pending(0), 0);
});

test("reopening an XHR in flight ends its count, since that request never fires loadend", () => {
  const t = setup();
  const xhr = new t.window.XMLHttpRequest();
  xhr.open();
  xhr.send();
  xhr.open();
  assert.equal(t.pending(0), 0);
  xhr.send();
  assert.equal(t.pending(0), 1);
  xhr.dispatchEvent(new Event("loadend"));
  assert.equal(t.pending(0), 0);
});

test("a body read that fails leaves no count behind", async () => {
  const t = setup();
  const text = t.Response.prototype.text;
  await assert.rejects(text.call({}));
  assert.equal(t.pending(0), 0);
});

test("a second send the browser refuses keeps counting the request in flight", () => {
  const t = setup(StrictXHR);
  const xhr = new t.window.XMLHttpRequest();
  xhr.open();
  xhr.send();
  assert.throws(() => xhr.send());
  assert.equal(t.pending(0), 1);
  xhr.dispatchEvent(new Event("loadend"));
  assert.equal(t.pending(0), 0);
});

const ok = (body: string, extra: Record<string, unknown> = {}) => ({
  ok: true,
  status: 200,
  statusText: "OK",
  url: "https://site.test/models-json?q=1",
  headers: [["Content-Type", "application/json"], ["X-Total", "3"]],
  body: new TextEncoder().encode(body).buffer,
  ...extra,
});
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
// Objects from the page script's realm, for deepEqual.
const plain = (value: unknown) => JSON.parse(JSON.stringify(value));

test("a fetch to the page's own site goes through the panel and reads as same-origin", async () => {
  const t = setup(NativeXHR, "https://site.test/models");
  const request = t.window.fetch("/models-json?q=1");
  await tick();
  assert.equal(t.native.length, 0);
  const sent = t.posted[0];
  assert.equal(sent.type, "request");
  // Resolved against the page's address, not the shell's.
  assert.equal(sent.url, "https://site.test/models-json?q=1");
  assert.equal(sent.method, "GET");
  assert.equal(sent.body, null);
  // Counted until answered, as a native request is.
  assert.equal(t.pending(0), 1);
  t.reply(sent.id, ok('{"models":[1]}'));
  const response = await request;
  assert.equal(t.pending(0), 0);
  assert.equal(response.status, 200);
  assert.equal(response.type, "basic");
  assert.equal(response.url, "https://site.test/models-json?q=1");
  assert.equal(response.redirected, false);
  // Same-origin: every header is readable, not only the CORS-safelisted ones.
  assert.equal(response.headers.get("x-total"), "3");
  assert.equal(response.clone().url, "https://site.test/models-json?q=1");
  assert.deepEqual(await response.json(), { models: [1] });
});

test("a page whose site answers any origin keeps its own requests, which the browser can read", async () => {
  const t = setup(NativeXHR, "https://site.test/", true, () => true);
  const own = t.window.fetch("/stream");
  assert.equal(t.native.length, 1);
  const xhr = new t.window.XMLHttpRequest();
  xhr.open("GET", "/api");
  xhr.send();
  await tick();
  assert.equal(t.posted.length, 0);
  assert.deepEqual(xhr.calls.at(-1), ["send", undefined]);
  t.answer(0);
  await own;
});

test("a page whose site answers any origin still relays a GET its API doesn't answer", async () => {
  const t = setup(NativeXHR, "https://site.test/", true, (url) => !url.endsWith("/same"));
  const read = t.window.fetch("/same");
  await tick();
  assert.equal(t.refused.length, 1);
  assert.equal(t.posted.length, 1);
  t.reply(t.posted[0].id, ok('{"a":1}'));
  assert.deepEqual(await (await read).json(), { a: 1 });
  // A write stays the browser's: its failure is the page's, with no panel request.
  await assert.rejects(t.window.fetch("/same", { method: "POST", body: "x" }), TypeError);
  assert.equal(t.posted.length, 1);
  assert.equal(t.refused.length, 2);
});

test("an own-site GET the browser can read stays the browser's, streamed as it was", async () => {
  const t = setup(NativeXHR, "https://site.test/", false, (url) => url.endsWith("/api/models"));
  const read = t.window.fetch("/api/models");
  await tick();
  assert.equal(t.native.length, 1);
  assert.equal(t.posted.length, 0);
  t.answer(0, "[1]");
  assert.deepEqual(await (await read).json(), [1]);
});

test("one the browser can't read goes to the panel, and that address goes there directly after", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  const first = t.window.fetch("/models-json?search=a");
  await tick();
  assert.equal(t.refused.length, 1);
  assert.equal(t.posted.length, 1);
  t.reply(t.posted[0].id, ok("{}"));
  await first;
  const second = t.window.fetch("/models-json?search=ab");
  await tick();
  // Same address, another query: no second try in the browser.
  assert.equal(t.refused.length, 1);
  assert.equal(t.posted.length, 2);
  t.reply(t.posted[1].id, ok("{}"));
  await second;
});

test("a write never tries the browser first", async () => {
  const t = setup(NativeXHR, "https://site.test/", false, () => true);
  const write = t.window.fetch("/api", { method: "POST", body: "x" });
  await tick();
  assert.equal(t.native.length, 0);
  assert.equal(t.posted.length, 1);
  t.reply(t.posted[0].id, ok("{}"));
  await write;
});

test("an XHR GET the browser can read loads the browser's answer, with no panel request", async () => {
  const t = setup(NativeXHR, "https://site.test/", false, () => true);
  const xhr = new t.window.XMLHttpRequest() as PageXHR;
  xhr.open("GET", "/poll");
  xhr.send();
  await tick();
  assert.equal(t.posted.length, 0);
  t.answer(0, "late");
  await tick();
  await tick();
  const [open] = xhr.calls.slice(-2);
  assert.match(String(open[2]), /^blob:/);
});

test("requests to other sites, and from a page with no address, stay the browser's", async () => {
  const t = setup(NativeXHR, "https://site.test/models");
  const other = t.window.fetch("https://api.other.test/x");
  assert.equal(t.native.length, 1);
  assert.equal(t.posted.length, 0);
  t.answer(0);
  await other;
  const local = setup();
  const file = local.window.fetch("/x");
  assert.equal(local.native.length, 1);
  local.answer(0);
  await file;
});

test("a fetch body reaches the panel with the headers the browser would send", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  void t.window.fetch("/api", { method: "POST", headers: { "X-Test": "1" }, body: "hello", integrity: "sha256-abc" });
  await tick();
  await tick();
  const sent = t.posted[0];
  assert.equal(sent.method, "POST");
  assert.equal(sent.integrity, "sha256-abc");
  assert.equal(new TextDecoder().decode(sent.body as ArrayBuffer), "hello");
  assert.deepEqual(Object.fromEntries(sent.headers as Array<[string, string]>), {
    "content-type": "text/plain;charset=UTF-8",
    "x-test": "1",
  });
});

test("a failed relay is fetch's network error; an abort rejects and tells the panel", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  const failing = t.window.fetch("/a");
  await tick();
  t.reply(t.posted[0].id, { ok: false });
  await assert.rejects(failing, (error: Error) => error.name === "TypeError");

  const controller = new AbortController();
  const aborted = t.window.fetch("/b", { signal: controller.signal});
  await tick();
  controller.abort();
  await assert.rejects(aborted, (error: Error) => error.name === "AbortError");
  assert.deepEqual(plain(t.posted.at(-1)), { type: "requestAbort", id: t.posted[1].id });
  assert.equal(t.pending(0), 0);
});

test("an answer from anything but the shell is ignored", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  let settled = false;
  void t.window.fetch("/a").then(() => (settled = true));
  await tick();
  t.reply(t.posted[0].id, ok("{}"), {});
  await tick();
  assert.equal(settled, false);
});

type PageXHR = NativeXHR & {
  readyState: number;
  status: number;
  responseURL: string;
  setRequestHeader: (name: string, value: string) => void;
  getResponseHeader: (name: string) => string | null;
  getAllResponseHeaders: () => string;
};

test("an XHR to the page's site loads the panel's answer and shows the site's status and headers", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  const xhr = new t.window.XMLHttpRequest() as PageXHR;
  const events: string[] = [];
  for (const type of ["loadstart", "readystatechange", "load", "loadend"]) xhr.addEventListener(type, () => events.push(type));
  xhr.open("GET", "/api");
  xhr.setRequestHeader("X-Test", "1");
  xhr.send();
  await tick();
  assert.deepEqual(events, ["loadstart"]);
  const sent = t.posted[0];
  assert.deepEqual(plain(sent.headers), [["X-Test", "1"]]);
  // OPENED while the panel makes it, as one in flight.
  assert.equal(xhr.readyState, 1);
  t.reply(sent.id, ok("{}", { status: 404, statusText: "Not Found" }));
  await tick();
  // The browser now loads the answer from a blob: URL.
  const [open, send] = xhr.calls.slice(-2);
  assert.equal(open[0], "open");
  assert.match(String(open[2]), /^blob:/);
  assert.deepEqual(send, ["send"]);
  xhr.state = 4;
  assert.equal(xhr.status, 404);
  assert.equal(xhr.responseURL, "https://site.test/models-json?q=1");
  assert.equal(xhr.getResponseHeader("X-TOTAL"), "3");
  assert.equal(xhr.getAllResponseHeaders(), "content-type: application/json\r\nx-total: 3\r\n");
  assert.equal(t.pending(0), 1);
  xhr.dispatchEvent(new Event("loadend"));
  assert.equal(t.pending(0), 0);
});

test("an XHR the panel can't make ends as a network error", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  const xhr = new t.window.XMLHttpRequest() as PageXHR;
  const events: string[] = [];
  for (const type of ["readystatechange", "error", "loadend"]) xhr.addEventListener(type, () => events.push(`${type}:${xhr.readyState}`));
  xhr.open("GET", "/api");
  xhr.send();
  await tick();
  t.reply(t.posted[0].id, { ok: false });
  await tick();
  assert.deepEqual(events, ["readystatechange:4", "error:4", "loadend:4"]);
  assert.equal(xhr.status, 0);
  assert.equal(t.pending(0), 0);
  assert.throws(() => xhr.send(), (error: Error) => error.name === "InvalidStateError");
});

test("aborting an XHR the panel is making ends it as the browser would", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  const xhr = new t.window.XMLHttpRequest() as PageXHR;
  const events: string[] = [];
  for (const type of ["readystatechange", "abort", "loadend"]) xhr.addEventListener(type, () => events.push(`${type}:${xhr.readyState}`));
  xhr.open("GET", "/api");
  xhr.send();
  await tick();
  xhr.abort();
  assert.deepEqual(events, ["readystatechange:4", "abort:4", "loadend:4"]);
  assert.equal(xhr.readyState, 0);
  assert.deepEqual(plain(t.posted.at(-1)), { type: "requestAbort", id: t.posted[0].id });
  // A late answer changes nothing.
  t.reply(t.posted[0].id, ok("{}"));
  await tick();
  assert.equal(xhr.calls.filter(([name]) => name === "send").length, 0);
});

test("a synchronous XHR, which can't wait for the panel, stays the browser's", () => {
  const t = setup(NativeXHR, "https://site.test/");
  const xhr = new t.window.XMLHttpRequest() as PageXHR;
  xhr.open("GET", "/api", false);
  xhr.send();
  assert.equal(t.posted.length, 0);
  assert.deepEqual(xhr.calls.at(-1), ["send", undefined]);
});

test("a fetch's redirect mode reaches the panel, and manual stops at an opaque redirect", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  const request = t.window.fetch("/login", { redirect: "manual"});
  await tick();
  assert.equal(t.posted[0].redirect, "manual");
  t.reply(t.posted[0].id, { ...ok(""), opaqueRedirect: true, status: 0, headers: [], url: "https://site.test/login" });
  const response = await request;
  assert.equal(response.type, "opaqueredirect");
  assert.equal(response.status, 0);
  assert.equal(response.url, "https://site.test/login");
  assert.equal([...response.headers].length, 0);
});

test("a response a redirect brought from another site reads as a CORS one", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  const request = t.window.fetch("/download");
  await tick();
  t.reply(t.posted[0].id, ok("{}", { url: "https://cdn.test/file", redirected: true, cors: true }));
  const response = await request;
  assert.equal(response.type, "cors");
  assert.equal(response.redirected, true);
  assert.equal(response.url, "https://cdn.test/file");
});

test("a chain that went through another site and back is still a redirected CORS response", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  const request = t.window.fetch("/models-json?q=1");
  await tick();
  t.reply(t.posted[0].id, ok("{}", { redirected: true, cors: true }));
  const response = await request;
  assert.equal(response.url, "https://site.test/models-json?q=1");
  assert.equal(response.type, "cors");
  assert.equal(response.redirected, true);
});

test("an XHR's Document body is sent serialized, as UTF-8 HTML or XML", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  for (const [document, markup, type] of [
    [new PageDocument("application/xml", "<envelope/>"), '<?xml version="1.0"?><envelope/>', "application/xml;charset=UTF-8"],
    [new PageDocument("text/html", "<html><body>hi</body></html>"), "<!DOCTYPE html><html><body>hi</body></html>", "text/html;charset=UTF-8"],
  ] as const) {
    const xhr = new t.window.XMLHttpRequest() as PageXHR;
    xhr.open("POST", "/soap");
    xhr.send(document as unknown as string);
    await tick();
    const sent = t.posted.at(-1) as Record<string, unknown>;
    assert.equal(new TextDecoder().decode(sent.body as ArrayBuffer), markup);
    assert.deepEqual(plain(sent.headers), [["Content-Type", type]]);
  }
});

test("an XHR's text body relabels the page's charset as the UTF-8 it is sent in", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  for (const [type, body, sent] of [
    ["text/plain;charset=iso-8859-1", "caf\u00e9", "text/plain;charset=UTF-8"],
    ['text/plain; charset="Shift_JIS"; foo=bar', "x", 'text/plain; charset="UTF-8"; foo=bar'],
    ["text/plain;Charset=latin1;format=flowed", new URLSearchParams("a=1"), "text/plain;Charset=UTF-8;format=flowed"],
    ["text/plain", "x", "text/plain"],
    ["application/octet-stream;charset=latin1", new Uint8Array([233]).buffer, "application/octet-stream;charset=latin1"],
  ] as const) {
    const xhr = new t.window.XMLHttpRequest() as PageXHR;
    xhr.open("POST", "/api");
    xhr.setRequestHeader("Content-Type", type);
    xhr.send(body as unknown as string);
    await tick();
    assert.deepEqual(plain((t.posted.at(-1) as Record<string, unknown>).headers), [["Content-Type", sent]]);
  }
});

test("an XHR always follows redirects, as the browser's does", async () => {
  const t = setup(NativeXHR, "https://site.test/");
  const xhr = new t.window.XMLHttpRequest() as PageXHR;
  xhr.open("GET", "/api");
  xhr.send();
  await tick();
  assert.equal(t.posted[0].redirect, "follow");
});
