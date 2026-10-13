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

/** A stand-in for the browser's own XHR. Its state is on prototype accessors, as on the real one, so the
 *  subclass's getters aren't shadowed by own properties. */
class NativeXHR extends EventTarget {
  #state = 0;
  #type = "";
  get readyState() {
    return this.#state;
  }
  get responseType() {
    return this.#type;
  }
  set responseType(value: string) {
    this.#type = value;
  }
  get withCredentials() {
    return false;
  }
  get timeout() {
    return 0;
  }
  open() {
    this.#state = 1;
  }
  setRequestHeader() {}
  send() {
    nativeSends.push(this);
  }
  abort() {
    if (this.#state === 4) this.#state = 0;
  }
}
/** Every XHR the browser itself was asked to send. */
const nativeSends: NativeXHR[] = [];

function setup() {
  const posted: Array<Record<string, unknown>> = [];
  const native: Request[] = [];
  let now = 0;
  // The script patches its Response's body readers: a class of its own keeps that out of other tests.
  class PageResponse extends Response {}
  const parent = {};
  const window = Object.assign(new EventTarget(), {
    fetch: async (request: Request) => {
      native.push(request);
      return new PageResponse("native");
    },
    XMLHttpRequest: NativeXHR,
  });
  class Progress extends Event {}
  runInNewContext(source, {
    window, parent, cfg: { url: "https://huggingface.co/models" }, document: { baseURI: "https://huggingface.co/models" },
    post: (message: Record<string, unknown>) => posted.push(message), Date: { now: () => now },
    Request, Response: PageResponse, Headers, URL, Uint8Array, TextDecoder, AbortController, DOMException, Event, ProgressEvent: Progress,
    Blob, atob, setTimeout, clearTimeout, Symbol, Object, JSON, String, Promise, Array, Map, Set, Math, Number, TypeError,
  });
  const reply = (id: unknown, result: unknown, source: unknown = parent) =>
    window.dispatchEvent(Object.assign(new Event("message"), { source, data: { type: "unsloth:browser-command", command: "resourceResult", id, result } }));
  const ok = (body: string, status = 200) => ({
    status, url: "https://huggingface.co/models-json", headers: { "content-type": "application/json" }, body: btoa(body),
  });
  const pending = (since: number) => (window as unknown as Record<symbol, (since: number) => number>)[Symbol.for("unsloth.browser.pending")](since);
  const page = window as unknown as PageWindow;
  return { window: page, posted, native, reply, ok, pending, advance: (ms: number) => (now += ms) };
}

/** What the page sees once the shell script has run. */
type PageWindow = {
  fetch: (input: string, init?: RequestInit) => Promise<Response>;
  XMLHttpRequest: new () => XMLHttpRequest;
};

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

test("a same-origin GET goes to the panel and comes back as the site's response", async () => {
  const t = setup();
  const response = t.window.fetch("/models-json?search=qwen");
  await flush();
  assert.deepEqual(
    { type: t.posted[0].type, url: t.posted[0].url, headers: t.posted[0].headers },
    { type: "resource", url: "https://huggingface.co/models-json?search=qwen", headers: {} },
  );
  // Only the shell's parent answers; the page posting to itself can't.
  t.reply(t.posted[0].id, t.ok('{"models":["spoofed"]}'), {});
  t.reply(t.posted[0].id, t.ok('{"models":["Qwen/Qwen3-VL-8B-Instruct"]}', 404));
  const result = await response;
  assert.equal(result.status, 404);
  assert.equal(result.url, "https://huggingface.co/models-json");
  assert.deepEqual(await result.json(), { models: ["Qwen/Qwen3-VL-8B-Instruct"] });
});

test("a failed read is a network error to the page, as a refused CORS request is", async () => {
  const t = setup();
  const response = t.window.fetch("/models-json");
  await flush();
  t.reply(t.posted[0].id, { error: "HTTP 502" });
  await assert.rejects(response, (error: Error) => error.name === "TypeError");
});

test("writes, credentials, other sites, ranges and conditionals keep the native request", async () => {
  const t = setup();
  await t.window.fetch("/api/settings", { method: "POST", body: "x" });
  await t.window.fetch("/models-json", { credentials: "include" });
  await t.window.fetch("/models-json", { headers: { Authorization: "Bearer x" } });
  await t.window.fetch("/weights.bin", { headers: { Range: "bytes=0-1023" } });
  await t.window.fetch("/models-json", { headers: { "If-None-Match": '"abc"' } });
  await t.window.fetch("https://api.github.com/repos");
  // Past what the panel takes: native at once, rather than a relayed read nobody answers.
  await t.window.fetch("/models-json", { headers: { "X-Big": "v".repeat(5000) } });
  await t.window.fetch("/models-json", { headers: Object.fromEntries(Array.from({ length: 33 }, (_, i) => [`x-${i}`, "v"])) });
  assert.equal(t.posted.length, 0);
  assert.equal(t.native.length, 8);
  assert.equal(t.native[3].headers.get("range"), "bytes=0-1023");
});

test("the page's other headers go along with a relayed read", async () => {
  const t = setup();
  const relayed = t.window.fetch("/models-json", {
    headers: { Accept: "application/json", "X-Request-Id": "Root=1-abc", "X-Requested-With": "XMLHttpRequest" },
  });
  await flush();
  assert.deepEqual(t.posted[0].headers, {
    accept: "application/json", "x-request-id": "Root=1-abc", "x-requested-with": "XMLHttpRequest",
  });
  t.reply(t.posted[0].id, t.ok("{}"));
  await relayed;
});

test("a relayed XHR takes its headers along, repeats joined as the browser joins them", async () => {
  const t = setup();
  const xhr = new t.window.XMLHttpRequest();
  xhr.open("GET", "/models-json");
  xhr.setRequestHeader("X-Requested-With", "XMLHttpRequest");
  xhr.setRequestHeader("X-Tag", "a");
  xhr.setRequestHeader("X-Tag", "b");
  const done = new Promise((resolve) => xhr.addEventListener("loadend", resolve));
  xhr.send();
  assert.deepEqual(t.posted[0].headers, { "x-requested-with": "XMLHttpRequest", "x-tag": "a, b" });
  t.reply(t.posted[0].id, t.ok("{}"));
  await done;
  assert.equal(xhr.status, 200);
});

test("an XHR asking for a range goes out natively, header and all", async () => {
  const t = setup();
  const xhr = new t.window.XMLHttpRequest();
  xhr.open("GET", "/weights.bin");
  xhr.setRequestHeader("Range", "bytes=0-1023");
  xhr.send();
  assert.equal(t.posted.length, 0);
  assert.equal(nativeSends.at(-1), xhr);
});

test("only requests started since the action count, however long they stay open", async () => {
  const t = setup();
  const stream = t.window.fetch("/stream");
  await flush();
  // A long poll the page already had open isn't the action's.
  t.advance(1000);
  assert.equal(t.pending(1000), 0);
  // Cross-origin requests count too: a list fetched from another API is still content on its way.
  const api = t.window.fetch("https://api.github.com/x");
  t.advance(60_000);
  assert.equal(t.pending(1000), 1);
  assert.equal(t.pending(0), 2);
  t.reply(t.posted[0].id, t.ok("{}"));
  await Promise.all([stream, api]);
  assert.equal(t.pending(0), 0);
});

test("reading a response's body still counts, since the page draws only once it has it", async () => {
  const t = setup();
  const response = await t.window.fetch("https://api.github.com/repos/x/git/trees/y");
  assert.equal(t.pending(0), 0);
  const body = response.text();
  assert.equal(t.pending(0), 1);
  assert.equal(await body, "native");
  assert.equal(t.pending(0), 0);
});

test("an XHR keeps the responseType set before open and fires its events in order", async () => {
  const t = setup();
  const xhr = new t.window.XMLHttpRequest();
  xhr.responseType = "json";
  xhr.open("GET", "/models-json");
  const states: number[] = [];
  xhr.addEventListener("readystatechange", () => states.push(xhr.readyState));
  const done = new Promise((resolve) => xhr.addEventListener("loadend", resolve));
  xhr.send();
  assert.equal(t.pending(0), 1);
  t.reply(t.posted[0].id, t.ok('{"models":[{"id":"Qwen/Qwen3-VL-8B-Instruct"}]}'));
  await done;
  assert.deepEqual(states, [2, 3, 4]);
  assert.equal(xhr.status, 200);
  assert.equal(xhr.responseType, "json");
  assert.equal(xhr.response.models[0].id, "Qwen/Qwen3-VL-8B-Instruct");
  assert.equal(xhr.getResponseHeader("Content-Type"), "application/json");
  await flush();
  assert.equal(t.pending(0), 0);
});

test("reopening an XHR mid-flight drops the old read without events; HTML is a document only on request", async () => {
  const t = setup();
  const xhr = new t.window.XMLHttpRequest();
  const events: string[] = [];
  for (const type of ["abort", "load", "loadend"]) xhr.addEventListener(type, () => events.push(type));
  xhr.open("GET", "/first");
  xhr.send();
  xhr.open("GET", "/fragment");
  xhr.send();
  const [first, second] = t.posted.filter((message) => message.type === "resource");
  assert.ok(t.posted.some((message) => message.type === "resourceCancel" && message.id === first.id));
  t.reply(first.id, t.ok("stale"));
  t.reply(second.id, { status: 200, url: "https://huggingface.co/fragment", headers: { "content-type": "text/html" }, body: btoa("<p>new</p>") });
  await flush();
  assert.deepEqual(events, ["load", "loadend"]);
  assert.equal(xhr.responseText, "<p>new</p>");
  assert.equal(xhr.responseXML, null);
});

test("an XHR aborted mid-flight cancels the panel's read and never loads", async () => {
  const t = setup();
  const xhr = new t.window.XMLHttpRequest();
  const events: string[] = [];
  for (const type of ["abort", "load", "loadend"]) xhr.addEventListener(type, () => events.push(type));
  xhr.open("GET", "/models-json");
  xhr.send();
  await flush();
  xhr.abort();
  t.reply(t.posted[0].id, t.ok("{}"));
  await flush();
  assert.deepEqual(events, ["abort", "loadend"]);
  assert.ok(t.posted.some((message) => message.type === "resourceCancel" && message.id === t.posted[0].id));
  assert.equal(xhr.readyState, 0);
});

test("a relayed XHR decodes text in the response's charset, and JSON as UTF-8", async () => {
  const t = setup();
  const read = async (type: string, contentType: string, bytes: string) => {
    const xhr = new t.window.XMLHttpRequest();
    xhr.open("GET", "/legacy");
    xhr.responseType = type as XMLHttpRequestResponseType;
    const done = new Promise((resolve) => xhr.addEventListener("loadend", resolve));
    xhr.send();
    const sent = t.posted.filter((message) => message.type === "resource").at(-1);
    t.reply(sent?.id, { status: 200, url: "https://huggingface.co/legacy", headers: { "content-type": contentType }, body: btoa(bytes) });
    await done;
    return xhr;
  };
  assert.equal((await read("", "text/plain; charset=ISO-8859-1", "caf\xe9")).responseText, "café");
  assert.equal((await read("", 'text/plain; charset="windows-1251"', "\xcf\xf0\xe8")).responseText, "При");
  assert.equal((await read("", "text/plain; charset=no-such-charset", "caf\xc3\xa9")).responseText, "café");
  assert.deepEqual((await read("json", "application/json; charset=iso-8859-1", '{"a":"\xc3\xa9"}')).response, { a: "é" });
});
